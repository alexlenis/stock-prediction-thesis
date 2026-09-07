import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.predict_engine import (
    predict_stock,
    get_cached_data,
    invalidate_cache,
    create_features,
    build_lstm_features,
    load_latest_sentiment,
    rf, xgb, lgbm, log_model, log_scaler,
    lstm_cls_model, lstm_cls_scaler,
    LSTM_FEATURES, compute_rsi,
)
from src.backtest_engine import run_backtest

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "stock_prediction_db"
analyzer  = SentimentIntensityAnalyzer()

SOURCE_CONFIG = {
    "benzinga":      {"label": "Benzinga",      "reliability": 0.85},
    "cnbc":          {"label": "CNBC",          "reliability": 0.90},
    "marketwatch":   {"label": "MarketWatch",   "reliability": 0.80},
    "finviz":        {"label": "FinViz",        "reliability": 0.80},
    "yahoo_finance": {"label": "Yahoo Finance", "reliability": 0.85},
    "reddit":        {"label": "Reddit",        "reliability": 0.60},
}

def classify_sentiment(avg_score):
    if avg_score >  0.15: return "Bullish", "#22c55e"
    if avg_score < -0.15: return "Bearish", "#ef4444"
    return "Neutral", "#eab308"

def next_trading_day(d):
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d

def build_full_features(df, ticker: str = ""):
    df = df.copy()
    df["Return"]       = df["Close"].pct_change()
    df["MA_5"]         = df["Close"].rolling(5).mean()
    df["MA_10"]        = df["Close"].rolling(10).mean()
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Momentum_5"]   = df["Close"] - df["Close"].shift(5)
    df["Return_lag_1"] = df["Return"].shift(1)
    df["Return_lag_2"] = df["Return"].shift(2)
    df["RSI_14"]       = compute_rsi(df["Close"])
    sent = load_latest_sentiment(ticker) if ticker else {}
    for col in ["sent_mean", "sent_std", "sent_count",
                "sentiment_lag_1", "sentiment_lag_2",
                "sentiment_3d", "sentiment_7d", "sentiment_momentum"]:
        if col not in df.columns:
            df[col] = sent.get(col, 0.0)
    return df.dropna()

# ── Ensemble / Meta Decision Engine ─────────────────────────────────
MODEL_WEIGHTS = {
    "rf":       0.18,
    "xgb":      0.24,
    "lgbm":     0.24,
    "logistic": 0.12,
    "lstm":     0.22,
}
SIGNAL_SCORE = {"SELL": -1.0, "HOLD": 0.0, "BUY": 1.0}
SCORE_SIGNAL = [(-0.33, "SELL"), (0.33, "HOLD"), (1.01, "BUY")]


def _safe_number(value, fallback=0.0):
    try:
        if value is None:
            return fallback
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return fallback
        return v
    except Exception:
        return fallback


def _recent_performance(last10):
    if not last10:
        return None
    valid = [d for d in last10 if isinstance(d, dict) and "correct" in d]
    if not valid:
        return None
    return round(sum(1 for d in valid if d.get("correct")) / len(valid) * 100, 1)


def _score_to_signal(score):
    for threshold, signal in SCORE_SIGNAL:
        if score < threshold:
            return signal
    return "BUY"


def build_ensemble_decision(models: dict):
    """
    Meta decision layer over the five base models.
    It combines static model diversity weights with current confidence,
    historical accuracy, and recent last-10 performance.
    """
    if not models:
        return None

    votes = {"BUY": 0, "HOLD": 0, "SELL": 0}
    weighted_scores = []
    contribution_rows = []

    for key, model in models.items():
        if key not in MODEL_WEIGHTS or not isinstance(model, dict):
            continue

        signal = model.get("signal") or "HOLD"
        if signal not in SIGNAL_SCORE:
            signal = "HOLD"

        base_weight = MODEL_WEIGHTS[key]
        confidence = _safe_number(model.get("confidence"), 50.0)
        accuracy = _safe_number(model.get("accuracy"), 50.0)
        recent = _recent_performance(model.get("last10"))
        recent_val = _safe_number(recent, accuracy)

        # Hybrid reliability: long-term quality + current confidence + recent behavior.
        reliability = (0.40 * accuracy) + (0.40 * confidence) + (0.20 * recent_val)
        reliability_factor = max(0.35, min(reliability / 100.0, 1.15))
        final_weight = base_weight * reliability_factor
        signed_score = SIGNAL_SCORE[signal] * final_weight

        votes[signal] += 1
        weighted_scores.append(signed_score)
        contribution_rows.append({
            "key": key,
            "signal": signal,
            "base_weight": round(base_weight, 3),
            "effective_weight": round(final_weight, 3),
            "confidence": round(confidence, 1),
            "accuracy": round(accuracy, 1) if model.get("accuracy") is not None else None,
            "recent_performance": recent,
            "reliability": round(reliability, 1),
            "impact": round(signed_score, 3),
        })

    total_effective_weight = sum(abs(r["effective_weight"]) for r in contribution_rows) or 1.0
    normalized_score = sum(weighted_scores) / total_effective_weight
    final_signal = _score_to_signal(normalized_score)

    agreement = round((votes[final_signal] / max(1, sum(votes.values()))) * 100, 1)
    avg_conf = np.mean([r["confidence"] for r in contribution_rows]) if contribution_rows else 50.0
    avg_rel = np.mean([r["reliability"] for r in contribution_rows]) if contribution_rows else 50.0

    # Confidence becomes stronger when models agree and reliability is good.
    meta_confidence = round((0.45 * avg_conf) + (0.35 * avg_rel) + (0.20 * agreement), 1)

    if agreement < 45 or meta_confidence < 45:
        risk = "High"
        risk_color = "#ef4444"
    elif agreement < 70 or meta_confidence < 62:
        risk = "Medium"
        risk_color = "#eab308"
    else:
        risk = "Low"
        risk_color = "#22c55e"

    if normalized_score > 0.35:
        bias = "Bullish"
        bias_color = "#22c55e"
    elif normalized_score < -0.35:
        bias = "Bearish"
        bias_color = "#ef4444"
    else:
        bias = "Neutral"
        bias_color = "#eab308"

    return {
        "signal": final_signal,
        "confidence": meta_confidence,
        "agreement": agreement,
        "risk": risk,
        "risk_color": risk_color,
        "bias": bias,
        "bias_color": bias_color,
        "score": round(normalized_score, 3),
        "votes": votes,
        "method": "Hybrid Weighted Ensemble",
        "weights": MODEL_WEIGHTS,
        "contributions": sorted(contribution_rows, key=lambda x: abs(x["impact"]), reverse=True),
        "summary": f"{votes['BUY']} BUY / {votes['HOLD']} HOLD / {votes['SELL']} SELL",
    }


# ── /predict/{ticker} ────────────────────────────────────────────────
@app.get("/predict/{ticker}")
def predict(ticker: str):
    result = predict_stock(ticker)
    history = result.get("history", [])
    future  = result.get("forecast", [])
    upper   = result.get("confidence_upper", [])
    lower   = result.get("confidence_lower", [])

    if not history:
        return {"error": "No history data"}

    # Generate trading-day dates only (skip weekends)
    current_date     = datetime.strptime(history[-1]["Date"], "%Y-%m-%d")
    forecast         = []
    confidence_upper = []
    confidence_lower = []

    for i, price in enumerate(future):
        current_date = next_trading_day(current_date)
        date_str     = current_date.strftime("%Y-%m-%d")
        forecast.append({"Date": date_str, "Close": float(price)})
        if i < len(upper):
            confidence_upper.append({"Date": date_str, "Close": float(upper[i])})
        if i < len(lower):
            confidence_lower.append({"Date": date_str, "Close": float(lower[i])})

    return {
        "history":           history,
        "forecast":          forecast,
        "confidence_upper":  confidence_upper,
        "confidence_lower":  confidence_lower,
        "forecast_horizon":  result.get("forecast_horizon", len(forecast)),
        "rf":                result.get("rf"),
        "xgb":               result.get("xgb"),
        "lgbm":              result.get("lgbm"),
        "logistic":          result.get("logistic"),
        "lstm":              result.get("lstm"),
        "final":             result.get("final"),
        "risk_metrics":      result.get("risk_metrics", {}),
    }


# ── /refresh/{ticker} ────────────────────────────────────────────────
@app.post("/refresh/{ticker}")
def refresh_data(ticker: str):
    """Force re-download of price data from yfinance, bypassing cache."""
    invalidate_cache(ticker)
    result = predict_stock(ticker)
    return {"status": "refreshed", "last_date": result["history"][-1]["Date"] if result.get("history") else None}


# ── /backtest/{ticker} ───────────────────────────────────────────────
@app.get("/backtest/{ticker}")
def backtest(ticker: str):
    try:
        return run_backtest(ticker)
    except Exception as e:
        print(f"[BACKTEST ERROR] {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


# ── /shap/{ticker} ───────────────────────────────────────────────────
@app.get("/shap/{ticker}")
def shap_values(ticker: str):
    """
    Compute SHAP values using TreeExplainer on XGBoost.
    Returns signed contributions per feature for today's prediction:
      positive  → pushes toward BUY
      negative  → pushes toward SELL
    """
    import shap as shap_lib

    df_raw = get_cached_data(ticker)
    df     = create_features(df_raw)
    sent   = load_latest_sentiment(ticker)

    FEATURES = list(rf.feature_names_in_)
    row  = df.iloc[-1]
    fdata = {}
    for col in FEATURES:
        if col in sent:
            fdata[col] = sent[col]
        elif col in df.columns:
            try:    fdata[col] = float(row[col])
            except: fdata[col] = 0.0
        else:
            fdata[col] = 0.0

    X = pd.DataFrame([fdata])

    try:
        # TreeExplainer is exact and fast for tree-based models
        explainer   = shap_lib.TreeExplainer(xgb)
        shap_vals   = explainer.shap_values(X)   # shape: (3, 1, n_feat) or (1, n_feat)

        # TreeExplainer output for multi-class XGBoost: ndarray (n_samples, n_features, n_classes)
        sv = np.array(shap_vals)
        if sv.ndim == 3:
            buy_shap = sv[0, :, 2]    # sample 0, all features, BUY class (index 2)
        elif sv.ndim == 2:
            buy_shap = sv[0, :]       # single class fallback
        elif isinstance(shap_vals, list) and len(shap_vals) >= 3:
            buy_shap = np.array(shap_vals[2]).flatten()[:len(FEATURES)]
        else:
            buy_shap = sv.flatten()[:len(FEATURES)]

        shap_df = pd.DataFrame({
            "feature": FEATURES,
            "value":   X.iloc[0].values.tolist(),
            "shap":    buy_shap,
        })
        shap_df["abs_shap"] = shap_df["shap"].abs()
        top15 = shap_df.nlargest(15, "abs_shap")

        return {
            "features":     top15[["feature", "value", "shap"]].to_dict(orient="records"),
            "model":        "XGBoost (TreeExplainer SHAP)",
            "target_class": "BUY class — positive = pushes toward BUY",
        }

    except Exception as e:
        print(f"[SHAP ERROR] {e}")
        import traceback; traceback.print_exc()
        # Signed fallback: RF importances × sign of feature deviation from mean
        try:
            ds_path = os.path.join(ROOT_DIR, "data", "processed", "final_dataset.csv")
            ds_means = pd.read_csv(ds_path, usecols=FEATURES).mean()
            imp = rf.feature_importances_
            signed = []
            for i, col in enumerate(FEATURES):
                deviation = float(X.iloc[0][col]) - float(ds_means.get(col, 0))
                signed.append(float(imp[i]) * np.sign(deviation))
            shap_df = pd.DataFrame({"feature": FEATURES, "value": X.iloc[0].values, "shap": signed})
            top15 = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=False).index).head(15)
            return {
                "features":     top15.to_dict(orient="records"),
                "model":        "RF importance × deviation (signed fallback)",
                "target_class": "positive = above avg → likely bullish feature",
            }
        except Exception as e2:
            return {"features": [], "error": str(e2)}

# ── /details/{ticker} ────────────────────────────────────────────────
@app.get("/details/{ticker}")
def get_details(ticker: str):
    try:
        import tensorflow as tf

        df_raw = get_cached_data(ticker)
        df     = build_full_features(df_raw, ticker=ticker)

        FEATURES = list(rf.feature_names_in_)
        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        df["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
        df["signal"]        = df["future_return"].apply(
            lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1)
        )
        df = df.dropna()

        X = df[FEATURES].fillna(0)
        y = df["signal"]

        split  = int(len(X) * 0.8)
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

        def model_details(model, X_t, y_t, scaler=None):
            X_input  = scaler.transform(X_t) if scaler else X_t
            pred     = model.predict(X_input)
            acc      = float(accuracy_score(y_t, pred))
            buy_mask = pred == 2
            win_rate = float((y_t[buy_mask] == 2).mean()) if buy_mask.sum() > 0 else 0.0

            try:
                proba      = model.predict_proba(X_input)
                confidence = float(np.mean(np.max(proba, axis=1)))
            except:
                confidence = 0.5

            top_factors = []
            try:
                imp      = model.feature_importances_
                imp_norm = imp / imp.sum()
                top3_idx = np.argsort(imp_norm)[::-1][:3]
                top_factors = [
                    {"name": FEATURES[i], "value": round(float(imp_norm[i]), 3)}
                    for i in top3_idx
                ]
            except:
                try:
                    coef      = np.abs(model.coef_)
                    mean_coef = coef.mean(axis=0)
                    coef_norm = mean_coef / mean_coef.sum()
                    top3_idx  = np.argsort(coef_norm)[::-1][:3]
                    top_factors = [
                        {"name": FEATURES[i], "value": round(float(coef_norm[i]), 3)}
                        for i in top3_idx
                    ]
                except Exception as e:
                    print(f"[COEF ERROR] {e}")

            last10_X     = X.iloc[-10:]
            last10_y     = y.iloc[-10:]
            last10_input = scaler.transform(last10_X) if scaler else last10_X
            last10_pred  = model.predict(last10_input)
            last10 = [
                {
                    "pred":    label_map[int(p)],
                    "actual":  label_map[int(a)],
                    "correct": bool(p == a),
                }
                for p, a in zip(last10_pred, last10_y)
            ]

            latest_X     = X.iloc[[-1]]
            latest_input = scaler.transform(latest_X) if scaler else latest_X
            try:
                cur_proba = model.predict_proba(latest_input)[0]
                cur_conf  = float(np.max(cur_proba))
                # Only fire BUY/SELL when model has majority confidence
                cur_pred  = int(np.argmax(cur_proba)) if cur_conf >= 0.50 else 1
            except:
                cur_pred = int(model.predict(latest_input)[0])
                cur_conf = confidence

            return {
                "signal":      label_map[int(cur_pred)],
                "confidence":  round(cur_conf * 100, 1),
                "accuracy":    round(acc * 100, 1),
                "win_rate":    round(win_rate * 100, 1),
                "top_factors": top_factors,
                "last10":      last10,
            }

        # ── LSTM details ──────────────────────────────────────────────
        def lstm_details():
            try:
                SEQ_LEN    = 10
                N_FEATURES = len(LSTM_FEATURES)

                lstm_df = build_lstm_features(df_raw, ticker=ticker)
                for col in LSTM_FEATURES:
                    if col not in lstm_df.columns:
                        lstm_df[col] = 0.0

                X_lstm = lstm_df[LSTM_FEATURES].values
                n      = len(X_lstm)

                if n < SEQ_LEN + 1:
                    raise ValueError("Not enough data for LSTM")

                scaled = lstm_cls_scaler.transform(X_lstm)

                # Current prediction (latest sequence)
                seq_tensor = tf.constant(
                    scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES),
                    dtype=tf.float32
                )
                pred_proba = lstm_cls_model(seq_tensor, training=False).numpy()[0]
                cur_pred   = int(np.argmax(pred_proba))
                cur_conf   = float(np.max(pred_proba))

                # Accuracy: evaluate on the test split (same 80/20 as tree models)
                # Build sequence dataset from the scaled feature array
                test_start = int(n * 0.8)
                y_arr      = y.values  # aligned to df (not lstm_df), use offset
                y_offset   = len(y_arr) - n

                test_preds, test_actuals = [], []
                for idx in range(test_start, n):
                    if idx < SEQ_LEN:
                        continue
                    s = tf.constant(
                        scaled[idx - SEQ_LEN: idx].reshape(1, SEQ_LEN, N_FEATURES),
                        dtype=tf.float32
                    )
                    p       = lstm_cls_model(s, training=False).numpy()[0]
                    pred_   = int(np.argmax(p))
                    y_idx   = y_offset + idx
                    actual_ = int(y_arr[y_idx]) if 0 <= y_idx < len(y_arr) else 1
                    test_preds.append(pred_)
                    test_actuals.append(actual_)

                acc_val = round(
                    sum(p == a for p, a in zip(test_preds, test_actuals)) / len(test_preds) * 100, 1
                ) if test_preds else None

                buy_preds = [(p, a) for p, a in zip(test_preds, test_actuals) if p == 2]
                win_val   = round(
                    sum(1 for p, a in buy_preds if p == a) / len(buy_preds) * 100, 1
                ) if buy_preds else 0.0

                # Last 10 predictions for the dot display
                last10 = []
                for i in range(10, 0, -1):
                    end_idx = n - i
                    if end_idx < SEQ_LEN:
                        continue
                    s = tf.constant(
                        scaled[end_idx - SEQ_LEN: end_idx].reshape(1, SEQ_LEN, N_FEATURES),
                        dtype=tf.float32
                    )
                    p       = lstm_cls_model(s, training=False).numpy()[0]
                    pred_   = int(np.argmax(p))
                    y_idx   = y_offset + end_idx
                    actual_ = int(y_arr[y_idx]) if 0 <= y_idx < len(y_arr) else 1
                    last10.append({
                        "pred":    label_map[pred_],
                        "actual":  label_map[actual_],
                        "correct": bool(pred_ == actual_),
                    })

                # Gradient-based saliency for the LSTM: |∂ P(predicted class) / ∂ input|
                # averaged over the lookback window → per-feature contribution.
                # (Neural nets lack feature_importances_; saliency is the standard
                #  interpretability analogue.)
                top_factors = []
                try:
                    inp = tf.convert_to_tensor(
                        scaled[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES), dtype=tf.float32
                    )
                    with tf.GradientTape() as tape:
                        tape.watch(inp)
                        out    = lstm_cls_model(inp, training=False)
                        target = out[0, cur_pred]
                    grads = tape.gradient(target, inp).numpy()[0]       # (SEQ_LEN, N_FEATURES)
                    sal   = np.abs(grads).mean(axis=0)                  # (N_FEATURES,)
                    if sal.sum() > 0:
                        sal_norm = sal / sal.sum()
                        top3     = np.argsort(sal_norm)[::-1][:3]
                        top_factors = [
                            {"name": LSTM_FEATURES[i], "value": round(float(sal_norm[i]), 3)}
                            for i in top3
                        ]
                except Exception as e:
                    print(f"[LSTM SALIENCY ERROR] {e}")

                return {
                    "signal":      label_map[cur_pred],
                    "confidence":  round(cur_conf * 100, 1),
                    "accuracy":    acc_val,
                    "win_rate":    win_val,
                    "top_factors": top_factors,
                    "last10":      last10,
                }

            except Exception as e:
                print(f"[LSTM DETAILS ERROR] {e}")
                import traceback; traceback.print_exc()
                return {
                    "signal": "HOLD", "confidence": 50.0,
                    "accuracy": None, "win_rate": None,
                    "top_factors": [], "last10": [],
                }

        models = {
            "rf":       model_details(rf,        X_test, y_test),
            "xgb":      model_details(xgb,       X_test, y_test),
            "lgbm":     model_details(lgbm,      X_test, y_test),
            "logistic": model_details(log_model, X_test, y_test, log_scaler),
            "lstm":     lstm_details(),
        }

        ensemble = build_ensemble_decision(models)

        # Compute training period from final_dataset.csv
        training_period = "2019 – 2026"
        try:
            dataset_path = os.path.join(ROOT_DIR, "data", "processed", "final_dataset.csv")
            ds = pd.read_csv(dataset_path, usecols=["date"])
            year_min = pd.to_datetime(ds["date"]).dt.year.min()
            year_max = pd.to_datetime(ds["date"]).dt.year.max()
            training_period = f"{year_min} – {year_max}"
        except Exception:
            pass

        return {
            "ticker": ticker.upper(),
            "models": models,
            "ensemble": ensemble,
            "training_period": training_period,
        }

    except Exception as e:
        print(f"[DETAILS ERROR] {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}

# ── /signals/{ticker} ────────────────────────────────────────────────
@app.get("/signals/{ticker}")
def get_signals(ticker: str):
    try:
        client     = MongoClient(MONGO_URI)
        collection = client[DB_NAME]["raw_documents"]
        cutoff     = datetime.now(timezone.utc) - timedelta(hours=48)
        cutoff_str = cutoff.isoformat()
        ticker_up  = ticker.upper()

        query = {
            "$and": [
                {"scraped_at": {"$gte": cutoff_str}},
                {"$or": [
                    {"ticker":  ticker_up},
                    {"tickers": ticker_up},
                ]}
            ]
        }

        docs = list(collection.find(query))
        print(f"[SIGNALS] {ticker_up} → {len(docs)} docs in last 48h")

        source_scores: dict[str, list[float]] = {k: [] for k in SOURCE_CONFIG}

        for doc in docs:
            source = doc.get("source", "").lower().strip()
            if source not in SOURCE_CONFIG:
                continue
            text = doc.get("title") or doc.get("text") or ""
            if not text:
                continue
            pre = doc.get("sentiment_score")
            if pre is not None:
                try:    score = float(pre)
                except: score = analyzer.polarity_scores(str(text))["compound"]
            else:
                score = analyzer.polarity_scores(str(text))["compound"]
            weight = SOURCE_CONFIG[source]["reliability"]
            source_scores[source].append(score * weight)

        signals = []
        for source_key, scores in source_scores.items():
            if not scores:
                continue
            avg          = float(np.mean(scores))
            value, color = classify_sentiment(avg)
            cfg          = SOURCE_CONFIG[source_key]
            signals.append({
                "label":       cfg["label"],
                "source":      source_key,
                "value":       value,
                "color":       color,
                "score":       round(avg, 3),
                "count":       len(scores),
                "reliability": cfg["reliability"],
            })

        signals.sort(key=lambda x: x["count"], reverse=True)

        all_scores = [s for v in source_scores.values() for s in v]
        if all_scores:
            comp_val, comp_color = classify_sentiment(float(np.mean(all_scores)))
        else:
            comp_val, comp_color = "Neutral", "#eab308"

        return {
            "signals": signals,
            "composite": {
                "value":          comp_val,
                "color":          comp_color,
                "total_articles": len(docs),
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        print(f"[SIGNALS ERROR] {e}")
        return {
            "signals":   [],
            "composite": {"value": "Neutral", "color": "#eab308", "total_articles": 0},
            "error":     str(e),
        }