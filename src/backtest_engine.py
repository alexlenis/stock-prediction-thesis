"""
Walk-forward backtest: evaluate all 5 trained models on the held-out test split
(last 20% of final_dataset.csv, never seen during training).

Strategy simulated:
  BUY  → long position for the next 5 trading days
  SELL → short position for the next 5 trading days
  HOLD → stay in cash (0% return)

Non-overlapping 5-day windows to avoid autocorrelation in PnL.
"""

import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_BT_CACHE: dict = {}


def _load_models():
    rf         = joblib.load(os.path.join(MODELS_DIR, "rf_final.pkl"))
    xgb        = joblib.load(os.path.join(MODELS_DIR, "xgboost_final.pkl"))
    lgbm       = joblib.load(os.path.join(MODELS_DIR, "lightgbm_final.pkl"))
    log_model  = joblib.load(os.path.join(MODELS_DIR, "logistic_final.pkl"))
    log_scaler = joblib.load(os.path.join(MODELS_DIR, "logistic_scaler.pkl"))
    lstm_model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm_final.keras"))
    lstm_scaler = joblib.load(os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    return rf, xgb, lgbm, log_model, log_scaler, lstm_model, lstm_scaler


def _sharpe(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    daily = np.diff(equity_curve) / equity_curve[:-1]
    if daily.std() < 1e-8:
        return 0.0
    return round(float(np.mean(daily) / daily.std() * np.sqrt(252)), 2)


def _max_drawdown(equity_curve):
    peak = equity_curve[0]
    dd   = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd   = min(dd, (v - peak) / peak)
    return round(float(dd) * 100, 2)


def run_backtest(ticker: str) -> dict:
    if ticker in _BT_CACHE:
        return _BT_CACHE[ticker]

    rf, xgb, lgbm, log_model, log_scaler, lstm_model, lstm_scaler = _load_models()

    ds_path = os.path.join(BASE_DIR, "data", "processed", "final_dataset.csv")
    df = pd.read_csv(ds_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    tdf = df[df["ticker"] == ticker.upper()].copy().reset_index(drop=True)
    if len(tdf) < 30:
        return {"error": f"Not enough data for {ticker}"}

    FEATURES = list(rf.feature_names_in_)
    for col in FEATURES:
        if col not in tdf.columns:
            tdf[col] = 0.0

    tdf = tdf.dropna(subset=["future_return", "signal"]).reset_index(drop=True)

    # Use the SAME per-ticker chronological hold-out the models trained against
    # (is_train column from merge_dataset.py). This makes the backtest genuinely
    # out-of-sample — the models never saw these rows during training.
    if "is_train" in tdf.columns:
        split = int(tdf["is_train"].astype(bool).sum())
    else:
        split = int(len(tdf) * 0.8)
    test_df  = tdf.iloc[split:].copy().reset_index(drop=True)

    if len(test_df) < 10:
        return {"error": "Not enough test data (need ≥ 10 rows)"}

    # Models predict in label space {0:SELL, 1:HOLD, 2:BUY}; the stored signal
    # column is {-1,0,1}. Remap so accuracy/correctness compare like-for-like.
    test_df["signal"] = test_df["signal"].replace({-1: 0, 0: 1, 1: 2})

    X_test = test_df[FEATURES].fillna(0)
    y_true = test_df["signal"].values
    actual_returns = test_df["future_return"].values
    dates = test_df["date"].dt.strftime("%Y-%m-%d").tolist()

    # ── Per-model predictions ────────────────────────────────────────
    rf_preds   = rf.predict(X_test)
    xgb_preds  = xgb.predict(X_test)
    lgbm_preds = lgbm.predict(X_test)
    log_preds  = log_model.predict(log_scaler.transform(X_test))

    # LSTM — sequential predictions on test set
    from src.predict_engine import LSTM_FEATURES
    lstm_preds = []
    SEQ = 10
    lstm_df = tdf.copy()
    for col in LSTM_FEATURES:
        if col not in lstm_df.columns:
            lstm_df[col] = 0.0
    lstm_X = lstm_df[LSTM_FEATURES].values
    scaled  = lstm_scaler.transform(lstm_X)

    for idx in range(split, len(lstm_df)):
        if idx < SEQ:
            lstm_preds.append(1)
            continue
        seq = scaled[idx - SEQ: idx].reshape(1, SEQ, len(LSTM_FEATURES))
        p   = lstm_model(tf.constant(seq, dtype=tf.float32), training=False).numpy()[0]
        lstm_preds.append(int(np.argmax(p)) if float(np.max(p)) >= 0.40 else 1)  # match live operating point
    lstm_preds = np.array(lstm_preds)

    # Ensemble: simple majority vote (4 sklearn + 1 LSTM)
    all_votes = np.stack([rf_preds, xgb_preds, lgbm_preds, log_preds, lstm_preds], axis=1)
    ensemble_preds = np.array([
        np.bincount(row, minlength=3).argmax()
        for row in all_votes
    ])

    # ── Non-overlapping 5-day windows for clean PnL ──────────────────
    window_idx = list(range(0, len(test_df), 5))
    w_dates    = [dates[i]          for i in window_idx]
    w_signals  = [ensemble_preds[i] for i in window_idx]
    w_rf       = [rf_preds[i]       for i in window_idx]
    w_xgb      = [xgb_preds[i]      for i in window_idx]
    w_lgbm     = [lgbm_preds[i]     for i in window_idx]
    w_log      = [log_preds[i]      for i in window_idx]
    w_lstm     = [lstm_preds[i]     for i in window_idx]
    w_returns  = [float(actual_returns[i]) for i in window_idx]
    w_actual   = [int(y_true[i])    for i in window_idx]

    # Equity curves (start $10,000)
    initial = 10_000.0
    strat_eq   = [initial]
    bh_eq      = [initial]
    rf_eq      = [initial]
    xgb_eq_c   = [initial]
    lgbm_eq_c  = [initial]

    trades = []
    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    for sig, ret, date, act, r_s, x_s, l_s in zip(
        w_signals, w_returns, w_dates, w_actual,
        w_rf, w_xgb, w_lgbm
    ):
        pnl_strat = ret if sig == 2 else (-ret if sig == 0 else 0.0)
        pnl_rf    = ret if r_s == 2 else (-ret if r_s == 0 else 0.0)
        pnl_xgb   = ret if x_s == 2 else (-ret if x_s == 0 else 0.0)
        pnl_lgbm  = ret if l_s == 2 else (-ret if l_s == 0 else 0.0)

        strat_eq.append(strat_eq[-1]  * (1 + pnl_strat))
        bh_eq.append(bh_eq[-1]        * (1 + ret))
        rf_eq.append(rf_eq[-1]        * (1 + pnl_rf))
        xgb_eq_c.append(xgb_eq_c[-1] * (1 + pnl_xgb))
        lgbm_eq_c.append(lgbm_eq_c[-1]* (1 + pnl_lgbm))

        if sig != 1:
            trades.append({
                "date":    date,
                "signal":  label_map[sig],
                "return":  round(ret  * 100, 2),
                "pnl":     round(pnl_strat * 100, 2),
                "correct": bool(sig == act),
            })

    strat_arr = np.array(strat_eq)
    bh_arr    = np.array(bh_eq)

    strat_pct = ((strat_arr / initial) - 1) * 100
    bh_pct    = ((bh_arr   / initial) - 1) * 100

    # ── Per-model accuracy on ALL test rows ──────────────────────────
    def acc(pred, true): return round(float(np.mean(pred == true)) * 100, 1)

    model_acc = {
        "RF":       acc(rf_preds,       y_true),
        "XGBoost":  acc(xgb_preds,      y_true),
        "LightGBM": acc(lgbm_preds,     y_true),
        "Logistic": acc(log_preds,      y_true),
        "LSTM":     acc(lstm_preds,     y_true),
        "Ensemble": acc(ensemble_preds, y_true),
    }

    # BUY win rate: how often ensemble BUY predicts price actually rises
    buy_mask = ensemble_preds == 2
    buy_win  = round(float(np.mean(actual_returns[buy_mask] > 0)) * 100, 1) if buy_mask.sum() > 0 else 0.0

    # ── Monthly accuracy ─────────────────────────────────────────────
    test_copy = test_df.copy()
    test_copy["pred"]  = ensemble_preds
    test_copy["month"] = test_copy["date"].dt.to_period("M").astype(str)
    monthly_acc = {}
    for month, grp in test_copy.groupby("month"):
        monthly_acc[month] = round(float(np.mean(grp["pred"].values == grp["signal"].values)) * 100, 1)

    result = {
        "dates":            w_dates,
        "strategy_equity":  [round(x, 2) for x in strat_pct.tolist()[1:]],
        "buyhold_equity":   [round(x, 2) for x in bh_pct.tolist()[1:]],
        "model_accuracy":   model_acc,
        "monthly_accuracy": monthly_acc,
        "trades":           trades[-30:],
        "summary": {
            "test_period":       f"{dates[0]} → {dates[-1]}",
            "test_rows":         len(test_df),
            "total_trades":      len(trades),
            "buy_win_rate":      buy_win,
            "strategy_return":   round(float(strat_pct[-1]), 2),
            "buyhold_return":    round(float(bh_pct[-1]), 2),
            "outperformance":    round(float(strat_pct[-1] - bh_pct[-1]), 2),
            "sharpe_ratio":      _sharpe(strat_arr),
            "max_drawdown":      _max_drawdown(strat_arr),
            "bh_sharpe":         _sharpe(bh_arr),
            "bh_max_drawdown":   _max_drawdown(bh_arr),
        }
    }

    _BT_CACHE[ticker] = result
    return result
