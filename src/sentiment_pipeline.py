"""
Sentiment pipeline: reads all documents from MongoDB, scores them with
FinBERT (domain-specific financial NLP), and aggregates by date+ticker
with reliability-weighted statistics.

FinBERT (ProsusAI/finbert) labels: 0=positive, 1=negative, 2=neutral
Compound score = P(positive) - P(negative), range [-1, 1]

Replaces the original VADER-based pipeline.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymongo import MongoClient

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "sentiment_daily.csv")

# ==============================
# CONFIG
# ==============================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction_db"
COLLECTION_NAME = "raw_documents"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE = 64  # safe for CPU; reduce to 16 if OOM

# ==============================
# LOAD FINBERT
# ==============================
print("[INFO] Loading FinBERT model (downloads ~440 MB on first run)...")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
finbert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert.eval()
print("[INFO] FinBERT ready.")


def finbert_batch(texts: list) -> list:
    """Batch-score texts. Returns compound score = P(pos) - P(neg) for each."""
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,  # financial headlines are short; 128 >> VADER
        )
        with torch.no_grad():
            logits = finbert(**enc).logits
        probs = F.softmax(logits, dim=-1).numpy()
        # id2label: 0=positive, 1=negative, 2=neutral
        scores.extend(float(p[0] - p[1]) for p in probs)
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  [{done}/{len(texts)}] scored...", end="\r", flush=True)
    print()
    return scores


# ==============================
# LOAD DOCUMENTS
# ==============================
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
docs = list(collection.find({}))
print(f"[INFO] Loaded {len(docs)} documents from MongoDB.")

if not docs:
    print("[WARNING] No documents found. Run the scrapers first.")
    sys.exit(0)

# ==============================
# EXTRACT TEXT + METADATA
# Cached docs (finbert_score already set) skip re-scoring.
# ==============================
texts_to_score = []   # texts that need FinBERT
ids_to_update  = []   # MongoDB _ids for those texts
meta_new       = []   # metadata for uncached docs
meta_cached    = []   # metadata for already-scored docs
scores_cached  = []   # scores from cache

for doc in docs:
    title = doc.get("title") or ""
    body  = doc.get("text")  or ""
    text  = (title + " " + body).strip()
    if not text:
        continue

    raw_date = doc.get("published_at") or doc.get("scraped_at")
    try:
        date = pd.to_datetime(raw_date).date()
    except Exception:
        try:
            date = pd.to_datetime(doc.get("scraped_at")).date()
        except Exception:
            continue

    tickers = doc.get("tickers") or (
        [doc.get("ticker")] if doc.get("ticker") else []
    )
    if not tickers:
        continue

    reliability = float(doc.get("reliability_score") or 0.5)
    m = {"date": date, "tickers": tickers, "reliability": reliability}

    if "finbert_score" in doc:
        # Use cached score — no re-inference needed
        meta_cached.append(m)
        scores_cached.append(float(doc["finbert_score"]))
    else:
        texts_to_score.append(text)
        ids_to_update.append(doc["_id"])
        meta_new.append(m)

print(f"[INFO] {len(scores_cached)} docs with cached FinBERT scores.")
print(f"[INFO] {len(texts_to_score)} docs need FinBERT scoring...")

if texts_to_score:
    new_scores = finbert_batch(texts_to_score)
    print(f"[INFO] Scoring complete. Writing scores back to MongoDB...")
    # Persist scores so future runs skip these docs
    ops = [
        {"_id": oid, "score": s}
        for oid, s in zip(ids_to_update, new_scores)
    ]
    for op in ops:
        collection.update_one({"_id": op["_id"]}, {"$set": {"finbert_score": op["score"]}})
    print(f"[INFO] {len(ops)} scores cached in MongoDB.")
else:
    new_scores = []
    print("[INFO] All docs already cached — no FinBERT inference needed.")

# Merge cached + new
meta   = meta_cached   + meta_new
scores = scores_cached + new_scores

# ==============================
# BUILD ROWS
# ==============================
rows = []
for info, score in zip(meta, scores):
    for ticker in info["tickers"]:
        rows.append({
            "date":        info["date"],
            "ticker":      ticker,
            "sentiment":   score,
            "reliability": info["reliability"],
        })

df = pd.DataFrame(rows)
print(f"[INFO] {len(df)} raw ticker-day rows before aggregation.")

# ==============================
# RELIABILITY-WEIGHTED AGGREGATION
# ==============================
def weighted_agg(group):
    w = group["reliability"].values
    s = group["sentiment"].values
    total_w = w.sum()
    if total_w == 0:
        w_mean, w_std = 0.0, 0.0
    else:
        w_mean    = float(np.average(s, weights=w))
        variance  = float(np.average((s - w_mean) ** 2, weights=w))
        w_std     = float(np.sqrt(variance))
    return pd.Series({
        "sent_mean":       w_mean,
        "sent_std":        w_std,
        "sent_count":      float(len(group)),
        "avg_reliability": float(w.mean()),
    })

print("[INFO] Aggregating by date+ticker...")
agg = df.groupby(["date", "ticker"]).apply(weighted_agg).reset_index()

# ==============================
# SAVE
# ==============================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
agg.to_csv(OUTPUT_PATH, index=False)
print(f"[DONE] {len(agg)} rows saved to {OUTPUT_PATH}")
print(agg[["sent_mean", "avg_reliability"]].describe().round(4))
