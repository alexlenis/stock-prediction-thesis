"""
Alpha Vantage historical news backfill scraper.

Fetches financial news with sentiment scores going back to 2024 and stores
them in MongoDB using the same schema as the other scrapers.

Free API key (takes 2 min): https://www.alphavantage.co/support/#api-key
Free tier: 25 requests/day.

Plan: 12 tickers × 5 date-ranges = 60 requests total → done in 3 days.
Each request returns up to 1000 articles for that ticker+period.

Usage:
    set AV_API_KEY=your_key_here
    python src/scrapers/alpha_vantage_historical.py

Or edit AV_API_KEY directly in this file.
"""
import os
import sys
import time
from datetime import datetime, timezone

import requests
from pymongo import MongoClient

# ==============================
# CONFIG — set your key here or via environment variable
# ==============================
AV_API_KEY = os.environ.get("AV_API_KEY", "S8GC7IPFBTFNQI68")
AV_BASE_URL = "https://www.alphavantage.co/query"

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ",
]

# 5 half-year windows → 60 total requests at 25/day = 3 days
DATE_RANGES = [
    ("20240101T0000", "20240630T2359"),
    ("20240701T0000", "20241231T2359"),
    ("20250101T0000", "20250630T2359"),
    ("20250701T0000", "20251231T2359"),
    ("20260101T0000", "20260630T2359"),
]

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction_db"
COLLECTION_NAME = "raw_documents"

RELIABILITY_SCORE = 0.92   # Alpha Vantage is a premium financial data provider
SLEEP_BETWEEN_REQUESTS = 15  # seconds — keeps well below 5 req/min free limit


# ==============================
# HELPERS
# ==============================
def get_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


def doc_exists(col, url: str) -> bool:
    return col.find_one({"url": url}) is not None


def fetch_av_news(ticker: str, time_from: str, time_to: str) -> list:
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "time_from": time_from,
        "time_to":   time_to,
        "limit":     1000,
        "sort":      "EARLIEST",
        "apikey":    AV_API_KEY,
    }
    resp = requests.get(AV_BASE_URL, params=params, timeout=30)
    data = resp.json()

    if "Information" in data:
        print(f"  [RATE LIMIT] {data['Information']}")
        return []
    if "Note" in data:
        print(f"  [NOTE] {data['Note']}")
        return []

    articles = data.get("feed", [])
    return articles


def parse_article(item: dict, target_ticker: str) -> dict | None:
    url   = (item.get("url") or "").strip()
    title = (item.get("title") or "").strip()
    if not url or not title:
        return None

    summary = (item.get("summary") or "").strip()
    source  = item.get("source", "")
    time_published = item.get("time_published", "")  # "20240315T143000"

    try:
        dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
        published_at = dt.replace(tzinfo=timezone.utc).isoformat()
    except Exception:
        published_at = ""

    # Alpha Vantage provides ticker-specific sentiment scores
    ticker_sentiment = 0.0
    for ts in item.get("ticker_sentiment", []):
        if ts.get("ticker") == target_ticker:
            try:
                ticker_sentiment = float(ts.get("ticker_sentiment_score", 0))
            except Exception:
                pass
            break

    return {
        "source":           "alpha_vantage",
        "source_type":      "news",
        "ticker":           target_ticker,
        "title":            title,
        "text":             summary or title,
        "url":              url,
        "author":           source,
        "published_at":     published_at,
        "scraped_at":       datetime.now(timezone.utc).isoformat(),
        "reliability_score": RELIABILITY_SCORE,
        "sentiment_score":  ticker_sentiment,  # AV's own score (stored for reference)
        "language":         "en",
        "raw_metadata": {
            "av_overall_sentiment_score": item.get("overall_sentiment_score"),
            "av_overall_sentiment_label": item.get("overall_sentiment_label"),
        },
    }


# ==============================
# MAIN
# ==============================
def main():
    if not AV_API_KEY or AV_API_KEY == "YOUR_KEY_HERE":
        print("[ERROR] Alpha Vantage API key not set.")
        print()
        print("  1. Go to: https://www.alphavantage.co/support/#api-key")
        print("  2. Enter your email → get a free key instantly")
        print("  3. Then run:")
        print("       set AV_API_KEY=your_key_here")
        print("       python src/scrapers/alpha_vantage_historical.py")
        sys.exit(1)

    col = get_collection()
    total_inserted = 0
    request_count  = 0

    print(f"[START] Backfilling {len(TICKERS)} tickers × {len(DATE_RANGES)} periods")
    print(f"        = {len(TICKERS) * len(DATE_RANGES)} total requests")
    print(f"        Free tier limit: 25/day → complete in "
          f"{(len(TICKERS)*len(DATE_RANGES))//25 + 1} days\n")

    for ticker in TICKERS:
        ticker_inserted = 0
        for time_from, time_to in DATE_RANGES:
            print(f"[FETCH] {ticker} | {time_from[:8]} → {time_to[:8]}", end="  ")

            try:
                articles = fetch_av_news(ticker, time_from, time_to)
                request_count += 1
                print(f"got {len(articles)} articles", end="  ")

                new_docs = []
                for item in articles:
                    doc = parse_article(item, ticker)
                    if doc and not doc_exists(col, doc["url"]):
                        new_docs.append(doc)

                if new_docs:
                    col.insert_many(new_docs)
                    total_inserted   += len(new_docs)
                    ticker_inserted  += len(new_docs)
                    print(f"-> inserted {len(new_docs)}")
                else:
                    print("-> all duplicates")

            except Exception as e:
                print(f"[ERROR] {e}")

            print(f"  [SLEEP] {SLEEP_BETWEEN_REQUESTS}s ...", end="\r")
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            print()

        print(f"  [{ticker}] Total new: {ticker_inserted}\n")

    print("=" * 60)
    print(f"[DONE] {total_inserted} documents inserted ({request_count} API calls)")
    print()
    print("Next steps:")
    print("  1. python src/sentiment_pipeline.py   (re-score all docs with FinBERT)")
    print("  2. python src/merge_dataset.py         (rebuild final_dataset.csv)")
    print("  3. python src/train_random_forest.py  (retrain RF)")
    print("  4. python src/train_xgboost.py         (retrain XGB)")
    print("  5. python src/train_lightgbm.py        (retrain LGBM)")
    print("  6. python src/train_baseline_improved.py (retrain Logistic)")
    print("  7. python src/train_lstm.py            (retrain LSTM — optional, slow)")


if __name__ == "__main__":
    main()
