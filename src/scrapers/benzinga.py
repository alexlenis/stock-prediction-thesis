import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests
from pymongo import MongoClient

# ---------------------------------
# SETTINGS
# ---------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction_db"
COLLECTION_NAME = "raw_documents"

POLL_INTERVAL_SECONDS = 20 * 60
RELIABILITY_SCORE = 0.85

# Benzinga public RSS feed
RSS_URL = "https://www.benzinga.com/feed"

COMPANY_KEYWORDS = {
    "AAPL":  ["AAPL", "APPLE"],
    "MSFT":  ["MSFT", "MICROSOFT"],
    "AMZN":  ["AMZN", "AMAZON"],
    "GOOGL": ["GOOGL", "GOOG", "GOOGLE", "ALPHABET"],
    "TSLA":  ["TSLA", "TESLA"],
    "NVDA":  ["NVDA", "NVIDIA"],
    "META":  ["META PLATFORMS", "FACEBOOK"],
    "NFLX":  ["NFLX", "NETFLIX"],
    "AMD":   ["AMD", "ADVANCED MICRO DEVICES"],
    "INTC":  ["INTC", "INTEL"],
    "SPY":   ["SPY", "S&P 500", "SPDR"],
    "QQQ":   ["QQQ", "NASDAQ 100", "INVESCO QQQ"],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------
# DB
# ---------------------------------
def get_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]


# ---------------------------------
# EVENT HANDLER
# ---------------------------------
def handle_new_documents_event(docs):
    if not docs:
        print("[INFO] No new Benzinga documents.")
        return
    col = get_collection()
    res = col.insert_many(docs)
    print(f"[EVENT] NEW_BENZINGA -> inserted {len(res.inserted_ids)} documents")


# ---------------------------------
# DEDUP
# ---------------------------------
def exists(col, url):
    return col.find_one({"source": "benzinga", "url": url}) is not None


# ---------------------------------
# TICKER DETECTION
# All matching tickers returned to avoid missing multi-company articles.
# META keyword is "META PLATFORMS" (not bare "META") to avoid false
# positives from words like "metadata".
# ---------------------------------
def detect_tickers(text):
    upper = text.upper()
    matched = []
    for ticker, keywords in COMPANY_KEYWORDS.items():
        if any(kw in upper for kw in keywords):
            matched.append(ticker)
    return matched


# ---------------------------------
# SCRAPER
# ---------------------------------
def scrape_benzinga():
    col = get_collection()
    docs = []

    try:
        r = requests.get(RSS_URL, headers=HEADERS, timeout=20)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        items = root.findall(".//item")
        new_count = 0

        for item in items:
            title       = (item.findtext("title") or "").strip()
            url         = (item.findtext("link") or "").strip()
            pub_date    = (item.findtext("pubDate") or "").strip()
            description = (item.findtext("description") or "").strip()

            if not title or not url:
                continue

            if exists(col, url):
                continue

            combined = f"{title} {description}"
            tickers = detect_tickers(combined)
            if not tickers:
                continue

            text_body = description if description else title

            for ticker in tickers:
                doc = {
                    "source": "benzinga",
                    "source_type": "news",
                    "ticker": ticker,
                    "title": title,
                    "text": text_body,
                    "url": url,
                    "author": "Benzinga",
                    "published_at": pub_date,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "reliability_score": RELIABILITY_SCORE,
                    "sentiment_score": None,
                    "language": "en",
                    "raw_metadata": {"feed": RSS_URL},
                }
                docs.append(doc)
                new_count += 1

        print(f"[OK] Benzinga -> {new_count} new articles")

    except Exception as e:
        print(f"[ERROR] Benzinga: {e}")

    return docs


# ---------------------------------
# LOOP
# ---------------------------------
def start_polling():
    print("[START] Benzinga RSS poller started.")

    while True:
        print(f"\n[POLL] Benzinga at {datetime.now().isoformat()}")

        docs = scrape_benzinga()
        handle_new_documents_event(docs)

        print(f"[SLEEP] {POLL_INTERVAL_SECONDS // 60} min")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_polling()
