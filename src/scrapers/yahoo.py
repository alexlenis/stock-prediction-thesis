import time
from datetime import datetime, timezone

import yfinance as yf
from pymongo import MongoClient

# ---------------------------------
# SETTINGS
# ---------------------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction_db"
COLLECTION_NAME = "raw_documents"

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]
POLL_INTERVAL_SECONDS = 20 * 60  # 20 minutes
RELIABILITY_SCORE = 0.85


# ---------------------------------
# DB
# ---------------------------------
def get_collection():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


# ---------------------------------
# EVENT HANDLER
# ---------------------------------
def handle_new_documents_event(new_documents):
    if not new_documents:
        print("[INFO] No new Yahoo documents found.")
        return

    collection = get_collection()
    result = collection.insert_many(new_documents)
    print(f"[EVENT] NEW_YAHOO_DOCUMENTS_FOUND -> inserted {len(result.inserted_ids)} documents")


# ---------------------------------
# DEDUP CHECK
# ---------------------------------
def document_exists(collection, url):
    return collection.find_one({"url": url}) is not None


# ---------------------------------
# SCRAPER
# ---------------------------------
def scrape_yahoo_finance():
    collection = get_collection()
    new_documents = []

    for ticker in TICKERS:
        try:
            news_items = yf.Ticker(ticker).news or []
            new_count = 0

            for item in news_items:
                # yfinance news item keys: title, link, publisher, providerPublishTime, uuid, type
                content = item.get("content", {})
                title = content.get("title") or item.get("title", "").strip()
                url = (
                    content.get("canonicalUrl", {}).get("url")
                    or item.get("link", "").strip()
                )
                publisher = content.get("provider", {}).get("displayName") or item.get("publisher", "Yahoo Finance")

                # Older yfinance: providerPublishTime (Unix timestamp int).
                # Newer yfinance: content.pubDate (ISO-8601 string).
                pub_ts = item.get("providerPublishTime") or content.get("pubDate")
                published_at = ""
                if pub_ts:
                    try:
                        published_at = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc).isoformat()
                    except (ValueError, TypeError):
                        # ISO-8601 string from newer API — store as-is (pipeline parses it)
                        published_at = str(pub_ts)

                if not url or not title:
                    continue

                if document_exists(collection, url):
                    continue

                document = {
                    "source": "yahoo_finance",
                    "source_type": "news",
                    "ticker": ticker,
                    "title": title,
                    "text": title,
                    "url": url,
                    "author": publisher,
                    "published_at": published_at,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "reliability_score": RELIABILITY_SCORE,
                    "sentiment_score": None,
                    "language": "en",
                    "raw_metadata": {}
                }

                new_documents.append(document)
                new_count += 1

            print(f"[OK] Yahoo checked for {ticker} -> {new_count} new articles")
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Yahoo scraper failed for {ticker}: {e}")

    return new_documents


# ---------------------------------
# POLLING LOOP
# ---------------------------------
def start_polling():
    print("[START] Yahoo Finance poller started.")
    print(f"[INFO] Polling interval: {POLL_INTERVAL_SECONDS // 60} minutes")

    while True:
        print("\n" + "=" * 60)
        print(f"[POLL] Checking Yahoo Finance at {datetime.now().isoformat()}")

        try:
            new_documents = scrape_yahoo_finance()
            handle_new_documents_event(new_documents)
        except Exception as e:
            print(f"[ERROR] Polling cycle failed: {e}")

        print(f"[SLEEP] Sleeping for {POLL_INTERVAL_SECONDS // 60} minutes...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_polling()
