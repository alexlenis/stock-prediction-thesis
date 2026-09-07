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
RELIABILITY_SCORE = 0.90

# CNBC RSS feeds (static HTML — no JS rendering required)
RSS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # Top News
    "https://www.cnbc.com/id/20910258/device/rss/rss.html",    # Markets
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",    # Technology
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",    # Business
]

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
    ),
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
        print("[INFO] No new CNBC documents found.")
        return
    col = get_collection()
    res = col.insert_many(docs)
    print(f"[EVENT] NEW_CNBC_DOCUMENTS_FOUND -> inserted {len(res.inserted_ids)} documents")


# ---------------------------------
# DEDUP
# ---------------------------------
def exists(col, url):
    return col.find_one({"source": "cnbc", "url": url}) is not None


# ---------------------------------
# TICKER DETECTION
# All matching tickers are returned (not just the first)
# to avoid losing multi-company articles.
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
def scrape_cnbc():
    col = get_collection()
    all_docs = []
    seen_urls = set()

    for feed_url in RSS_FEEDS:
        try:
            r = requests.get(feed_url, headers=HEADERS, timeout=20)
            r.raise_for_status()

            root = ET.fromstring(r.content)
            items = root.findall(".//item")
            new_count = 0

            for item in items:
                title = (item.findtext("title") or "").strip()
                url   = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                description = (item.findtext("description") or "").strip()

                if not title or not url:
                    continue

                if url in seen_urls or exists(col, url):
                    continue
                seen_urls.add(url)

                # Use title + description for ticker detection
                combined_text = f"{title} {description}"
                tickers = detect_tickers(combined_text)
                if not tickers:
                    continue

                text_body = description if description else title

                for ticker in tickers:
                    doc = {
                        "source": "cnbc",
                        "source_type": "news",
                        "ticker": ticker,
                        "title": title,
                        "text": text_body,
                        "url": url,
                        "author": "CNBC",
                        "published_at": pub_date,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "reliability_score": RELIABILITY_SCORE,
                        "sentiment_score": None,
                        "language": "en",
                        "raw_metadata": {"feed": feed_url},
                    }
                    all_docs.append(doc)
                    new_count += 1

            print(f"[OK] CNBC feed: {feed_url} -> {new_count} new articles")
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] CNBC feed failed: {feed_url}: {e}")

    return all_docs


# ---------------------------------
# LOOP
# ---------------------------------
def start_polling():
    print("[START] CNBC RSS poller started.")
    print(f"[INFO] Polling interval: {POLL_INTERVAL_SECONDS // 60} minutes")

    while True:
        print("\n" + "=" * 60)
        print(f"[POLL] Checking CNBC at {datetime.now().isoformat()}")

        try:
            new_docs = scrape_cnbc()
            handle_new_documents_event(new_docs)
        except Exception as e:
            print(f"[ERROR] Polling cycle failed: {e}")

        print(f"[SLEEP] Sleeping for {POLL_INTERVAL_SECONDS // 60} minutes...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_polling()
