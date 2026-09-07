import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
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
POLL_INTERVAL_SECONDS = 20 * 60
RELIABILITY_SCORE = 0.80

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.marketwatch.com/",
}


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
        print("[INFO] No new MarketWatch documents found. No event triggered.")
        return

    collection = get_collection()
    result = collection.insert_many(new_documents)

    print(f"[EVENT] NEW_MARKETWATCH_DOCUMENTS_FOUND -> inserted {len(result.inserted_ids)} documents")


# ---------------------------------
# DEDUP CHECK
# ---------------------------------
def document_exists(collection, source, ticker, title, url):
    query = {
        "source": source,
        "ticker": ticker,
        "title": title,
        "url": url
    }
    return collection.find_one(query) is not None


# ---------------------------------
# SCRAPER
# ---------------------------------
def scrape_marketwatch():
    collection = get_collection()
    session = requests.Session()
    session.headers.update(HEADERS)

    new_documents = []

    for ticker in TICKERS:
        url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"

        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Target article links directly — MarketWatch article URLs contain /story/ or /articles/
            all_links = soup.find_all("a", href=True)

            new_count_for_ticker = 0
            seen_urls = set()

            for link_tag in all_links:
                article_url = link_tag.get("href", "").strip()

                if article_url.startswith("/"):
                    article_url = f"https://www.marketwatch.com{article_url}"

                # Only keep actual article pages
                if not any(seg in article_url for seg in ["/story/", "/articles/", "/amp/story/"]):
                    continue

                if "marketwatch.com" not in article_url:
                    continue

                title = link_tag.get_text(" ", strip=True)
                if not title or len(title) < 15:
                    continue

                if article_url in seen_urls:
                    continue
                seen_urls.add(article_url)

                # Look for a nearby <time> tag and author within the link's parent container
                parent = link_tag.parent
                time_tag = parent.find("time") if parent else None
                published_at = time_tag.get("datetime", "") if time_tag else ""

                author_tag = parent.find(attrs={"class": lambda x: x and "author" in x.lower()}) if parent else None
                author = author_tag.get_text(" ", strip=True) if author_tag else "MarketWatch"

                text = title  # Full body requires JS rendering; headline is sufficient for sentiment

                if document_exists(
                    collection=collection,
                    source="marketwatch",
                    ticker=ticker,
                    title=title,
                    url=article_url
                ):
                    continue

                document = {
                    "source": "marketwatch",
                    "source_type": "news",
                    "ticker": ticker,
                    "title": title,
                    "text": text if text else title,
                    "url": article_url,
                    "author": author,
                    "published_at": published_at,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "reliability_score": RELIABILITY_SCORE,
                    "sentiment_score": None,
                    "language": "en",
                    "raw_metadata": {
                        "stock_page": url
                    }
                }

                new_documents.append(document)
                new_count_for_ticker += 1

            print(f"[OK] MarketWatch checked for {ticker} -> {new_count_for_ticker} new articles")

            time.sleep(3)

        except Exception as e:
            print(f"[ERROR] MarketWatch scraper failed for {ticker}: {e}")

    return new_documents


# ---------------------------------
# POLLING LOOP
# ---------------------------------
def start_polling():
    print("[START] MarketWatch event-driven poller started.")
    print(f"[INFO] Polling interval: {POLL_INTERVAL_SECONDS // 60} minutes")

    while True:
        print("\n" + "=" * 60)
        print(f"[POLL] Checking MarketWatch pages at {datetime.now().isoformat()}")

        try:
            new_documents = scrape_marketwatch()
            handle_new_documents_event(new_documents)

        except Exception as e:
            print(f"[ERROR] Polling cycle failed: {e}")

        print(f"[SLEEP] Sleeping for {POLL_INTERVAL_SECONDS // 60} minutes...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_polling()