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
POLL_INTERVAL_SECONDS = 20 * 60  # 20 minutes
RELIABILITY_SCORE = 0.80

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finviz.com/",
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
        print("[INFO] No new Finviz documents found. No event triggered.")
        return

    collection = get_collection()
    result = collection.insert_many(new_documents)

    print(f"[EVENT] NEW_FINVIZ_DOCUMENTS_FOUND -> inserted {len(result.inserted_ids)} documents")

    # Μελλοντικά:
    # run_sentiment_pipeline(new_documents)
    # update_daily_features(new_documents)
    # trigger_prediction_refresh()


# ---------------------------------
# DEDUP CHECK
# ---------------------------------
def document_exists(collection, source, ticker, title, published_at):
    query = {
        "source": source,
        "ticker": ticker,
        "title": title,
        "published_at": published_at
    }
    return collection.find_one(query) is not None


# ---------------------------------
# DATETIME NORMALIZATION
# ---------------------------------
def normalize_finviz_datetime(raw_dt: str):
    """
    Normalizes Finviz datetime strings to a parseable format.
    Finviz uses 'Today HH:MMam' for same-day articles; resolve to actual date.
    - 'Today 09:30AM'  → 'Jun-23-26 09:30AM'
    - 'Mar-20-26 09:30AM' → unchanged
    """
    raw_dt = raw_dt.strip()
    if raw_dt.upper().startswith("TODAY"):
        today_str = datetime.now().strftime("%b-%d-%y")
        raw_dt = today_str + raw_dt[5:]  # replace "Today" prefix
    return raw_dt


# ---------------------------------
# SCRAPER
# ---------------------------------
def scrape_finviz():
    collection = get_collection()
    session = requests.Session()
    session.headers.update(HEADERS)

    new_documents = []

    for ticker in TICKERS:
        url = f"https://finviz.com/quote.ashx?t={ticker}"

        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Finviz quote page συνήθως έχει table με id="news-table"
            news_table = soup.find("table", id="news-table")

            if not news_table:
                print(f"[WARNING] No news table found for {ticker}")
                time.sleep(2)
                continue

            rows = news_table.find_all("tr")
            new_count_for_ticker = 0

            current_date_context = None

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                datetime_text = cells[0].get_text(" ", strip=True)
                link_tag = cells[1].find("a")

                if not link_tag:
                    continue

                title = link_tag.get_text(" ", strip=True)
                article_url = link_tag.get("href", "").strip()

                source_span = cells[1].find("span")
                source_name = source_span.get_text(" ", strip=True) if source_span else "Finviz"

                # Finviz συχνά βάζει είτε:
                # - "Today 09:30AM"
                # - μόνο "09:30AM" όταν η ημερομηνία είναι ίδια με το προηγούμενο row
                parts = datetime_text.split()

                if len(parts) == 2:
                    current_date_context = parts[0]
                    published_at = f"{parts[0]} {parts[1]}"
                elif len(parts) == 1 and current_date_context:
                    published_at = f"{current_date_context} {parts[0]}"
                else:
                    published_at = datetime_text

                published_at = normalize_finviz_datetime(published_at)

                if not title:
                    continue

                if document_exists(
                    collection=collection,
                    source="finviz",
                    ticker=ticker,
                    title=title,
                    published_at=published_at
                ):
                    continue

                document = {
                    "source": "finviz",
                    "source_type": "news",
                    "ticker": ticker,
                    "title": title,
                    "text": title,  # προς το παρόν headline-only
                    "url": article_url,
                    "author": source_name,
                    "published_at": published_at,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "reliability_score": RELIABILITY_SCORE,
                    "sentiment_score": None,
                    "language": "en",
                    "raw_metadata": {
                        "quote_page": url
                    }
                }

                new_documents.append(document)
                new_count_for_ticker += 1

            print(f"[OK] Finviz checked for {ticker} -> {new_count_for_ticker} new articles")

            time.sleep(3)

        except Exception as e:
            print(f"[ERROR] Finviz scraper failed for {ticker}: {e}")

    return new_documents


# ---------------------------------
# POLLING LOOP
# ---------------------------------
def start_polling():
    print("[START] Finviz event-driven poller started.")
    print(f"[INFO] Polling interval: {POLL_INTERVAL_SECONDS // 60} minutes")

    while True:
        print("\n" + "=" * 60)
        print(f"[POLL] Checking Finviz pages at {datetime.now().isoformat()}")

        try:
            new_documents = scrape_finviz()
            handle_new_documents_event(new_documents)

        except Exception as e:
            print(f"[ERROR] Polling cycle failed: {e}")

        print(f"[SLEEP] Sleeping for {POLL_INTERVAL_SECONDS // 60} minutes...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_polling()