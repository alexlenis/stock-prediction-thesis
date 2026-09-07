import time
import re
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

SUBREDDITS = ["stocks", "investing", "wallstreetbets"]

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]

# Company name → ticker mapping for Reddit posts that don't use symbols
COMPANY_ALIASES = {
    "AAPL":  ["apple", "iphone", "ipad", "macbook", "tim cook", "app store"],
    "MSFT":  ["microsoft", "windows", "azure", "xbox", "copilot", "satya nadella", "office 365", "teams"],
    "AMZN":  ["amazon", "aws", "prime", "bezos", "andy jassy", "whole foods"],
    "GOOGL": ["google", "alphabet", "youtube", "gemini", "waymo", "deepmind", "sundar pichai", "android"],
    "TSLA":  ["tesla", "elon musk", "cybertruck", "model 3", "model y", "model s", "supercharger", "elon"],
    "NVDA":  ["nvidia", "rtx", "geforce", "jensen huang", "cuda", "h100", "blackwell", "gpu"],
    "META":  ["meta", "facebook", "instagram", "whatsapp", "zuckerberg", "oculus", "threads", "reels"],
    "NFLX":  ["netflix", "streaming", "reed hastings"],
    "AMD":   ["amd", "radeon", "ryzen", "lisa su", "epyc"],
    "INTC":  ["intel", "core i", "pat gelsinger", "xeon", "arc gpu"],
    "SPY":   ["s&p 500", "s&p500", "sp500", "spy etf", "index fund"],
    "QQQ":   ["nasdaq", "qqq etf", "tech stocks", "tech index"],
}
POLL_INTERVAL_SECONDS = 20 * 60
RELIABILITY_SCORE = 0.60

BASE_URL = "https://old.reddit.com/r/{}/"
MAX_PAGES = 3  # pagination depth

HEADERS = {
    "User-Agent": "reddit-alpha-scraper/2.0"
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
        print("[INFO] No new Reddit documents found.")
        return

    collection = get_collection()
    result = collection.insert_many(new_documents)

    print(f"[EVENT] 🚨 NEW_REDDIT_ALPHA -> {len(result.inserted_ids)} docs inserted")

# ---------------------------------
# DEDUP
# ---------------------------------
def document_exists(collection, url):
    return collection.find_one({"url": url}) is not None

# ---------------------------------
# NLP HELPERS
# ---------------------------------
POSITIVE_WORDS = ["buy", "bull", "moon", "long", "pump", "breakout"]
NEGATIVE_WORDS = ["sell", "bear", "dump", "short", "crash"]

def detect_tickers(text):
    found = []
    text_upper = text.upper()
    text_lower = text.lower()

    for ticker in TICKERS:
        if ticker in text_upper:
            found.append(ticker)

    for ticker, aliases in COMPANY_ALIASES.items():
        if ticker not in found:
            for alias in aliases:
                if alias in text_lower:
                    found.append(ticker)
                    break

    return list(set(found))


def sentiment_score(text):
    text = text.lower()

    score = 0
    for w in POSITIVE_WORDS:
        if w in text:
            score += 1

    for w in NEGATIVE_WORDS:
        if w in text:
            score -= 1

    return score


def hype_score(score, num_comments):
    return score * 2 + num_comments * 0.1

# ---------------------------------
# HELPERS
# ---------------------------------
def normalize_url(url):
    if url.startswith("/"):
        return "https://old.reddit.com" + url
    return url

# ---------------------------------
# SCRAPER
# ---------------------------------
def scrape_subreddit(session, subreddit):
    collection = get_collection()

    url = BASE_URL.format(subreddit)
    page_count = 0

    new_documents = []

    while url and page_count < MAX_PAGES:
        response = session.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        posts = soup.find_all("div", class_="thing")

        for post in posts:
            title_tag = post.find("a", class_="title")
            if not title_tag:
                continue

            title = title_tag.get_text(" ", strip=True)
            post_url = normalize_url(title_tag.get("href", "").strip())

            if not title or not post_url:
                continue

            if document_exists(collection, post_url):
                continue

            # metadata
            author = post.get("data-author", "unknown")

            # old.reddit .thing divs carry data-timestamp (epoch milliseconds)
            published_at = None
            ts_ms = post.get("data-timestamp")
            if ts_ms:
                try:
                    published_at = datetime.fromtimestamp(
                        int(ts_ms) / 1000, tz=timezone.utc
                    ).isoformat()
                except (ValueError, TypeError):
                    published_at = None

            comments_tag = post.find("a", string=re.compile("comment"))
            num_comments = 0
            if comments_tag:
                match = re.search(r"\d+", comments_tag.text)
                if match:
                    num_comments = int(match.group())

            tickers = detect_tickers(title)
            sent_score = sentiment_score(title)
            hype = hype_score(sent_score, num_comments)

            document = {
                "source": "reddit",
                "source_type": "social",
                "subreddit": subreddit,
                "tickers": tickers,
                "title": title,
                "text": title,
                "url": post_url,
                "author": author,
                "num_comments": num_comments,
                "sentiment_score": sent_score,
                "hype_score": hype,
                "published_at": published_at,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
                "reliability_score": RELIABILITY_SCORE,
                "language": "en",
                "raw_metadata": {}
            }

            new_documents.append(document)

        # pagination
        next_button = soup.find("span", class_="next-button")
        url = next_button.find("a")["href"] if next_button else None

        page_count += 1
        time.sleep(2)

    print(f"[OK] r/{subreddit} scraped {len(new_documents)} new posts")
    return new_documents

# ---------------------------------
# MAIN SCRAPER
# ---------------------------------
def scrape_reddit():
    session = requests.Session()
    session.headers.update(HEADERS)

    all_new_documents = []

    for subreddit in SUBREDDITS:
        try:
            docs = scrape_subreddit(session, subreddit)
            all_new_documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] r/{subreddit}: {e}")

    return all_new_documents

# ---------------------------------
# POLLING LOOP
# ---------------------------------
def start_polling():
    print("[START] Reddit Alpha Poller started 🚀")

    while True:
        print("\n" + "=" * 60)
        print(f"[POLL] {datetime.now().isoformat()}")

        try:
            new_docs = scrape_reddit()
            handle_new_documents_event(new_docs)

        except Exception as e:
            print(f"[ERROR] Polling failed: {e}")

        print(f"[SLEEP] {POLL_INTERVAL_SECONDS}s")
        time.sleep(POLL_INTERVAL_SECONDS)

# ---------------------------------
# RUN
# ---------------------------------
if __name__ == "__main__":
    start_polling()