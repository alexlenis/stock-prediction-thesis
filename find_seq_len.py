from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
col    = client["stock_prediction_db"]["raw_documents"]

total = col.count_documents({})
print(f"Total docs in MongoDB: {total}")
print("=" * 40)

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "NFLX", "AMD", "INTC",
    "SPY", "QQQ"
]

enough = 0
not_enough = 0

for ticker in TICKERS:
    # handles both "ticker" and "tickers" field formats
    n = col.count_documents({
        "$or": [
            {"ticker":  ticker},
            {"tickers": ticker},
        ]
    })
    status = "✅ Good" if n >= 500 else ("⚠️  Low" if n >= 100 else "❌ Not enough")
    print(f"  {ticker:6} : {n:5} articles  {status}")
    if n >= 500:
        enough += 1
    else:
        not_enough += 1

print("=" * 40)
print(f"Ready to retrain: {enough}/{len(TICKERS)} tickers")
if not_enough > 0:
    print(f"⚠️  {not_enough} tickers need more data before retraining")
else:
    print("✅ All tickers have enough data — safe to retrain")