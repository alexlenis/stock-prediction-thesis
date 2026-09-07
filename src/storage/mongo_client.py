from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "stock_prediction_db"

def get_database():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db