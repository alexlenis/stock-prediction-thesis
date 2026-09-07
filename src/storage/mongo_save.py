from src.storage.mongo_client import get_database

COLLECTION_NAME = "raw_documents"

def save_documents(documents):
    if not documents:
        print("[INFO] No documents to save.")
        return

    db = get_database()
    collection = db[COLLECTION_NAME]

    result = collection.insert_many(documents)
    print(f"[OK] Inserted {len(result.inserted_ids)} documents into '{COLLECTION_NAME}'")