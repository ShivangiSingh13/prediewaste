"""Database connection helper.

Connects to a real MongoDB server through PyMongo.
Set MONGO_URI to use a different server (e.g. MongoDB Atlas):
    export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"

If no MongoDB server can be reached, the code falls back to `mongomock`
(an in-memory, PyMongo-compatible database) so the project still runs
end-to-end. Use a real MongoDB server for your final demo / viva.
"""
import os
from pathlib import Path

try:                                   # optional: read MONGO_URI / MONGO_DB from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "data" / "waste_fee.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MONGO_URI_ENV = os.environ.get("MONGO_URI")          # set this for MongoDB Atlas
MONGO_URI = MONGO_URI_ENV or "mongodb://localhost:27017"
DB_NAME = os.environ.get("MONGO_DB", "waste_db")

_client = None
USING_MOCK = False


def get_db():
    """Return the `waste_db` database (real MongoDB if available)."""
    global _client, USING_MOCK
    if _client is None:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
            client.admin.command("ping")          # fails fast if no server
            _client = client
            print("[config] Connected to MongoDB" + (" (MONGO_URI)" if MONGO_URI_ENV else " at localhost"))
        except PyMongoError as e:
            if MONGO_URI_ENV:   # user explicitly asked for a server (e.g. Atlas) -> do not hide the error
                raise SystemExit(f"[config] Could not connect to MongoDB using MONGO_URI: {e}\n"
                                 "Check the password, the Network Access (IP allow-list) and the URI.")
            import mongomock
            _client = mongomock.MongoClient()
            USING_MOCK = True
            print("[config] WARNING: no MongoDB server found -> using in-memory "
                  "mongomock fallback (data is NOT saved between runs).")
    return _client[DB_NAME]
