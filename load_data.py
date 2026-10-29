"""Step 1 - Load the CSV dataset into MongoDB using PyMongo."""
import numpy as np
import pandas as pd

from config import DATA_CSV, get_db


def load(drop_existing=True, verbose=True):
    db = get_db()
    col = db["municipalities"]

    df = pd.read_csv(DATA_CSV)
    if verbose:
        print(f"[load] CSV read: {df.shape[0]} rows x {df.shape[1]} columns")

    # MongoDB documents: replace NaN with None (stored as BSON null)
    df = df.astype(object).where(df.notna(), None)
    records = df.to_dict("records")
    # make sure numpy types become plain Python types for BSON
    for r in records:
        for k, v in r.items():
            if isinstance(v, np.integer):
                r[k] = int(v)
            elif isinstance(v, np.floating):
                r[k] = float(v)

    if drop_existing:
        col.drop()
    result = col.insert_many(records)
    col.create_index("region")
    col.create_index("istat", unique=True)
    if verbose:
        print(f"[load] Inserted {len(result.inserted_ids)} documents into "
              f"{db.name}.municipalities (indexes: region, istat[unique])")
    return col


def ensure_loaded():
    """Load the CSV into MongoDB only if the collection is empty."""
    col = get_db()["municipalities"]
    if col.count_documents({}) == 0:
        load()
    return col


if __name__ == "__main__":
    load()
