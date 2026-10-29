"""Step 3 - MongoDB / PyMongo operations demo (CRUD, queries, aggregation, index).

Run after load_data.py / train_models.py. Output is also saved to
outputs/mongodb_operations_log.txt (used in the report).
"""
import io
from contextlib import redirect_stdout
from pprint import pformat

from config import OUT_DIR, get_db
from load_data import ensure_loaded


def demo():
    ensure_loaded()
    db = get_db()
    col = db["municipalities"]

    print("COLLECTIONS:", sorted(db.list_collection_names()))
    print("Documents in municipalities:", col.count_documents({}))

    # ---- READ ----------------------------------------------------------
    print("\n[READ] find_one - one municipality (selected fields)")
    print(pformat(col.find_one({"name": "Comacchio"},
                               {"_id": 0, "name": 1, "region": 1, "pop": 1, "msw": 1, "sor": 1, "fee": 1})))

    print("\n[READ] find with filter - 3 PAYT municipalities with sor > 90, sorted by sor")
    for d in col.find({"fee": "PAYT", "sor": {"$gt": 90}},
                      {"_id": 0, "name": 1, "region": 1, "sor": 1}).sort("sor", -1).limit(3):
        print("  ", d)

    print("\n[READ] count_documents - municipalities per fee type")
    for fee in ("PAYT", "STANDARD"):
        print(f"   {fee}: {col.count_documents({'fee': fee})}")

    # ---- CREATE / UPDATE / DELETE -------------------------------------
    print("\n[CREATE] insert_one - add a test municipality")
    test_doc = {"istat": 999999, "name": "Test Town", "region": "Lombardia", "province": "Test",
                "pop": 5000, "msw": 2_000_000, "sor": 55.0, "fee": "STANDARD", "d_fee": 0, "sample": 1}
    print("   inserted_id:", col.insert_one(test_doc).inserted_id)

    print("\n[UPDATE] update_one - change fee system of the test town to PAYT")
    r = col.update_one({"istat": 999999}, {"$set": {"fee": "PAYT", "d_fee": 1}})
    print("   matched:", r.matched_count, "modified:", r.modified_count)
    print("  ", col.find_one({"istat": 999999}, {"_id": 0, "name": 1, "fee": 1, "d_fee": 1}))

    print("\n[UPDATE] update_many - flag very high recyclers (sor >= 85)")
    r = col.update_many({"sor": {"$gte": 85}}, {"$set": {"top_recycler": True}})
    print("   modified:", r.modified_count)

    print("\n[DELETE] delete_one - remove the test town")
    print("   deleted:", col.delete_one({"istat": 999999}).deleted_count)

    # ---- AGGREGATION ---------------------------------------------------
    print("\n[AGGREGATE] average separate-collection % (sor) and municipalities by region (top 5)")
    pipeline = [{"$group": {"_id": "$region", "avg_sor": {"$avg": "$sor"}, "municipalities": {"$sum": 1}}},
                {"$sort": {"avg_sor": -1}}, {"$limit": 5}]
    for d in col.aggregate(pipeline):
        print(f"   {d['_id']:<22} avg_sor={d['avg_sor']:.2f}  n={d['municipalities']}")

    print("\n[AGGREGATE] average sor by fee system")
    for d in col.aggregate([{"$group": {"_id": "$fee", "avg_sor": {"$avg": "$sor"}, "n": {"$sum": 1}}}]):
        print(f"   {d['_id']:<10} avg_sor={d['avg_sor']:.2f}  n={d['n']}")

    # ---- INDEXES ---------------------------------------------------------
    print("\n[INDEX] indexes on municipalities:", sorted(col.index_information().keys()))

    # ---- RESULTS COLLECTIONS ------------------------------------------
    if "model_metrics" in db.list_collection_names():
        print("\n[READ] stored model metrics")
        for d in db["model_metrics"].find({}, {"_id": 0, "task": 1, "model": 1}):
            print("  ", d)
        print("   predictions stored:", db["predictions"].count_documents({}))


def main():
    buf = io.StringIO()
    with redirect_stdout(buf):
        demo()
    text = buf.getvalue()
    print(text)
    (OUT_DIR / "mongodb_operations_log.txt").write_text(text)


if __name__ == "__main__":
    main()
