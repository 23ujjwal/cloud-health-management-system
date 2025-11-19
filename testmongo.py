# test_mongo_conn.py
from mongo_db import client, db, users_coll
print("Loaded DB name:", db.name)
try:
    print("Users count:", users_coll.count_documents({}))
except Exception as e:
    print("Count error:", type(e).__name__, e)
