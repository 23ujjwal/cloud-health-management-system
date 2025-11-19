# test_mongo_conn.py
from mongo_db import client, db, users_coll
print("Connected to DB:", db.name)
try:
    cnt = users_coll.count_documents({})
    print("Users documents count ->", cnt)
except Exception as e:
    print("Count error:", type(e).__name__, e)

# insert_test_user.py
from mongo_db import users_coll
import hashlib
u = "test_user_debug_1"
pw = "pass123"
h = hashlib.sha256(pw.encode()).hexdigest()
doc = {"username": u, "password": h, "role": "patient"}
res = users_coll.insert_one(doc)
print("Inserted id:", res.inserted_id)
