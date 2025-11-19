# migrate_to_mongo.py
# Script to migrate existing sqlite data (health_management.db) to MongoDB Atlas.
# Usage:
# 1) Ensure .env contains MONGO_URI and (optional) MONGO_DB_NAME and SQLITE_DB
# 2) Run: python migrate_to_mongo.py
#
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from mongo_db import users_coll, patients_coll, doctors_coll, appointments_coll, symptom_records_coll, notes_coll

SQLITE_DB = os.getenv("SQLITE_DB", "health_management.db")

def table_exists(conn, table_name):
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return c.fetchone() is not None

def row_to_dict(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

def try_parse_iso(value):
    if not isinstance(value, str):
        return value
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    return value

def migrate_table(table_name, query, coll):
    if not os.path.exists(SQLITE_DB):
        print(f"[SKIP] SQLite DB not found at {SQLITE_DB}")
        return
    conn = sqlite3.connect(SQLITE_DB)
    if not table_exists(conn, table_name):
        print(f"[SKIP] Table '{table_name}' does not exist in SQLite DB.")
        conn.close()
        return

    c = conn.cursor()
    try:
        c.execute(query)
    except Exception as e:
        print(f"[ERROR] Failed to execute query for table {table_name}: {e}")
        conn.close()
        return

    rows = c.fetchall()
    docs = []
    for r in rows:
        d = row_to_dict(c, r)
        # convert ISO-like timestamp strings to datetime objects where possible
        for k, v in list(d.items()):
            d[k] = try_parse_iso(v)
        docs.append(d)

    if docs:
        try:
            res = coll.insert_many(docs)
            print(f"Inserted {len(res.inserted_ids)} documents into {coll.name}")
        except Exception as e:
            print(f"[ERROR] Failed to insert documents into {coll.name}: {e}")
    else:
        print(f"[INFO] No rows found for {table_name}")
    conn.close()

def main():
    migrate_table('users', 'SELECT * FROM users', users_coll)
    migrate_table('patients', 'SELECT * FROM patients', patients_coll)
    migrate_table('doctors', 'SELECT * FROM doctors', doctors_coll)
    migrate_table('appointments', 'SELECT * FROM appointments', appointments_coll)
    migrate_table('symptom_records', 'SELECT * FROM symptom_records', symptom_records_coll)
    migrate_table('notes', 'SELECT * FROM notes', notes_coll)  # will be skipped if missing
    print('Migration finished.')

if __name__ == '__main__':
    main()
