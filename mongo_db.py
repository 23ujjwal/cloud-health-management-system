# mongo_db.py - MongoDB connection helper (auto-generated)
import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors
import gridfs

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'health_management')

if not MONGO_URI:
    raise RuntimeError('MONGO_URI missing in environment (.env). Add it locally before running the app.')

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
db = client[MONGO_DB_NAME]

users_coll = db['users']
patients_coll = db['patients']
doctors_coll = db['doctors']
appointments_coll = db['appointments']
symptom_records_coll = db['symptom_records']
triage_coll = db['triage_records']
notes_coll = db['notes']
reports_coll = db['reports']
fs = gridfs.GridFS(db)
