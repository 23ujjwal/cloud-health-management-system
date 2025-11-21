
import streamlit as st
import hashlib
import json
from datetime import datetime, time, date
import pandas as pd
import difflib
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()


from bson import ObjectId
from bson.json_util import dumps as bson_dumps
import pymongo
import gridfs
import certifi   # <-- ensure certifi is installed and imported

# Try helper module first
try:
    from mongo_db import db, users_coll, patients_coll, doctors_coll, appointments_coll, symptom_records_coll, triage_coll, notes_coll, reports_coll, fs
except Exception:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "health_management")
    if not MONGO_URI:
        raise RuntimeError("MONGO_URI is not set. Please add it to .env.")
    # Create MongoClient forcing certifi CA bundle and explicit TLS options
    # This helps avoid SSL/TLS handshake errors on some cloud hosts (Render, etc.)
    client = pymongo.MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000
    )
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

# Utility: serialize a Mongo document to friendly JSON-compatible python dict
def serialize_doc(doc):
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        # Convert ObjectId to string
        if isinstance(v, ObjectId):
            out[k] = str(v)
        # Convert datetime to ISO
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        # Convert bytes if any (unlikely)
        elif isinstance(v, bytes):
            try:
                out[k] = v.decode()
            except Exception:
                out[k] = str(v)
        # For nested documents or lists, attempt recursive conversion
        elif isinstance(v, dict):
            out[k] = serialize_doc(v)
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(serialize_doc(item))
                elif isinstance(item, ObjectId):
                    new_list.append(str(item))
                elif isinstance(item, datetime):
                    new_list.append(item.isoformat())
                else:
                    new_list.append(item)
            out[k] = new_list
        else:
            out[k] = v
    return out
def get_latest_triage_per_patient() -> pd.DataFrame:
    """
    Return a pandas DataFrame containing the latest triage record per patient.
    Works with MongoDB triage_records (triage_coll) and patients_coll.
    Columns: patient_id, recommended_specialty, severity, created_at, full_name, age, gender
    """
    try:
        # Aggregate latest triage per patient using Mongo aggregation
        pipeline = [
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": "$patient_id",
                "doc": {"$first": "$$ROOT"}   # take newest per patient
            }},
            {"$replaceRoot": {"newRoot": "$doc"}},
            {"$project": {
                "_id": 0,
                "patient_id": 1,
                "recommended_specialty": 1,
                "severity": 1,
                "created_at": 1,
                "selected_symptoms": 1,
                "top_conditions": 1,
                "note": 1
            }}
        ]
        cursor = triage_coll.aggregate(pipeline, allowDiskUse=True)
        rows = list(cursor)
        if not rows:
            return pd.DataFrame()

        # Normalize patient ids to strings for joining
        for r in rows:
            # keep patient id as string for consistent handling in DataFrame
            pid = r.get("patient_id")
            if isinstance(pid, ObjectId):
                r["patient_id"] = str(pid)
            else:
                r["patient_id"] = str(pid) if pid is not None else None

            # make sure created_at is a python datetime (serialize_doc will convert later if needed)
            ca = r.get("created_at")
            # if it's a Mongo date object it will be OK; otherwise leave it for pandas to parse
            r["created_at"] = ca

        df = pd.DataFrame(rows)

        # If for some rows patient_id is missing, drop them
        if "patient_id" not in df.columns or df["patient_id"].isnull().all():
            return pd.DataFrame()

        # Fetch patient metadata in one query for all patient_ids
        patient_ids = list(df["patient_id"].dropna().unique())
        # Build query to match numeric or ObjectId or string forms if needed.
        # We'll query by string-form patient id stored in patients_coll as either 'id' or '_id'
        patients_map = {}
        # Try to find patients whose _id matches ObjectId or whose id field matches numeric
        # Build two query sets: ObjectId candidates and string candidates
        obj_ids = [ObjectId(pid) for pid in patient_ids if ObjectId.is_valid(pid)]
        str_ids = [pid for pid in patient_ids if not ObjectId.is_valid(pid)]

        q_or = []
        if obj_ids:
            q_or.append({"_id": {"$in": obj_ids}})
        if str_ids:
            # maybe patients store patient id in 'id' numeric field or in string '_id' forms
            q_or.append({"_id": {"$in": [ObjectId(s) for s in str_ids if ObjectId.is_valid(s)]}})
            q_or.append({"id": {"$in": [int(s) for s in str_ids if s.isdigit()]}})
            q_or.append({"user_id": {"$in": str_ids}})

        if q_or:
            pat_cursor = patients_coll.find({"$or": q_or}) if len(q_or) > 1 else patients_coll.find(q_or[0])
        else:
            pat_cursor = patients_coll.find({})

        for p in pat_cursor:
            pid_key = None
            # prefer string _id if present
            try:
                pid_key = str(p.get("_id")) if p.get("_id") is not None else None
            except Exception:
                pid_key = None
            if not pid_key and "id" in p:
                pid_key = str(p.get("id"))
            # fallback to user_id
            if not pid_key and "user_id" in p:
                pid_key = str(p.get("user_id"))
            if pid_key:
                patients_map[pid_key] = {
                    "full_name": p.get("full_name") or p.get("name") or p.get("fullName"),
                    "age": p.get("age"),
                    "gender": p.get("gender")
                }

        # Map patient metadata onto df
        df["patient_id"] = df["patient_id"].astype(str)
        df["full_name"] = df["patient_id"].apply(lambda x: patients_map.get(x, {}).get("full_name"))
        df["age"] = df["patient_id"].apply(lambda x: patients_map.get(x, {}).get("age"))
        df["gender"] = df["patient_id"].apply(lambda x: patients_map.get(x, {}).get("gender"))

        # Convert created_at to pandas datetime for sorting & compatibility
        if "created_at" in df.columns:
            try:
                df["created_at"] = pd.to_datetime(df["created_at"])
            except Exception:
                # if conversion fails, leave as-is
                pass

        # Ensure columns order expected by UI
        wanted_cols = ["patient_id", "recommended_specialty", "severity", "created_at", "full_name", "age", "gender", "selected_symptoms", "top_conditions", "note"]
        existing = [c for c in wanted_cols if c in df.columns]
        df = df[existing]

        # sort by created_at desc
        if "created_at" in df.columns:
            df = df.sort_values("created_at", ascending=False).reset_index(drop=True)

        return df

    except Exception:
        # on error, return empty DataFrame rather than crashing the UI
        return pd.DataFrame()

# Init DB (create indexes)
def init_db():
    try:
        users_coll.create_index("username", unique=True)
    except Exception:
        pass
    try:
        patients_coll.create_index("user_id")
        doctors_coll.create_index("user_id")
        appointments_coll.create_index("patient_id")
        symptom_records_coll.create_index("patient_id")
        triage_coll.create_index("patient_id")
    except Exception:
        pass
    return True

# Reports / GridFS helpers
def upload_report(file_bytes, filename, content_type, patient_id, uploaded_by):
    meta = {
        "filename": filename,
        "content_type": content_type,
        "patient_id": str(patient_id),
        "uploaded_by": str(uploaded_by),
        "uploaded_at": datetime.utcnow()
    }
    fid = fs.put(file_bytes, filename=filename, content_type=content_type, metadata=meta)
    reports_coll.insert_one({"file_id": fid, **meta})
    return str(fid)

def list_reports_for_patient(patient_id, limit=20, skip=0):
    cursor = reports_coll.find({"patient_id": str(patient_id)}).sort("uploaded_at", -1).skip(skip).limit(limit)
    return [serialize_doc(r) for r in cursor]

def count_reports_for_patient(patient_id):
    return reports_coll.count_documents({"patient_id": str(patient_id)})

def get_report_file(file_id):
    try:
        if isinstance(file_id, str) and ObjectId.is_valid(file_id):
            grid_out = fs.get(ObjectId(file_id))
        else:
            grid_out = fs.get(file_id)
        meta = {
            "filename": grid_out.filename,
            "content_type": getattr(grid_out, 'content_type', None),
            "length": grid_out.length,
            "uploaded_at": grid_out.upload_date
        }
        return grid_out.read(), meta
    except Exception:
        return None, None


THEMES = {
    "Azure": {
        "--primary": "#3B82F6",
        "--accent": "#60A5FA",
        "--ok": "#10B981",
        "--warn": "#F59E0B",
        "--danger": "#EF4444",
        "--text": "#0F172A",
        "--muted": "#64748B",
        "--glass": "rgba(255,255,255,0.6)",
        "--glass-dark": "rgba(15,23,42,0.55)",
        "--bg-grad-light": "linear-gradient(135deg,#F8FAFC 0%, #EEF2FF 100%)",
        "--bg-grad-dark": "linear-gradient(135deg,#0B1220 0%, #0F172A 100%)",
    },
    "Emerald": {
        "--primary": "#10B981",
        "--accent": "#34D399",
        "--ok": "#22C55E",
        "--warn": "#EAB308",
        "--danger": "#F43F5E",
        "--text": "#052E2B",
        "--muted": "#4B5563",
        "--glass": "rgba(255,255,255,0.6)",
        "--glass-dark": "rgba(5,46,43,0.55)",
        "--bg-grad-light": "linear-gradient(135deg,#F0FDF4 0%, #ECFDF5 100%)",
        "--bg-grad-dark": "linear-gradient(135deg,#021614 0%, #052E2B 100%)",
    },
    "Crimson": {
        "--primary": "#EF4444",
        "--accent": "#FB7185",
        "--ok": "#F59E0B",
        "--warn": "#F97316",
        "--danger": "#DC2626",
        "--text": "#1F2937",
        "--muted": "#6B7280",
        "--glass": "rgba(255,255,255,0.6)",
        "--glass-dark": "rgba(31,41,55,0.55)",
        "--bg-grad-light": "linear-gradient(135deg,#FFF1F2 0%, #FFE4E6 100%)",
        "--bg-grad-dark": "linear-gradient(135deg,#1F0E11 0%, #2B0F13 100%)",
    },
}

def inject_css(theme_name: str, dark: bool):
    t = THEMES.get(theme_name, THEMES["Azure"])
    glass = t["--glass-dark"] if dark else t["--glass"]
    bg = t["--bg-grad-dark"] if dark else t["--bg-grad-light"]
    text = "#E5E7EB" if dark else t["--text"]
    muted = "#9CA3AF" if dark else t["--muted"]

    st.markdown(
        f"""
        <style>
        :root {{
          --primary: {t["--primary"]};
          --accent: {t["--accent"]};
          --ok: {t["--ok"]};
          --warn: {t["--warn"]};
          --danger: {t["--danger"]};
          --text: {text};
          --muted: {muted};
        }}
        .stApp {{ background: {bg} !important; }}
        .hero {{
          border-radius: 18px; padding: 24px 22px; margin-bottom: 16px;
          background: {glass}; backdrop-filter: saturate(120%) blur(12px);
          border: 1px solid rgba(255,255,255,0.25);
          box-shadow: 0 10px 28px rgba(0,0,0,0.06);
        }}
        .hero h1 {{ color: var(--text); margin: 0; letter-spacing: -0.02em; }}
        .hero p {{ color: var(--muted); margin: 6px 0 0 0; font-size: 0.95rem; }}
        .card {{
          background: {glass}; border-radius: 16px; padding: 16px 16px;
          border: 1px solid rgba(255,255,255,0.25);
          box-shadow: 0 6px 22px rgba(0,0,0,0.05); margin-bottom: 12px;
        }}
        .stButton>button {{
          background: var(--primary) !important; color: white !important; border: none !important;
          border-radius: 12px !important; padding: 0.6rem 0.9rem !important; font-weight: 600 !important;
          transition: transform .05s ease;
        }}
        .stButton>button:hover {{ transform: translateY(-1px); filter: brightness(1.02); }}
        .stButton>button:active {{ transform: translateY(0px) scale(.98); }}
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stMultiSelect>div>div, .stDateInput input, .stTimeInput input {{
          border-radius: 12px !important;
        }}
        .stDataFrame{{ border-radius: 12px; overflow: hidden; }}
        .pill {{ display:inline-block; padding:6px 10px; border-radius:999px;
                 background: var(--accent); color:white; font-size:.8rem; font-weight:700; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
          <h1>🏥 {title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def role_card(emoji: str, title: str, body: str, buttons: list[tuple]):
    with st.container():
        st.markdown(f"<div class='card'><h3 style='margin:0;color:var(--text)'>{emoji} {title}</h3>"
                    f"<p style='color:var(--muted);margin:.25rem 0 .75rem 0'>{body}</p>", unsafe_allow_html=True)
        cols = st.columns(len(buttons))
        for i, (label, key) in enumerate(buttons):
            with cols[i]:
                if st.button(label, use_container_width=True, key=key):
                    st.session_state.page = key
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --------------------
# Database-backed functions (Mongo)
# --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password, role, **extra):
    try:
        hashed_pw = hash_password(password)
        doc = {
            "username": username,
            "password": hashed_pw,
            "role": role,
            "created_at": datetime.utcnow()
        }
        if extra:
            doc.update(extra)
        res = users_coll.insert_one(doc)
        print(f"DEBUG create_user: inserted id {res.inserted_id} for {username}")
        return str(res.inserted_id)
    except pymongo.errors.DuplicateKeyError:
        return None
    except Exception as e:
        print("create_user error:", repr(e))
        return None

def verify_user(username, password, role=None):
    hashed_pw = hash_password(password)
    q = {"username": username, "password": hashed_pw}
    if role:
        q["role"] = role
    doc = users_coll.find_one(q)
    if doc:
        return str(doc.get("_id"))
    return None

def create_patient(user_id, full_name, age, gender, phone, email, address):
    doc = {
        "user_id": str(user_id),
        "full_name": full_name,
        "age": int(age) if (age is not None and str(age).isdigit()) else None,
        "gender": gender,
        "phone": phone,
        "email": email,
        "address": address,
        "created_at": datetime.utcnow()
    }
    res = patients_coll.insert_one(doc)
    print("DEBUG create_patient inserted", res.inserted_id)
    return str(res.inserted_id)

def create_doctor(user_id, full_name, specialization, hospital_clinic, phone, email):
    doc = {
        "user_id": str(user_id),
        "full_name": full_name,
        "specialization": specialization,
        "hospital_clinic": hospital_clinic,
        "phone": phone,
        "email": email,
        "created_at": datetime.utcnow()
    }
    res = doctors_coll.insert_one(doc)
    return str(res.inserted_id)

def get_patient_by_user_id(user_id):
    doc = patients_coll.find_one({"user_id": str(user_id)})
    if not doc:
        return None
    pid = str(doc.get("_id"))
    return (pid, doc.get("user_id"), doc.get("full_name"), doc.get("age"),
            doc.get("gender"), doc.get("phone"), doc.get("email"), doc.get("address"))

def get_doctor_by_user_id(user_id):
    doc = doctors_coll.find_one({"user_id": str(user_id)})
    if not doc:
        return None
    did = str(doc.get("_id"))
    return (did, doc.get("user_id"), doc.get("full_name"), doc.get("specialization"),
            doc.get("hospital_clinic"), doc.get("phone"), doc.get("email"))

def add_symptom_record(patient_id, symptoms, duration, previous_diagnosis):
    doc = {
        "patient_id": str(patient_id),
        "symptoms": symptoms,
        "duration": duration,
        "previous_diagnosis": previous_diagnosis,
        "recorded_at": datetime.utcnow()
    }
    res = symptom_records_coll.insert_one(doc)
    return str(res.inserted_id)

def get_all_symptom_records_df():
    rows = []
    for sr in symptom_records_coll.find().sort("recorded_at", -1):
        s = serialize_doc(sr)
        # try to find patient details
        patient = patients_coll.find_one({"user_id": s.get("patient_id")}) or patients_coll.find_one({"_id": ObjectId(s.get("patient_id"))}) or {}
        rows.append({
            "id": s.get("_id"),
            "patient_id": s.get("patient_id"),
            "full_name": patient.get("full_name"),
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "symptoms": s.get("symptoms"),
            "duration": s.get("duration"),
            "previous_diagnosis": s.get("previous_diagnosis"),
            "recorded_at": s.get("recorded_at")
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def save_triage(patient_id: int, selected_symptoms: list, severity: str,
                top_conditions: list, specialty: str, note: str):
    doc = {
        "patient_id": str(patient_id),
        "selected_symptoms": selected_symptoms,
        "severity": severity,
        "top_conditions": top_conditions,
        "recommended_specialty": specialty,
        "note": note,
        "created_at": datetime.utcnow()
    }
    res = triage_coll.insert_one(doc)
    return str(res.inserted_id)

def get_triage_df():
    rows = []
    for t in triage_coll.find().sort("created_at", -1):
        tr = serialize_doc(t)
        patient = patients_coll.find_one({"user_id": tr.get("patient_id")}) or {}
        rows.append({
            "id": tr.get("_id"),
            "created_at": tr.get("created_at"),
            "patient_id": tr.get("patient_id"),
            "full_name": patient.get("full_name"),
            "age": patient.get("age"),
            "gender": patient.get("gender"),
            "severity": tr.get("severity"),
            "recommended_specialty": tr.get("recommended_specialty"),
            "selected_symptoms": json.dumps(tr.get("selected_symptoms") or []),
            "top_conditions": json.dumps(tr.get("top_conditions") or []),
            "note": tr.get("note")
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

def create_appointment(patient_id, doctor_id, meeting_iso=None, when_dt=None, note=None, notes=None):
    """
    Create an appointment in MongoDB (appointments_coll) or fallback to SQLite if Mongo not available.
    Accepts flexible args:
      - when_dt: a datetime.datetime object (preferred)
      - meeting_iso: ISO datetime string (fallback)
      - note or notes: free text
    Returns: inserted id (str) or None on failure.
    """
    # choose note value
    note_val = notes if notes is not None else note

    # normalize meeting datetime to a Python datetime if possible
    meeting_dt = None
    if when_dt is not None:
        meeting_dt = when_dt
    elif meeting_iso:
        try:
            if isinstance(meeting_iso, str):
                # try common ISO parse
                meeting_dt = datetime.fromisoformat(meeting_iso)
            elif isinstance(meeting_iso, (int, float)):
                # unix timestamp
                meeting_dt = datetime.fromtimestamp(meeting_iso)
        except Exception:
            meeting_dt = None

    # Helper: normalize id into ObjectId / int / str depending on format
    def _norm_id_for_storage(x):
        if x is None:
            return None
        # if already ObjectId
        if isinstance(x, ObjectId):
            return x
        # if numeric
        if isinstance(x, int):
            return x
        # string: check ObjectId-like
        if isinstance(x, str):
            if ObjectId.is_valid(x):
                try:
                    return ObjectId(x)
                except Exception:
                    pass
            # numeric string -> int
            if x.isdigit():
                try:
                    return int(x)
                except Exception:
                    pass
            return x
        # fallback to string
        return str(x)

    # If Mongo collections exist, insert into Mongo
    try:
        if 'appointments_coll' in globals() and appointments_coll is not None:
            doc = {
                "patient_id": _norm_id_for_storage(patient_id),
                "doctor_id": _norm_id_for_storage(doctor_id),
                "created_at": datetime.utcnow()
            }
            # store ISO/datetime field under "when" for Mongo
            if meeting_dt is not None:
                doc["when"] = meeting_dt
            elif meeting_iso:
                doc["when"] = meeting_iso

            if note_val:
                doc["notes"] = note_val

            res = appointments_coll.insert_one(doc)
            return str(res.inserted_id)
    except Exception:
        # If Mongo insert fails, we'll try SQLite fallback below
        pass

    # Fallback to SQLite (keep original behavior)
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        # meeting_at stored as ISO string in sqlite
        meeting_iso_str = None
        if meeting_dt is not None:
            meeting_iso_str = meeting_dt.isoformat()
        elif meeting_iso:
            meeting_iso_str = meeting_iso if isinstance(meeting_iso, str) else str(meeting_iso)

        c.execute("""INSERT INTO appointments(patient_id, doctor_id, meeting_at, note)
                     VALUES(?,?,?,?)""",
                  (str(patient_id), str(doctor_id), meeting_iso_str, note_val))
        conn.commit()
        rowid = c.lastrowid
        conn.close()
        return rowid
    except Exception:
        # Could not insert anywhere
        return None


def get_patient_appointments(patient_id: int) -> pd.DataFrame:
    rows = []
    for a in appointments_coll.find({"patient_id": str(patient_id)}).sort("meeting_at", -1):
        ad = serialize_doc(a)
        # doctor lookup
        doc = doctors_coll.find_one({"_id": ObjectId(ad.get("doctor_id"))}) if ad.get("doctor_id") and ObjectId.is_valid(ad.get("doctor_id")) else doctors_coll.find_one({"user_id": ad.get("doctor_id")})
        ad["doctor_name"] = doc.get("full_name") if doc else None
        ad["specialization"] = doc.get("specialization") if doc else None
        ad["hospital_clinic"] = doc.get("hospital_clinic") if doc else None
        rows.append(ad)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # normalize date columns if present
    if 'meeting_at' in df.columns:
        try:
            df['meeting_at'] = pd.to_datetime(df['meeting_at'])
        except Exception:
            pass
    if 'created_at' in df.columns:
        try:
            df['created_at'] = pd.to_datetime(df['created_at'])
        except Exception:
            pass
    return df

# Helper: try to convert an id-like value into ObjectId or int when possible
# Helper: try to convert an id-like value into ObjectId or int when possible



    # enrich appointment with doctor info where possible
    doc_id = appt.get("doctor_id")
    normalized_doc_id = _normalize_id_for_query(doc_id)
    doctor = None
    if normalized_doc_id is not None:
        doctor = doctors_coll.find_one({"_id": normalized_doc_id}) or doctors_coll.find_one({"id": normalized_doc_id}) or doctors_coll.find_one({"user_id": normalized_doc_id})
    if not doctor and doc_id is not None:
        doctor = doctors_coll.find_one({"_id": _normalize_id_for_query(str(doc_id))}) or doctors_coll.find_one({"user_id": str(doc_id)})

    out = serialize_doc(appt)
    if doctor:
        out["doctor_name"] = doctor.get("full_name") or doctor.get("name") or doctor.get("doctor_name")
        out["doctor_specialization"] = doctor.get("specialization") or doctor.get("speciality") or doctor.get("specialty")
    return out


# ---------------------------------------------------------
# FIXED HELPERS FOR MONGO IDs (ObjectId / int / string)
# ---------------------------------------------------------

# ----------------------------
# Robust appointment helpers
# ----------------------------
def _normalize_id_for_query(val):
    """Normalize id for queries: returns ObjectId, int, or string as appropriate."""
    if val is None:
        return None
    if isinstance(val, ObjectId):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        # ObjectId-like hex
        if ObjectId.is_valid(val):
            try:
                return ObjectId(val)
            except Exception:
                pass
        # numeric string
        if val.isdigit():
            try:
                return int(val)
            except Exception:
                pass
        return val
    return str(val)


def get_latest_appointment(patient_id):
    """
    Return the latest appointment document (serialized dict) for the given patient_id.
    Works when patient_id is ObjectId / ObjectId-string / int / numeric-string.
    """
    norm = _normalize_id_for_query(patient_id)

    # try multiple variants (stored data may use different types)
    queries = []
    if norm is not None:
        queries.append({"patient_id": norm})
    # string form
    queries.append({"patient_id": str(patient_id)})

    appt = None
    for q in queries:
        appt = (
            appointments_coll.find_one(q, sort=[("when", -1)]) or
            appointments_coll.find_one(q, sort=[("meeting_at", -1)])
        )
        if appt:
            break

    if not appt:
        return None

    out = serialize_doc(appt)

    # attach doctor info if possible
    doc_id = appt.get("doctor_id")
    if doc_id is not None:
        dnorm = _normalize_id_for_query(doc_id)
        doctor = (
            doctors_coll.find_one({"_id": dnorm}) or
            doctors_coll.find_one({"id": dnorm}) or
            doctors_coll.find_one({"user_id": dnorm}) or
            doctors_coll.find_one({"_id": _normalize_id_for_query(str(doc_id))})
        )
        if doctor:
            out["doctor_name"] = doctor.get("full_name") or doctor.get("name")
            out["doctor_specialization"] = doctor.get("specialization") or doctor.get("speciality") or doctor.get("specialty")
    return out


def patient_taken_by_specialty(patient_id, specialty) -> bool:
    """
    Returns True when this patient's latest appointment is with a doctor
    whose specialization matches `specialty` (case-insensitive substring tolerant).
    Accepts mixed id types (ObjectId/string/int).
    """
    if not specialty:
        return False
    latest = get_latest_appointment(patient_id)
    if not latest:
        return False

    # prefer attached doctor_specialization if present
    doc_spec = (latest.get("doctor_specialization") or "").strip()
    if not doc_spec:
        # attempt to find doctor via raw appointment doc
        try:
            raw_appt = appointments_coll.find_one({"_id": _normalize_id_for_query(latest.get("_id"))})
            if raw_appt:
                d_id = raw_appt.get("doctor_id")
                if d_id is not None:
                    dnorm = _normalize_id_for_query(d_id)
                    doctor = (
                        doctors_coll.find_one({"_id": dnorm}) or
                        doctors_coll.find_one({"id": dnorm}) or
                        doctors_coll.find_one({"user_id": dnorm})
                    )
                    if doctor:
                        doc_spec = (doctor.get("specialization") or doctor.get("speciality") or doctor.get("specialty") or "").strip()
        except Exception:
            doc_spec = ""

    latest_spec = (doc_spec or "").lower()
    my_spec = (specialty or "").strip().lower()
    if not latest_spec or not my_spec:
        return False
    return (my_spec in latest_spec) or (latest_spec in my_spec)



def add_note(patient_id, doctor_id, note):
    doc = {
        "patient_id": str(patient_id),
        "doctor_id": str(doctor_id),
        "note": note,
        "created_at": datetime.utcnow()
    }
    res = notes_coll.insert_one(doc)
    return str(res.inserted_id)

def get_notes_for_patient(patient_id):
    cursor = notes_coll.find({"patient_id": str(patient_id)}).sort("created_at", -1)
    return [serialize_doc(d) for d in cursor]

# ============================
# DISEASE KB & CLASSIFIER
# ============================
DISEASE_KB = [
    {"name":"Common Cold","severity":"mild","specialty":"General Physician","keywords":["cold","common cold","rhinitis","runny nose","sore throat"]},
    {"name":"Seasonal Allergy","severity":"mild","specialty":"Allergist / Immunologist","keywords":["allergy","allergic rhinitis","hay fever"]},
    {"name":"Conjunctivitis","severity":"mild","specialty":"Ophthalmologist","keywords":["pink eye","conjunctivitis","eye redness"]},
    {"name":"Tension Headache","severity":"mild","specialty":"General Physician","keywords":["tension headache","stress headache"]},
    {"name":"Gastritis / Acid Reflux","severity":"mild","specialty":"Gastroenterologist","keywords":["gastritis","acid reflux","acidity","gerd","heartburn"]},
    {"name":"Dermatitis (mild)","severity":"mild","specialty":"Dermatologist","keywords":["dermatitis","eczema","mild rash","itchy skin"]},
    {"name":"Viral Fever (low-grade)","severity":"mild","specialty":"General Physician","keywords":["viral fever","low grade fever"]},
    {"name":"Acne (mild)","severity":"mild","specialty":"Dermatologist","keywords":["acne","pimples"]},
    {"name":"Sinusitis (mild)","severity":"mild","specialty":"ENT","keywords":["sinusitis","sinus infection","sinus pain"]},
    {"name":"Mouth Ulcer (aphthous)","severity":"mild","specialty":"Dentist","keywords":["mouth ulcer","aphthous ulcer","canker sore"]},
    {"name":"Otitis Externa (mild)","severity":"mild","specialty":"ENT","keywords":["swimmer's ear","otitis externa"]},
    {"name":"Mild Diarrhea","severity":"mild","specialty":"General Physician","keywords":["mild diarrhea","loose motions"]},
    {"name":"Influenza","severity":"moderate","specialty":"General Physician","keywords":["influenza","flu","high fever body ache"]},
    {"name":"Acute Gastroenteritis","severity":"moderate","specialty":"Gastroenterologist","keywords":["gastroenteritis","food poisoning","vomiting and diarrhea"]},
    {"name":"Urinary Tract Infection","severity":"moderate","specialty":"Urologist","keywords":["uti","urinary tract infection","burning urination"]},
    {"name":"Migraine","severity":"moderate","specialty":"Neurologist","keywords":["migraine","one sided headache","throbbing headache"]},
    {"name":"Asthma Exacerbation","severity":"moderate","specialty":"Pulmonologist","keywords":["asthma","wheezing","shortness of breath"]},
    {"name":"Bronchitis","severity":"moderate","specialty":"Pulmonologist","keywords":["bronchitis","chest congestion","productive cough"]},
    {"name":"Bacterial Sinusitis","severity":"moderate","specialty":"ENT","keywords":["bacterial sinusitis","purulent nasal discharge"]},
    {"name":"Iron-deficiency Anemia","severity":"moderate","specialty":"Hematologist","keywords":["anemia","iron deficiency","low hemoglobin"]},
    {"name":"Typhoid","severity":"moderate","specialty":"General Physician","keywords":["typhoid","enteric fever"]},
    {"name":"COVID-like Illness","severity":"moderate","specialty":"General Physician","keywords":["covid","loss of smell","covid like"]},
    {"name":"Lung Cancer","severity":"serious","specialty":"Oncologist / Pulmonologist","keywords":["lung cancer","blood in cough","chronic cough weight loss"]},
    {"name":"Colorectal Cancer","severity":"serious","specialty":"Oncologist / Gastroenterologist","keywords":["colorectal cancer","blood in stool","weight loss bowel"]},
    {"name":"Breast Cancer","severity":"serious","specialty":"Oncologist","keywords":["breast cancer","breast lump","nipple discharge"]},
    {"name":"Gastric (Stomach) Cancer","severity":"serious","specialty":"Oncologist / Gastroenterologist","keywords":["stomach cancer","gastric cancer","early satiety weight loss"]},
    {"name":"Leukemia","severity":"serious","specialty":"Oncologist / Hematologist","keywords":["leukemia","abnormal blood counts","frequent infections"]},
]

TERM_TO_DISEASE = {}
for item in DISEASE_KB:
    TERM_TO_DISEASE[item["name"].lower()] = item
    for k in item["keywords"]:
        TERM_TO_DISEASE[k.lower()] = item

def classify_disease_from_text(query: str):
    q = (query or "").strip().lower()
    if not q:
        return {"name":"Undetermined Condition","score":0.0,"base_severity":"mild","specialty":"General Physician"}, "mild", "General Physician"

    candidates = list(set(list(TERM_TO_DISEASE.keys()) + [d["name"].lower() for d in DISEASE_KB]))
    scored = []

    # exact
    for item in DISEASE_KB:
        if item["name"].lower() == q:
            scored.append((item, 1.0))

    # keyword contains
    for term, meta in TERM_TO_DISEASE.items():
        if term in q or q in term:
            base = 0.95 if term == meta["name"].lower() else 0.85
            scored.append((meta, base))

    # fuzzy
    for term in candidates:
        ratio = difflib.SequenceMatcher(None, q, term).ratio()
        if ratio >= 0.6:
            meta = TERM_TO_DISEASE.get(term)
            if meta:
                scored.append((meta, max(0.70, min(0.90, ratio))))

    if not scored:
        best = {"name":"Undetermined Condition","score":0.0,"base_severity":"mild","specialty":"General Physician"}
        return best, "mild", "General Physician"

    best_by_name = {}
    for meta, s in scored:
        nm = meta["name"]
        if (nm not in best_by_name) or (s > best_by_name[nm][1]):
            best_by_name[nm] = (meta, s)

    best_meta, best_score = max(best_by_name.values(), key=lambda t: t[1])
    best = {
        "name": best_meta["name"],
        "score": float(f"{best_score:.3f}"),
        "base_severity": best_meta["severity"],
        "specialty": best_meta["specialty"],
    }
    overall = best["base_severity"]
    specialty = best["specialty"]
    return best, overall, specialty

DISEASE_ABOUT = {
    "Common Cold": "Viral upper-respiratory infection causing runny nose and sore throat; usually self-limited.",
    "Seasonal Allergy": "Allergic response to pollen/dust with sneezing and itchy/watery eyes.",
    "Conjunctivitis": "Inflammation of the eye’s conjunctiva; often contagious if infectious.",
    "Tension Headache": "Band-like, stress-related headache; typically non-throbbing.",
    "Gastritis / Acid Reflux": "Stomach acid irritation causing burning chest/upper-abdominal discomfort.",
    "Dermatitis (mild)": "Irritated/itchy skin; often triggered by contact or dryness.",
    "Viral Fever (low-grade)": "Short-lived fever with mild body aches; usually resolves with rest/fluids.",
    "Acne (mild)": "Clogged skin pores causing comedones and small inflamed bumps.",
    "Sinusitis (mild)": "Inflamed sinuses with facial pressure/congestion; often post-cold.",
    "Mouth Ulcer (aphthous)": "Small painful oral ulcers; heal in 1–2 weeks.",
    "Otitis Externa (mild)": "Ear-canal inflammation; often ‘swimmer’s ear’.",
    "Mild Diarrhea": "Short-term loose stools; maintain hydration.",
    "Influenza": "Acute viral illness with high fever, myalgia, headache and cough.",
    "Acute Gastroenteritis": "Vomiting/diarrhea from gut inflammation; risk of dehydration.",
    "Urinary Tract Infection": "Burning urination, frequency; needs urine test ± antibiotics.",
    "Migraine": "Throbbing headache ± nausea/photophobia; often unilateral.",
    "Asthma Exacerbation": "Wheezing and breathlessness due to airway narrowing.",
    "Bronchitis": "Bronchial inflammation causing productive cough/chest congestion.",
    "Bacterial Sinusitis": "Persistent purulent discharge/facial pain; may need antibiotics.",
    "Iron-deficiency Anemia": "Low hemoglobin from iron lack → fatigue, breathlessness on exertion.",
    "Typhoid": "Prolonged fever from salmonella; needs testing/antibiotics.",
    "COVID-like Illness": "Fever/cough ± loss of smell; get tested if available.",
    "Lung Cancer": "Persistent cough, weight loss, blood in sputum; needs imaging/oncology referral.",
    "Colorectal Cancer": "Blood in stool, weight loss, bowel changes; needs colon evaluation.",
    "Breast Cancer": "Breast lump or skin/nipple changes; needs imaging/biopsy.",
    "Gastric (Stomach) Cancer": "Early satiety, weight loss, persistent epigastric pain; needs endoscopy.",
    "Leukemia": "Abnormal blood counts, infections/bruising; needs hematology work-up.",
}

# ==============================
# PATIENT UI
# ==============================
def patient_registration():
    st.subheader("👤 Patient Registration")
    st.caption("Create your account to securely check diseases and save records.")
    with st.form("patient_reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username*")
            password = st.text_input("Password*", type="password")
            full_name = st.text_input("Full Name*")
            age = st.number_input("Age*", min_value=1, max_value=120, value=25)
        with col2:
            gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
            phone = st.text_input("Phone Number*")
            email = st.text_input("Email*")
            address = st.text_area("Address")
        submitted = st.form_submit_button("Register")
    if submitted:
        if username and password and full_name and phone and email:
            user_id = create_user(username, password, "patient", email=email)
            st.write("DEBUG -> create_user returned:", user_id)
            if user_id:
                try:
                    patient_doc = {
                        "user_id": user_id,
                        "full_name": full_name,
                        "age": int(age) if age is not None else None,
                        "gender": gender,
                        "phone": phone,
                        "email": email,
                        "address": address,
                        "created_at": datetime.utcnow()
                    }
                    pres = patients_coll.insert_one(patient_doc)
                    print("DEBUG create_patient: inserted patient id", pres.inserted_id)
                    st.success("Registration successful! Please login.")
                    st.balloons()
                    st.session_state.page = 'main'
                    st.rerun()
                except Exception as e:
                    print("DEBUG create_patient: Exception:", repr(e))
                    st.error("Failed to create patient profile — check logs.")
            else:
                st.error("Registration failed — username may already exist or there was a backend error.")
        else:
            st.error("Please fill all required fields!")

def patient_login():
    st.subheader("🔐 Patient Login")
    with st.form("patient_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Login", use_container_width=True)
        with c2:
            back = st.form_submit_button("Back", use_container_width=True)
    if submitted:
        user_id = verify_user(username, password, "patient")
        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_role = "patient"
            st.session_state.user_id = user_id
            st.toast("Welcome back!", icon="✅")
            st.rerun()
        else:
            st.error("Invalid credentials!")
    if back:
        st.session_state.page = 'main'
        st.rerun()

def patient_disease_checker(patient_id: int):
    st.subheader("🩺 Quick Disease Checker")
    st.caption("Educational triage only — not a diagnosis. For emergencies, seek immediate care.")

    k = st.session_state.checker_key
    disease_guess = st.text_input(
        "What disease do you think you have?",
        placeholder="e.g., common cold, migraine, gastritis, breast cancer…",
        key=f"disease_guess_{k}"
    )
    note_text = st.text_area("Anything else you'd like to add? (optional)",
                             key=f"note_text_{k}")

    cols = st.columns([1,1,1])
    with cols[0]:
        check_save = st.button("Check & Save", use_container_width=True, key=f"check_{k}")
    with cols[1]:
        new_check = st.button("Start New Check", use_container_width=True, key=f"newcheck_{k}")
    with cols[2]:
        start_chat_btn = st.button("Start New Chat 💬", use_container_width=True, key=f"chat_{k}")

    if new_check:
        st.session_state.checker_key += 1
        st.rerun()

    if start_chat_btn:
        st.session_state.chat_messages = []
        st.session_state.show_chat = True
        st.toast("New chat started", icon="💬")

    if check_save:
        if not disease_guess.strip():
            st.error("Please type a disease name.")
            return

        best, severity, specialty = classify_disease_from_text(disease_guess)

        sev_color = {"mild":"var(--ok)","moderate":"var(--warn)","serious":"var(--danger)"}[severity]
        st.markdown(
            f"#### Most likely: **{best['name']}**  ·  match **{int(best['score']*100)}%**",
            unsafe_allow_html=True
        )
        about = DISEASE_ABOUT.get(best["name"], "No brief available.")
        st.markdown(f"<div class='card'><b>About:</b> {about}</div>", unsafe_allow_html=True)

        st.markdown(
            f"**Final verdict:** <span class='pill' style='background:{sev_color}'> {severity.title()} </span>",
            unsafe_allow_html=True
        )
        st.success(f"Recommended doctor: **{specialty}**")

        top_one = [{"name": best["name"], "score": best["score"], "base_severity": best["base_severity"], "specialty": best["specialty"]}]
        save_triage(
            patient_id=patient_id,
            selected_symptoms=[f"disease:{disease_guess.strip()}"],
            severity=severity,
            top_conditions=top_one,
            specialty=specialty,
            note=note_text.strip()
        )
        st.toast("Triage saved", icon="💾")
        if severity == "serious":
            st.warning("Your input suggests a serious condition. Please seek prompt medical evaluation.")

def patient_chat_ui(assigned_doctor_name: str):
    st.subheader("💬 Chat with your assigned doctor")
    if st.session_state.show_chat:
        for msg in st.session_state.chat_messages:
            speaker = "You" if msg["role"] == "patient" else assigned_doctor_name
            st.markdown(f"**{speaker}:** {msg['text']}")
        with st.form("chat_send"):
            txt = st.text_input("Type your message")
            sent = st.form_submit_button("Send")
        if sent and txt.strip():
            st.session_state.chat_messages.append({"role":"patient","text":txt.strip()})
            st.rerun()
    else:
        st.info("Click **Start New Chat** above to begin a conversation.")

def patient_dashboard():
    """
    Robust patient dashboard supporting both SQLite tuple rows and Mongo dict rows.
    Handles report upload (GridFS), listing, preview (images), and download.
    """
    row = get_patient_by_user_id(st.session_state.user_id)
    if not row:
        st.info("No patient profile found. Please register.")
        return

    # helpers to safely extract fields from tuple (sqlite) or dict (mongo)
    def _get_field(r, idx_or_key, fallback=None):
        try:
            if isinstance(r, dict):
                return r.get(idx_or_key, fallback)
            else:
                # tuple-like
                return r[idx_or_key] if idx_or_key < len(r) else fallback
        except Exception:
            return fallback

    # determine patient_id (prefer stable string form)
    if isinstance(row, dict):
        # prefer string _id if present
        pid = row.get("_id") or row.get("id") or row.get("patient_id")
        patient_name = row.get("full_name") or row.get("name") or row.get("fullName")
    else:
        # sqlite tuple layout: (id, user_id, full_name, age, gender, phone, email, address)
        try:
            pid = row[0]
            patient_name = row[2]
        except Exception:
            pid = row[0] if len(row) > 0 else None
            patient_name = row[2] if len(row) > 2 else None

    # Normalize patient id to string for use in Mongo queries / storage keys
    patient_id = str(pid) if pid is not None else None

    st.markdown(f"### 🏥 Patient Dashboard — <span class='pill'>Secure</span>", unsafe_allow_html=True)
    st.write(f"Hello, **{patient_name or 'Patient'}**!")

    # Profile box
    with st.expander("👤 My Profile", expanded=False):
        age = _get_field(row, 3, "-")
        gender = _get_field(row, 4, "-")
        phone = _get_field(row, 5, "-")
        email = _get_field(row, 6, "-")
        address = _get_field(row, 7, "-")
        c1, c2, c3 = st.columns(3)
        c1.metric("Age", age or "-")
        c2.metric("Gender", gender or "-")
        c3.metric("Phone", phone or "-")
        st.caption(f"Email: {email or '-'}  |  Address: {address or '-'}")

    st.divider()

    # Assigned doctor & Date of Approval (latest appointment)
    st.subheader("✅ Doctor Approval")
    latest = get_latest_appointment(patient_id)
    assigned_doctor_name = None
    if latest:
        # If dict (mongo) or tuple (sqlite)
        if isinstance(latest, dict):
            meet_val = latest.get("when") or latest.get("meeting_at") or latest.get("meetingAt") or latest.get("meeting_at_iso")
            assigned_doctor_name = latest.get("doctor_name") or latest.get("doctor") or latest.get("doctor_full_name")
            doc_spec = latest.get("doctor_specialization") or latest.get("specialization")
            note = latest.get("notes") or latest.get("note")
        else:
            # tuple-like legacy: (id, meeting_at, note, created_at, doctor_name, specialization, hospital_clinic)
            try:
                meet_val = latest[1]
                note = latest[2] if len(latest) > 2 else None
                assigned_doctor_name = latest[4] if len(latest) > 4 else None
                doc_spec = latest[5] if len(latest) > 5 else None
            except Exception:
                meet_val = None
                note = None
                assigned_doctor_name = None
                doc_spec = None

        # parse meeting datetime robustly
        meet_dt = None
        if isinstance(meet_val, str):
            try:
                meet_dt = datetime.fromisoformat(meet_val)
            except Exception:
                try:
                    meet_dt = pd.to_datetime(meet_val)
                    if hasattr(meet_dt, "to_pydatetime"):
                        meet_dt = meet_dt.to_pydatetime()
                except Exception:
                    meet_dt = None
        elif isinstance(meet_val, datetime):
            meet_dt = meet_val

        if meet_dt:
            st.success(f"Doctor **{assigned_doctor_name or '-'}** ({doc_spec or '-'}) at **{meet_dt.strftime('%d/%m/%y %H:%M')}**")
            if note:
                st.caption(f"Note: {note}")
        else:
            if assigned_doctor_name:
                st.success(f"Doctor **{assigned_doctor_name}** ({doc_spec or '-'})")
                if note:
                    st.caption(f"Note: {note}")
            else:
                st.info("No doctor assigned yet. Once a doctor approves and schedules a meeting, it will appear here.")
    else:
        st.info("No doctor assigned yet. Once a doctor approves and schedules a meeting, it will appear here.")

    # Chat area if doctor assigned
    if assigned_doctor_name:
        st.markdown(" ")
        patient_chat_ui(assigned_doctor_name)

    st.divider()
    patient_disease_checker(patient_id)

    st.divider()
    st.subheader("📅 Appointment History")
    appts = get_patient_appointments(patient_id)
    if not appts.empty:
        appts_display = appts.copy()
        if "meeting_at" in appts_display.columns:
            try:
                appts_display["meeting_at"] = pd.to_datetime(appts_display["meeting_at"]).dt.strftime("%d/%m/%y %H:%M")
            except Exception:
                pass
        if "created_at" in appts_display.columns:
            try:
                appts_display["created_at"] = pd.to_datetime(appts_display["created_at"]).dt.strftime("%d/%m/%y %H:%M")
            except Exception:
                pass
        st.dataframe(
            appts_display.rename(columns={
                "meeting_at": "Meeting At (dd/mm/yy)",
                "doctor_name": "Doctor",
                "specialization": "Specialty",
                "hospital_clinic": "Hospital/Clinic",
                "note": "Note",
                "created_at": "Created"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No appointments yet.")

    st.divider()
    st.subheader("📝 Optional: Add a Free-Text Note")
    with st.form("symptom_form", clear_on_submit=True):
        symptoms = st.text_area("Notes", placeholder="Anything else you'd like your doctor to know…", height=100)
        duration = st.text_input("Duration", placeholder="e.g., 3 days, 2 weeks (optional)")
        previous_diagnosis = st.text_area("Previous Diagnosis (optional)", placeholder="Any past diagnosis…")
        c1, c2 = st.columns([1, 1])
        with c1:
            submitted = st.form_submit_button("Save Note", use_container_width=True)
        with c2:
            exit_btn = st.form_submit_button("Logout", use_container_width=True)
    if submitted:
        if symptoms.strip():
            add_symptom_record(patient_id, symptoms.strip(), duration.strip(), previous_diagnosis.strip())
            st.success("Saved!")
            st.toast("Saved", icon="💾")
        else:
            st.error("Please write something in Notes before saving.")
    if exit_btn:
        logout_reset()

    # ----- Upload Reports UI -----
    st.divider()
    st.subheader("📎 Upload Medical Report")
    with st.form("upload_report_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload report (PDF, image). Max 10MB", type=["pdf", "png", "jpg", "jpeg"], key="report_upload")
        submit_report = st.form_submit_button("Upload Report")
    if submit_report:
        if uploaded_file is not None:
            try:
                data = uploaded_file.read()
                # safe size check (10 MB)
                max_bytes = 10 * 1024 * 1024
                if len(data) > max_bytes:
                    st.error("File too large. Max 10 MB.")
                else:
                    # upload_report should accept patient_id (string) and uploader id
                    fid = upload_report(data, uploaded_file.name, uploaded_file.type, patient_id, st.session_state.user_id)
                    st.success("Report uploaded. File id: " + str(fid))
                    # force a rerun to refresh listing
                    st.experimental_rerun()
            except Exception as e:
                print("DEBUG upload exception:", repr(e))
                st.error("Upload failed.")
        else:
            st.error("Please select a file to upload.")

    # List uploaded reports
    st.markdown("**Uploaded reports for this patient:**")
    try:
        total_reports = count_reports_for_patient(patient_id)
    except Exception:
        total_reports = 0

    per_page = 5
    page = st.number_input("Page", min_value=1, value=1, step=1, key="reports_page")
    skip = (page - 1) * per_page
    reports = list_reports_for_patient(patient_id, limit=per_page, skip=skip)

    if reports:
        for r in reports:
            fname = r.get("filename") or "unknown"
            uploaded_at = r.get("uploaded_at")
            ftype = (r.get("content_type") or "").lower()
            # show basic info
            colA, colB = st.columns([6, 1])
            with colA:
                st.write(f"**{fname}** — uploaded at {uploaded_at}")
            with colB:
                # unique key per file for buttons
                dl_key = f"dl_{str(r.get('file_id'))}_{page}"
                # Fetch file bytes (could be heavy — fetch on demand right before download)
                data, meta = get_report_file(r.get("file_id"))
                if data:
                    # If image — preview inline
                    if ftype.startswith("image/"):
                        try:
                            st.image(data, caption=fname, use_column_width=True)
                        except Exception:
                            st.write("(Image preview failed)")
                    # Provide direct download button (Streamlit will stream the bytes)
                    try:
                        st.download_button(label="Download", data=data, file_name=fname, mime=meta.get("content_type"), key=dl_key)
                    except Exception:
                        st.button("Download (prepare)", key=dl_key + "_btn")
                else:
                    st.error("Could not fetch file from storage.")
        # show page info
        start_num = skip + 1
        end_num = min(total_reports, skip + len(reports))
        st.write(f"Showing page {page} — {start_num} to {end_num} of {total_reports}")
    else:
        st.caption("No reports uploaded yet for this patient.")

# ==============================
# DOCTOR UI
# ==============================
def doctor_registration():
    st.subheader("👨‍⚕️ Doctor Registration")
    with st.form("doctor_reg_form"):
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username*")
            password = st.text_input("Password*", type="password")
            full_name = st.text_input("Full Name*")
            specialization = st.text_input("Specialization*", placeholder="e.g., Pulmonologist")
        with col2:
            hospital_clinic = st.text_input("Hospital/Clinic Name*")
            phone = st.text_input("Phone Number*")
            email = st.text_input("Email*")
        submitted = st.form_submit_button("Register")
    if submitted:
        if username and password and full_name and specialization and hospital_clinic:
            user_id = create_user(username, password, "doctor", email=email)
            if user_id:
                create_doctor(user_id, full_name, specialization, hospital_clinic, phone, email)
                st.success("Registration successful! Please login.")
                st.balloons()
                st.session_state.page = 'main'
                st.rerun()
            else:
                st.error("Username already exists!")
        else:
            st.error("Please fill all required fields!")

def doctor_login():
    st.subheader("🔐 Doctor Login")

    with st.form("doctor_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("Login", use_container_width=True)
        with c2:
            back = st.form_submit_button("Back", use_container_width=True)

    if submitted:
        user_id = verify_user(username, password, "doctor")
        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_role = "doctor"
            st.session_state.user_id = user_id
            st.toast("Welcome, Doctor!", icon="🩺")
            st.rerun()
        else:
            st.error("Invalid credentials!")

    if back:
        st.session_state.page = 'main'
        st.rerun()


def doctor_dashboard():
    drow = get_doctor_by_user_id(st.session_state.user_id)
    if not drow:
        st.info("No doctor profile found. Please register.")
        return

    doctor_id = drow["id"] if isinstance(drow, dict) else drow[0]
    doctor_spec = (drow.get("specialization") if isinstance(drow, dict) else drow[3] or "").strip()

    st.markdown(
        f"### 👨‍⚕️ Doctor Dashboard — <span class='pill'>{doctor_spec}</span>",
        unsafe_allow_html=True
    )
    st.write(f"Welcome, **Dr. {drow.get('full_name') if isinstance(drow, dict) else drow[2]}** ({doctor_spec})")

    # Logout button
    c1, _ = st.columns([1, 3])
    with c1:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.user_id = None
            st.toast("Logged out", icon="👋")
            st.rerun()

    st.divider()

    # Load triage
    latest_triage = get_latest_triage_per_patient()

    if not latest_triage.empty and doctor_spec:
        routed = latest_triage[
            latest_triage["recommended_specialty"].str.contains(
                doctor_spec, case=False, na=False
            )
        ].copy()
    else:
        routed = pd.DataFrame()

    # ---------- FIXED _is_taken FUNCTION ----------
    if not routed.empty:

        def _is_taken(pid):
            """
            Check if the patient already has an appointment with a doctor
            of the same specialty.
            """
            try:
                return patient_taken_by_specialty(pid, doctor_spec)
            except Exception:
                return False

        routed["taken_same_spec"] = routed["patient_id"].apply(_is_taken)
        triage_for_me = routed[~routed["taken_same_spec"]].drop(columns=["taken_same_spec"]).copy()
        hidden_count = int(routed["taken_same_spec"].sum())

    else:
        triage_for_me = pd.DataFrame()
        hidden_count = 0

    # UI section
    st.subheader("🧭 Triage (patients routed to you)")

    if hidden_count > 0:
        st.caption(f"ℹ️ {hidden_count} patient(s) hidden due to already having a doctor assigned in same specialty.")

    if not triage_for_me.empty:
        with st.expander("🔎 Filter", expanded=False):
            name_q = st.text_input("Patient name contains…")
            sev = st.multiselect("Severity", ["mild", "moderate", "serious"])
            df = triage_for_me.copy()
            if name_q:
                df = df[df["full_name"].str.contains(name_q, case=False, na=False)]
            if sev:
                df = df[df["severity"].isin(sev)]

        st.dataframe(df if "df" in locals() else triage_for_me,
                     use_container_width=True, hide_index=True)
    else:
        st.info("No routed patients available to you right now.")

    # ----------------------
    # Assignment UI
    # ----------------------
    st.divider()
    st.subheader("📅 Assign Meeting (Approval)")

    if triage_for_me.empty:
        st.caption("No eligible patients to schedule.")
    else:
        pat_opts = (
            triage_for_me[["patient_id", "full_name"]]
            .drop_duplicates()
            .sort_values("full_name")
        )

        # ⚠️ DO NOT CONVERT patient_id TO INT
        pat_display = {str(r["patient_id"]): r["full_name"] for _, r in pat_opts.iterrows()}

        colA, colB = st.columns(2)
        with colA:
            chosen_pid = st.selectbox(
                "Select patient",
                options=list(pat_display.keys()),
                format_func=lambda pid: f"{pat_display[pid]}"
            )

        with colB:
            meet_date = st.date_input("Meeting date", value=date.today())
            meet_time = st.time_input("Meeting time", value=time(10, 0))

        note = st.text_input("Note (optional)", placeholder="e.g., bring previous reports")

        if st.button("Assign Meeting", use_container_width=True):
            if patient_taken_by_specialty(chosen_pid, doctor_spec):
                st.error("This patient has already been assigned to another doctor of the same specialty.")
            else:
                meeting_dt = datetime.combine(meet_date, meet_time)
                create_appointment(
                    patient_id=chosen_pid,
                    doctor_id=doctor_id,
                    when_dt=meeting_dt,
                    notes=note.strip()
                )
                st.success("Meeting assigned! Patient will see this as Date of Approval.")
                st.toast("Appointment saved", icon="📅")
                st.rerun()

    # ----------------------
    # Notes Viewer
    # ----------------------
    st.divider()
    st.subheader("📋 Patient Notes (free text) — only your routed patients")

    notes_df = get_all_symptom_records_df()
    if not notes_df.empty and not triage_for_me.empty:
        allowed_ids = set(triage_for_me["patient_id"].astype(str))
        notes_df["patient_id"] = notes_df["patient_id"].astype(str)
        notes_df = notes_df[notes_df["patient_id"].isin(allowed_ids)].copy()
    else:
        notes_df = pd.DataFrame()

    if not notes_df.empty:
        st.dataframe(
            notes_df.drop(columns=["patient_id"]),
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            label="📥 Download Notes (CSV)",
            data=notes_df.drop(columns=["patient_id"]).to_csv(index=False),
            file_name=f"patient_notes_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No notes for your routed patients yet.")
# ==============================
# MAIN & SESSION
# ==============================
def init_session_state():
    defaults = {
        'logged_in': False,
        'user_role': None,
        'user_id': None,
        'page': 'main',
        'theme': 'Azure',
        'dark': False,
        'chat_messages': [],
        'checker_key': 0,
        'show_chat': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def logout_reset():
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.user_id = None
    st.session_state.page = 'main'
    st.session_state.chat_messages = []
    st.session_state.show_chat = False
    st.toast("Logged out", icon="👋")
    st.rerun()

def admin_stats_panel():
    st.subheader("📊 Statistics & Exports")
    stats = {
        'users': users_coll.count_documents({}),
        'patients': patients_coll.count_documents({}),
        'doctors': doctors_coll.count_documents({}),
        'appointments': appointments_coll.count_documents({}),
        'symptom_records': symptom_records_coll.count_documents({}),
        'reports': reports_coll.count_documents({}),
    }
    cols = st.columns(3)
    cols[0].metric("Users", stats['users'])
    cols[1].metric("Patients", stats['patients'])
    cols[2].metric("Doctors", stats['doctors'])
    cols2 = st.columns(3)
    cols2[0].metric("Appointments", stats['appointments'])
    cols2[1].metric("Symptom Records", stats['symptom_records'])
    cols2[2].metric("Reports", stats['reports'])

    st.markdown("**Recent symptom records**")
    for s in list(symptom_records_coll.find().sort("recorded_at", -1).limit(5)):
        s = serialize_doc(s)
        st.write(f"- {s.get('patient_id')} | {str(s.get('symptoms'))[:80]} | {s.get('recorded_at')}")

    st.markdown("**Recent reports**")
    for r in list(reports_coll.find().sort("uploaded_at", -1).limit(5)):
        r = serialize_doc(r)
        st.write(f"- {r.get('filename')} — patient {r.get('patient_id')} — {r.get('uploaded_at')}")

    st.markdown("---")
    if st.button("Export all patients (CSV)"):
        rows = list(patients_coll.find())
        if rows:
            csv = pd.DataFrame([serialize_doc(r) for r in rows]).to_csv(index=False)
            st.download_button("Download patients CSV", data=csv, file_name="patients_export.csv", mime="text/csv")
        else:
            st.info("No patients to export.")

def main():
    st.set_page_config(
        page_title="Health Management System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_db()
    init_session_state()

    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        st.session_state.theme = st.selectbox("Theme", list(THEMES.keys()),
                                              index=list(THEMES.keys()).index(st.session_state.theme))
        st.session_state.dark = st.toggle("Dark mode", value=st.session_state.dark)
        st.caption("Applied instantly ✨")
        st.markdown("---")
        if st.button("Admin: Stats & Exports"):
            st.session_state.page = "admin_stats"

    inject_css(st.session_state.theme, st.session_state.dark)
    hero("Cloud-Based Health Management System",
         "Patient Disease Checker + Doctor Portal with Scheduling (single app).")

    if st.session_state.logged_in:
        if st.session_state.user_role == "patient":
            patient_dashboard()
        elif st.session_state.user_role == "doctor":
            doctor_dashboard()
        return

    # Landing
    c1, c2 = st.columns(2)
    with c1:
        role_card("👤", "Patient",
                  "Tell us the disease you think you have. We’ll triage and a suitable doctor can approve a meeting.",
                  [("Patient Login", "patient_login"), ("Patient Registration", "patient_registration")])
    with c2:
        role_card("👨‍⚕️", "Doctor",
                  "See only patients routed to your specialty (excluding those already assigned) and assign meetings.",
                  [("Doctor Login", "doctor_login"), ("Doctor Registration", "doctor_registration")])

    page = st.session_state.page
    if page == 'patient_login':
        st.divider(); patient_login()
    elif page == 'patient_registration':
        st.divider(); patient_registration()
    elif page == 'doctor_login':
        st.divider(); doctor_login()
    elif page == 'doctor_registration':
        st.divider(); doctor_registration()
    elif page == "admin_stats":
        st.divider(); admin_stats_panel()

    st.divider()
    st.markdown(
        "<div style='text-align:center;color:var(--muted);padding:8px;'>"
        "🏥 HMS • Educational triage only — not a medical diagnosis."
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
