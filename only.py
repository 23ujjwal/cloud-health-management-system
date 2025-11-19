# app.py
# Basic Streamlit login/register/delete using FILE HANDLING (users.json)
# No Google Cloud, no DB. 2nd-year friendly.

import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from werkzeug.security import generate_password_hash, check_password_hash
import streamlit as st

USERS_FILE = Path("users.json")


# ------------------ FILE HELPERS ------------------
def load_users() -> Dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_users(data: Dict[str, dict]) -> None:
    USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def user_exists(email: str) -> bool:
    users = load_users()
    return email in users


def create_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    email = email.strip().lower()
    if not name or not email or not password:
        return False, "Please fill all fields."

    users = load_users()
    if email in users:
        return False, "Email already exists."

    users[email] = {
        "name": name.strip(),
        "password_hash": generate_password_hash(password.strip()),
    }
    save_users(users)
    return True, "Account created successfully!"


def verify_login(email: str, password: str) -> Tuple[bool, Optional[str]]:
    email = email.strip().lower()
    users = load_users()
    user = users.get(email)
    if user and check_password_hash(user["password_hash"], password.strip()):
        return True, user["name"]
    return False, None


def delete_user(email: str, password: str) -> Tuple[bool, str]:
    email = email.strip().lower()
    users = load_users()
    user = users.get(email)
    if not user:
        return False, "No account found with this email."

    if not check_password_hash(user["password_hash"], password.strip()):
        return False, "Incorrect password."

    users.pop(email, None)
    save_users(users)
    return True, "Account deleted successfully."


# ------------------ UI SECTIONS ------------------
def page_home():
    st.title("Health Record System — Basic (File Handling)")
    st.write(
        "This demo stores users in a local **users.json** file. "
        "Features: **Register / Login / Delete Account**."
    )
    st.info(
        "Note: JSON is for simple demos only. For real projects, use a database."
    )


def page_register():
    st.subheader("Register")
    with st.form("register-form", clear_on_submit=False):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        pw2 = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if pw != pw2:
            st.error("Passwords do not match!")
        else:
            ok, msg = create_user(name, email, pw)
            st.success(msg) if ok else st.warning(msg)


def page_login():
    st.subheader("Login")
    with st.form("login-form", clear_on_submit=False):
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        ok, name = verify_login(email, pw)
        if ok:
            st.session_state["auth"] = True
            st.session_state["email"] = email.strip().lower()
            st.session_state["name"] = name
            st.success(f"Welcome, {name}! 🎉")
            st.rerun()
        else:
            st.error("Invalid email or password.")


def page_delete():
    st.subheader("Delete Account")
    # If logged in, lock the email field to the current user for safety
    if st.session_state.get("auth"):
        email = st.text_input("Email", value=st.session_state["email"], disabled=True)
    else:
        email = st.text_input("Email")

    pw = st.text_input("Password", type="password")
    if st.button("Delete my account"):
        e = st.session_state["email"] if st.session_state.get("auth") else email
        ok, msg = delete_user(e, pw)
        if ok:
            st.success(msg)
            if st.session_state.get("auth"):
                st.session_state.clear()
                st.rerun()
        else:
            st.error(msg)


def page_dashboard():
    st.subheader("Dashboard")
    st.info("Logged in successfully. Add your features here later.")
    st.write(
        f"**Name:** {st.session_state.get('name','—')}  \n"
        f"**Email:** {st.session_state.get('email','—')}"
    )


def sidebar_menu() -> str:
    st.sidebar.title("Menu")
    options = ["Home", "Register", "Login", "Delete Account"]
    if st.session_state.get("auth"):
        options.append("Dashboard")
    choice = st.sidebar.radio("Go to", options, index=0)

    if st.session_state.get("auth"):
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()
    return choice


# ------------------ MAIN ------------------
def main():
    st.set_page_config(page_title="Basic Streamlit (File Handling)", page_icon="🗂️")
    if "auth" not in st.session_state:
        st.session_state["auth"] = False

    choice = sidebar_menu()

    if choice == "Home":
        page_home()
    elif choice == "Register":
        page_register()
    elif choice == "Login":
        if st.session_state.get("auth"):
            st.success("Already logged in.")
            page_dashboard()
        else:
            page_login()
    elif choice == "Delete Account":
        page_delete()
    elif choice == "Dashboard":
        if st.session_state.get("auth"):
            page_dashboard()
        else:
            st.error("Please login first.")

    st.markdown("---")
    st.caption(f"Users file: `{USERS_FILE.resolve()}`")


if __name__ == "__main__":
    main()
