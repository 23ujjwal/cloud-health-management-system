# app.py
# Basic Flask site (Register/Login/Delete) using FILE HANDLING (users.json)
# No databases, no cloud. Perfect for a simple webpage demo.

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template_string, request, redirect, url_for, session, flash

USERS_FILE = Path("users.json")

app = Flask(__name__)
app.secret_key = "change-this-in-production"  # required for sessions


# ------------------ FILE HELPERS ------------------
def load_users() -> Dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_users(data: Dict[str, dict]) -> None:
    USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def user_exists(email: str) -> bool:
    return email in load_users()

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
        # add more fields later (role, etc.)
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


# ------------------ TEMPLATES ------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title or 'App' }}</title>
  <style>
    body{font-family:system-ui, Arial;margin:0;background:#f8fafc;color:#111}
    header{background:#0ea5e9;color:#fff;padding:12px 18px;display:flex;justify-content:space-between;align-items:center}
    a{color:#0ea5e9;text-decoration:none}
    .btn{padding:6px 10px;border:1px solid #0ea5e9;border-radius:6px;display:inline-block}
    .btn.primary{background:#0ea5e9;color:#fff}
    .container{max-width:720px;margin:20px auto;padding:0 16px}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin:10px 0}
    input{width:100%;padding:8px;margin:6px 0;border:1px solid #cbd5e1;border-radius:6px}
    .row{display:flex;gap:8px;flex-wrap:wrap}
    .flash{background:#fef3c7;border:1px solid #f59e0b;padding:8px;border-radius:6px;margin:10px 0}
    .muted{color:#6b7280;font-size:12px}
  </style>
</head>
<body>
  <header>
    <div><strong>Basic Web Login</strong></div>
    <nav>
      <a href="{{ url_for('home') }}">Home</a>
      {% if session.get('auth') %}
        | <span class="muted">{{ session.get('name') }} ({{ session.get('email') }})</span>
        | <a href="{{ url_for('dashboard') }}">Dashboard</a>
        | <a href="{{ url_for('logout') }}">Logout</a>
      {% else %}
        | <a href="{{ url_for('login') }}">Login</a>
        | <a href="{{ url_for('register') }}">Register</a>
      {% endif %}
      | <a href="{{ url_for('delete_account') }}">Delete</a>
    </nav>
  </header>
  <div class="container">
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        {% for m in messages %}<div class="flash">{{ m }}</div>{% endfor %}
      {% endif %}
    {% endwith %}
    {% block content %}{% endblock %}
  </div>
</body>
</html>
"""

HOME_HTML = """
{% extends 'base.html' %}
{% block content %}
  <h2>Welcome</h2>
  <div class="card">
    <p>This is a simple **web page** version using Flask + users.json for storage.</p>
    <p>Features: Register, Login, Delete Account, Dashboard.</p>
  </div>
{% endblock %}
"""

REGISTER_HTML = """
{% extends 'base.html' %}
{% block content %}
<h2>Register</h2>
<form method="post" class="card">
  <input name="name" placeholder="Full name" required>
  <input name="email" placeholder="Email" required>
  <input name="password" placeholder="Password" type="password" required>
  <input name="confirm" placeholder="Confirm password" type="password" required>
  <button class="btn primary">Create Account</button>
</form>
{% endblock %}
"""

LOGIN_HTML = """
{% extends 'base.html' %}
{% block content %}
<h2>Login</h2>
<form method="post" class="card">
  <input name="email" placeholder="Email" required>
  <input name="password" placeholder="Password" type="password" required>
  <button class="btn primary">Login</button>
</form>
{% endblock %}
"""

DELETE_HTML = """
{% extends 'base.html' %}
{% block content %}
<h2>Delete Account</h2>
<form method="post" class="card">
  <input name="email" placeholder="Email" value="{{ session.get('email','') }}" {% if session.get('auth') %}readonly{% endif %} required>
  <input name="password" placeholder="Password" type="password" required>
  <button class="btn">Delete my account</button>
</form>
{% endblock %}
"""

DASHBOARD_HTML = """
{% extends 'base.html' %}
{% block content %}
<h2>Dashboard</h2>
<div class="card">
  <p><strong>Name:</strong> {{ session.get('name') }}</p>
  <p><strong>Email:</strong> {{ session.get('email') }}</p>
  <p class="muted">Add more pages/features here later.</p>
</div>
{% endblock %}
"""


# ------------------ ROUTES ------------------
from jinja2 import DictLoader
app.jinja_loader = DictLoader({
    "base.html": BASE_HTML,
    "home.html": HOME_HTML,
    "register.html": REGISTER_HTML,
    "login.html": LOGIN_HTML,
    "delete.html": DELETE_HTML,
    "dashboard.html": DASHBOARD_HTML,
})

@app.route("/")
def home():
    return render_template_string(app.jinja_loader.get_source(app.jinja_env, "home.html")[0], title="Home")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip().lower()
        pw = request.form.get("password","").strip()
        confirm = request.form.get("confirm","").strip()
        if pw != confirm:
            flash("Passwords do not match!")
        else:
            ok, msg = create_user(name, email, pw)
            flash(msg)
            if ok:
                return redirect(url_for("login"))
    return render_template_string(app.jinja_loader.get_source(app.jinja_env, "register.html")[0], title="Register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        pw = request.form.get("password","").strip()
        ok, name = verify_login(email, pw)
        if ok:
            session["auth"] = True
            session["email"] = email
            session["name"] = name
            flash(f"Welcome, {name}! 🎉")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.")
    return render_template_string(app.jinja_loader.get_source(app.jinja_env, "login.html")[0], title="Login")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("home"))

@app.route("/delete", methods=["GET", "POST"])
def delete_account():
    if request.method == "POST":
        email = (session.get("email") if session.get("auth") else request.form.get("email","").strip().lower())
        pw = request.form.get("password","").strip()
        ok, msg = delete_user(email, pw)
        flash(msg)
        if ok:
            session.clear()
            return redirect(url_for("home"))
    return render_template_string(app.jinja_loader.get_source(app.jinja_env, "delete.html")[0], title="Delete Account")

@app.route("/dashboard")
def dashboard():
    if not session.get("auth"):
        flash("Please login first.")
        return redirect(url_for("login"))
    return render_template_string(app.jinja_loader.get_source(app.jinja_env, "dashboard.html")[0], title="Dashboard")


if __name__ == "__main__":
    # host='0.0.0.0' lets it work on your LAN too; remove if not needed
    app.run(debug=True, host="0.0.0.0", port=5000)
