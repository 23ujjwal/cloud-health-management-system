from fastapi import FastAPI, Form, Request, Response, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import jwt, JWTError
from passlib.context import CryptContext
from datetime import datetime, timedelta
from google.cloud import firestore

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
SECRET_KEY = "SUPER_SECRET_KEY_CHANGE_THIS"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Firestore setup
db = firestore.Client()
users_ref = db.collection("Users")

app = FastAPI(title="Health Record System – Login Page")

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str):
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict, expires_delta=None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_email(email: str):
    doc = users_ref.document(email).get()
    return doc.to_dict() if doc.exists else None

def create_user(email, name, hashed_pw):
    users_ref.document(email).set({
        "email": email,
        "name": name,
        "password_hash": hashed_pw,
        "created_at": datetime.utcnow().isoformat()
    })

def delete_user(email):
    users_ref.document(email).delete()

def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# -------------------------------------------------------
# HTML Layout
# -------------------------------------------------------
def html_base(title: str, body: str):
    return f"""
    <!doctype html>
    <html lang='en'>
    <head>
      <meta charset='utf-8'/>
      <title>{title}</title>
      <meta name='viewport' content='width=device-width,initial-scale=1'>
      <style>
        body {{ font-family: Arial, sans-serif; background:#f3f4f6; }}
        .container {{ max-width:480px; margin:60px auto; background:white;
                      padding:24px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.08); }}
        h1 {{ font-size:22px; color:#1f2937; margin-bottom:16px; }}
        label {{ display:block; margin-top:12px; font-size:13px; color:#374151; }}
        input[type="text"], input[type="email"], input[type="password"] {{
          width:100%; padding:10px; margin-top:6px; border:1px solid #ddd; border-radius:6px;
        }}
        button {{ margin-top:16px; padding:10px 14px; border:none; border-radius:6px;
                  background:#2563eb; color:white; cursor:pointer; }}
        .error {{ color:#dc2626; font-weight:600; margin-top:10px; }}
        .muted {{ color:#6b7280; font-size:13px; margin-top:10px; }}
        a {{ color:#2563eb; text-decoration:none; }}
      </style>
    </head>
    <body><div class="container">{body}</div></body>
    </html>
    """

# -------------------------------------------------------
# ROUTES
# -------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse(url="/login")

# LOGIN
@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = None):
    msg = f"<div class='error'>{error}</div>" if error else ""
    html = f"""
    <h1>Health Record System — Login Page</h1>
    {msg}
    <form method='post' action='/login'>
      <label>Email</label>
      <input type='email' name='email' required placeholder='you@example.com'>
      <label>Password</label>
      <input type='password' name='password' required>
      <button type='submit'>Login</button>
    </form>
    <div class='muted'>New user? <a href='/register'>Create an account</a></div>
    """
    return html_base("Login – Health Record System", html)

@app.post("/login", response_class=HTMLResponse)
def login(response: Response, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email)
    if not user or not verify_password(password, user["password_hash"]):
        return login_page("Invalid email or password")
    token = create_access_token({"sub": email})
    resp = RedirectResponse(url="/dashboard", status_code=302)
    resp.set_cookie("access_token", token, httponly=True)
    return resp

# REGISTER
@app.get("/register", response_class=HTMLResponse)
def register_page(error: str = None):
    msg = f"<div class='error'>{error}</div>" if error else ""
    html = f"""
    <h1>Register — Health Record System</h1>
    {msg}
    <form method='post' action='/register'>
      <label>Full Name</label>
      <input type='text' name='name' required placeholder='Your name'>
      <label>Email</label>
      <input type='email' name='email' required placeholder='you@example.com'>
      <label>Password</label>
      <input type='password' name='password' required>
      <label>Confirm Password</label>
      <input type='password' name='confirm_password' required>
      <button type='submit'>Register</button>
    </form>
    <div class='muted'>Already have an account? <a href='/login'>Login</a></div>
    """
    return html_base("Register – Health Record System", html)

@app.post("/register", response_class=HTMLResponse)
def register(name: str = Form(...), email: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password:
        return register_page("Passwords do not match")
    if get_user_by_email(email):
        return register_page("User already exists")
    if len(password) < 6:
        return register_page("Password must be at least 6 characters")
    hashed = hash_password(password)
    create_user(email, name, hashed)
    return RedirectResponse(url="/login", status_code=302)

# DASHBOARD
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(user: dict = Depends(get_current_user)):
    html = f"""
    <h1>Welcome, {user['name']}</h1>
    <p><b>Email:</b> {user['email']}</p>
    <p><b>Account created:</b> {user['created_at']}</p>

    <form method='post' action='/logout'>
      <button type='submit'>Logout</button>
    </form>
    <form method='post' action='/delete-account'>
      <button type='submit' style='background:#ef4444;margin-top:10px;'>Delete Account</button>
    </form>
    """
    return html_base("Patient Dashboard", html)

# LOGOUT
@app.post("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp

# DELETE ACCOUNT
@app.post("/delete-account")
def delete_account(user: dict = Depends(get_current_user)):
    delete_user(user["email"])
    resp = RedirectResponse(url="/register", status_code=302)
    resp.delete_cookie("access_token")
    return resp
