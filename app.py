# app.py (updated)
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from pathlib import Path
import os
import io
import csv
import pandas as pd
import json
import sqlite3
from datetime import datetime
from flask import jsonify

# suppress noisy ML logs early
import warnings, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf.*")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("werkzeug").setLevel(logging.INFO)

# local paths: app.py is in project root
ROOT = Path(__file__).resolve().parent

# Import your Predictor implementation (keeps existing behavior)
from models.predictor import Predictor

# ML utilities
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# password hashing
import bcrypt

# config
DB_PATH = os.environ.get("FPD_DB_PATH", str(ROOT / "app_data.db"))
MODELS_DIR = os.environ.get("FPD_MODELS_DIR", str(ROOT / "models"))
ACTIVE_MODEL_PATH = os.path.join(MODELS_DIR, "active_model.pkl")
AUTO_FLAG_THRESHOLD = float(os.environ.get("FPD_AUTO_FLAG_THRESHOLD", 0.5))
ADMIN_USERNAME = os.environ.get("FPD_DEFAULT_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("FPD_DEFAULT_ADMIN_PASS", "admin")

os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "replace-this-with-secure-key")
# make some handy values available to all templates
@app.context_processor
def inject_globals():
    # provide a timezone-naive UTC now and current year for templates
    from datetime import datetime as _dt
    now = _dt.utcnow()
    # also expose session username and whether current user is admin (templates use session too)
    return {
        "now": now,
        "current_year": now.year,
        # convenience: expose current user object if available
        "current_user": current_user() if 'current_user' in globals() else None,
        # expose flask session so templates that used session.username still work
        "session": session
    }


# instantiate your Predictor (will load models you already have)
P = Predictor()

# ---------- Database helpers ----------
# ---------- Database helpers (use flask.g to avoid cross-thread SQLite issues) ----------
import sqlite3
from flask import g

def get_db():
    """
    Returns a per-request SQLite connection stored on flask.g.
    This avoids "SQLite objects created in a thread can only be used in that same thread."
    """
    if "db_conn" not in g:
        # allow connections to be used by other threads if necessary by setting check_same_thread=False
        # but prefer to keep per-request connections (so this flag is not strictly required). We set it to False
        # to be robust to dev-server reloader behavior.
        g.db_conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        g.db_conn.row_factory = sqlite3.Row
        g.db_conn.execute("PRAGMA foreign_keys = ON;")
    return g.db_conn

def close_db(e=None):
    db_conn = g.pop("db_conn", None)
    if db_conn is not None:
        try:
            db_conn.close()
        except Exception:
            pass

# register teardown so connection is closed after each request context
app.teardown_appcontext(close_db)

def init_db():
    # Create tables idempotently
    db = get_db()
    sql = """
    BEGIN;
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT,
        password_hash BLOB NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        is_blocked INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        last_login TEXT,
        notes TEXT
    );
    CREATE TABLE IF NOT EXISTS flagged_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reported_by_user_id INTEGER,
        profile_id_or_handle TEXT,
        profile_raw TEXT,
        model_score REAL,
        model_prediction TEXT,
        status TEXT DEFAULT 'pending',
        admin_action TEXT,
        admin_notes TEXT,
        created_at TEXT,
        reviewed_at TEXT,
        reviewed_by_admin_id INTEGER,
        FOREIGN KEY (reported_by_user_id) REFERENCES users(id),
        FOREIGN KEY (reviewed_by_admin_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        profile_raw TEXT,
        score REAL,
        prediction TEXT,
        created_at TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_name TEXT,
        trained_by_admin_id INTEGER,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        model_file TEXT,
        created_at TEXT,
        notes TEXT,
        FOREIGN KEY (trained_by_admin_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS admin_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        admin_id INTEGER,
        action TEXT,
        details TEXT,
        ip TEXT,
        created_at TEXT,
        FOREIGN KEY (admin_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS blocked_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_handle TEXT UNIQUE NOT NULL,
        profile_raw TEXT,
        blocked_by_admin_id INTEGER,
        blocked_at TEXT,
        reason TEXT,
        flagged_profile_id INTEGER,
        FOREIGN KEY (blocked_by_admin_id) REFERENCES users(id),
        FOREIGN KEY (flagged_profile_id) REFERENCES flagged_profiles(id)
    );
    COMMIT;
    """
    db.executescript(sql)
    db.commit()
    
    # Run migrations for existing databases
    _run_migrations(db)

def _run_migrations(db):
    """Apply schema migrations for existing databases"""
    try:
        # Check if admin_notes column exists in flagged_profiles
        cursor = db.execute("PRAGMA table_info(flagged_profiles)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'admin_notes' not in columns:
            print("[MIGRATION] Adding admin_notes column to flagged_profiles...")
            db.execute("ALTER TABLE flagged_profiles ADD COLUMN admin_notes TEXT")
            db.commit()
            print("[MIGRATION] admin_notes column added successfully")
        
        if 'admin_action' not in columns:
            print("[MIGRATION] Adding admin_action column to flagged_profiles...")
            db.execute("ALTER TABLE flagged_profiles ADD COLUMN admin_action TEXT")
            db.commit()
            print("[MIGRATION] admin_action column added successfully")
            
    except Exception as e:
        print(f"[MIGRATION] Migration failed (non-fatal): {e}")

def seed_default_admin():
    """
    Create default admin (admin:admin) only once. Uses same per-request db helper.
    Call this after init_db() in startup (remember that this runs in the main thread)
    """
    db = get_db()
    row = db.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USERNAME,)).fetchone()
    if row:
        return
    pw_hash = bcrypt.hashpw(ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt())
    db.execute("INSERT INTO users (username, email, password_hash, role, is_blocked, created_at) VALUES (?, ?, ?, ?, ?, ?)",
               (ADMIN_USERNAME, f"{ADMIN_USERNAME}@example.com", pw_hash, "admin", 0, datetime.utcnow().isoformat()))
    db.commit()
    print(f"[INIT] seeded default admin -> {ADMIN_USERNAME}:{ADMIN_PASSWORD}")


# ----- AJAX endpoints for user dashboard integration -----
# Paste into app.py (after imports and after get_db/predict_with_model/ helpers)

@app.route("/predict_ajax", methods=["POST"])
def predict_ajax():
    """
    Accepts either:
      - raw_json: JSON body or form field 'raw_json' (string) OR
      - form fields f1..f4 OR manual CLI-like profile fields (screen_name, followers_count, ...)

    Returns JSON: { score: float, label: "FAKE"/"GENUINE", profile: {...} }
    """
    if not is_logged_in() or is_admin():
        return jsonify({"error": "authentication required"}), 401

    # prefer JSON request body
    if request.is_json:
        payload = request.get_json()
    else:
        payload = {}
        raw_json = request.form.get("raw_json", "").strip()
        if raw_json:
            try:
                payload = json.loads(raw_json)
            except Exception:
                return jsonify({"error": "invalid raw_json"}), 400
        else:
            # try the CLI-style manual fields
            # check both f1..f4 and the detailed fields
            if any(k in request.form for k in ["f1","f2","f3","f4"]):
                payload = {k: float(request.form.get(k, 0) or 0) for k in ["f1","f2","f3","f4"]}
            else:
                # detailed profile fields
                payload = {
                    "screen_name": request.form.get("screen_name","manual_user"),
                    "description": request.form.get("description",""),
                    "followers_count": int(request.form.get("followers_count") or 0),
                    "friends_count": int(request.form.get("friends_count") or 0),
                    "statuses_count": int(request.form.get("statuses_count") or 0),
                    "favourites_count": int(request.form.get("favourites_count") or 0),
                    "listed_count": int(request.form.get("listed_count") or 0),
                    "utc_offset": int(request.form.get("utc_offset") or 0),
                    "profile_image_url": request.form.get("profile_image_url",""),
                    "lang_label": int(request.form.get("lang_label") or 0)
                }

    # run prediction using existing predictor wrapper
    try:
        # Prefer the project-level wrapper predict_with_model if present
        if 'predict_with_model' in globals():
            score, label = predict_with_model(payload)
        else:
            out = P.predict_manual(payload, screen_name=payload.get("screen_name"))
            label = out.get("label") or out.get("prediction") or "GENUINE"
            score = float(out.get("score") or out.get("final_score") or 0.0)
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 500

    # store prediction in DB (best-effort)
    try:
        db = get_db()
        db.execute("INSERT INTO predictions (user_id, profile_raw, score, prediction, created_at) VALUES (?, ?, ?, ?, ?)",
                   (session.get("user_id"), json.dumps(payload), float(score), label, datetime.utcnow().isoformat()))
        db.commit()
    except Exception:
        app.logger.debug("Failed to save prediction to DB (non-fatal)")

    # Auto-flag if FAKE - this is the workflow where fake profiles are flagged for admin review
    if label == "FAKE":
        try:
            db = get_db()
            # Check if already flagged by this user
            existing = db.execute("""SELECT id FROM flagged_profiles 
                                    WHERE reported_by_user_id = ? AND profile_id_or_handle = ?""",
                                 (session.get("user_id"), payload.get("screen_name"))).fetchone()
            if not existing:
                db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, model_score, model_prediction, status, created_at)
                              VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                           (session.get("user_id"), payload.get("screen_name") or payload.get("profile_id") or "", json.dumps(payload), float(score), label, datetime.utcnow().isoformat()))
                db.commit()
                app.logger.info(f"Auto-flagged fake profile: {payload.get('screen_name')} by user {session.get('user_id')}")
        except Exception as e:
            app.logger.debug(f"Auto-flag failed (non-fatal): {e}")

    return jsonify({"score": float(score), "label": label, "profile": payload})

@app.route("/test_ajax", methods=["POST"])
def test_ajax():
    """
    Two modes:
      - upload CSV file via FormData (field 'upload_csv') -> run prediction on first row or search input
      - choose existing test.csv and pass 'search_input' (form field) to select row by ID or screen_name
    Returns JSON: { score, label, profile, search_value }
    """
    if not is_logged_in() or is_admin():
        return jsonify({"error": "authentication required"}), 401

    test_csv_path = ROOT / "data" / "test" / "test.csv"
    # If a file was uploaded, replace test_csv_path for this call
    if "upload_csv" in request.files and request.files["upload_csv"].filename:
        f = request.files["upload_csv"]
        temp_path = ROOT / "data" / "test" / f.filename
        f.save(str(temp_path))
        csv_path = temp_path
    else:
        csv_path = test_csv_path

    if not csv_path.exists():
        return jsonify({"error": f"test CSV not found at {csv_path}"}), 400

    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception as e:
        return jsonify({"error": f"failed to read csv: {e}"}), 500

    # Get search input - can be ID or screen_name
    search_input = request.form.get("search_input","").strip()
    
    # Legacy support for old 'id_input' parameter
    if not search_input:
        search_input = request.form.get("id_input","").strip()
    
    if search_input == "":
        # If empty, use first row
        row = df.iloc[0]
        search_value = row.get("id", "") or row.get("screen_name", "")
    else:
        # Try to find by ID first (if numeric)
        row = None
        if "id" in df.columns:
            match = df[df["id"].astype(str) == search_input]
            if match.empty:
                # Try case-insensitive ID match
                match = df[df["id"].astype(str).str.lower() == search_input.lower()]
            if not match.empty:
                row = match.iloc[0]
        
        # If not found by ID, try screen_name
        if row is None and "screen_name" in df.columns:
            match = df[df["screen_name"].astype(str) == search_input]
            if match.empty:
                # Try case-insensitive screen_name match
                match = df[df["screen_name"].astype(str).str.lower() == search_input.lower()]
            if match.empty:
                # Try partial match (contains)
                match = df[df["screen_name"].astype(str).str.lower().str.contains(search_input.lower())]
            if not match.empty:
                row = match.iloc[0]
        
        if row is None:
            return jsonify({"error": f"No profile found with ID or username '{search_input}' in test.csv"}), 404
        
        search_value = search_input

    # build profile (same mapping as your test_mode)
    profile = {
        "screen_name": row.get("screen_name","") or row.get("profile_name","") or f"test_{row.get('id','')}",
        "description": row.get("description","") or "",
        "followers_count": int(float(row.get("followers_count",0) or 0)),
        "friends_count": int(float(row.get("friends_count",0) or 0)),
        "statuses_count": int(float(row.get("statuses_count",0) or 0)),
        "favourites_count": int(float(row.get("favourites_count",0) or 0)),
        "listed_count": int(float(row.get("listed_count",0) or 0)),
        "utc_offset": int(float(row.get("utc_offset",0) or 0)),
        "profile_image_url": row.get("profile_image_url","") or "",
        "lang_label": int(float(row.get("lang_label",0) or 0)) if row.get("lang_label","")!="" else 0
    }

    try:
        if 'predict_with_model' in globals():
            score, label = predict_with_model(profile)
        else:
            out = P.predict_manual(profile, screen_name=profile.get("screen_name"))
            score = float(out.get("score") or out.get("final_score") or 0.0)
            label = out.get("label") or out.get("prediction") or ("FAKE" if score >= AUTO_FLAG_THRESHOLD else "GENUINE")
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 500

    # Save prediction entry
    try:
        db = get_db()
        db.execute("INSERT INTO predictions (user_id, profile_raw, score, prediction, created_at) VALUES (?, ?, ?, ?, ?)",
                   (session.get("user_id"), json.dumps(profile), float(score), label, datetime.utcnow().isoformat()))
        db.commit()
    except Exception:
        pass

    # Auto-flag if FAKE - this is the workflow where fake profiles are flagged for admin review
    if label == "FAKE":
        try:
            db = get_db()
            # Check if already flagged by this user
            existing = db.execute("""SELECT id FROM flagged_profiles 
                                    WHERE reported_by_user_id = ? AND profile_id_or_handle = ?""",
                                 (session.get("user_id"), profile.get("screen_name"))).fetchone()
            if not existing:
                db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, model_score, model_prediction, status, created_at)
                              VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                           (session.get("user_id"), profile.get("screen_name") or "", json.dumps(profile), float(score), label, datetime.utcnow().isoformat()))
                db.commit()
                app.logger.info(f"Auto-flagged fake profile: {profile.get('screen_name')} by user {session.get('user_id')}")
        except Exception as e:
            app.logger.debug(f"Auto-flag failed (non-fatal): {e}")

    return jsonify({"score": float(score), "label": label, "profile": profile, "search_value": search_value})

@app.route("/report_ajax", methods=["POST"])
def report_ajax():
    """Report a profile to admin via AJAX. Expects 'profile_id' and optional 'profile_raw'."""
    if not is_logged_in() or is_admin():
        return jsonify({"error":"authentication required"}), 401
    profile_id = request.form.get("profile_id","").strip()
    profile_raw = request.form.get("profile_raw","").strip() or None
    notes = request.form.get("notes","").strip() or None
    if request.is_json:
        body = request.get_json(silent=True) or {}
        profile_id = body.get("profile_id", profile_id)
        profile_raw = body.get("profile_raw", profile_raw)
        notes = body.get("notes", notes)
    if not profile_id:
        return jsonify({"error":"profile_id required"}), 400
    try:
        db = get_db()
        db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, admin_notes, created_at)
                      VALUES (?, ?, ?, ?, ?)""",
                   (session.get("user_id"), profile_id, profile_raw, notes, datetime.utcnow().isoformat()))
        db.commit()
        return jsonify({"ok": True, "message": "Reported to admin"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ----------------------------
# User: fetch my reports/status
# ----------------------------
@app.route("/user/reports_json", methods=["GET"])
def user_reports_json():
    if not is_logged_in():
        return jsonify({"error": "authentication required"}), 401

    try:
        db = get_db()   # uses your existing DB helper
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "no user id in session"}), 400

        rows = db.execute("""
            SELECT id,
                   profile_id_or_handle,
                   COALESCE(model_prediction,'') AS model_prediction,
                   COALESCE(model_score, 0.0) AS model_score,
                   COALESCE(status,'pending') AS status,
                   COALESCE(admin_action,'') AS admin_action,
                   COALESCE(admin_notes,'') AS admin_notes,
                   created_at,
                   reviewed_at
            FROM flagged_profiles
            WHERE reported_by_user_id = ? 
            AND model_prediction = 'FAKE'
            ORDER BY created_at DESC
        """, (user_id,)).fetchall()

        out = []
        for r in rows:
            out.append({
                "id": r["id"],
                "profile": r["profile_id_or_handle"],
                "prediction": r["model_prediction"],
                "score": float(r["model_score"]) if r["model_score"] is not None else 0.0,
                "status": r["status"],
                "admin_notes": r["admin_notes"] or "",
                "created_at": r["created_at"],
                "reviewed_at": r["reviewed_at"]
            })
        return jsonify({"ok": True, "rows": out})
    except Exception as e:
        # return error to client for debugging; remove or hide message in production
        return jsonify({"error": str(e)}), 500
# POST /report_retract - user withdraws their own report
@app.route("/report_retract", methods=["POST"])
def report_retract():
    if not is_logged_in():
        return jsonify({"error": "authentication required"}), 401

    fid = request.form.get("fid") or request.json.get("fid") if request.is_json else None
    if not fid:
        return jsonify({"error": "missing report id (fid)"}), 400

    try:
        db = get_db()
        # verify ownership
        row = db.execute("SELECT reported_by_user_id, status FROM flagged_profiles WHERE id = ?", (fid,)).fetchone()
        if not row:
            return jsonify({"error": "report not found"}), 404
        if row["reported_by_user_id"] != session.get("user_id"):
            return jsonify({"error": "not allowed"}), 403
        if row["status"] == "withdrawn":
            return jsonify({"ok": True, "message": "already withdrawn"}), 200

        # update status to withdrawn
        db.execute("UPDATE flagged_profiles SET status=?, admin_notes=?, reviewed_at=? WHERE id=?",
                   ("withdrawn", "Withdrawn by reporter", datetime.utcnow().isoformat(), fid))
        db.commit()
        return jsonify({"ok": True, "message": "Report withdrawn"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Auth helpers ----------
def hash_pw(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_pw(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def create_user_db(username, password, email=None, role='user'):
    db = get_db()
    pw = hash_pw(password)
    db.execute("INSERT INTO users (username, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
               (username, email, pw, role, datetime.utcnow().isoformat()))
    db.commit()

def get_user_by_username(username):
    db = get_db()
    return db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

def current_user():
    if session.get("user_id"):
        return get_db().execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    return None

def login_session_from_row(row):
    session["logged_in"] = True
    session["username"] = row["username"]
    session["user_id"] = row["id"]
    session["is_admin"] = (row["role"] == "admin")
    session["role"] = row["role"]

def is_logged_in():
    return session.get("logged_in", False)

def is_admin():
    return session.get("is_admin", False)

# ---------- Model helpers ----------
def save_active_model(model_obj, feature_cols, metrics=None):
    payload = (model_obj, feature_cols, metrics or {})
    joblib.dump(payload, ACTIVE_MODEL_PATH)

def load_active_model():
    if os.path.exists(ACTIVE_MODEL_PATH):
        try:
            return joblib.load(ACTIVE_MODEL_PATH)
        except Exception as e:
            app.logger.error("Failed to load active model: %s", e)
    return (None, None, None)

# Keep your existing predictor instance for the heavy models
# Provide a small wrapper that tries Predictor first, then active sklearn model
def predict_with_model(profile):
    # Try your Predictor's predict_manual (it returns a dict-like response)
    try:
        out = P.predict_manual(profile, screen_name=profile.get("screen_name"))
        # Expect out to contain keys: label, score (or similar). Normalize:
        if isinstance(out, dict):
            label = out.get("label") or out.get("pred") or out.get("prediction")
            score = out.get("score") or out.get("prob") or out.get("confidence") or 0.0
            # ensure numeric
            try:
                score = float(score)
            except:
                score = 1.0 if str(label).lower() == "fake" else 0.0
            return score, ("FAKE" if str(label).lower() == "fake" or score>=AUTO_FLAG_THRESHOLD else "GENUINE")
    except Exception as e:
        app.logger.debug("Predictor manual failed: %s", e)

    # fallback: active sklearn model if present
    try:
        model, feature_cols, metrics = load_active_model()
        if model:
            # build vector from feature_cols (fall back to zeros)
            X = []
            for c in feature_cols:
                val = profile.get(c, 0)
                try:
                    X.append(float(val))
                except:
                    X.append(0.0)
            import numpy as np
            X = np.array(X).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                score = float(model.predict_proba(X)[0,1])
            else:
                score = float(model.predict(X)[0])
            label = "FAKE" if score >= AUTO_FLAG_THRESHOLD else "GENUINE"
            return score, label
    except Exception as e:
        app.logger.error("Active sklearn model failed: %s", e)

    raise RuntimeError("No model available to predict.")

# ---------- Routes ----------
@app.route("/")
def index():
    if not is_logged_in():
        return redirect(url_for("login"))
    if is_admin():
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("user_dashboard"))

# REGISTER (DB-backed)
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        email = request.form.get("email","") or None
        if not username or not password:
            flash("Provide username and password", "warning")
            return redirect(url_for("register"))
        if get_user_by_username(username):
            flash("Username already exists", "danger")
            return redirect(url_for("register"))
        try:
            create_user_db(username, password, email, role='user')
            flash("Account created. Please login.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Registration failed: {e}", "danger")
            return redirect(url_for("register"))
    return render_template("register.html")

# LOGIN
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username","").strip()
        pw = request.form.get("password","")
        # first check DB
        row = get_user_by_username(uname)
        if row:
            if row["is_blocked"]:
                flash("Account blocked", "danger"); return redirect(url_for("login"))
            if check_pw(pw, row["password_hash"]):
                login_session_from_row(row)
                # update last_login
                get_db().execute("UPDATE users SET last_login=? WHERE id=?", (datetime.utcnow().isoformat(), row["id"]))
                get_db().commit()
                flash("Logged in", "success")
                return redirect(url_for("admin_dashboard") if row["role"]=="admin" else url_for("user_dashboard"))
        # fallback to legacy credentials support (in case)
        # If you had a USER_CREDENTIALS dict, we won't use it now (DB takes precedence)
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

# User dashboard route (manual predict)
@app.route("/user", methods=["GET","POST"])
def user_dashboard():
    if not is_logged_in() or is_admin():
        return redirect(url_for("login"))
    result = None
    if request.method == "POST":
        prof = {
            "screen_name": request.form.get("screen_name","manual_user"),
            "description": request.form.get("description",""),
            "followers_count": int(request.form.get("followers_count") or 0),
            "friends_count": int(request.form.get("friends_count") or 0),
            "statuses_count": int(request.form.get("statuses_count") or 0),
            "favourites_count": int(request.form.get("favourites_count") or 0),
            "listed_count": int(request.form.get("listed_count") or 0),
            "utc_offset": int(request.form.get("utc_offset") or 0),
            "profile_image_url": request.form.get("profile_image_url",""),
            "lang_label": int(request.form.get("lang_label") or 0)
        }
        try:
            score, label = predict_with_model(prof)
            result = {"score": score, "label": label, "profile": prof}
            # save prediction to DB
            try:
                db = get_db()
                db.execute("INSERT INTO predictions (user_id, profile_raw, score, prediction, created_at) VALUES (?, ?, ?, ?, ?)",
                           (session.get("user_id"), json.dumps(prof), score, label, datetime.utcnow().isoformat()))
                db.commit()
            except Exception as e:
                app.logger.debug("Failed to store prediction: %s", e)
            # auto-flag if needed
            if label == "FAKE" and score >= AUTO_FLAG_THRESHOLD:
                try:
                    db = get_db()
                    db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, model_score, model_prediction, created_at)
                                  VALUES (?, ?, ?, ?, ?, ?)""",
                               (None, prof.get("screen_name") or "", json.dumps(prof), score, label, datetime.utcnow().isoformat()))
                    db.commit()
                except Exception as e:
                    app.logger.debug("Failed to auto-flag: %s", e)
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
    return render_template("user_dashboard.html", result=result)

# Predict form (matches templates' predict endpoint)
@app.route("/predict", methods=["GET","POST"])
def predict_form():
    if not is_logged_in() or is_admin():
        return redirect(url_for("login"))
    if request.method == "POST":
        raw_json = request.form.get("raw_json","").strip()
        if raw_json:
            try:
                profile = json.loads(raw_json)
            except Exception:
                flash("Invalid JSON", "danger"); return redirect(url_for("predict_form"))
        else:
            profile = {k: request.form.get(k, 0) for k in ['f1','f2','f3','f4']}
        try:
            score, label = predict_with_model(profile)
            # store
            try:
                db = get_db()
                db.execute("INSERT INTO predictions (user_id, profile_raw, score, prediction, created_at) VALUES (?, ?, ?, ?, ?)",
                           (session.get("user_id"), json.dumps(profile), score, label, datetime.utcnow().isoformat()))
                db.commit()
            except Exception:
                pass
            if label == "FAKE" and score >= AUTO_FLAG_THRESHOLD:
                try:
                    db = get_db()
                    db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, model_score, model_prediction, created_at)
                                  VALUES (?, ?, ?, ?, ?, ?)""",
                               (None, profile.get("profile_id") or profile.get("username") or "", json.dumps(profile), score, label, datetime.utcnow().isoformat()))
                    db.commit()
                except Exception:
                    pass
            flash(f"Prediction: {label} (score={float(score):.3f})", "success" if label!="FAKE" else "danger")
            return render_template("predict.html", result={"score":score,"label":label,"profile":profile})
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger"); return redirect(url_for("predict_form"))
    return render_template("predict.html")

# Report form (users submit suspicious profile id)
@app.route("/report", methods=["GET","POST"])
def report_form():
    if not is_logged_in() or is_admin():
        return redirect(url_for("login"))
    if request.method == "POST":
        profile_id = request.form.get("profile_id","").strip()
        profile_raw = request.form.get("profile_raw","").strip() or None
        try:
            db = get_db()
            db.execute("""INSERT INTO flagged_profiles (reported_by_user_id, profile_id_or_handle, profile_raw, created_at)
                          VALUES (?, ?, ?, ?)""",
                       (session.get("user_id"), profile_id, profile_raw, datetime.utcnow().isoformat()))
            db.commit()
            flash("Profile reported to admin", "success")
            return redirect(url_for("user_dashboard"))
        except Exception as e:
            flash(f"Report failed: {e}", "danger")
    return render_template("report.html")

# Test CSV mode (keeps your earlier behavior)
@app.route("/test", methods=["GET","POST"])
def test_mode():
    if not is_logged_in() or is_admin():
        return redirect(url_for("login"))
    test_csv_path = ROOT / "data" / "test" / "test.csv"
    df = None
    msg = None
    prediction = None
    if request.method == "POST":
        # upload or select
        if "upload_csv" in request.files and request.files["upload_csv"].filename:
            f = request.files["upload_csv"]
            upload_path = ROOT / "data" / "test" / f.filename
            f.save(str(upload_path))
            msg = f"Saved uploaded CSV to {upload_path.name}"
            test_csv_path = upload_path
        if not test_csv_path.exists():
            flash(f"test.csv not found at {test_csv_path}. Upload one.", "warning")
        else:
            df = pd.read_csv(test_csv_path, dtype=str).fillna("")
            id_input = request.form.get("id_input","").strip()
            if id_input == "":
                row = df.iloc[0]
            else:
                if "id" in df.columns:
                    match = df[df["id"].astype(str) == id_input]
                    if match.empty:
                        match = df[df["id"].astype(str).str.lower() == id_input.lower()]
                    if match.empty:
                        flash(f"No id '{id_input}' found in test.csv", "danger")
                        row = None
                    else:
                        row = match.iloc[0]
                else:
                    flash("test.csv has no 'id' column; using index 0", "warning")
                    row = df.iloc[0]
            if row is not None:
                prof = {
                    "screen_name": row.get("screen_name","") or row.get("profile_name","") or f"test_{row.get('id','')}",
                    "description": row.get("description","") or "",
                    "followers_count": int(float(row.get("followers_count",0) or 0)),
                    "friends_count": int(float(row.get("friends_count",0) or 0)),
                    "statuses_count": int(float(row.get("statuses_count",0) or 0)),
                    "favourites_count": int(float(row.get("favourites_count",0) or 0)),
                    "listed_count": int(float(row.get("listed_count",0) or 0)),
                    "utc_offset": int(float(row.get("utc_offset",0) or 0)),
                    "profile_image_url": row.get("profile_image_url","") or "",
                    "lang_label": int(float(row.get("lang_label",0) or 0)) if row.get("lang_label","")!="" else 0
                }
                prediction = P.predict_manual(prof, screen_name=prof.get("screen_name"))
    else:
        if test_csv_path.exists():
            try:
                df = pd.read_csv(test_csv_path, dtype=str).fillna("")
            except:
                df = None
    return render_template("test_mode.html", df=df, msg=msg, prediction=prediction)

# ----------------- Admin routes (paste this block into app.py) -----------------
# Requires: get_db(), is_logged_in(), is_admin(), P, ROOT, MODELS_DIR, ACTIVE_MODEL_PATH, save_active_model(), joblib, datetime, pd

from flask import request, send_file, jsonify

@app.route("/admin", methods=["GET"])
def admin_dashboard():
    """Admin dashboard with tab support: flagged, users, charts, retrain"""
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    db = get_db()

    # load flagged profiles from DB - only show auto-flagged FAKE profiles from user searches
    # Filter: must have model_prediction = 'FAKE' (auto-flagged from searches)
    flagged_rows = db.execute("""
        SELECT 
            fp.id,
            fp.reported_by_user_id,
            fp.profile_id_or_handle,
            fp.profile_raw,
            fp.model_score,
            fp.model_prediction,
            fp.status,
            fp.admin_action,
            fp.admin_notes,
            fp.created_at,
            fp.reviewed_at,
            fp.reviewed_by_admin_id,
            u.username AS reporter
        FROM flagged_profiles fp
        LEFT JOIN users u ON fp.reported_by_user_id = u.id
        WHERE fp.model_prediction = 'FAKE'
        ORDER BY fp.created_at DESC
    """).fetchall()

    # load users
    users = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()

    # attempt to load predictions CSV (optional)
    pred_path = ROOT / "data" / "test" / "test_predictions.csv"
    preds = None
    if pred_path.exists():
        try:
            preds = pd.read_csv(pred_path).to_dict(orient="records")
        except Exception:
            preds = None

    # metrics
    total_flagged = db.execute("SELECT COUNT(*) as cnt FROM flagged_profiles").fetchone()['cnt']
    pending = db.execute("SELECT COUNT(*) as cnt FROM flagged_profiles WHERE status='pending'").fetchone()['cnt']
    blocked_users = db.execute("SELECT COUNT(*) as cnt FROM users WHERE is_blocked=1").fetchone()['cnt']
    latest_run = db.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 1").fetchone()
    metrics = {"total_flagged": total_flagged, "pending": pending, "blocked_users": blocked_users}
    if latest_run:
        metrics.update({
            "accuracy": latest_run["accuracy"],
            "precision": latest_run["precision"],
            "recall": latest_run["recall"],
            "f1": latest_run["f1"],
            "last_trained_at": latest_run["created_at"]
        })

    tab = request.args.get("tab", "flagged")
    
    # Create response with no-cache headers to prevent stale data
    response = app.make_response(render_template("admin_dashboard.html",
                           preds=preds or [],
                           rows=flagged_rows,
                           users=users,
                           metrics=metrics,
                           last_run=latest_run,
                           tab=tab))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/admin/flagged/<int:fid>/action", methods=["POST"])
def admin_flagged_action(fid):
    """Actions: reviewed, dismiss, block_user"""
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    action = request.form.get("action", "")
    db = get_db()
    try:
        now = datetime.utcnow().isoformat()
        admin_id = session.get("user_id")
        
        # Get the current state BEFORE update
        before = db.execute("SELECT admin_notes, admin_action, status FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
        print(f"\n{'='*80}")
        print(f"BEFORE UPDATE - Flag ID: {fid}")
        print(f"  Status: {before['status']}")
        print(f"  Admin Action: {before['admin_action']}")
        print(f"  Admin Notes: {before['admin_notes']}")
        print(f"  Action requested: {action}")
        print(f"{'='*80}\n")
        
        if action == "dismiss":
            # Update with dismiss notes
            cursor = db.execute(
                "UPDATE flagged_profiles SET status=?, admin_notes=?, admin_action=?, reviewed_at=?, reviewed_by_admin_id=? WHERE id=?",
                ("dismissed", "Dismissed by admin", "dismissed", now, admin_id, fid)
            )
            db.commit()
            rows_affected = cursor.rowcount
            
            # Verify the update worked
            after = db.execute("SELECT admin_notes, admin_action, status FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
            print(f"\nAFTER UPDATE - Flag ID: {fid}")
            print(f"  Rows affected: {rows_affected}")
            print(f"  Status: {after['status']}")
            print(f"  Admin Action: {after['admin_action']}")
            print(f"  Admin Notes: {after['admin_notes']}")
            print(f"{'='*80}\n")
            
            app.logger.info(f"Admin {admin_id} dismissed flag {fid} - BEFORE: notes='{before['admin_notes']}' AFTER: notes='{after['admin_notes']}'")
            flash(f"✅ Flag dismissed. Current notes: '{after['admin_notes']}'", "success")
            
        elif action == "reviewed":
            # Update with reviewed notes
            cursor = db.execute(
                "UPDATE flagged_profiles SET status=?, admin_notes=?, admin_action=?, reviewed_at=?, reviewed_by_admin_id=? WHERE id=?",
                ("reviewed", "Reviewed by admin", "reviewed", now, admin_id, fid)
            )
            db.commit()
            rows_affected = cursor.rowcount
            
            # Verify the update worked
            after = db.execute("SELECT admin_notes, admin_action, status FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
            print(f"\nAFTER UPDATE - Flag ID: {fid}")
            print(f"  Rows affected: {rows_affected}")
            print(f"  Status: {after['status']}")
            print(f"  Admin Action: {after['admin_action']}")
            print(f"  Admin Notes: {after['admin_notes']}")
            print(f"{'='*80}\n")
            
            app.logger.info(f"Admin {admin_id} reviewed flag {fid} - BEFORE: notes='{before['admin_notes']}' AFTER: notes='{after['admin_notes']}'")
            flash(f"✅ Marked as reviewed. Current notes: '{after['admin_notes']}'", "success")
            
        elif action == "block_user":
            # Mark the fake profile as confirmed/blocked (for demonstration)
            # Since these profiles aren't registered users, we just update the flagged record
            r = db.execute("SELECT profile_id_or_handle, profile_raw FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
            if r:
                profile_handle = r["profile_id_or_handle"]
                profile_raw = r["profile_raw"]
                
                # Update the flagged profile record
                cursor = db.execute(
                    "UPDATE flagged_profiles SET admin_action=?, admin_notes=?, status=?, reviewed_at=?, reviewed_by_admin_id=? WHERE id=?",
                    ("blocked", f"Fake profile '{profile_handle}' confirmed and blocked from dataset", "actioned", now, admin_id, fid)
                )
                
                # Also add to blocked_profiles table (for demonstration tracking)
                try:
                    db.execute("""
                        INSERT OR REPLACE INTO blocked_profiles 
                        (profile_handle, profile_raw, blocked_by_admin_id, blocked_at, reason, flagged_profile_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (profile_handle, profile_raw, admin_id, now, "Confirmed fake profile from dataset", fid))
                except Exception as e:
                    app.logger.warning(f"Could not insert into blocked_profiles: {e}")
                
                db.commit()
                rows_affected = cursor.rowcount
                
                # Verify the update worked
                after = db.execute("SELECT admin_notes, admin_action, status FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
                print(f"\nAFTER UPDATE - Flag ID: {fid}")
                print(f"  Rows affected: {rows_affected}")
                print(f"  Status: {after['status']}")
                print(f"  Admin Action: {after['admin_action']}")
                print(f"  Admin Notes: {after['admin_notes']}")
                print(f"  Profile '{profile_handle}' added to blocked_profiles table")
                print(f"{'='*80}\n")
                
                app.logger.info(f"Admin {admin_id} confirmed fake profile '{profile_handle}' for flag {fid}")
                flash(f"✅ Fake profile '{profile_handle}' confirmed and blocked from dataset.", "success")
            else:
                flash("Cannot block — profile not found.", "warning")
        else:
            flash("Unknown action.", "warning")
    except Exception as e:
        app.logger.error(f"Admin action failed for flag {fid}: {e}")
        flash(f"Action failed: {e}", "danger")
        db.rollback()
        
    return redirect(url_for("admin_dashboard", tab="flagged"))

@app.route("/admin/flagged/<int:fid>/debug", methods=["GET"])
def admin_flagged_debug(fid):
    """Debug endpoint to see raw database values for a flag"""
    if not is_logged_in() or not is_admin():
        return jsonify({"error": "admin required"}), 403
    
    db = get_db()
    row = db.execute("SELECT * FROM flagged_profiles WHERE id=?", (fid,)).fetchone()
    
    if not row:
        return jsonify({"error": f"Flag {fid} not found"}), 404
    
    # Convert sqlite3.Row to dict
    result = {key: row[key] for key in row.keys()}
    
    return jsonify({
        "flag_id": fid,
        "current_values": result,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/admin/users/<int:uid>/block", methods=["POST"])
def admin_block_user(uid):
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    try:
        db = get_db()
        db.execute("UPDATE users SET is_blocked=1 WHERE id=?", (uid,))
        db.commit()
        flash("User blocked.", "success")
    except Exception as e:
        flash(f"Failed to block user: {e}", "danger")
    return redirect(url_for("admin_dashboard", tab="users"))

@app.route("/admin/users/<int:uid>/unblock", methods=["POST"])
def admin_unblock_user(uid):
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    try:
        db = get_db()
        db.execute("UPDATE users SET is_blocked=0 WHERE id=?", (uid,))
        db.commit()
        flash("User unblocked.", "success")
    except Exception as e:
        flash(f"Failed to unblock user: {e}", "danger")
    return redirect(url_for("admin_dashboard", tab="users"))

@app.route("/admin/charts_data", methods=["GET"])
def admin_charts_data():
    if not is_logged_in() or not is_admin():
        return jsonify({"error": "admin required"}), 403
    db = get_db()
    try:
        # Basic metrics
        total_flagged = db.execute("SELECT COUNT(*) as cnt FROM flagged_profiles").fetchone()['cnt']
        pending = db.execute("SELECT COUNT(*) as cnt FROM flagged_profiles WHERE status='pending'").fetchone()['cnt']
        blocked_users = db.execute("SELECT COUNT(*) as cnt FROM users WHERE is_blocked=1").fetchone()['cnt']
        total_users = db.execute("SELECT COUNT(*) as cnt FROM users").fetchone()['cnt']
        
        # Status distribution for pie chart
        status_data = db.execute("""
            SELECT COALESCE(status, 'pending') as status, COUNT(*) as count 
            FROM flagged_profiles 
            GROUP BY status
        """).fetchall()
        status_distribution = {row['status']: row['count'] for row in status_data}
        
        # Ensure at least one status exists for the chart
        if not status_distribution:
            status_distribution = {"pending": 0}
        
        # Flagged profiles over time (last 7 days)
        timeline_data = db.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM flagged_profiles
            WHERE created_at >= date('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """).fetchall()
        timeline = [{"date": row['date'] or 'N/A', "count": row['count']} for row in timeline_data]
        
        # If no timeline data, create empty array for chart
        if not timeline:
            timeline = []
        
        # Model performance metrics (last 5 training runs)
        training_history = db.execute("""
            SELECT created_at, accuracy, precision, recall, f1
            FROM training_runs
            ORDER BY created_at DESC
            LIMIT 5
        """).fetchall()
        
        model_metrics = []
        for row in training_history:
            try:
                model_metrics.append({
                    "date": row['created_at'][:10] if row['created_at'] else 'N/A',
                    "accuracy": float(row['accuracy']) if row['accuracy'] is not None else 0,
                    "precision": float(row['precision']) if row['precision'] is not None else 0,
                    "recall": float(row['recall']) if row['recall'] is not None else 0,
                    "f1": float(row['f1']) if row['f1'] is not None else 0
                })
            except (ValueError, TypeError) as e:
                app.logger.warning(f"Skipping training run with invalid metrics: {e}")
                continue
        
        model_metrics.reverse()  # Show oldest to newest
        
        # Latest model metrics
        latest_run = db.execute("SELECT * FROM training_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        latest_metrics = {}
        if latest_run:
            latest_metrics = {
                "accuracy": float(latest_run["accuracy"]) if latest_run["accuracy"] is not None else 0,
                "precision": float(latest_run["precision"]) if latest_run["precision"] is not None else 0,
                "recall": float(latest_run["recall"]) if latest_run["recall"] is not None else 0,
                "f1": float(latest_run["f1"]) if latest_run["f1"] is not None else 0,
                "last_trained_at": latest_run["created_at"] or "N/A"
            }
        
        result = {
            "summary": {
                "total_flagged": total_flagged,
                "pending": pending,
                "blocked_users": blocked_users,
                "total_users": total_users
            },
            "status_distribution": status_distribution,
            "timeline": timeline,
            "model_metrics": model_metrics,
            "latest_metrics": latest_metrics
        }
        
        app.logger.info(f"Charts data returned: {len(timeline)} timeline points, {len(model_metrics)} model metrics")
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in charts_data: {e}", exc_info=True)
        return jsonify({"error": str(e), "details": "Check server logs for more info"}), 500

@app.route("/admin/retrain", methods=["POST"])
def admin_retrain():
    """Upload CSV, pick feature cols and label col, train a simple logistic regression, save model & record run."""
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))

    csv_file = request.files.get("csv")
    features_raw = request.form.get("features", "").strip()
    labelcol = request.form.get("labelcol", "label").strip()

    if not csv_file:
        flash("CSV file is required for retrain.", "danger")
        return redirect(url_for("admin_dashboard", tab="retrain"))

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        flash(f"Failed to read CSV: {e}", "danger")
        return redirect(url_for("admin_dashboard", tab="retrain"))

    # Determine feature columns
    if features_raw:
        feature_cols = [c.strip() for c in features_raw.split(",") if c.strip() in df.columns]
    else:
        # pick numeric columns excluding label
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != labelcol][:10]

    if labelcol not in df.columns:
        flash(f"Label column '{labelcol}' not found in CSV.", "danger")
        return redirect(url_for("admin_dashboard", tab="retrain"))
    if not feature_cols:
        flash("No feature columns found/determined.", "danger")
        return redirect(url_for("admin_dashboard", tab="retrain"))

    try:
        X = df[feature_cols].fillna(0).values
        y = df[labelcol].astype(int).values
        # simple train/test split
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mdl = LogisticRegression(max_iter=1000)
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1s = float(f1_score(y_test, y_pred, zero_division=0))

        # persist model file
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_file = os.path.join(MODELS_DIR, f"model_{ts}.pkl")
        joblib.dump((mdl, feature_cols, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1s}), model_file)

        # set as active model
        joblib.dump((mdl, feature_cols, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1s}), ACTIVE_MODEL_PATH)

        # record run in DB
        db = get_db()
        db.execute("""INSERT INTO training_runs (dataset_name, trained_by_admin_id, accuracy, precision, recall, f1, model_file, created_at, notes)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (f"uploaded_{ts}", session.get("user_id"), acc, prec, rec, f1s, model_file, datetime.utcnow().isoformat(), "admin retrain"))
        db.commit()

        flash(f"Retrain finished — accuracy: {acc:.4f}", "success")
    except Exception as e:
        flash(f"Training failed: {e}", "danger")
    return redirect(url_for("admin_dashboard", tab="retrain"))

@app.route("/admin/download_active_model", methods=["GET"])
def download_active_model():
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    if os.path.exists(ACTIVE_MODEL_PATH):
        return send_file(ACTIVE_MODEL_PATH, as_attachment=True, download_name="active_model.pkl")
    flash("No active model available.", "warning")
    return redirect(url_for("admin_dashboard", tab="retrain"))

@app.route("/admin/download", methods=["GET"])
def admin_download():
    if not is_logged_in() or not is_admin():
        return redirect(url_for("login"))
    pred_path = ROOT / "data" / "test" / "test_predictions.csv"
    if not pred_path.exists():
        flash("No predictions file present", "warning")
        return redirect(url_for("admin_dashboard", tab="charts"))
    return send_file(str(pred_path), as_attachment=True, download_name="test_predictions.csv")
# ----------------- end admin block -----------------


# JSON predict API (keeps your existing endpoint)
@app.route("/predict_json", methods=["POST"])
def predict_json():
    if not is_logged_in():
        return {"error":"unauthenticated"}, 401
    j = request.get_json() or {}
    try:
        score, label = predict_with_model(j)
        return {"score": score, "label": label}
    except Exception as e:
        return {"error": str(e)}, 500

# ---------- Startup (safe: run DB init inside app context) ----------
if __name__ == "__main__":
    # Ensure DB/tables and default admin are created inside an app context
    with app.app_context():
        init_db()
        seed_default_admin()

    # Run dev server. Use reloader off to avoid duplicate init/logging if you prefer.
    # Set use_reloader=False if you saw threading/duplicate-init issues before.
    app.run(debug=True, port=5000, use_reloader=False)
