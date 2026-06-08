"""
Maize Price Prediction System — Flask Backend
Run: python app.py
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
import sqlite3, joblib, numpy as np, pandas as pd
from datetime import datetime, date
import os, hashlib, io
import africastalking
from datetime import date, timedelta, datetime
import calendar

# Replace with your credentials
africastalking.initialize(
    username="sandbox",  # or your live username
    api_key="atsk_0dfc1797f1220dd3ec5108de4132da9757f7ca4fe5a458c175c2333415ae1bd2764a95ed"
)

sms = africastalking.SMS

app = Flask(__name__)
app.secret_key = "maize_predict_2024_secret"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "maize.db")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
FEAT_PATH  = os.path.join(BASE_DIR, "model_features.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "merged_maize_weather.csv")

# County to encoded value mapping
COUNTY_MAP = {
    'Baringo': 0, 'Bomet': 1, 'Bungoma': 2, 'Busia': 3, 'Embu': 4,
    'Garissa': 5, 'Homa-bay': 6, 'Kajiado': 7, 'Kakamega': 8,
    'Kirinyaga': 9, 'Kisii': 10, 'Kisumu': 11, 'Kitui': 12,
    'Kwale': 13, 'Laikipia': 14, 'Lamu': 15, 'Makueni': 16,
    'Meru': 17, 'Migori': 18, 'Nairobi': 19, 'Nandi': 20,
    'Nyamira': 21, 'Nyandarua': 22, 'Nyeri': 23, 'Siaya': 24,
    'Taita-Taveta': 25, 'Tharaka-Nithi': 26, 'Trans-Nzoia': 27,
    'Uasin-Gishu': 28, 'Vihiga': 29, 'West-Pokot': 30
}

# Classification to encoded value mapping
CLASS_MAP = {
    'White Maize': 1,
    'Mixed-Traditional': 0,
    'Yellow Maize': 2
}

def send_sms_direct(phone: str, message: str) -> bool:
    """Send SMS — used during registration before full sms_ussd module loads."""
    try:
        from sms_ussd import send_sms
        return send_sms(phone, message)
    except Exception as e:
        print(f"[SMS] Could not send to {phone}: {e}")
        return False


# Load model once at startup
model    = joblib.load(MODEL_PATH)
features = joblib.load(FEAT_PATH)

# County → encoded int mapping
COUNTY_MAP = {
    'Baringo':0,'Bomet':1,'Bungoma':2,'Busia':3,'Embu':4,'Garissa':5,
    'Homa-bay':6,'Kajiado':7,'Kakamega':8,'Kirinyaga':9,'Kisii':10,
    'Kisumu':11,'Kitui':12,'Kwale':13,'Laikipia':14,'Lamu':15,
    'Makueni':16,'Meru':17,'Migori':18,'Nairobi':19,'Nandi':20,
    'Nyamira':21,'Nyandarua':22,'Nyeri':23,'Siaya':24,'Taita-Taveta':25,
    'Tharaka-Nithi':26,'Trans-Nzoia':27,'Uasin-Gishu':28,'Vihiga':29,'West-Pokot':30
}
CLASS_MAP = {'White Maize':1,'Mixed-Traditional':0,'Yellow Maize':2}

def get_seasonal_temperature(month):
    """Get average temperature for a given month in Kenya (based on historical data)"""
    # Average temperatures for Kenyan maize-growing regions
    monthly_temps = {
        1: 22.0,   # January
        2: 23.0,   # February
        3: 23.5,   # March
        4: 22.5,   # April
        5: 22.0,   # May
        6: 21.0,   # June
        7: 20.5,   # July
        8: 21.0,   # August
        9: 22.0,   # September
        10: 22.5,  # October
        11: 22.0,  # November
        12: 21.5   # December
    }
    return monthly_temps.get(month, 22.0)

def get_seasonal_rainfall(month):
    """Get average rainfall for a given month in Kenya (mm)"""
    # Average rainfall patterns for maize-growing regions
    monthly_rain = {
        1: 35.0,    # January - dry
        2: 45.0,    # February - dry
        3: 95.0,    # March - long rains start
        4: 145.0,   # April - peak long rains
        5: 125.0,   # May - long rains end
        6: 55.0,    # June - dry
        7: 50.0,    # July - dry
        8: 55.0,    # August - dry
        9: 60.0,    # September - short rains start
        10: 85.0,   # October - short rains peak
        11: 110.0,  # November - short rains end
        12: 55.0    # December - dry
    }
    return monthly_rain.get(month, 60.0)

def estimate_future_price_lags(current_price, days_ahead):
    """
    Estimate future price lags based on current price and days ahead.
    This uses a simplified model of price momentum.
    """
    # Short-term price momentum (price tends to continue in same direction)
    momentum = 0.01  # 1% price change per week assumption
    
    if days_ahead <= 7:  # 1 week
        price_lag_7d = current_price
        price_lag_14d = current_price * (1 - momentum)
        price_lag_30d = current_price * (1 - momentum * 2)
    elif days_ahead <= 30:  # 1 month
        price_lag_7d = current_price * (1 + momentum)
        price_lag_14d = current_price
        price_lag_30d = current_price * (1 - momentum)
    else:  # > 1 month
        price_lag_7d = current_price * (1 + momentum * 2)
        price_lag_14d = current_price * (1 + momentum)
        price_lag_30d = current_price
    
    return price_lag_7d, price_lag_14d, price_lag_30d

def get_seasonal_adjustment(month):
    """
    Get seasonal price adjustment factor.
    Prices are typically higher in lean seasons, lower during harvest.
    """
    # Seasonal factors (>1 means higher prices, <1 means lower prices)
    seasonal_factors = {
        1: 1.12,   # January - lean season (high prices)
        2: 1.10,   # February - lean season (high prices)
        3: 1.05,   # March - planting (moderate)
        4: 1.00,   # April - normal
        5: 0.95,   # May - pre-harvest (prices start dropping)
        6: 0.90,   # June - harvest begins (prices drop)
        7: 0.88,   # July - peak harvest (lowest prices)
        8: 0.90,   # August - harvest continues (low prices)
        9: 0.95,   # September - post-harvest (prices stabilize)
        10: 1.00,  # October - normal
        11: 1.05,  # November - short rains (prices rise)
        12: 1.08   # December - festive season (higher demand)
    }
    return seasonal_factors.get(month, 1.00)

def get_season_encoded(month):
    m = int(month)
    if m in [3,4,5]:   return 0   # long_rains
    elif m in [10,11,12]: return 1 # short_rains
    elif m in [6,7,8,9]:  return 2 # dry_mid
    else:                 return 3 # dry_jan_feb


def get_historical_wholesale(county, classification, month):
    """Get historical wholesale price for similar conditions"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Filter by county, maize type, and similar month
        filtered = df[
            (df['County'] == county) & 
            (df['Classification'].str.contains(classification, case=False, na=False)) &
            (df['Date'].dt.month == month)
        ]
        
        if len(filtered) > 0:
            return filtered['Wholesale'].median()
        
        # Fallback: same county any month
        filtered = df[df['County'] == county]
        if len(filtered) > 0:
            return filtered['Wholesale'].median()
        
        # Final fallback: national average
        return df['Wholesale'].median()
    except Exception as e:
        print(f"Historical lookup error: {e}")
        return 50.0  # Default fallback

def get_historical_retail(county, classification, month):
    """Get historical retail price for similar conditions"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        
        filtered = df[
            (df['County'] == county) & 
            (df['Classification'].str.contains(classification, case=False, na=False)) &
            (df['Date'].dt.month == month)
        ]
        
        if len(filtered) > 0:
            return filtered['Retail'].median()
        
        filtered = df[df['County'] == county]
        if len(filtered) > 0:
            return filtered['Retail'].median()
        
        return df['Retail'].median()
    except Exception as e:
        print(f"Retail lookup error: {e}")
        return 65.0

def estimate_wholesale_from_retail(retail_price, county):
    """Estimate wholesale price from retail using regional markup"""
    # Markup rates by county (wholesale to retail)
    markup_rates = {
        'Bomet': 0.25, 'Kirinyaga': 0.28, 'Kisumu': 0.30,
        'Kakamega': 0.27, 'Meru': 0.26, 'Kitui': 0.22,
        'Trans-Nzoia': 0.24, 'Nakuru': 0.26, 'Nairobi': 0.32,
        'default': 0.25
    }
    markup = markup_rates.get(county, markup_rates['default'])
    wholesale = retail_price / (1 + markup)
    return round(wholesale, 2)

def estimate_retail_from_wholesale(wholesale_price, county):
    """Estimate retail price from wholesale using regional markup"""
    markup_rates = {
        'Bomet': 0.25, 'Kirinyaga': 0.28, 'Kisumu': 0.30,
        'Kakamega': 0.27, 'Meru': 0.26, 'Kitui': 0.22,
        'Trans-Nzoia': 0.24, 'Nakuru': 0.26, 'Nairobi': 0.32,
        'default': 0.25
    }
    markup = markup_rates.get(county, markup_rates['default'])
    retail = wholesale_price * (1 + markup)
    return round(retail, 2)

def get_current_wholesale_from_db(county, market=None):
    """Get most recent wholesale price from database"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Get latest date in dataset
        latest_date = df['Date'].max()
        
        filtered = df[df['County'] == county]
        if market:
            filtered = filtered[filtered['Market'] == market]
        
        filtered = filtered[filtered['Date'] == latest_date]
        
        if len(filtered) > 0:
            return filtered['Wholesale'].median()
        return None
    except Exception as e:
        print(f"Current wholesale lookup error: {e}")
        return None

# ─────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        password TEXT NOT NULL,
        role TEXT DEFAULT 'farmer',
        county TEXT,
        verified INTEGER DEFAULT 0,
        otp TEXT,
        otp_expires TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    # Add missing columns to existing DB gracefully
    for col, defn in [
        ("phone",       "TEXT"),
        ("verified",    "INTEGER DEFAULT 0"),
        ("otp",         "TEXT"),
        ("otp_expires", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except Exception:
            pass  # column already exists

    c.execute("""CREATE TABLE IF NOT EXISTS sms_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone TEXT NOT NULL,
        message TEXT,
        direction TEXT DEFAULT 'incoming',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        county TEXT,
        classification TEXT,
        supply_volume REAL,
        temperature REAL,
        rainfall REAL,
        predicted_price REAL,
        prediction_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    # Seed admin
    admin_pw = hashlib.sha256("admin123".encode()).hexdigest()
    c.execute("INSERT OR IGNORE INTO users (name,email,password,role,verified) VALUES (?,?,?,?,?)",
              ("Admin","admin@maize.co.ke", admin_pw, "admin", 1))
    conn.commit()
    conn.close()

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("home.html", user=session.get("user"))

@app.route("/login", methods=["GET","POST"])
def login():
    verified_flash = session.pop("verified_flash", False)
    if request.method == "POST":
        email    = request.form.get("email","").strip().lower()
        password = hash_pw(request.form.get("password",""))
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                            (email, password)).fetchone()
        conn.close()
        if user:
            if not user["verified"] and user["role"] != "admin":
                session["pending_email"] = email
                session["pending_phone"] = user["phone"] or ""
                return render_template("login.html",
                    error="Please verify your phone number first.",
                    show_verify=True)
            session["user"] = {
                "id"    : user["id"],
                "name"  : user["name"],
                "email" : user["email"],
                "phone" : user["phone"] or "",
                "role"  : user["role"],
                "county": user["county"],
            }
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html", verified_flash=verified_flash)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name   = request.form.get("name","").strip()
        email  = request.form.get("email","").strip().lower()
        phone  = request.form.get("phone","").strip()
        pw     = request.form.get("password","")
        role   = request.form.get("role","farmer")
        county = request.form.get("county","")

        # Validation
        if not name or not email or not phone or not pw:
            return render_template("register.html", error="All fields are required",
                                   counties=sorted(COUNTY_MAP.keys()))
        if len(pw) < 6:
            return render_template("register.html", error="Password must be at least 6 characters",
                                   counties=sorted(COUNTY_MAP.keys()))
        if not phone.startswith("+"):
            return render_template("register.html", error="Phone must start with country code e.g. +254...",
                                   counties=sorted(COUNTY_MAP.keys()))
        
        # Validate admin access code
        ADMIN_ACCESS_CODE = "maizeiq2024admin"
        if role == "admin":
            admin_code = request.form.get("admin_code","")
            if admin_code != ADMIN_ACCESS_CODE:
                return render_template("register.html",
                    error="Incorrect admin access code.",
                    counties=sorted(COUNTY_MAP.keys()))
        if role not in ("farmer","trader","admin"):
            role = "farmer"

        # Generate OTP
        import random
        otp      = str(random.randint(100000, 999999))
        expires  = str(datetime.now() + __import__("datetime").timedelta(minutes=10))

        conn = None
        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (name,email,phone,password,role,county,otp,otp_expires,verified) "
                "VALUES (?,?,?,?,?,?,?,?,0)",
                (name, email, phone, hash_pw(pw), role, county, otp, expires)
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already registered",
                                   counties=sorted(COUNTY_MAP.keys()))
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                return render_template("register.html", error="System busy, please try again",
                                       counties=sorted(COUNTY_MAP.keys()))
            raise
        finally:
            if conn:
                conn.close()

        # Send OTP via SMS
        otp_msg = f"MaizeIQ: Your verification code is {otp}. Valid for 10 minutes."
        sms_ok  = send_sms_direct(phone, otp_msg)
        if not sms_ok:
            # Store OTP in session for dev/testing fallback
            session["dev_otp"]   = otp
            session["dev_phone"] = phone

        session["pending_email"] = email
        session["pending_phone"] = phone
        return redirect(url_for("verify_otp"))

    return render_template("register.html", counties=sorted(COUNTY_MAP.keys()))


@app.route("/verify", methods=["GET","POST"])
def verify_otp():
    email = session.get("pending_email")
    if not email:
        return redirect(url_for("register"))

    dev_otp = session.get("dev_otp","")   # shown in dev when SMS is unavailable

    if request.method == "POST":
        entered = request.form.get("otp","").strip()
        conn = None
        try:
            conn = get_db()
            user = conn.execute(
                "SELECT * FROM users WHERE email=?", (email,)
            ).fetchone()

            if not user:
                return render_template("verify.html", error="Account not found", dev_otp=dev_otp)

            now = str(datetime.now())
            if user["otp"] == entered and user["otp_expires"] >= now:
                conn.execute(
                    "UPDATE users SET verified=1, otp=NULL, otp_expires=NULL WHERE email=?",
                    (email,)
                )
                conn.commit()
                session.pop("pending_email", None)
                session.pop("pending_phone", None)
                session.pop("dev_otp", None)
                session["verified_flash"] = True
                return redirect(url_for("login"))
            else:
                msg = "Incorrect code." if user["otp"] != entered else "Code has expired."
                return render_template("verify.html", error=msg, dev_otp=dev_otp,
                                       phone=session.get("pending_phone",""))
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                return render_template("verify.html", error="System busy, please try again",
                                       dev_otp=dev_otp, phone=session.get("pending_phone",""))
            raise
        finally:
            if conn:
                conn.close()

    return render_template("verify.html", dev_otp=dev_otp,
                           phone=session.get("pending_phone",""))


@app.route("/verify/resend", methods=["POST"])
def resend_otp():
    email = session.get("pending_email")
    if not email:
        return jsonify({"error":"No pending registration"}), 400
    
    import random
    otp     = str(random.randint(100000, 999999))
    expires = str(datetime.now() + __import__("datetime").timedelta(minutes=10))
    
    conn = None
    try:
        conn = get_db()
        user = conn.execute("SELECT phone FROM users WHERE email=?", (email,)).fetchone()
        if not user:
            return jsonify({"error":"User not found"}), 404
        
        conn.execute("UPDATE users SET otp=?, otp_expires=? WHERE email=?", (otp, expires, email))
        conn.commit()
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            return jsonify({"error":"System busy, please try again"}), 503
        raise
    finally:
        if conn:
            conn.close()
    
    msg  = f"MaizeIQ: Your new verification code is {otp}. Valid for 10 minutes."
    ok   = send_sms_direct(user["phone"], msg)
    session["dev_otp"] = otp
    return jsonify({"success": True, "dev_otp": otp if not ok else None})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ─────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login"))
    user = session["user"]
    conn = get_db()
    
    if user["role"] == "admin":
        # Get recent predictions with user names - CONVERT TO DICT
        preds_raw = conn.execute(
            """SELECT p.*, u.name as username 
               FROM predictions p 
               LEFT JOIN users u ON p.user_id=u.id 
               ORDER BY p.created_at DESC LIMIT 50"""
        ).fetchall()
        
        # Convert Row objects to dictionaries for JSON serialization
        predictions = []
        for row in preds_raw:
            pred_dict = {}
            for key in row.keys():
                value = row[key]
                # Handle date/datetime objects
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                pred_dict[key] = value
            predictions.append(pred_dict)
        
        # Get total users
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        
        # Get predictions by day for chart - CONVERT TO DICT
        day_rows = conn.execute(
            """SELECT prediction_date as date, COUNT(*) as count 
               FROM predictions 
               WHERE prediction_date IS NOT NULL
               GROUP BY prediction_date 
               ORDER BY prediction_date DESC LIMIT 30"""
        ).fetchall()
        
        preds_by_day = []
        for row in day_rows:
            preds_by_day.append({
                "date": str(row["date"]) if row["date"] else None,
                "count": row["count"]
            })
        preds_by_day = preds_by_day[::-1]  # Reverse to chronological order
        
        # Get predictions by county - CONVERT TO DICT
        cty_rows = conn.execute(
            """SELECT county, COUNT(*) as count 
               FROM predictions 
               WHERE county IS NOT NULL
               GROUP BY county 
               ORDER BY count DESC LIMIT 8"""
        ).fetchall()
        
        preds_by_county = []
        for row in cty_rows:
            preds_by_county.append({
                "county": row["county"],
                "count": row["count"]
            })
        
        conn.close()
        
        return render_template("admin_dashboard.html", 
                               user=user,
                               predictions=predictions,
                               total_users=total_users,
                               preds_by_day=preds_by_day,
                               preds_by_county=preds_by_county)
    
    elif user["role"] == "trader":
        preds_raw = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
            (user["id"],)
        ).fetchall()
        
        # Convert to dict
        predictions = []
        for row in preds_raw:
            pred_dict = {}
            for key in row.keys():
                value = row[key]
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                pred_dict[key] = value
            predictions.append(pred_dict)
        
        # Trader: county price overview
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=30)]
        county_avg = recent.groupby("County")["Wholesale"].mean().round(2).to_dict()
        conn.close()
        
        return render_template("trader_dashboard.html", user=user,
                               predictions=predictions, county_avg=county_avg)
    
    else:  # farmer
        preds_raw = conn.execute(
            "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 20",
            (user["id"],)
        ).fetchall()
        
        # Convert to dict
        predictions = []
        for row in preds_raw:
            pred_dict = {}
            for key in row.keys():
                value = row[key]
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                pred_dict[key] = value
            predictions.append(pred_dict)
        
        conn.close()
        return render_template("farmer_dashboard.html", user=user, predictions=predictions)

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
@app.route("/prediction", methods=["GET"])
def prediction():
    """Render the prediction page with counties and classifications"""
    
    counties = []
    classifications = []
    
    # Method 1: Try to load from COUNTY_MAP if it exists
    if 'COUNTY_MAP' in globals() and COUNTY_MAP:
        counties = sorted(COUNTY_MAP.keys())
    else:
        # Method 2: Try to load from CSV file
        try:
            if os.path.exists(DATA_PATH):
                df = pd.read_csv(DATA_PATH)
                if 'County' in df.columns:
                    counties = sorted([c for c in df['County'].unique() if pd.notna(c)])
                if 'Classification' in df.columns:
                    classifications = sorted([c for c in df['Classification'].unique() if pd.notna(c)])
        except Exception as e:
            print(f"Error reading CSV: {e}")
    
    # Method 3: Fallback to hardcoded lists
    if not counties:
        counties = [
            'Bomet', 'Kirinyaga', 'Kisumu', 'Kakamega', 'Meru', 'Kitui',
            'Trans-Nzoia', 'Nakuru', 'Nairobi', 'Kiambu', 'Machakos',
            'Makueni', 'Embu', 'Tharaka-Nithi', 'Laikipia', 'Nyeri'
        ]
    
    if not classifications:
        classifications = ['White Maize', 'Yellow Maize', 'Mixed-Traditional']
    
    # Sort for better UX
    counties = sorted(counties)
    classifications = sorted(classifications)
    
    print(f"[DEBUG] Loaded {len(counties)} counties and {len(classifications)} classifications")
    
    return render_template("prediction.html", 
                         user=session.get("user"),
                         counties=counties,
                         classifications=classifications)

# ========== PREDICTION API - POST ROUTE ==========
@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict future prices for specific dates with proper differentiation"""
    try:
        data = request.get_json()
        print(f"[DEBUG] Prediction request: {data}")
        
        county = data.get("county")
        if not county:
            return jsonify({"success": False, "error": "County is required"}), 400
            
        classification = data.get("classification", "White Maize")
        supply_volume = float(data.get("supply_volume", 5000))
        
        # Get current wholesale price
        current_wholesale = data.get("wholesale_price")
        
        # If user doesn't know wholesale, estimate from retail
        if not current_wholesale and data.get("retail_price"):
            retail_price = float(data.get("retail_price"))
            current_wholesale = retail_price / 1.3
        
        # If still no price, get from historical data or use default
        if not current_wholesale:
            current_wholesale = get_historical_wholesale(county, classification, date.today().month)
            if not current_wholesale:
                current_wholesale = 50.0
        
        current_wholesale = float(current_wholesale)
        
        # Get weather data
        temperature = data.get("temperature")
        rainfall = data.get("rainfall")
        
        if temperature is None:
            temperature = 22.0
        if rainfall is None:
            rainfall = 2.0
        
        temperature = float(temperature)
        rainfall = float(rainfall)
        
        # Get prediction timeframe
        prediction_type = data.get("prediction_type", "current")
        custom_date_str = data.get("target_date")
        
        # Determine target date
        today = date.today()
        
        if prediction_type == "current":
            target_date = today
            days_ahead = 0
        elif prediction_type == "next_week":
            target_date = today + timedelta(days=7)
            days_ahead = 7
        elif prediction_type == "next_month":
            target_date = today + timedelta(days=30)
            days_ahead = 30
        elif prediction_type == "next_season":
            # Next planting season (March)
            target_date = date(today.year, 3, 15)
            if target_date < today:
                target_date = date(today.year + 1, 3, 15)
            days_ahead = (target_date - today).days
        elif prediction_type == "custom" and custom_date_str:
            target_date = datetime.strptime(custom_date_str, "%Y-%m-%d").date()
            days_ahead = (target_date - today).days
        else:
            target_date = today
            days_ahead = 0
        
        print(f"[DEBUG] Target date: {target_date}, Days ahead: {days_ahead}")
        
        # ========== DIFFERENT PRICE FOR EACH TIMEFRAME ==========
        
        # Base prediction using current price as reference
        base_price = current_wholesale
        
        # Apply different factors based on days ahead
        if days_ahead == 0:
            # TODAY - use current price with small adjustment
            predicted_price = base_price * 1.02  # 2% markup for retail
            confidence_margin = 0.05  # ±5%
            
        elif days_ahead <= 7:
            # NEXT WEEK - slight seasonal trend
            # During harvest season (June-August) prices drop, during lean season (Jan-Feb) prices rise
            month = target_date.month
            if month in [6, 7, 8]:  # Harvest season
                seasonal_effect = 0.95  # 5% drop
            elif month in [1, 2, 12]:  # Lean/festive season
                seasonal_effect = 1.08  # 8% rise
            else:
                seasonal_effect = 1.02  # 2% moderate rise
            
            # Price momentum (if price has been rising, continue trend)
            momentum = 0.01  # 1% per week
            predicted_price = base_price * seasonal_effect * (1 + momentum)
            confidence_margin = 0.08  # ±8%
            
        elif days_ahead <= 30:
            # NEXT MONTH - stronger seasonal effects
            month = target_date.month
            
            # Detailed seasonal factors based on Kenya's maize price patterns
            seasonal_factors = {
                1: 1.15,   # January - peak lean season (highest prices)
                2: 1.12,   # February - still high
                3: 1.05,   # March - planting season, moderate
                4: 1.00,   # April - normal
                5: 0.95,   # May - pre-harvest, prices start dropping
                6: 0.90,   # June - harvest begins (prices drop significantly)
                7: 0.88,   # July - peak harvest (lowest prices)
                8: 0.89,   # August - harvest continues
                9: 0.93,   # September - post-harvest, prices rising
                10: 1.00,  # October - normalizing
                11: 1.06,  # November - short rains, prices rising
                12: 1.10   # December - festive season demand
            }
            
            seasonal_effect = seasonal_factors.get(month, 1.00)
            
            # Calculate trend from current to target
            # If going toward harvest, prices drop; toward lean season, prices rise
            current_month = today.month
            if month in [6, 7, 8] and current_month not in [6, 7, 8]:
                # Moving into harvest - prices drop
                price_change = -0.10  # -10%
            elif month in [1, 2, 12] and current_month not in [1, 2, 12]:
                # Moving into lean season - prices rise
                price_change = 0.12  # +12%
            else:
                # Normal seasonal transition
                price_change = (seasonal_effect - 1.00) * 0.5
            
            predicted_price = base_price * (1 + price_change)
            confidence_margin = 0.12  # ±12%
            
        else:
            # LONG TERM (next season or custom date far in future)
            month = target_date.month
            
            # More pronounced seasonal effects for long-term
            strong_seasonal = {
                1: 1.18, 2: 1.15, 3: 1.08, 4: 1.02, 5: 0.92,
                6: 0.85, 7: 0.83, 8: 0.85, 9: 0.92, 10: 1.02,
                11: 1.10, 12: 1.15
            }
            
            seasonal_effect = strong_seasonal.get(month, 1.00)
            
            # Also factor in supply and demand
            if supply_volume > 10000:
                supply_effect = 0.95  # High supply lowers price
            elif supply_volume < 2000:
                supply_effect = 1.08  # Low supply raises price
            else:
                supply_effect = 1.00
            
            predicted_price = base_price * seasonal_effect * supply_effect
            
            # If planting season (March/Oct), add planting demand premium
            if month in [3, 10]:
                predicted_price *= 1.03  # 3% planting demand premium
            
            confidence_margin = 0.15  # ±15% for long-term
        
        # Round to 2 decimal places
        predicted_price = round(predicted_price, 2)
        
        # Calculate confidence range
        low = round(predicted_price * (1 - confidence_margin), 2)
        high = round(predicted_price * (1 + confidence_margin), 2)
        
        # Ensure price is realistic (not too low or high)
        predicted_price = max(30, min(150, predicted_price))
        low = max(25, low)
        high = min(160, high)
        
        # Format date label
        if days_ahead == 0:
            date_label = "Today"
        elif days_ahead == 7:
            date_label = "Next Week (7 days)"
        elif days_ahead == 30:
            date_label = "Next Month (30 days)"
        elif days_ahead <= 60:
            date_label = target_date.strftime("%B %d, %Y")
        else:
            # For seasonal predictions
            if target_date.month == 3:
                date_label = f"Next Planting Season ({target_date.strftime('%B %Y')})"
            elif target_date.month in [6, 7, 8]:
                date_label = f"Next Harvest Season ({target_date.strftime('%B %Y')})"
            else:
                date_label = target_date.strftime("%B %d, %Y")
        
        # Calculate price change for advice
        price_change_percent = ((predicted_price - current_wholesale) / current_wholesale) * 100
        
        # Generate advice
        if days_ahead > 0:
            if price_change_percent > 10:
                advice = {
                    "recommendation": "📈 WAIT",
                    "message": f"Prices expected to RISE by {price_change_percent:.1f}%. Consider storing your maize and selling later.",
                    "color": "success"
                }
            elif price_change_percent < -8:
                advice = {
                    "recommendation": "⚠️ SELL NOW",
                    "message": f"Prices expected to DROP by {abs(price_change_percent):.1f}%. Consider selling now before prices go down.",
                    "color": "danger"
                }
            else:
                advice = {
                    "recommendation": "📊 MONITOR",
                    "message": f"Prices expected to remain relatively stable ({price_change_percent:+.1f}% change). Monitor market conditions.",
                    "color": "warning"
                }
        else:
            advice = None
        
        # Debug output
        print(f"[DEBUG] Prediction result - Days ahead: {days_ahead}, Price: {predicted_price}, Change: {price_change_percent:+.1f}%")
        
        return jsonify({
            "success": True,
            "predicted_price": predicted_price,
            "confidence_low": low,
            "confidence_high": high,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "date_label": date_label,
            "days_ahead": days_ahead,
            "county": county,
            "classification": classification,
            "current_price": round(current_wholesale, 2),
            "price_change_percent": round(price_change_percent, 1),
            "advice": advice,
            "method_used": "future_prediction" if days_ahead > 0 else "current_prediction"
        })
        
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400

# ─────────────────────────────────────────
# HISTORICAL DATA
# ─────────────────────────────────────────
@app.route("/historical")
def historical():
    return render_template("historical.html", user=session.get("user"),
                           counties=sorted(COUNTY_MAP.keys()))

@app.route("/api/historical")
def api_historical():
    county = request.args.get("county","")
    year   = request.args.get("year","")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    if county:
        df = df[df['County'] == county]
    if year:
        df = df[df['Date'].dt.year == int(year)]
    df = df[['Date','County','Classification','Market','Wholesale','Retail',
             'Supply Volume','temperature_mean_c','rain_sum_mm']].sort_values('Date', ascending=False)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return jsonify(df.head(500).to_dict(orient="records"))

@app.route("/api/historical/download")
def download_historical():
    county = request.args.get("county","")
    year   = request.args.get("year","")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    if county: df = df[df['County'] == county]
    if year:   df = df[df['Date'].dt.year == int(year)]
    df = df[['Date','County','Classification','Market','Wholesale','Retail',
             'Supply Volume','temperature_mean_c','rain_sum_mm']]
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    fname = f"maize_prices_{county or 'all'}_{year or 'all'}.csv"
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype='text/csv',
                     as_attachment=True,
                     download_name=fname)

@app.route("/api/chart-data")
def chart_data():
    county = request.args.get("county","Trans-Nzoia")
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['County']==county].sort_values('Date')
    monthly = df.groupby(df['Date'].dt.to_period('M')).agg(
        avg_price=('Wholesale','mean'), avg_rain=('rain_sum_mm','mean')
    ).reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    return jsonify(monthly.to_dict(orient="records"))


# ─────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))


# ─────────────────────────────────────────
# ADMIN — DELETE USER
# ─────────────────────────────────────────
@app.route("/admin/delete-user/<int:uid>", methods=["POST"])
def delete_user(uid):
    if not session.get("user") or session["user"]["role"] != "admin":
        return jsonify({"error":"Unauthorized"}), 403
    conn = get_db()
    conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})

# ─────────────────────────────────────────
# PROFILE
# ─────────────────────────────────────────
@app.route("/profile", methods=["GET","POST"])
def profile():
    if not session.get("user"):
        return redirect(url_for("login"))
    user = session["user"]
    msg, err = None, None
    if request.method == "POST":
        action = request.form.get("action")
        conn   = get_db()
        if action == "update":
            name   = request.form.get("name","").strip()
            county = request.form.get("county","")
            conn.execute("UPDATE users SET name=?, county=? WHERE id=?",
                         (name, county, user["id"]))
            conn.commit()
            session["user"]["name"]   = name
            session["user"]["county"] = county
            msg = "Profile updated successfully"
        elif action == "password":
            current = hash_pw(request.form.get("current",""))
            new_pw  = request.form.get("new_password","")
            ok = conn.execute("SELECT id FROM users WHERE id=? AND password=?",
                              (user["id"], current)).fetchone()
            if not ok:
                err = "Current password is incorrect"
            elif len(new_pw) < 6:
                err = "New password must be at least 6 characters"
            else:
                conn.execute("UPDATE users SET password=? WHERE id=?",
                             (hash_pw(new_pw), user["id"]))
                conn.commit()
                msg = "Password changed successfully"
        conn.close()
    return render_template("profile.html", user=session["user"],
                           counties=sorted(COUNTY_MAP.keys()), msg=msg, err=err)

@app.route("/api/county-comparison")
def county_comparison():
    df  = pd.read_csv(DATA_PATH)
    avg = df.groupby("County")["Wholesale"].mean().round(2).sort_values()
    return jsonify([{"county": k, "avg_price": v} for k, v in avg.items()])

@app.route("/api/price-trend")
def price_trend():
    county = request.args.get("county", "Trans-Nzoia")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["County"] == county].sort_values("Date")
    weekly = df.set_index("Date")["Wholesale"].resample("W").mean().dropna()
    return jsonify([{"date": str(d.date()), "price": round(p, 2)}
                    for d, p in weekly.items()])

@app.route("/insights")
def insights():
    if not session.get("user") or session["user"]["role"] != "admin":
        return redirect(url_for("login"))
    df  = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    top5_high = df.groupby("County")["Wholesale"].mean().nlargest(5).round(2).to_dict()
    top5_low  = df.groupby("County")["Wholesale"].mean().nsmallest(5).round(2).to_dict()
    monthly   = df.groupby(df["Date"].dt.to_period("M"))["Wholesale"].mean().round(2)
    monthly   = {str(k): v for k, v in monthly.items()}
    return render_template("insights.html", user=session["user"],
                           top5_high=top5_high, top5_low=top5_low, monthly=monthly)

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html", user=session.get("user")), 404



# ── County comparison page ───────────────────────────────────────────────────
@app.route("/compare")
def compare():
    return render_template("compare.html", user=session.get("user"),
                           counties=sorted(COUNTY_MAP.keys()))

# ── User predictions CSV download ───────────────────────────────────────────
@app.route("/api/predictions/download")
def download_predictions():
    if not session.get("user"):
        return redirect(url_for("login"))
    conn = get_db()
    rows = conn.execute(
        "SELECT prediction_date,county,classification,predicted_price,"
        "temperature,rainfall,supply_volume FROM predictions WHERE user_id=? ORDER BY created_at DESC",
        (session["user"]["id"],)
    ).fetchall()
    conn.close()
    lines = ["Date,County,Classification,Predicted Price (KSh),Temp (C),Rainfall (mm),Supply (kg)"]
    for r in rows:
        lines.append(",".join(str(v or "") for v in r))
    buf = io.BytesIO("\n".join(lines).encode())
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="my_predictions.csv")

# ── Admin dashboard — augment with chart data ────────────────────────────────
# Override the existing dashboard route with enriched data
# (patch the existing function by appending a new route alias)
@app.route("/dashboard/data")
def dashboard_data():
    """API used by admin dashboard charts."""
    if not session.get("user") or session["user"]["role"] != "admin":
        return jsonify({"error":"Unauthorized"}), 403
    conn = get_db()
    # Predictions per day (last 30 days)
    rows = conn.execute(
        "SELECT prediction_date as date, COUNT(*) as count "
        "FROM predictions GROUP BY prediction_date "
        "ORDER BY prediction_date DESC LIMIT 30"
    ).fetchall()
    preds_by_day = [{"date": r["date"], "count": r["count"]} for r in rows][::-1]
    # Predictions by county (top 8)
    rows2 = conn.execute(
        "SELECT county, COUNT(*) as count FROM predictions "
        "GROUP BY county ORDER BY count DESC LIMIT 8"
    ).fetchall()
    preds_by_county = [{"county": r["county"], "count": r["count"]} for r in rows2]
    conn.close()
    return jsonify({"preds_by_day": preds_by_day, "preds_by_county": preds_by_county})

# ── Security headers ─────────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "SAMEORIGIN"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    return response

# ── Session timeout (30 minutes of inactivity) ───────────────────────────────
@app.before_request
def check_session_timeout():
    session.permanent = True
    app.permanent_session_lifetime = __import__("datetime").timedelta(minutes=30)


# ═══════════════════════════════════════════════════════════
# ADMIN — REAL-TIME DATA APIs
# ═══════════════════════════════════════════════════════════

# ── Real-time users JSON (for admin dashboard live reload) ──────────────────
@app.route("/admin/users-json")
def admin_users_json():
    if not session.get("user") or session["user"]["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403
    conn = get_db()
    users = conn.execute(
        "SELECT id, name, email, phone, role, county, verified, created_at "
        "FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([{
        "id"      : u["id"],
        "name"    : u["name"],
        "email"   : u["email"],
        "phone"   : u["phone"] or "",
        "role"    : u["role"],
        "county"  : u["county"] or "—",
        "verified": bool(u["verified"]),
        "joined"  : str(u["created_at"])[:10]
    } for u in users])


# ── Dataset status API ────────────────────────────────────────────────────────
@app.route("/admin/dataset-status")
def admin_dataset_status():
    if not session.get("user") or session["user"]["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    status = []
    try:
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        status.append({
            "name"     : "Market Price Data",
            "records"  : len(df),
            "date_from": str(df["Date"].min().date()),
            "date_to"  : str(df["Date"].max().date()),
            "missing"  : int(df["Wholesale"].isnull().sum()),
            "status"   : "Ready"
        })
        status.append({
            "name"     : "Merged Dataset (Price + Weather)",
            "records"  : len(df),
            "date_from": str(df["Date"].min().date()),
            "date_to"  : str(df["Date"].max().date()),
            "missing"  : int(df.isnull().sum().sum()),
            "status"   : "Ready"
        })
        sv_missing = int(df["Supply Volume"].isnull().sum()) if "Supply Volume" in df.columns else 0
        sv_pct     = round(sv_missing / len(df) * 100, 1) if len(df) > 0 else 0
        status.append({
            "name"     : "Supply Volume Column",
            "records"  : len(df) - sv_missing,
            "date_from": "—",
            "date_to"  : "—",
            "missing"  : f"{sv_missing} ({sv_pct}%)",
            "status"   : "Partial" if sv_missing > 0 else "Ready"
        })
    except Exception as e:
        status.append({"name": "Dataset", "records": 0, "date_from":"—",
                        "date_to":"—", "missing": str(e), "status": "Error"})
    return jsonify(status)


# ── File upload and dataset ingestion ────────────────────────────────────────
@app.route("/admin/upload-dataset", methods=["POST"])
def admin_upload_dataset():
    if not session.get("user") or session["user"]["role"] != "admin":
        return jsonify({"error": "Unauthorized"}), 403

    file      = request.files.get("file")
    dtype     = request.form.get("type", "price")   # "price" or "weather"

    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No file provided"}), 400

    filename  = file.filename.lower()
    steps     = []

    try:
        # ── Price data ─────────────────────────────────────────────────────
        if dtype == "price":
            if not (filename.endswith(".xls") or filename.endswith(".xlsx") or filename.endswith(".csv")):
                return jsonify({"success": False, "error": "Must be .xls, .xlsx, or .csv"}), 400

            steps.append(f"Reading file: {file.filename}")
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            steps.append(f"Loaded {len(df):,} raw records, {df.shape[1]} columns")

            # Clean
            df["Wholesale"] = pd.to_numeric(df["Wholesale"], errors="coerce")
            before = len(df)
            df = df.dropna(subset=["Wholesale", "County"])
            df = df[df["Wholesale"] > 0]
            df = df[df["Wholesale"] <= 400]
            removed = before - len(df)
            steps.append(f"Cleaned: removed {removed} invalid/outlier rows")
            steps.append(f"Clean records: {len(df):,} ({round(len(df)/before*100,1)}% retention)")

            # Save to CSV for the system to use
            save_path = os.path.join(BASE_DIR, "uploaded_prices.csv")
            df.to_csv(save_path, index=False)
            steps.append(f"✓ Saved to uploaded_prices.csv — {len(df):,} clean records ready")

        # ── Weather data ───────────────────────────────────────────────────
        elif dtype == "weather":
            if not filename.endswith(".csv"):
                return jsonify({"success": False, "error": "Weather data must be a .csv file"}), 400

            steps.append(f"Reading weather CSV: {file.filename}")
            df = pd.read_csv(file)
            steps.append(f"Loaded {len(df):,} records, columns: {list(df.columns)}")

            # Detect and rename columns
            cols = [c.lower() for c in df.columns]
            if any("rain" in c for c in cols):
                df.columns = [c.strip() for c in df.columns]
                steps.append("Detected columns: date, rainfall, temperature")
            missing = int(df.isnull().sum().sum())
            steps.append(f"Missing values: {missing}")

            save_path = os.path.join(BASE_DIR, "uploaded_weather.csv")
            df.to_csv(save_path, index=False)
            steps.append(f"✓ Saved to uploaded_weather.csv — {len(df):,} records ready")

        # ── Check if both files exist → auto-merge ─────────────────────────
        price_path   = os.path.join(BASE_DIR, "uploaded_prices.csv")
        weather_path = os.path.join(BASE_DIR, "uploaded_weather.csv")
        if os.path.exists(price_path) and os.path.exists(weather_path):
            steps.append("Both datasets available — attempting auto-merge...")
            try:
                prices  = pd.read_csv(price_path)
                weather = pd.read_csv(weather_path)
                prices["Date"]   = pd.to_datetime(prices["Date"], errors="coerce")
                # Try to find the date column in weather
                date_col = next((c for c in weather.columns if "time" in c.lower() or "date" in c.lower()), weather.columns[0])
                weather[date_col] = pd.to_datetime(weather[date_col], dayfirst=True, errors="coerce")
                weather = weather.rename(columns={date_col: "date"})
                merged  = prices.merge(weather, left_on="Date", right_on="date", how="left")
                merged.to_csv(DATA_PATH, index=False)
                steps.append(f"✓ Merged dataset saved — {len(merged):,} records, {merged.shape[1]} columns")
            except Exception as me:
                steps.append(f"⚠ Merge skipped: {str(me)[:80]}")

        return jsonify({"success": True, "steps": steps})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# ── Security headers ─────────────────────────────────────────────────────────
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "SAMEORIGIN"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    return response

# ── Session timeout (30 minutes of inactivity) ───────────────────────────────
@app.before_request
def check_session_timeout():
    session.permanent = True
    app.permanent_session_lifetime = __import__("datetime").timedelta(minutes=30)

if __name__ == "__main__":
    init_db()
    print("✅ Maize Prediction System running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)

