from flask import Flask, jsonify, send_from_directory, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import requests
from datetime import datetime
import numpy as np
import random
import math
import os

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./sensor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# =========================
# Configuration
# =========================
PHONE_IP = "10.168.100.75:8080"   # Your phone's phyphox server
USE_SYNTHETIC = False              # Try real phone first; fallback to synthetic if unreachable
PHONE_TIMEOUT = 2                  # seconds

# Tracks whether last fetch was real or synthetic
_data_source = {"mode": "unknown", "phone_ok": False}

# Simulated person activity state
_sim_state = {
    "activity": "walking",
    "step": 0,
}

ACTIVITIES = ["walking", "running", "sitting"]


# =========================
# Database Models
# =========================

class SensorData(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    accX        = db.Column(db.Float)
    accY        = db.Column(db.Float)
    accZ        = db.Column(db.Float)
    gyroX       = db.Column(db.Float)
    gyroY       = db.Column(db.Float)
    gyroZ       = db.Column(db.Float)
    magnitude   = db.Column(db.Float)
    activity    = db.Column(db.String(32))
    anomaly     = db.Column(db.Boolean, default=False)
    fall_detected = db.Column(db.Boolean, default=False)
    source      = db.Column(db.String(16), default="synthetic")   # "real" | "synthetic"
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)


class Alert(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    alert_type  = db.Column(db.String(64))   # "fall" | "anomaly" | "impact"
    severity    = db.Column(db.String(16))   # "low" | "medium" | "high"
    message     = db.Column(db.String(256))
    magnitude   = db.Column(db.Float)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    acknowledged = db.Column(db.Boolean, default=False)


with app.app_context():
    db.create_all()


# =========================
# Synthetic Sensor Generator
# =========================

def synthetic_data():
    """Generate realistic sensor data based on simulated activity."""
    s = _sim_state
    s["step"] += 1
    t = s["step"] * 0.1

    if s["step"] % 100 == 0:
        s["activity"] = random.choice(ACTIVITIES)
    if s["step"] % 300 == 0:
        s["activity"] = "falling"

    activity = s["activity"]

    if activity == "walking":
        accX  =  0.3 * math.sin(2 * math.pi * 2 * t)         + random.gauss(0, 0.10)
        accY  =  9.8 + 0.5 * math.sin(2 * math.pi * 2 * t + 0.5) + random.gauss(0, 0.15)
        accZ  =  0.2 * math.cos(2 * math.pi * 2 * t)         + random.gauss(0, 0.10)
        gyroX =  0.8 * math.sin(2 * math.pi * 1.5 * t)       + random.gauss(0, 0.20)
        gyroY =  0.5 * math.cos(2 * math.pi * 1.5 * t)       + random.gauss(0, 0.15)
        gyroZ =  0.3 * math.sin(2 * math.pi * t)              + random.gauss(0, 0.10)

    elif activity == "running":
        accX  =  1.2 * math.sin(2 * math.pi * 3.5 * t)       + random.gauss(0, 0.20)
        accY  =  9.8 + 2.0 * math.sin(2 * math.pi * 3.5 * t + 0.3) + random.gauss(0, 0.30)
        accZ  =  0.8 * math.cos(2 * math.pi * 3.5 * t)       + random.gauss(0, 0.20)
        gyroX =  2.5 * math.sin(2 * math.pi * 3 * t)         + random.gauss(0, 0.40)
        gyroY =  1.8 * math.cos(2 * math.pi * 3 * t)         + random.gauss(0, 0.30)
        gyroZ =  1.2 * math.sin(2 * math.pi * 2 * t)         + random.gauss(0, 0.20)

    elif activity == "sitting":
        accX  =  random.gauss(0, 0.05)
        accY  =  9.81 + random.gauss(0, 0.05)
        accZ  =  random.gauss(0, 0.05)
        gyroX =  random.gauss(0, 0.10)
        gyroY =  random.gauss(0, 0.10)
        gyroZ =  random.gauss(0, 0.10)

    elif activity == "falling":
        phase = s["step"] % 30
        if phase < 15:
            accX = random.gauss(0, 0.3)
            accY = random.gauss(0, 0.3)
            accZ = random.gauss(0, 0.3)
        else:
            accX = random.uniform(-8, 8)
            accY = random.uniform(-8, 8)
            accZ = random.uniform(-8, 8)
        gyroX = random.uniform(-15, 15)
        gyroY = random.uniform(-15, 15)
        gyroZ = random.uniform(-15, 15)
        if phase == 29:
            s["activity"] = "sitting"

    else:
        accX = accY = accZ = gyroX = gyroY = gyroZ = 0.0

    return {
        "accX":     round(accX,  4),
        "accY":     round(accY,  4),
        "accZ":     round(accZ,  4),
        "gyroX":    round(gyroX, 4),
        "gyroY":    round(gyroY, 4),
        "gyroZ":    round(gyroZ, 4),
        "activity": activity,
        "source":   "synthetic",
    }


# =========================
# Phyphox Phone Fetcher
# =========================

def _safe_val(buf, key):
    """Extract the latest value from a phyphox buffer dict."""
    try:
        entry = buf.get(key, {})
        # phyphox returns {"buffer":[...], "size":N, "updateTime":...}
        data = entry.get("buffer", [])
        if data:
            val = data[-1]
            return float(val) if val is not None else 0.0
        return 0.0
    except Exception:
        return 0.0


def fetch_real_sensor():
    """
    Fetch live accelerometer + gyroscope data from phyphox running on the phone.
    Endpoint: http://<phone_ip>/get?accX&accY&accZ&gyroX&gyroY&gyroZ
    Falls back to synthetic on any network / parse error.
    """
    try:
        url = f"http://{PHONE_IP}/get?accX&accY&accZ&gyroX&gyroY&gyroZ"
        resp = requests.get(url, timeout=PHONE_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()

        # phyphox wraps everything under "buffer" key at top level
        buf = payload.get("buffer", payload)   # handle both shapes

        result = {
            "accX":     _safe_val(buf, "accX"),
            "accY":     _safe_val(buf, "accY"),
            "accZ":     _safe_val(buf, "accZ"),
            "gyroX":    _safe_val(buf, "gyroX"),
            "gyroY":    _safe_val(buf, "gyroY"),
            "gyroZ":    _safe_val(buf, "gyroZ"),
            "activity": "unknown",
            "source":   "real",
        }

        _data_source["phone_ok"] = True
        _data_source["mode"]     = "real"
        return result

    except requests.exceptions.ConnectionError:
        _data_source["phone_ok"] = False
        _data_source["mode"]     = "synthetic (phone unreachable)"
        return synthetic_data()

    except requests.exceptions.Timeout:
        _data_source["phone_ok"] = False
        _data_source["mode"]     = "synthetic (phone timeout)"
        return synthetic_data()

    except Exception as e:
        print(f"[phyphox] fetch error: {e}")
        _data_source["phone_ok"] = False
        _data_source["mode"]     = "synthetic (parse error)"
        return synthetic_data()


# =========================
# Activity Classifier
# =========================

def classify_activity(accX, accY, accZ, gyroX, gyroY, gyroZ):
    mag      = math.sqrt(accX**2 + accY**2 + accZ**2)
    gyro_mag = math.sqrt(gyroX**2 + gyroY**2 + gyroZ**2)

    if mag < 1.5:
        return "free_fall"
    elif mag > 18:
        return "impact"
    elif mag < 3:
        return "sitting"
    elif gyro_mag > 8 or mag > 14:
        return "running"
    elif gyro_mag > 2 or abs(mag - 9.81) > 1.5:
        return "walking"
    else:
        return "sitting"


# =========================
# Anomaly Detection (Z-score)
# =========================

def detect_anomaly(latest_mag):
    records = SensorData.query.order_by(SensorData.id.desc()).limit(30).all()
    if len(records) < 10:
        return False, 0.0
    values = [r.magnitude for r in records if r.magnitude is not None]
    if len(values) < 5:
        return False, 0.0
    mean = np.mean(values)
    std  = np.std(values)
    if std < 0.01:
        return False, 0.0
    z_score = abs(latest_mag - mean) / std
    return z_score > 2.5, round(float(z_score), 3)


# =========================
# Fall Detection
# =========================

def detect_fall(accX, accY, accZ, activity):
    mag       = math.sqrt(accX**2 + accY**2 + accZ**2)
    free_fall = mag < 3.0
    impact    = mag > 22.0
    return free_fall or impact or activity in ("falling", "free_fall", "impact")


# =========================
# Alert Creator
# =========================

def create_alert(alert_type, severity, message, magnitude):
    alert = Alert(
        alert_type=alert_type,
        severity=severity,
        message=message,
        magnitude=round(float(magnitude), 3),
    )
    db.session.add(alert)
    db.session.commit()


# =========================
# Routes
# =========================

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')


@app.route('/data')
def get_data():
    try:
        if USE_SYNTHETIC:
            sensor = synthetic_data()
        else:
            sensor = fetch_real_sensor()

        accX  = sensor["accX"]
        accY  = sensor["accY"]
        accZ  = sensor["accZ"]
        gyroX = sensor["gyroX"]
        gyroY = sensor["gyroY"]
        gyroZ = sensor["gyroZ"]
        source = sensor.get("source", "synthetic")

        magnitude = round(math.sqrt(accX**2 + accY**2 + accZ**2), 4)

        # Use classifier for real data; trust label for synthetic
        if source == "real" or sensor.get("activity") == "unknown":
            activity = classify_activity(accX, accY, accZ, gyroX, gyroY, gyroZ)
        else:
            activity = sensor.get("activity") or classify_activity(accX, accY, accZ, gyroX, gyroY, gyroZ)

        is_anomaly, z_score = detect_anomaly(magnitude)
        fall_detected       = detect_fall(accX, accY, accZ, activity)

        record = SensorData(
            accX=accX, accY=accY, accZ=accZ,
            gyroX=gyroX, gyroY=gyroY, gyroZ=gyroZ,
            magnitude=magnitude,
            activity=activity,
            anomaly=bool(is_anomaly or fall_detected),
            fall_detected=bool(fall_detected),
            source=source,
        )
        db.session.add(record)
        db.session.commit()

        if fall_detected:
            create_alert("fall", "high",
                         f"Fall detected! Magnitude: {magnitude:.2f} g  |  Source: {source}",
                         magnitude)
        elif is_anomaly:
            severity = "high" if z_score > 4 else "medium"
            create_alert("anomaly", severity,
                         f"Anomaly detected. Z-score: {z_score:.2f}  |  Source: {source}",
                         magnitude)

        return jsonify({
            "accX": accX, "accY": accY, "accZ": accZ,
            "gyroX": gyroX, "gyroY": gyroY, "gyroZ": gyroZ,
            "magnitude": magnitude,
            "activity": activity,
            "anomaly": bool(is_anomaly or fall_detected),
            "fall_detected": bool(fall_detected),
            "z_score": z_score,
            "source": source,
            "phone_ok": _data_source["phone_ok"],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/history')
def history():
    limit   = request.args.get('limit', 100, type=int)
    records = SensorData.query.order_by(SensorData.id.desc()).limit(limit).all()
    data = [{
        "id":           r.id,
        "accX":         r.accX,  "accY": r.accY,  "accZ": r.accZ,
        "gyroX":        r.gyroX, "gyroY": r.gyroY, "gyroZ": r.gyroZ,
        "magnitude":    r.magnitude,
        "activity":     r.activity,
        "anomaly":      r.anomaly,
        "fall_detected": r.fall_detected,
        "source":       r.source,
        "time":         r.timestamp.strftime("%H:%M:%S"),
        "timestamp":    r.timestamp.isoformat(),
    } for r in reversed(records)]
    return jsonify(data)


@app.route('/alerts')
def get_alerts():
    limit      = request.args.get('limit', 20, type=int)
    unack_only = request.args.get('unacknowledged', 'false').lower() == 'true'
    q = Alert.query.order_by(Alert.id.desc())
    if unack_only:
        q = q.filter_by(acknowledged=False)
    alerts = q.limit(limit).all()
    return jsonify([{
        "id":           a.id,
        "type":         a.alert_type,
        "severity":     a.severity,
        "message":      a.message,
        "magnitude":    a.magnitude,
        "time":         a.timestamp.strftime("%H:%M:%S"),
        "timestamp":    a.timestamp.isoformat(),
        "acknowledged": a.acknowledged,
    } for a in alerts])


@app.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    alert.acknowledged = True
    db.session.commit()
    return jsonify({"success": True})


@app.route('/alerts/acknowledge_all', methods=['POST'])
def acknowledge_all():
    Alert.query.filter_by(acknowledged=False).update({"acknowledged": True})
    db.session.commit()
    return jsonify({"success": True})


@app.route('/stats')
def stats():
    total        = SensorData.query.count()
    falls        = SensorData.query.filter_by(fall_detected=True).count()
    anomalies    = SensorData.query.filter_by(anomaly=True).count()
    unacked      = Alert.query.filter_by(acknowledged=False).count()
    real_count   = SensorData.query.filter_by(source='real').count()
    synth_count  = SensorData.query.filter_by(source='synthetic').count()

    activities = db.session.query(
        SensorData.activity, db.func.count(SensorData.id)
    ).group_by(SensorData.activity).all()
    activity_counts = {a: c for a, c in activities}

    recent = SensorData.query.order_by(SensorData.id.desc()).limit(100).all()
    mags   = [r.magnitude for r in recent if r.magnitude]
    mag_stats = {
        "mean": round(float(np.mean(mags)), 3) if mags else 0,
        "max":  round(float(np.max(mags)),  3) if mags else 0,
        "min":  round(float(np.min(mags)),  3) if mags else 0,
        "std":  round(float(np.std(mags)),  3) if mags else 0,
    }

    return jsonify({
        "total_records":       total,
        "fall_events":         falls,
        "anomaly_events":      anomalies,
        "unacknowledged_alerts": unacked,
        "real_readings":       real_count,
        "synthetic_readings":  synth_count,
        "activity_breakdown":  activity_counts,
        "magnitude_stats":     mag_stats,
        "data_source":         _data_source["mode"],
        "phone_ok":            _data_source["phone_ok"],
    })


@app.route('/source_status')
def source_status():
    """Quick endpoint for frontend to poll connection health."""
    return jsonify({
        "phone_ip":  PHONE_IP,
        "phone_ok":  _data_source["phone_ok"],
        "mode":      _data_source["mode"],
        "use_synthetic_flag": USE_SYNTHETIC,
    })


@app.route('/simulate_fall', methods=['POST'])
def trigger_fall():
    _sim_state["activity"] = "falling"
    _sim_state["step"]     = 0
    return jsonify({"message": "Fall simulation triggered"})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    SensorData.query.delete()
    Alert.query.delete()
    db.session.commit()
    return jsonify({"message": "History cleared"})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
