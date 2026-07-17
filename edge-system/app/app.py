"""Edge Inference Engine — Receives sensor data, runs ML predictions, persists results.

Subscribes to sensor/water/raw (from sensor-sim or real sensors),
runs the chlorophyll prediction pipeline, and:
    - Stores every reading + prediction in SQLite (WAL mode, crash-safe)
    - Publishes predictions to sensor/water/prediction
    - Publishes alarms to sensor/water/alarm (QoS 2) when chlorophyll ≥ 10 µg/L
    - Logs per-sample performance metrics (inference_ms, db_write_ms)
"""

import os
import json
import time
import sqlite3
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt_broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_RAW = "sensor/water/raw"
TOPIC_PREDICTION = "sensor/water/prediction"
TOPIC_ALARM = "sensor/water/alarm"

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
DB_PATH = os.getenv("DB_PATH", "/data/sensor_data.db")

# WHO Alert Level 1 threshold (µg/L)
ALARM_THRESHOLD = 10.0

# --- DATABASE SETUP ---
def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize the SQLite database with WAL mode for crash resilience.
    
    WAL (Write-Ahead Logging) ensures that if the Pi loses power mid-write,
    the database automatically rolls back to the last complete transaction.
    No corruption, no manual recovery needed.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Enable WAL mode — critical for embedded reliability
    conn.execute("PRAGMA journal_mode=WAL;")
    # Sync on commit but not on every write — balances safety vs SD card wear
    conn.execute("PRAGMA synchronous=NORMAL;")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT    NOT NULL,
            temp_c                  REAL,
            spcond_us_cm            REAL,
            ph                      REAL,
            battery                 REAL,
            chlorophyll_predicted   REAL,
            chlorophyll_actual      REAL,
            alarm_triggered         INTEGER DEFAULT 0,
            inference_ms            REAL,
            processed_at            TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn


def print_db_stats(conn: sqlite3.Connection):
    """Print database statistics on startup — shows what survived the last reboot."""
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
        if row_count > 0:
            last_ts = conn.execute(
                "SELECT timestamp FROM readings ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
            print(f"  -> Database: {row_count} historical readings. Last: {last_ts}")
        else:
            print("  -> Database: Empty (fresh start).")
    except Exception as e:
        print(f"  -> Database stats unavailable: {e}")


def store_reading(conn: sqlite3.Connection, timestamp: str, features: dict,
                  prediction: float, ground_truth: float, alarm: bool,
                  inference_ms: float):
    """Insert a reading into the SQLite database."""
    conn.execute(
        """INSERT INTO readings 
           (timestamp, temp_c, spcond_us_cm, ph, battery,
            chlorophyll_predicted, chlorophyll_actual, alarm_triggered, inference_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            timestamp,
            features.get("EXO3(Temp_C)"),
            features.get("EXO3(spCond_uS_cm)"),
            features.get("EXO3(pH)"),
            features.get("SystemBattery"),
            prediction,
            ground_truth,
            1 if alarm else 0,
            inference_ms
        )
    )
    conn.commit()


# --- MODEL LOADING ---
print("=" * 65)
print("  EDGE INFERENCE ENGINE")
print("=" * 65)

print("\n1. Loading ML Pipeline...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"  -> Pipeline loaded from {MODEL_PATH}")
except Exception as e:
    print(f"  -> CRITICAL: Failed to load model: {e}")
    exit(1)

print("\n2. Initializing SQLite Database...")
db_conn = init_database(DB_PATH)
print(f"  -> Database at {DB_PATH} (WAL mode enabled)")
print_db_stats(db_conn)


# --- MQTT CALLBACKS ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"\n  Connected to broker! Subscribing to '{TOPIC_RAW}'...\n")
        client.subscribe(TOPIC_RAW, qos=1)
    else:
        print(f"  Connection failed: reason_code={reason_code}")


def on_message(client, userdata, msg):
    """Process an incoming sensor reading: parse → infer → store → publish."""
    try:
        # --- Parse ---
        payload = json.loads(msg.payload.decode("utf-8"))
        features = payload.get("features", {})
        ground_truth = payload.get("ground_truth", 0.0)
        timestamp = payload.get("timestamp", "Unknown")

        # --- Inference (timed) ---
        t0 = time.perf_counter()
        df = pd.DataFrame([features])
        prediction = model.predict(df)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        # --- Alarm check ---
        alarm = prediction >= ALARM_THRESHOLD
        alarm_flag = "🚨 ALARM LEVEL 1" if alarm else "✅ Normal"

        # --- Store in DB (timed) ---
        t1 = time.perf_counter()
        store_reading(db_conn, timestamp, features, prediction,
                      ground_truth, alarm, inference_ms)
        db_write_ms = (time.perf_counter() - t1) * 1000

        # --- Publish prediction ---
        pred_payload = json.dumps({
            "timestamp": timestamp,
            "predicted_chlorophyll": round(prediction, 3),
            "actual_chlorophyll": round(ground_truth, 3),
            "alarm": alarm,
            "inference_ms": round(inference_ms, 2)
        })
        client.publish(TOPIC_PREDICTION, pred_payload, qos=1)

        # --- Publish alarm (if triggered) ---
        if alarm:
            alarm_payload = json.dumps({
                "timestamp": timestamp,
                "predicted_chlorophyll": round(prediction, 3),
                "threshold": ALARM_THRESHOLD,
                "level": "WHO_LEVEL_1"
            })
            client.publish(TOPIC_ALARM, alarm_payload, qos=2)

        # --- Log ---
        print(f"[{timestamp}]")
        print(f"  Inputs:    Temp={features.get('EXO3(Temp_C)')}, "
              f"EC={features.get('EXO3(spCond_uS_cm)')}, "
              f"pH={features.get('EXO3(pH)')}, "
              f"Bat={features.get('SystemBattery')}")
        print(f"  Predicted: {prediction:.3f} µg/L  {alarm_flag}")
        print(f"  Actual:    {ground_truth:.3f} µg/L")
        print(f"  Perf:      inference={inference_ms:.1f}ms  db_write={db_write_ms:.1f}ms")
        print("-" * 65)

    except Exception as e:
        print(f"  ERROR processing message: {e}")


# --- NETWORK STARTUP ---
print("\n3. Connecting to MQTT Broker...")
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# Retry loop — Docker Compose race condition protection
while True:
    try:
        print(f"  Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        break
    except Exception as e:
        print(f"  Broker not ready ({e}). Retrying in 3s...")
        time.sleep(3)

print("\n" + "=" * 65)
print("  Engine ready. Waiting for sensor data...")
print("=" * 65 + "\n")

# Block the main thread and listen forever
client.loop_forever()