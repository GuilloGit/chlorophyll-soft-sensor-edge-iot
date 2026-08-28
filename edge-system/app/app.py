"""
===============================================================================
Module Name:       app.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Edge Processing Tier (Inference & Persistence Engine)

Description:       Receives streaming multi-parameter physicochemical telemetry,
                   executes machine learning inference pipeline for Chlorophyll-a
                   estimation, persists readings and hardware metrics in SQLite
                   with Write-Ahead Logging (WAL) mode, aggregates 24-hour daily
                   batches for low-power transmission, triggers immediate WHO
                   Alert Level 1 alarms upon threshold exceedance, and manages
                   remote model updates via the Test-Before-Swap OTA protocol.

Data Interfaces:
  - Upstream:      MQTT topics:
                     - sensor/water/raw (QoS 1, streaming sensor readings)
                     - buoy/ota/update (QoS 1, remote OTA model commands)
  - Downstream:    MQTT topics:
                     - sensor/water/daily_batch (QoS 1, 24-hour aggregated telemetry)
                     - sensor/water/alarm (QoS 2, critical threshold alerts)
                     - buoy/ota/status (QoS 1, OTA deployment status notifications)
  - Storage / IPC: SQLite database with WAL journal mode (/data/sensor_data.db);
                   Joblib serialized pipeline (/data/model.joblib).

References:        Mozo et al. (2022); WHO Guidelines for Safe Recreational
                   Water Environments (2003).
===============================================================================
"""

import os
import json
import time
import shutil
import sqlite3
import hashlib
import threading
import pandas as pd
import joblib
import requests
import paho.mqtt.client as mqtt
import psutil

# --- CONFIGURATION ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt_broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_RAW = "sensor/water/raw"
TOPIC_DAILY_BATCH = "sensor/water/daily_batch"
TOPIC_ALARM = "sensor/water/alarm"

# OTA topics (same broker)
TOPIC_OTA_COMMAND = "buoy/ota/update"
TOPIC_OTA_STATUS = "buoy/ota/status"

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
TEMP_MODEL_PATH = MODEL_PATH + ".tmp"
BAKED_MODEL_PATH = os.getenv("BAKED_MODEL_PATH", "model.joblib")
DB_PATH = os.getenv("DB_PATH", "/data/sensor_data.db")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 96))  # 96 readings = 24 hours at 15min intervals

# WHO Alert Level 1 threshold (µg/L)
ALARM_THRESHOLD = 10.0

# Thread-safe batching state
batch_lock = threading.Lock()
current_batch = []

# Thread-safe model access
model_lock = threading.Lock()
current_model = None  # Loaded during startup below


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
            db_write_ms             REAL,
            processed_at            TEXT    DEFAULT (datetime('now'))
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT    NOT NULL,
            cpu_percent             REAL,
            ram_mb                  REAL
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
                  inference_ms: float, db_write_ms: float):
    """Insert a reading into the SQLite database."""
    conn.execute(
        """INSERT INTO readings 
           (timestamp, temp_c, spcond_us_cm, ph, battery,
            chlorophyll_predicted, chlorophyll_actual, alarm_triggered, inference_ms, db_write_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(timestamp),
            float(features.get("EXO3(Temp_C)", 0.0)),
            float(features.get("EXO3(spCond_uS_cm)", 0.0)),
            float(features.get("EXO3(pH)", 0.0)),
            float(features.get("SystemBattery", 0.0)),
            float(prediction),
            float(ground_truth),
            1 if alarm else 0,
            float(inference_ms),
            float(db_write_ms)
        )
    )
    conn.commit()


# --- MODEL LOADING ---
def load_initial_model():
    """Load the ML model, copying the baked-in image to the persistent volume if needed.
    
    When MODEL_PATH points to a persistent volume (e.g. /data/model.joblib),
    the Dockerfile-baked model serves as the initial fallback. This function
    copies it to the volume on first boot so OTA updates persist across
    container rebuilds.
    """
    if not os.path.exists(MODEL_PATH) and MODEL_PATH != BAKED_MODEL_PATH:
        if os.path.exists(BAKED_MODEL_PATH):
            print(f"  -> First boot: copying baked model to {MODEL_PATH}")
            shutil.copy2(BAKED_MODEL_PATH, MODEL_PATH)
        else:
            print(f"  -> CRITICAL: No model found at {MODEL_PATH} or {BAKED_MODEL_PATH}")
            exit(1)

    return joblib.load(MODEL_PATH)


print("=" * 65)
print("  EDGE INFERENCE ENGINE (with OTA)")
print("=" * 65)

print("\n1. Loading ML Pipeline...")
try:
    current_model = load_initial_model()
    print(f"  -> Pipeline loaded from {MODEL_PATH}")
except Exception as e:
    print(f"  -> CRITICAL: Failed to load model: {e}")
    exit(1)

print("\n2. Initializing SQLite Database...")
db_conn = init_database(DB_PATH)
print(f"  -> Database at {DB_PATH} (WAL mode enabled)")
print_db_stats(db_conn)


# --- OTA UPDATE HANDLER ---
def _handle_ota_update(client, payload):
    """Test-Before-Swap OTA: download → hash → trial-load → hot-swap → persist.
    
    This is the core fail-safe mechanism:
    1. Download the new model to a temporary file
    2. Verify the SHA-256 hash matches the expected value
    3. Trial-load the model into a dummy variable (catches corrupt files)
    4. Acquire the model lock and swap the in-memory reference
    5. Atomically replace the on-disk model (survives reboots)
    """
    global current_model

    url = payload.get("url")
    expected_hash = payload.get("sha256")
    version = payload.get("version", "unknown")

    if not url or not expected_hash:
        print("[OTA] Invalid payload: missing 'url' or 'sha256'. Aborting.")
        return

    print(f"[OTA] Update received — version={version}")
    print(f"[OTA] Downloading from: {url}")

    # 1. Download to a temporary file
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
    except Exception as e:
        print(f"[OTA] Download failed: {e}. Aborting.")
        _publish_ota_status(client, version, "failed", f"Download error: {e}")
        return

    with open(TEMP_MODEL_PATH, "wb") as f:
        f.write(response.content)

    print(f"[OTA] Downloaded {len(response.content)} bytes.")

    # 2. Verify SHA-256
    actual_hash = hashlib.sha256(response.content).hexdigest()
    if actual_hash != expected_hash:
        print(f"[OTA] Hash mismatch! Expected: {expected_hash[:16]}... "
              f"Got: {actual_hash[:16]}... Aborting.")
        os.remove(TEMP_MODEL_PATH)
        _publish_ota_status(client, version, "failed", "Hash mismatch")
        return

    print("[INFO] [OTAManager] SHA-256 verified successfully.")

    # 3. Trial-load (the ultimate fail-safe)
    try:
        new_model = joblib.load(TEMP_MODEL_PATH)
    except Exception as e:
        print(f"[OTA] Model corrupted or incompatible: {e}. Aborting.")
        os.remove(TEMP_MODEL_PATH)
        _publish_ota_status(client, version, "failed", f"Load error: {e}")
        return

    print("[INFO] [OTAManager] Trial-load completed successfully.")

    # 4. Hot-swap in memory (thread-safe)
    with model_lock:
        current_model = new_model

    # 5. Persist atomically to disk (survives reboots)
    os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    print(f"[INFO] [OTAManager] Update successful: Version {version} is now active and persisted.")
    _publish_ota_status(client, version, "success", "Model updated")


def _publish_ota_status(client, version, status, message):
    """Publish OTA update status for confirmation."""
    status_payload = json.dumps({
        "version": version,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    client.publish(TOPIC_OTA_STATUS, status_payload, qos=1)


# (System monitoring thread removed in favor of polling during inference for batching)


# --- MQTT CALLBACKS (Single client, dispatched by topic) ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"\n  Connected to broker! Subscribing to topics...")
        client.subscribe(TOPIC_RAW, qos=1)
        client.subscribe(TOPIC_OTA_COMMAND, qos=1)
        print(f"    → {TOPIC_RAW} (sensor data)")
        print(f"    → {TOPIC_OTA_COMMAND} (OTA commands)\n")
    else:
        print(f"  Connection failed: reason_code={reason_code}")


def on_message(client, userdata, msg):
    """Dispatch incoming messages by topic."""
    if msg.topic == TOPIC_RAW:
        _handle_sensor_reading(client, msg)
    elif msg.topic == TOPIC_OTA_COMMAND:
        _handle_ota_message(client, msg)


def _handle_sensor_reading(client, msg):
    """Process an incoming sensor reading: parse → infer → store → batch publish."""
    global current_batch
    try:
        # --- Parse ---
        payload = json.loads(msg.payload.decode("utf-8"))
        features = payload.get("features", {})
        ground_truth = payload.get("ground_truth", 0.0)
        timestamp = payload.get("timestamp", "Unknown")

        # --- Inference (timed, thread-safe) ---
        t0 = time.perf_counter()
        df = pd.DataFrame([features])
        with model_lock:
            prediction = float(current_model.predict(df)[0])
        inference_ms = float((time.perf_counter() - t0) * 1000)
        
        # --- System Metrics ---
        cpu_usage = float(psutil.cpu_percent())
        ram_mb = float(psutil.virtual_memory().used / (1024 * 1024))

        # --- Alarm check ---
        alarm = bool(prediction >= ALARM_THRESHOLD)
        alarm_flag = "[ALERT] [WHO-Level-1]" if alarm else "[NORMAL]"

        # --- Store in DB (timed) ---
        t1 = time.perf_counter()
        db_conn.execute(
            "INSERT INTO system_metrics (timestamp, cpu_percent, ram_mb) VALUES (?, ?, ?)",
            (str(timestamp), float(cpu_usage), float(ram_mb))
        )
        db_conn.commit()
        db_write_ms = float((time.perf_counter() - t1) * 1000)

        store_reading(db_conn, timestamp, features, prediction,
                      float(ground_truth), alarm, inference_ms, db_write_ms)

        # --- Batching Logic ---
        reading_data = {
            "timestamp": str(timestamp),
            "predicted_chlorophyll": round(float(prediction), 3),
            "actual_chlorophyll": round(float(ground_truth), 3),
            "alarm": bool(alarm),
            "inference_ms": round(float(inference_ms), 2),
            "db_write_ms": round(float(db_write_ms), 2),
            "cpu_percent": round(float(cpu_usage), 1),
            "ram_mb": round(float(ram_mb), 1)
        }
        
        with batch_lock:
            current_batch.append(reading_data)
            batch_size = len(current_batch)
            
            if batch_size >= BATCH_SIZE:
                # Publish the entire batch
                batch_payload = json.dumps(current_batch)
                client.publish(TOPIC_DAILY_BATCH, batch_payload, qos=1)
                print(f"  [INFO] [InferenceEngine] Published daily batch of {batch_size} readings to {TOPIC_DAILY_BATCH}")
                current_batch = []

        # --- Publish alarm (Immediate override) ---
        if alarm:
            alarm_payload = json.dumps({
                "timestamp": str(timestamp),
                "predicted_chlorophyll": round(float(prediction), 3),
                "threshold": float(ALARM_THRESHOLD),
                "level": "WHO_LEVEL_1"
            })
            client.publish(TOPIC_ALARM, alarm_payload, qos=2)
            print(f"  [ALERT] [InferenceEngine] Published ALARM immediately to {TOPIC_ALARM}")

        # --- Log ---
        print(f"[{timestamp}] (Batch: {batch_size}/{BATCH_SIZE})")
        print(f"  Inputs:    Temp={features.get('EXO3(Temp_C)')}, "
              f"EC={features.get('EXO3(spCond_uS_cm)')}, "
              f"pH={features.get('EXO3(pH)')}, "
              f"Bat={features.get('SystemBattery')}")
        print(f"  Predicted: {prediction:.3f} µg/L  {alarm_flag}")
        print(f"  Actual:    {ground_truth:.3f} µg/L")
        print(f"  Perf:      inference={inference_ms:.1f}ms  db_write={db_write_ms:.1f}ms")
        print("-" * 65)

    except Exception as e:
        print(f"  [ERROR] [InferenceEngine] Error processing message: {e}")


def _handle_ota_message(client, msg):
    """Handle incoming OTA update commands."""
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        _handle_ota_update(client, payload)
    except json.JSONDecodeError as e:
        print(f"[OTA] Invalid JSON payload: {e}")
    except Exception as e:
        print(f"[OTA] Unexpected error: {e}")


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
print("  Engine ready. Waiting for sensor data and OTA commands...")
print("=" * 65 + "\n")

# System monitoring is now handled within the inference loop directly.

# Block the main thread and listen forever
client.loop_forever()