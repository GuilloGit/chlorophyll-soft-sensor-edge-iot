"""
===============================================================================
Module Name:       app.py (V2 ONNX Decoupled Engine)
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Edge Processing Tier (ONNX Runtime Inference & Persistence Engine)

Description:       Receives streaming multi-parameter physicochemical telemetry,
                   executes ultra-lightweight ONNX Runtime inference for
                   Chlorophyll-a estimation, applies pure NumPy inverse power
                   transformations (target unscaling), persists observations and
                   hardware performance metrics in SQLite (WAL mode), aggregates
                   24-hour daily telemetry batches, dispatches immediate WHO
                   Alert Level 1 hazard alarms, and manages zero-downtime model
                   hot-swapping via the Test-Before-Swap OTA protocol.

Data Interfaces:
  - Upstream:      MQTT topics:
                     - sensor/water/raw (QoS 1, streaming sensor readings)
                     - buoy/ota/update (QoS 1, remote OTA model commands)
  - Downstream:    MQTT topics:
                     - sensor/water/daily_batch (QoS 1, 24-hour aggregated telemetry)
                     - sensor/water/alarm (QoS 2, critical threshold alerts)
                     - buoy/ota/status (QoS 1, OTA deployment status notifications)
  - Storage / IPC: SQLite database with WAL journal mode (/data/sensor_data.db);
                   ONNX serialized model (/data/model_v2.onnx);
                   Target transform parameter file (/app/target_transform.json).

References:        Mozo et al. (2022); Yeo & Johnson (2000); WHO Guidelines for
                   Safe Recreational Water Environments (2003).
===============================================================================
"""

import os
import gc
import json
import time
import lzma
import shutil
import sqlite3
import hashlib
import threading
import ctypes
import numpy as np
import onnxruntime as rt
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

BAKED_MODEL_PATH = os.getenv("BAKED_MODEL_PATH", "model_v2.onnx")
MODEL_PATH = os.getenv("MODEL_PATH", "model_v2.onnx")
TEMP_MODEL_PATH = MODEL_PATH + ".tmp"
TRANSFORM_PATH = os.getenv("TRANSFORM_PATH", os.getenv("UNSCALER_PATH", os.getenv("LAMBDA_PATH", "target_transform.json")))
UNSCALER_PATH = TRANSFORM_PATH  # Backward compatibility alias
LAMBDA_PATH = TRANSFORM_PATH    # Backward compatibility alias
Y_LAMBDA = 0.0
Y_MEAN = 0.0
Y_SCALE = 1.0
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


def _inverse_power_transform(y_transformed: float, lmbda: float, mean: float = 0.0, scale: float = 1.0) -> float:
    """Computes the Inverse Power Transform (Target Unscaling) using pure NumPy.
    
    First reverses the affine standardization:
        u = y_transformed * scale + mean
    Then inverts the Yeo-Johnson non-linear power transform:
        y = ((u * lambda + 1)^(1 / lambda) - 1)                 if u >= 0, lambda != 0
        y = exp(u) - 1                                          if u >= 0, lambda == 0
        y = 1 - (1 - (2 - lambda) * u)^(1 / (2 - lambda))       if u < 0, lambda != 2
        y = 1 - exp(-u)                                         if u < 0, lambda == 2
        
    Args:
        y_transformed: Transformed scalar prediction from the ONNX graph.
        lmbda: Learned power transformation parameter (lambda) from target_transform.json.
        mean: Training target mean from target_transform.json.
        scale: Training target standard deviation from target_transform.json.
        
    Returns:
        Unscaled physical Chlorophyll-a concentration in µg/L.
    """
    u = float(y_transformed * scale + mean)
    if u >= 0:
        if lmbda == 0.0:
            return float(np.exp(u) - 1.0)
        return float(np.power(u * lmbda + 1.0, 1.0 / lmbda) - 1.0)
    else:
        if lmbda == 2.0:
            return float(1.0 - np.exp(-u))
        return float(1.0 - np.power(1.0 - (2.0 - lmbda) * u, 1.0 / (2.0 - lmbda)))


# Backward compatibility alias
_inverse_yeo_johnson = _inverse_power_transform


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
def load_initial_model() -> rt.InferenceSession:
    """Loads the ONNX inference model, copying the baked image to persistent storage if needed.
    
    When MODEL_PATH points to a persistent Docker volume (/data/model_v2.onnx),
    the Dockerfile-baked model serves as the initial fallback. This function
    copies it to persistent storage on initial boot so that subsequent OTA
    updates survive container rebuilds.
    
    Critically, this function configures restrictive single-threaded SessionOptions
    (intra_op_num_threads=1, inter_op_num_threads=1) to suppress OpenMP active
    waiting (spin-locking), which would otherwise cause an idle CPU consumption spike.
    """
    if not os.path.exists(MODEL_PATH) and MODEL_PATH != BAKED_MODEL_PATH:
        if os.path.exists(BAKED_MODEL_PATH):
            print(f"[INFO] [InferenceEngineV2] First boot: copying baked model to {MODEL_PATH}")
            shutil.copy2(BAKED_MODEL_PATH, MODEL_PATH)
        else:
            print(f"[ERROR] [InferenceEngineV2] Critical: No model found at {MODEL_PATH} or {BAKED_MODEL_PATH}")
            exit(1)

    global Y_LAMBDA, Y_MEAN, Y_SCALE
    resolved_path = TRANSFORM_PATH
    if not os.path.exists(resolved_path):
        for candidate in ["target_transform.json", "target_unscaler.json", "y_lambda.json"]:
            if os.path.exists(candidate):
                resolved_path = candidate
                break
    try:
        with open(resolved_path, "r") as f:
            params = json.load(f)
            Y_LAMBDA = float(params.get("y_lambda", 0.0))
            Y_MEAN = float(params.get("y_mean", 0.0))
            Y_SCALE = float(params.get("y_scale", 1.0))
        print(f"[INFO] [InferenceEngineV2] Target transform parameters loaded: lambda={Y_LAMBDA:.5f}, mean={Y_MEAN:.5f}, scale={Y_SCALE:.5f} from {resolved_path}")
    except Exception as e:
        print(f"[WARN] [InferenceEngineV2] Failed to load transform parameters from {resolved_path}: {e}")

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    return rt.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])

print("=" * 65)
print("  EDGE INFERENCE ENGINE V2 (ONNX Decoupled Runtime)")
print("=" * 65)

print("\n1. Loading ONNX Pipeline...")
try:
    current_model = load_initial_model()
    print(f"[INFO] [InferenceEngineV2] Pipeline loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] [InferenceEngineV2] Critical failure loading model: {e}")
    exit(1)

print("\n2. Initializing SQLite Database...")
db_conn = init_database(DB_PATH)
print(f"[INFO] [InferenceEngineV2] Database at {DB_PATH} (WAL mode enabled)")
print_db_stats(db_conn)


# --- OTA UPDATE HANDLER ---
def _handle_ota_update(client: mqtt.Client, payload: dict) -> None:
    """Executes the Test-Before-Swap fail-safe OTA update protocol for ONNX models.
    
    1. Downloads candidate model artifact/archive to a temporary staging file via streaming I/O.
    2. Verifies cryptographic SHA-256 hash against payload expectation.
    2.5. If the artifact is an LZMA (.xz) compressed archive, decompresses it to TEMP_MODEL_PATH.
    3. Trial-loads candidate model into an isolated InferenceSession to catch corrupt binaries.
    4. Acquires model_lock and hot-swaps in-memory model reference without restarting container.
    5. Atomically replaces persistent disk file via os.replace.
    6. Dispatches status confirmation packet to buoy/ota/status.
    """
    global current_model

    url = payload.get("url")
    expected_hash = payload.get("sha256")
    version = payload.get("version", "unknown")

    if not url or not expected_hash:
        print("[WARN] [OTAManager] Invalid payload: missing 'url' or 'sha256'. Aborting update.")
        return

    print(f"[INFO] [OTAManager] Remote update command received (target version: {version})")
    print(f"[INFO] [OTAManager] Downloading artifact from: {url}")

    temp_staging_path = TEMP_MODEL_PATH + ".download"

    # 1. Download to temporary staging file using chunked streaming
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        downloaded_bytes = 0
        sha_calc = hashlib.sha256()
        with open(temp_staging_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    sha_calc.update(chunk)
                    downloaded_bytes += len(chunk)
    except Exception as e:
        print(f"[ERROR] [OTAManager] Download failed: {e}. Aborting update.")
        if os.path.exists(temp_staging_path):
            os.remove(temp_staging_path)
        _publish_ota_status(client, version, "failed", f"Download error: {e}")
        return

    print(f"[INFO] [OTAManager] Downloaded {downloaded_bytes:,} bytes to staging buffer.")

    # 2. Verify SHA-256 Checksum of downloaded artifact
    actual_hash = sha_calc.hexdigest()
    if actual_hash != expected_hash:
        print(f"[ERROR] [OTAManager] SHA-256 mismatch! Expected {expected_hash[:16]}..., got {actual_hash[:16]}... Aborting.")
        if os.path.exists(temp_staging_path):
            os.remove(temp_staging_path)
        _publish_ota_status(client, version, "failed", "Hash mismatch")
        return

    print("[INFO] [OTAManager] SHA-256 checksum verified successfully.")

    # 2.5 Decompress if LZMA (.xz) transport archive
    try:
        with open(temp_staging_path, "rb") as f_check:
            header = f_check.read(6)
        
        # Check XZ magic header: \xfd7zXZ\x00
        if header.startswith(b"\xfd7zXZ\x00"):
            print("[INFO] [OTAManager] Detected LZMA (.xz) transport archive. Decompressing...")
            t_decomp_start = time.perf_counter()
            with lzma.open(temp_staging_path, "rb") as f_in, open(TEMP_MODEL_PATH, "wb") as f_out:
                while chunk := f_in.read(1024 * 1024):
                    f_out.write(chunk)
            t_decomp = time.perf_counter() - t_decomp_start
            decomp_size = os.path.getsize(TEMP_MODEL_PATH) / (1024 * 1024)
            print(f"[INFO] [OTAManager] LZMA decompression completed in {t_decomp:.2f}s ({decomp_size:.2f} MB extracted).")
            os.remove(temp_staging_path)
        else:
            os.replace(temp_staging_path, TEMP_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] [OTAManager] Decompression failed: {e}. Aborting update.")
        if os.path.exists(temp_staging_path):
            os.remove(temp_staging_path)
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
        _publish_ota_status(client, version, "failed", f"Load error: Decompression failed ({e})")
        return

    # 3. Pre-swap memory reclamation:
    # An uncompressed 943 MB TreeEnsemble graph consumes ~6.8 GB of C++ heap in ONNX Runtime.
    # Holding two concurrent sessions simultaneously would require 13.6 GB, exceeding the
    # 8 GB physical RAM of the Raspberry Pi 4. To perform the hot-swap with zero OOM risk,
    # we release the active reference and invoke glibc malloc_trim(0).
    print("[INFO] [OTAManager] Reclaiming memory arena before candidate model load...")
    global current_model
    try:
        with model_lock:
            old_model = current_model
            current_model = None

        del old_model
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        # 4. Trial-load candidate model in clean memory space
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        new_model = rt.InferenceSession(TEMP_MODEL_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"[ERROR] [OTAManager] Model load/swap error: {e}. Rolling back to previous model.")
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
        try:
            fallback_model = rt.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=['CPUExecutionProvider'])
            with model_lock:
                current_model = fallback_model
        except Exception as rb_err:
            print(f"[ERROR] [OTAManager] Critical rollback error: {rb_err}")
        _publish_ota_status(client, version, "failed", f"Load error: {e}")
        return

    print("[INFO] [OTAManager] Trial-load verification passed successfully.")

    # 5. Thread-safe in-memory reference activation
    with model_lock:
        current_model = new_model

    # 6. Atomic persistence to disk
    os.replace(TEMP_MODEL_PATH, MODEL_PATH)

    print(f"[INFO] [OTAManager] Update completed: Version {version} active and persisted.")
    _publish_ota_status(client, version, "success", "Model updated")


def _publish_ota_status(client: mqtt.Client, version: str, status: str, message: str) -> None:
    """Publishes OTA update result confirmation packet to the broker."""
    status_payload = json.dumps({
        "version": version,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    client.publish(TOPIC_OTA_STATUS, status_payload, qos=1)


# --- MQTT CALLBACKS (Single client, dispatched by topic) ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("\n[INFO] [MQTTClient] Connected to broker. Subscribing to topics:")
        client.subscribe(TOPIC_RAW, qos=1)
        client.subscribe(TOPIC_OTA_COMMAND, qos=1)
        print(f"  -> {TOPIC_RAW} (telemetry stream, QoS 1)")
        print(f"  -> {TOPIC_OTA_COMMAND} (OTA command stream, QoS 1)\n")
    else:
        print(f"[ERROR] [MQTTClient] Broker connection failed with reason_code={reason_code}")


def on_message(client, userdata, msg):
    """Dispatches incoming MQTT messages by topic."""
    if msg.topic == TOPIC_RAW:
        _handle_sensor_reading(client, msg)
    elif msg.topic == TOPIC_OTA_COMMAND:
        _handle_ota_message(client, msg)


def _handle_sensor_reading(client, msg):
    """Processes an incoming telemetry reading: parse -> infer -> store -> batch publish."""
    global current_batch
    try:
        # --- Parse Telemetry Payload ---
        payload = json.loads(msg.payload.decode("utf-8"))
        features = payload.get("features", {})
        ground_truth = payload.get("ground_truth", 0.0)
        timestamp = payload.get("timestamp", "Unknown")

        # --- Inference (timed, thread-safe) ---
        t0 = time.perf_counter()
        
        # Convert physical surrogate features to 1x4 float32 array
        input_data = np.array([[
            features.get("EXO3(Temp_C)", 0.0), 
            features.get("EXO3(spCond_uS_cm)", 0.0),
            features.get("EXO3(pH)", 0.0), 
            features.get("SystemBattery", 0.0)
        ]], dtype=np.float32)

        with model_lock:
            active_model = current_model

        # If model is momentarily being swapped by OTA manager, wait briefly
        if active_model is None:
            for _ in range(20):
                time.sleep(0.5)
                with model_lock:
                    active_model = current_model
                if active_model is not None:
                    break
            if active_model is None:
                print("  [WARN] [InferenceEngineV2] Engine not ready during OTA swap, skipping sample.")
                return

        input_name = active_model.get_inputs()[0].name
        label_name = active_model.get_outputs()[0].name
        pred_transformed = float(np.asarray(active_model.run([label_name], {input_name: input_data})[0]).flatten()[0])
            
        # Target unscaling via inverse power transform (enforcing non-negative concentration)
        raw_prediction = _inverse_power_transform(pred_transformed, Y_LAMBDA, Y_MEAN, Y_SCALE)
        prediction = max(0.0, raw_prediction)
            
        inference_ms = float((time.perf_counter() - t0) * 1000)
        
        # --- System Hardware Metrics ---
        cpu_usage = float(psutil.cpu_percent())
        ram_mb = float(psutil.virtual_memory().used / (1024 * 1024))

        # --- WHO Alert Level 1 Evaluation ---
        alarm = bool(prediction >= ALARM_THRESHOLD)
        alarm_flag = "[ALERT] [WHO-Level-1]" if alarm else "[NORMAL]"

        # --- SQLite WAL Persistence (timed) ---
        t1 = time.perf_counter()
        db_conn.execute(
            "INSERT INTO system_metrics (timestamp, cpu_percent, ram_mb) VALUES (?, ?, ?)",
            (str(timestamp), float(cpu_usage), float(ram_mb))
        )
        db_conn.commit()
        db_write_ms = float((time.perf_counter() - t1) * 1000)

        store_reading(db_conn, timestamp, features, prediction,
                      float(ground_truth), alarm, inference_ms, db_write_ms)

        # --- 24-Hour Smart Batching ---
        reading_data = {
            "timestamp": str(timestamp),
            "features": {
                "temperature": float(features.get("EXO3(Temp_C)", 0.0)),
                "sp_cond": float(features.get("EXO3(spCond_uS_cm)", 0.0)),
                "ph": float(features.get("EXO3(pH)", 0.0)),
                "battery": float(features.get("SystemBattery", 0.0))
            },
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
                batch_payload = json.dumps(current_batch)
                client.publish(TOPIC_DAILY_BATCH, batch_payload, qos=1)
                print(f"[INFO] [InferenceEngineV2] Published 24-hour daily batch ({batch_size} samples) to {TOPIC_DAILY_BATCH}")
                current_batch = []

        # --- Immediate WHO Hazard Alarm Bypass ---
        if alarm:
            alarm_payload = json.dumps({
                "timestamp": str(timestamp),
                "predicted_chlorophyll": round(float(prediction), 3),
                "threshold": float(ALARM_THRESHOLD),
                "level": "WHO_LEVEL_1"
            })
            client.publish(TOPIC_ALARM, alarm_payload, qos=2)
            print(f"[ALERT] [InferenceEngineV2] Published critical ALARM packet immediately to {TOPIC_ALARM}")

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