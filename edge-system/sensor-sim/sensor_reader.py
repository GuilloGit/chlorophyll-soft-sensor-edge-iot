"""
===============================================================================
Module Name:       sensor_reader.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Edge Telemetry Simulator (Software-in-the-Loop)

Description:       Simulates physical multi-parameter water quality probes
                   (e.g., EXO3 sonde) by reading chronological observations
                   from a reservoir dataset at a time-scaled interval
                   (effective interval = nominal interval / scale factor).
                   Publishes telemetry readings over local MQTT with QoS 1 and
                   maintains crash-safe atomic checkpoints on disk.

Data Interfaces:
  - Upstream:      Historical test CSV dataset (/app/data/simulation_test_data.csv)
  - Downstream:    MQTT topic sensor/water/raw (QoS 1)
  - Storage / IPC: Atomic JSON checkpoint file (/data/checkpoint.json)

References:        Mozo et al. (2022).
===============================================================================
"""

import os
import sys
import json
import time
import signal
import tempfile
import datetime
import pandas as pd
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt_broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_RAW = "sensor/water/raw"

DATASET_PATH = os.getenv("DATASET_PATH", "/app/data/simulation_test_data.csv")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/data/checkpoint.json")

# Time scaling: real sensors read every REAL_INTERVAL_SEC seconds.
# TIME_SCALE_FACTOR compresses that for testing (900x → 1 row/sec).
REAL_INTERVAL_SEC = int(os.getenv("REAL_INTERVAL_SEC", 900))
TIME_SCALE_FACTOR = int(os.getenv("TIME_SCALE_FACTOR", 900))
EFFECTIVE_INTERVAL = REAL_INTERVAL_SEC / TIME_SCALE_FACTOR

FEATURES = [
    "EXO3(Temp_C)",
    "EXO3(spCond_uS_cm)",
    "EXO3(pH)",
    "SystemBattery"
]
TARGET = "EXO3(Chlorophyll_ug_L)"

# --- GLOBAL STATE ---
_shutdown_requested = False


def handle_signal(signum, frame):
    """Handles SIGTERM and SIGINT OS signals for graceful process termination."""
    global _shutdown_requested
    print(f"\n[INFO] [SensorSim] [{time.strftime('%X')}] Received signal {signum}. Shutting down gracefully...")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# --- CHECKPOINT MANAGEMENT ---
def load_checkpoint() -> int:
    """Loads the last processed row index from the persistent checkpoint file.
    
    Returns 0 if no checkpoint exists (fresh simulation run).
    """
    try:
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
            index = data.get("last_row_index", -1) + 1
            print(f"[INFO] [SensorSim] Checkpoint found. Resuming from row {index}.")
            return index
    except (FileNotFoundError, json.JSONDecodeError):
        print("[INFO] [SensorSim] No checkpoint found. Starting from row 0.")
        return 0


def save_checkpoint(index: int):
    """Atomically persists the current row index to the checkpoint file.
    
    Employs write-to-temporary-file and atomic replace (os.replace) to
    prevent file corruption during ungraceful shutdowns or power cuts.
    """
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({"last_row_index": index}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, CHECKPOINT_PATH)
    except Exception as e:
        print(f"[WARN] [SensorSim] Failed to save checkpoint: {e}")


# --- MQTT SETUP ---
def on_connect(client, userdata, flags, reason_code, properties):
    """Callback triggered upon establishing connection to MQTT broker."""
    if reason_code == 0:
        print(f"[INFO] [SensorSim] [{time.strftime('%X')}] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[ERROR] [SensorSim] [{time.strftime('%X')}] Connection failed: reason_code={reason_code}")


def connect_mqtt() -> mqtt.Client:
    """Connects to the local MQTT broker with a fault-tolerant retry loop."""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect

    while not _shutdown_requested:
        try:
            print(f"[INFO] [SensorSim] Connecting to broker at {MQTT_BROKER}:{MQTT_PORT}...")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_start()
            return client
        except Exception as e:
            print(f"[INFO] [SensorSim] Broker not ready ({e}). Retrying in 3s...")
            time.sleep(3)

    sys.exit(0)


# --- MAIN ---
def main():
    """Initializes and runs the telemetry simulation streaming loop."""
    print("=" * 65)
    print("  SENSOR SIMULATOR — Software-in-the-Loop Telemetry Stream")
    print("=" * 65)
    print(f"  Real interval:      {REAL_INTERVAL_SEC}s ({REAL_INTERVAL_SEC/60:.0f} min)")
    print(f"  Time scale factor:  {TIME_SCALE_FACTOR}x")
    print(f"  Effective interval: {EFFECTIVE_INTERVAL:.2f}s per row")
    print(f"  Dataset:            {DATASET_PATH}")
    print(f"  MQTT topic:         {TOPIC_RAW}")
    print()

    # 1. Load dataset
    print("[INFO] [SensorSim] 1. Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH).dropna()
        print(f"[INFO] [SensorSim] {len(df)} valid records loaded.")
    except Exception as e:
        print(f"[ERROR] [SensorSim] Failed to load CSV: {e}")
        sys.exit(1)

    # 2. Load checkpoint
    print("[INFO] [SensorSim] 2. Checking for reboot checkpoint...")
    start_index = load_checkpoint()

    if start_index >= len(df):
        print(f"\n[INFO] [SensorSim] All {len(df)} rows already processed. Simulation complete.")
        print("[INFO] [SensorSim] Remove /data/checkpoint.json to restart simulation.")
        return

    remaining = len(df) - start_index
    print(f"[INFO] [SensorSim] {remaining} rows remaining to process.")

    # 3. Connect to broker
    print("[INFO] [SensorSim] 3. Connecting to MQTT broker...")
    client = connect_mqtt()

    # Calculate hybrid "Back-in-Time" start (simplified 24h lookback)
    now = datetime.datetime.now(datetime.timezone.utc)
    virtual_time = now - datetime.timedelta(hours=24)
    print(f"[INFO] [SensorSim] Virtual start time: {virtual_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # 4. Publishing loop
    print(f"\n{'=' * 65}")
    print(f"  Starting sensor simulation (row {start_index} of {len(df)})...")
    print(f"  Publishing 1 row every {EFFECTIVE_INTERVAL:.2f}s. Press Ctrl+C to terminate.")
    print(f"{'=' * 65}\n")

    try:
        for iloc_idx in range(start_index, len(df)):
            if _shutdown_requested:
                break

            row = df.iloc[iloc_idx]

            # Build payload matching the original schema
            payload_dict = {
                "timestamp": virtual_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "features": {feat: float(row[feat]) for feat in FEATURES},
                "ground_truth": float(row[TARGET])
            }
            
            virtual_time += datetime.timedelta(seconds=REAL_INTERVAL_SEC)

            payload_json = json.dumps(payload_dict)

            # Publish with QoS 1 (at-least-once delivery)
            result = client.publish(TOPIC_RAW, payload_json, qos=1)
            result.wait_for_publish()

            print(f"[INFO] [SensorSim] [Row {iloc_idx:>5}/{len(df)}] {payload_dict['timestamp']}  "
                  f"Temp={payload_dict['features']['EXO3(Temp_C)']:.1f}  "
                  f"Chl_actual={payload_dict['ground_truth']:.2f}")

            # Save checkpoint after successful publish
            save_checkpoint(iloc_idx)

            # Sleep for the scaled interval
            time.sleep(EFFECTIVE_INTERVAL)

        if not _shutdown_requested:
            print(f"\n{'=' * 65}")
            print("  Simulation complete. All records published.")
            print(f"{'=' * 65}")

    except KeyboardInterrupt:
        print("\n[INFO] [SensorSim] Simulation stopped manually by user.")
    finally:
        print(f"[INFO] [SensorSim] Saving final checkpoint at row {iloc_idx}...")
        save_checkpoint(iloc_idx)
        client.loop_stop()
        client.disconnect()
        print("[INFO] [SensorSim] Disconnected from MQTT broker.")


if __name__ == "__main__":
    main()
