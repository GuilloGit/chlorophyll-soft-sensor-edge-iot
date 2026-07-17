"""Sensor Simulator — Reads CSV data and publishes to the local MQTT broker.

This service replaces the external telemetry-sim by running on the Pi itself.
It simulates physical sensors (I²C/ADC) by reading rows from a pre-recorded
CSV dataset at a configurable, time-scaled interval.

Features:
    - Checkpoint-based reboot resume (atomic write to /data/checkpoint.json)
    - Configurable time scaling (real 10-min interval compressed for testing)
    - Graceful shutdown on SIGTERM (Docker stop)
    - Retry loop for MQTT broker connection
"""

import os
import sys
import json
import time
import signal
import tempfile
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
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    print(f"\n[{time.strftime('%X')}] Received signal {signum}. Shutting down gracefully...")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# --- CHECKPOINT MANAGEMENT ---
def load_checkpoint() -> int:
    """Load the last processed row index from the checkpoint file.
    
    Returns 0 if no checkpoint exists (fresh start).
    """
    try:
        with open(CHECKPOINT_PATH, "r") as f:
            data = json.load(f)
            index = data.get("last_row_index", -1) + 1
            print(f"  -> Checkpoint found. Resuming from row {index}.")
            return index
    except (FileNotFoundError, json.JSONDecodeError):
        print("  -> No checkpoint found. Starting from row 0.")
        return 0


def save_checkpoint(index: int):
    """Atomically save the current row index to the checkpoint file.
    
    Uses write-to-temp + rename for crash safety. If the Pi loses power
    between these two operations, the old checkpoint survives intact.
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
        print(f"  [WARN] Failed to save checkpoint: {e}")


# --- MQTT SETUP ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[{time.strftime('%X')}] Connected to broker at {MQTT_BROKER}:{MQTT_PORT}")
    else:
        print(f"[{time.strftime('%X')}] Connection failed: reason_code={reason_code}")


def connect_mqtt() -> mqtt.Client:
    """Connect to the local MQTT broker with a retry loop.
    
    Docker Compose may start this container before Mosquitto is ready,
    so we retry until the broker accepts connections.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect

    while not _shutdown_requested:
        try:
            print(f"Connecting to broker at {MQTT_BROKER}:{MQTT_PORT}...")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            client.loop_start()
            return client
        except Exception as e:
            print(f"  Broker not ready ({e}). Retrying in 3s...")
            time.sleep(3)

    sys.exit(0)


# --- MAIN ---
def main():
    print("=" * 65)
    print("  SENSOR SIMULATOR — On-Pi Sensor Reader")
    print("=" * 65)
    print(f"  Real interval:      {REAL_INTERVAL_SEC}s ({REAL_INTERVAL_SEC/60:.0f} min)")
    print(f"  Time scale factor:  {TIME_SCALE_FACTOR}x")
    print(f"  Effective interval: {EFFECTIVE_INTERVAL:.2f}s per row")
    print(f"  Dataset:            {DATASET_PATH}")
    print(f"  MQTT topic:         {TOPIC_RAW}")
    print()

    # 1. Load dataset
    print("1. Loading dataset...")
    try:
        df = pd.read_csv(DATASET_PATH).dropna()
        print(f"  -> {len(df)} valid records loaded.")
    except Exception as e:
        print(f"  -> CRITICAL: Failed to load CSV: {e}")
        sys.exit(1)

    # 2. Load checkpoint
    print("2. Checking for reboot checkpoint...")
    start_index = load_checkpoint()

    if start_index >= len(df):
        print(f"\n  All {len(df)} rows already processed. Simulation complete.")
        print("  Delete /data/checkpoint.json to restart from the beginning.")
        return

    remaining = len(df) - start_index
    print(f"  -> {remaining} rows remaining to process.")

    # 3. Connect to broker
    print("3. Connecting to MQTT broker...")
    client = connect_mqtt()

    # 4. Publishing loop
    print(f"\n{'=' * 65}")
    print(f"  Starting sensor simulation (row {start_index} of {len(df)})...")
    print(f"  Publishing 1 row every {EFFECTIVE_INTERVAL:.2f}s. Ctrl+C to stop.")
    print(f"{'=' * 65}\n")

    try:
        for iloc_idx in range(start_index, len(df)):
            if _shutdown_requested:
                break

            row = df.iloc[iloc_idx]

            # Build payload matching the original schema
            payload_dict = {
                "timestamp": str(row.get("date", f"row-{iloc_idx}")),
                "features": {feat: float(row[feat]) for feat in FEATURES},
                "ground_truth": float(row[TARGET])
            }

            payload_json = json.dumps(payload_dict)

            # Publish with QoS 1 (at-least-once delivery)
            result = client.publish(TOPIC_RAW, payload_json, qos=1)
            result.wait_for_publish()

            print(f"[Row {iloc_idx:>5}/{len(df)}] {payload_dict['timestamp']}  "
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
        print("\n  Stopped manually by user.")
    finally:
        print(f"  Saving final checkpoint at row {iloc_idx}...")
        save_checkpoint(iloc_idx)
        client.loop_stop()
        client.disconnect()
        print("  Disconnected from broker. Goodbye.")


if __name__ == "__main__":
    main()
