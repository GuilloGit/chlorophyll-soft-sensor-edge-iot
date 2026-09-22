"""
===============================================================================
Module Name:       analytics_subscriber.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Cloud Analytics Tier

Description:       Subscribes to telemetry topics published by edge nodes.
                   Consumes daily telemetry batches (aggregated 24-hour windows)
                   and real-time WHO Alert Level 1 alarm messages. Computes
                   rolling soft-sensor fidelity metrics (MAE, R²) and edge
                   hardware resource statistics (inference latency, CPU, RAM),
                   persisting summary state to edge_performance_report.json.

Data Interfaces:
  - Upstream:      MQTT topics:
                     - sensor/water/daily_batch (QoS 1, daily telemetry array)
                     - sensor/water/alarm (QoS 2, critical threshold alerts)
  - Downstream:    edge_performance_report.json (local JSON analytics digest)
  - Storage / IPC: In-memory statistical accumulators; atomic JSON write.

References:        Mozo et al. (2022); WHO Guidelines for Safe Recreational
                   Water Environments (2003).
===============================================================================
"""

import os
import sys
import csv
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.metrics import mean_absolute_error, r2_score

# Ensure immediate unbuffered output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")  # Assuming running on laptop pointing to local Docker port
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_DAILY_BATCH = "sensor/water/daily_batch"
TOPIC_ALARM = "sensor/water/alarm"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.getenv("REPORT_PATH", os.path.join(SCRIPT_DIR, "edge_performance_report.json"))
CSV_PATH = os.getenv("CSV_PATH", os.path.join(SCRIPT_DIR, "telemetry_measurements.csv"))

# State
all_actual = []
all_predicted = []
all_inference_ms = []
all_cpu = []
all_ram = []
total_alarms = 0


def generate_report():
    """Calculates cumulative performance metrics and writes the JSON report."""
    if len(all_actual) == 0:
        print("[INFO] [CloudSubscriber] No data received yet.")
        return

    mae = mean_absolute_error(all_actual, all_predicted)
    
    # R2 can be undefined if variance is 0, so handle it safely
    try:
        r2 = r2_score(all_actual, all_predicted)
    except:
        r2 = 0.0

    report = {
        "edge_fidelity": {
            "mae": round(mae, 3),
            "r2": round(r2, 3),
            "total_samples": len(all_actual)
        },
        "hardware_performance": {
            "inference_ms": {
                "avg": round(float(np.mean(all_inference_ms)), 2),
                "p95": round(float(np.percentile(all_inference_ms, 95)), 2)
            },
            "cpu_percent": {
                "avg": round(float(np.mean(all_cpu)), 1),
                "peak": round(float(np.max(all_cpu)), 1)
            },
            "ram_mb": {
                "avg": round(float(np.mean(all_ram)), 1),
                "peak": round(float(np.max(all_ram)), 1)
            }
        },
        "system_events": {
            "total_alarms_triggered": total_alarms
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[INFO] [CloudSubscriber] Generated updated performance report at {REPORT_PATH}")
    print(json.dumps(report, indent=2))


def on_connect(client, userdata, flags, reason_code, properties):
    """Handles MQTT broker connection and subscribes to telemetry and alarm topics."""
    if reason_code == 0:
        print(f"[INFO] [CloudSubscriber] Connected to Cloud MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_DAILY_BATCH)
        client.subscribe(TOPIC_ALARM)
        print(f"[INFO] [CloudSubscriber] Subscribed to {TOPIC_DAILY_BATCH} and {TOPIC_ALARM}")
    else:
        print(f"[ERROR] [CloudSubscriber] Connection failed: {reason_code}")


def on_message(client, userdata, msg):
    """Processes incoming MQTT messages from edge nodes."""
    global total_alarms

    if msg.topic == TOPIC_ALARM:
        print(f"\n[ALERT] [CloudSubscriber] Alarm received: {msg.payload.decode('utf-8')}")
        total_alarms += 1
        
    elif msg.topic == TOPIC_DAILY_BATCH:
        try:
            batch = json.loads(msg.payload.decode('utf-8'))
            print(f"\n[INFO] [CloudSubscriber] Received daily batch containing {len(batch)} readings.")
            
            # Persist raw measurements to CSV
            file_exists = os.path.exists(CSV_PATH)
            fieldnames = [
                "timestamp", "temperature", "sp_cond", "ph", "battery",
                "predicted_chlorophyll", "actual_chlorophyll", "alarm",
                "inference_ms", "db_write_ms", "cpu_percent", "ram_mb"
            ]
            
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                
                for reading in batch:
                    all_actual.append(reading.get("actual_chlorophyll", 0.0))
                    all_predicted.append(reading.get("predicted_chlorophyll", 0.0))
                    all_inference_ms.append(reading.get("inference_ms", 0.0))
                    all_cpu.append(reading.get("cpu_percent", 0.0))
                    all_ram.append(reading.get("ram_mb", 0.0))
                    
                    feat = reading.get("features", {})
                    row = {
                        "timestamp": reading.get("timestamp", ""),
                        "temperature": feat.get("temperature", reading.get("temp_c", "")),
                        "sp_cond": feat.get("sp_cond", reading.get("spcond_us_cm", "")),
                        "ph": feat.get("ph", reading.get("ph", "")),
                        "battery": feat.get("battery", reading.get("battery", "")),
                        "predicted_chlorophyll": reading.get("predicted_chlorophyll", ""),
                        "actual_chlorophyll": reading.get("actual_chlorophyll", ""),
                        "alarm": reading.get("alarm", False),
                        "inference_ms": reading.get("inference_ms", ""),
                        "db_write_ms": reading.get("db_write_ms", ""),
                        "cpu_percent": reading.get("cpu_percent", ""),
                        "ram_mb": reading.get("ram_mb", ""),
                    }
                    writer.writerow(row)
            
            print(f"[INFO] [CloudSubscriber] Appended {len(batch)} readings to {CSV_PATH}")
            generate_report()
            if exit_on_batch:
                print("\n[INFO] [CloudSubscriber] Daily batch processed. Exiting as requested by --exit-on-batch.")
                client.disconnect()
        except Exception as e:
            print(f"[ERROR] [CloudSubscriber] Error processing batch: {e}")


exit_on_batch = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cloud Analytics Subscriber for Edge Buoy Telemetry")
    parser.add_argument("--broker", type=str, default=os.getenv("MQTT_BROKER", "localhost"), help="MQTT broker address (default: localhost or MQTT_BROKER env)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", 1883)), help="MQTT broker port (default: 1883)")
    parser.add_argument("--exit-on-batch", action="store_true", help="Exit automatically after processing the first daily batch")
    args = parser.parse_args()
    exit_on_batch = args.exit_on_batch
    MQTT_BROKER = args.broker
    MQTT_PORT = args.port

    print("========================================")
    print("  CLOUD ANALYTICS & PERFORMANCE MONITOR")
    print("========================================")
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Try connect loop
    while True:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"[INFO] [CloudSubscriber] Waiting for broker at {MQTT_BROKER}:{MQTT_PORT}... ({e})")
            time.sleep(3)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[INFO] [CloudSubscriber] Shutting down Cloud Analytics...")
        generate_report()

