import os
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.metrics import mean_absolute_error, r2_score

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")  # Assuming running on laptop pointing to local Docker port
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_DAILY_BATCH = "sensor/water/daily_batch"
TOPIC_ALARM = "sensor/water/alarm"

REPORT_PATH = "edge_performance_report.json"

# State
all_actual = []
all_predicted = []
all_inference_ms = []
all_cpu = []
all_ram = []
total_alarms = 0

def generate_report():
    if len(all_actual) == 0:
        print("No data received yet.")
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
    print(f"\n[REPORT] Generated updated performance report at {REPORT_PATH}")
    print(json.dumps(report, indent=2))


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to Cloud MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(TOPIC_DAILY_BATCH)
        client.subscribe(TOPIC_ALARM)
        print(f"Listening for daily batches on {TOPIC_DAILY_BATCH}...")
    else:
        print(f"Connection failed: {reason_code}")

def on_message(client, userdata, msg):
    global total_alarms

    if msg.topic == TOPIC_ALARM:
        print(f"\n[CLOUD] 🚨 ALARM RECEIVED: {msg.payload.decode('utf-8')}")
        total_alarms += 1
        
    elif msg.topic == TOPIC_DAILY_BATCH:
        try:
            batch = json.loads(msg.payload.decode('utf-8'))
            print(f"\n[CLOUD] 📥 Received Daily Batch containing {len(batch)} readings.")
            
            for reading in batch:
                all_actual.append(reading["actual_chlorophyll"])
                all_predicted.append(reading["predicted_chlorophyll"])
                all_inference_ms.append(reading["inference_ms"])
                all_cpu.append(reading["cpu_percent"])
                all_ram.append(reading["ram_mb"])
                
            generate_report()
        except Exception as e:
            print(f"Error processing batch: {e}")

if __name__ == "__main__":
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
            print(f"Waiting for broker at {MQTT_BROKER}:{MQTT_PORT}... ({e})")
            time.sleep(3)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down Cloud Analytics...")
        generate_report()
