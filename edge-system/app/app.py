import json
import time
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
# Note: Because both containers are on the same Docker Compose network,
# the app can connect to the broker using its container name!
BROKER_HOST = "mqtt_broker" 
BROKER_PORT = 1883
TOPIC = "sensor/water/data"
MODEL_PATH = "model.joblib"

print("1. Initializing Edge Inference Engine...")
try:
    model = joblib.load(MODEL_PATH)
    print("  -> Machine Learning Pipeline loaded successfully.")
except Exception as e:
    print(f"  -> CRITICAL ERROR loading model: {e}")
    exit(1)

# --- MQTT CALLBACKS ---
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"\nConnected to Mosquitto Broker! Subscribing to '{TOPIC}'...\n")
        client.subscribe(TOPIC)
    else:
        print(f"Connection failed with code {reason_code}")

def on_message(client, userdata, msg):
    try:
        # Decode the JSON payload from the Simulator
        payload = json.loads(msg.payload.decode('utf-8'))
        features = payload.get("features", {})
        ground_truth = payload.get("ground_truth", 0.0)
        timestamp = payload.get("timestamp", "Unknown Time")

        # Convert dictionary directly into a Pandas DataFrame for the model
        df = pd.DataFrame([features])

        # Run the inference (Pipeline handles all the PowerTransformer math!)
        prediction = model.predict(df)[0]
        
        # WHO Alert Level 1 Logic (> 10.0 µg/L)
        alarm_flag = "🚨 ALARM LEVEL 1 TRIGGERED!" if prediction >= 10.0 else "✅ Normal"

        # Log the results to the terminal
        print(f"[{timestamp}]")
        print(f"  Inputs:    Temp={features.get('EXO3(Temp_C)')}, EC={features.get('EXO3(spCond_uS_cm)')}, pH={features.get('EXO3(pH)')}, Bat={features.get('SystemBattery')}")
        print(f"  Predicted: {prediction:.3f} µg/L {alarm_flag}")
        print(f"  Actual:    {ground_truth:.3f} µg/L")
        print("-" * 65)

    except Exception as e:
        print(f"Error processing incoming data: {e}")

# --- NETWORK STARTUP ---
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# Docker Compose race condition protection:
# Keep trying to connect until the Mosquitto container is fully booted
while True:
    try:
        print(f"Connecting to broker at {BROKER_HOST}...")
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        break
    except Exception as e:
        print("Broker not ready yet. Retrying in 3 seconds...")
        time.sleep(3)

# Block the main thread and listen forever
client.loop_forever()