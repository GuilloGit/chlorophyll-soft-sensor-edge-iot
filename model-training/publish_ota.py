"""Publish an OTA update command to the buoy via MQTT.

Run this on your PC after retraining to trigger a remote model update.
The buoy's inference engine will download the model from the provided URL,
verify its SHA-256 hash, trial-load it, and hot-swap it into memory.

Usage:
    python publish_ota.py \\
        --url https://github.com/<owner>/<repo>/releases/download/v1.1/model.joblib \\
        --version 1.1.0

Prerequisites:
    pip install paho-mqtt
"""

import json
import hashlib
import argparse
import paho.mqtt.client as mqtt

TOPIC_OTA_COMMAND = "buoy/ota/update"


def main():
    parser = argparse.ArgumentParser(
        description="Publish an OTA update command to the buoy."
    )
    parser.add_argument(
        "--model", default="../edge-system/app/model.joblib",
        help="Path to the local model file (used only for SHA-256 computation)"
    )
    parser.add_argument(
        "--url", required=True,
        help="GitHub Release URL, e.g. "
             "https://github.com/<owner>/<repo>/releases/download/v1.1/model.joblib"
    )
    parser.add_argument(
        "--version", default="1.0.0",
        help="Semantic version tag for this model update"
    )
    parser.add_argument(
        "--broker", default="localhost",
        help="MQTT broker address (default: localhost for LAN testing)"
    )
    parser.add_argument(
        "--port", type=int, default=1883,
        help="MQTT broker port (default: 1883)"
    )
    args = parser.parse_args()

    # Compute SHA-256 of the local model file
    print(f"Computing SHA-256 of {args.model}...")
    with open(args.model, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    # Build the OTA command payload
    payload = json.dumps({
        "url": args.url,
        "sha256": sha,
        "version": args.version
    })

    # Publish to the MQTT broker
    print(f"Connecting to {args.broker}:{args.port}...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port, 60)
    client.publish(TOPIC_OTA_COMMAND, payload, qos=1)
    client.disconnect()

    print()
    print("=" * 55)
    print("  OTA command published successfully!")
    print("=" * 55)
    print(f"  Broker:   {args.broker}:{args.port}")
    print(f"  Topic:    {TOPIC_OTA_COMMAND}")
    print(f"  Version:  {args.version}")
    print(f"  URL:      {args.url}")
    print(f"  SHA-256:  {sha}")
    print()
    print("The buoy will download, verify, and hot-swap the model.")
    print("Monitor the buoy logs for [OTA] status messages.")


if __name__ == "__main__":
    main()
