"""
===============================================================================
Module Name:       publish_ota_v2.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Model Management / Remote OTA Dispatcher (V2 ONNX)

Description:       CLI tool for dispatching remote Over-The-Air (OTA) model update
                   commands to edge buoys running the V2 ONNX architecture over
                   MQTT. Computes the cryptographic SHA-256 checksum of the target
                   ONNX model artifact, packages the payload with download URL
                   and semantic version tag, and publishes with QoS 1 to
                   buoy/ota/update.

Data Interfaces:
  - Upstream:      Local ONNX model file (edge-system/app-v2/model_v2.onnx)
  - Downstream:    MQTT topic buoy/ota/update (QoS 1)
  - Storage / IPC: Local file read; MQTT broker connection.

References:        Test-Before-Swap OTA fail-safe architecture.
===============================================================================
"""

import os
import json
import hashlib
import argparse
import paho.mqtt.client as mqtt

TOPIC_OTA_COMMAND = "buoy/ota/update"


def main():
    """Parses arguments, calculates SHA-256 hash, and dispatches OTA update payload."""
    parser = argparse.ArgumentParser(
        description="Publish a remote OTA update command to the edge buoy (V2 ONNX)."
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to the local model or archive file (defaults to model_v2.onnx.xz if present, else model_v2.onnx)"
    )
    parser.add_argument(
        "--url", required=True,
        help="Model artifact download URL (e.g., GitHub Release asset link)"
    )
    parser.add_argument(
        "--version", default="1.0.0",
        help="Semantic version tag for this model release"
    )
    parser.add_argument(
        "--broker", default="localhost",
        help="MQTT broker hostname or IP (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=1883,
        help="MQTT broker port (default: 1883)"
    )
    args = parser.parse_args()

    model_target = args.model
    if not model_target:
        xz_candidate = "../edge-system/app-v2/model_v2.onnx.xz"
        onnx_candidate = "../edge-system/app-v2/model_v2.onnx"
        model_target = xz_candidate if os.path.exists(xz_candidate) else onnx_candidate

    # Compute SHA-256 of the local target file in streaming 64 KB chunks
    print(f"[INFO] [OTAPublisher] Computing SHA-256 of target artifact: {model_target}...")
    h = hashlib.sha256()
    with open(model_target, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    sha = h.hexdigest()

    # Build the OTA command payload
    payload = json.dumps({
        "url": args.url,
        "sha256": sha,
        "version": args.version
    })

    # Publish to the MQTT broker
    print(f"[INFO] [OTAPublisher] Connecting to {args.broker}:{args.port}...")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.broker, args.port, 60)
    client.publish(TOPIC_OTA_COMMAND, payload, qos=1)
    client.disconnect()

    print()
    print("=" * 60)
    print("  [INFO] [OTAPublisher] OTA Command Dispatched Successfully")
    print("=" * 60)
    print(f"  Broker:   {args.broker}:{args.port}")
    print(f"  Topic:    {TOPIC_OTA_COMMAND}")
    print(f"  Version:  {args.version}")
    print(f"  URL:      {args.url}")
    print(f"  SHA-256:  {sha}")
    print("=" * 60)
    print("[INFO] [OTAPublisher] Edge device will download, verify, and hot-swap the model.")


if __name__ == "__main__":
    main()
