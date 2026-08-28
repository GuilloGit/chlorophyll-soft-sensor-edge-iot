"""
===============================================================================
Module Name:       model_test.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Machine Learning Quality Assurance (QA)

Description:       Validates the exported edge soft-sensor model pipeline against
                   the holdout simulation dataset. Simulates edge input formatting
                   and confirms end-to-end inference execution without errors.

Data Interfaces:
  - Upstream:      edge-system/app/model.joblib
                   edge-system/sensor-sim/data/simulation_test_data.csv
  - Downstream:    Console validation log
  - Storage / IPC: Local filesystem read.

References:        Mozo et al. (2022).
===============================================================================
"""

import os
import joblib
import pandas as pd

# 1. Point to the exported artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../edge-system/app/model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "../edge-system/sensor-sim/data/simulation_test_data.csv")

FEATURES = ["EXO3(Temp_C)", "EXO3(spCond_uS_cm)", "EXO3(pH)", "SystemBattery"]
TARGET = "EXO3(Chlorophyll_ug_L)"

print("[INFO] [ModelTest] Loading Edge Model and Simulation Data...")
try:
    edge_model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    print("[INFO] [ModelTest] Model and test dataset loaded successfully.\n")
except Exception as e:
    print(f"[ERROR] [ModelTest] Error loading files: {e}")
    exit(1)

print("[INFO] [ModelTest] Simulating Edge Inference (Sample Evaluation)...")
print("-" * 65)

# Grab the first 5 rows of the test dataset
sample_data = df.head(5)

for sample_number, (_, row) in enumerate(sample_data.iterrows(), start=1):
    # A. Format the data exactly as the edge application will format the MQTT payload
    payload = {feature: row[feature] for feature in FEATURES}
    features_df = pd.DataFrame([payload])
    
    # B. Run the prediction (Testing the full Pipeline)
    prediction = edge_model.predict(features_df)[0]
    
    # C. Compare against the real recorded value
    actual = row[TARGET]
    error = abs(prediction - actual)
    
    print(f"Sample {sample_number}:")
    print(f"  Inputs:     Temp={row['EXO3(Temp_C)']}, EC={row['EXO3(spCond_uS_cm)']}, pH={row['EXO3(pH)']}, Bat={row['SystemBattery']}")
    print(f"  Predicted:  {prediction:.3f} µg/L")
    print(f"  Actual:     {actual:.3f} µg/L")
    print(f"  Error:      {error:.3f} µg/L\n")

print("-" * 65)
print("[INFO] [ModelTest] QA Validation Complete. Model pipeline executed successfully.")