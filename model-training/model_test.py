"""This test script validates the exported edge model against sample simulation data.

It loads the trained pipeline and doeas a quick check on  a few rows from the
simulation dataset to confirm inference works end to end.
"""

import joblib
import pandas as pd

# 1. Point to the exported artifacts
MODEL_PATH = "../edge-system/app/model.joblib"
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../edge-system/sensor-sim/data/simulation_test_data.csv")

FEATURES = ["EXO3(Temp_C)", "EXO3(spCond_uS_cm)", "EXO3(pH)", "SystemBattery"]
TARGET = "EXO3(Chlorophyll_ug_L)"

print("Loading Edge Model and Simulation Data...")
try:
    edge_model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Files loaded successfully!\n")
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

print("Simulating Edge Inference (First 5 Rows)...")
print("-" * 60)

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

print("-" * 60)
print("QA Complete. If the script ran without crashing and predictions are reasonably close to actuals, the artifacts are valid and ready for deployment.")