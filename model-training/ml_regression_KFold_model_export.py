"""Export the chlorophyll soft-sensor training pipeline.

Run this to prepare the final edge-ready model and export the test dataset
used by the telemetry simulator.

Based on this repository:
https://github.com/stanislavvakaruk/Chlorophyll_soft-sensor_machine_learning_models

Reference report:
A. Mozo, J. Morón-López, S. Vakaruk, A. G. Pompa-Pernía,
A. González-Prieto, J. A. Pascual Aguilar, S. Gómez-Canaval,
J. M. Ortiz (2022). Chlorophyll soft-sensor based on machine learning models
for algal bloom predictions. Scientific Reports.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# --- CONFIGURATION ---
TRAIN_FILE = "data/Playa_UPM_resampled_24H_1H.csv"
TEST_FILE = "data/Presa_UPM_resampled_24H_1H.csv"
MODELS_RANDOM_STATE = 22

FEATURES = ["EXO3(Temp_C)", "EXO3(spCond_uS_cm)", "EXO3(pH)", "SystemBattery"]
TARGET = "EXO3(Chlorophyll_ug_L)"

print("1. Loading Datasets...")
Training_Dataset = pd.read_csv(TRAIN_FILE)
Testing_Dataset = pd.read_csv(TEST_FILE)

# Reproducing the original preprocessing
Training_Dataset["date"] = Training_Dataset["Unnamed: 0"]
Testing_Dataset["date"] = Testing_Dataset["Unnamed: 0"]
Training_Dataset = Training_Dataset.T[1:].T.dropna()
Testing_Dataset = Testing_Dataset.T[1:].T.dropna()

print("2. Generating K-Fold Split (Extracting Fold 0)...")
kf = KFold(n_splits=10, shuffle=False)

# Get the indices for the very first fold
boya_0_splits = list(kf.split(Training_Dataset))[0]
boya_1_splits = list(kf.split(Testing_Dataset))[0]

# Combine Beach and Dam data for Training (90%)
df_train = pd.concat([
    Training_Dataset.iloc[boya_0_splits[0]], 
    Testing_Dataset.iloc[boya_1_splits[0]]
], ignore_index=True)

# Combine Beach and Dam data for Testing (10%)
df_test = pd.concat([
    Training_Dataset.iloc[boya_0_splits[1]], 
    Testing_Dataset.iloc[boya_1_splits[1]]
], ignore_index=True)

X_train = df_train[FEATURES]
y_train = df_train[TARGET]

print("3. Building the Edge Pipeline...")
# This completely encapsulates the Random Forest AND both scalers
# The Pi will only need to call model.predict(X) and this handles all the math
rf_model = RandomForestRegressor(
    n_estimators=100, 
    n_jobs=-1, 
    random_state=MODELS_RANDOM_STATE
)

# Wrap the model so it scales the Y (Chlorophyll) automatically
target_regressor = TransformedTargetRegressor(
    regressor=rf_model,
    transformer=PowerTransformer()
)

# Create the final pipeline that scales the X (Features) automatically
edge_pipeline = Pipeline([
    ('x_scaler', PowerTransformer()),
    ('rf_model_with_y_scaler', target_regressor)
])

print("4. Training the Final Edge Model...")
edge_pipeline.fit(X_train, y_train)

print("5. Exporting Assets...")

edge_model_path = "../edge-system/app/model.joblib"
os.makedirs("../edge-system/app", exist_ok=True)
joblib.dump(edge_pipeline, edge_model_path)
print(f"  -> Deployed Model to: {edge_model_path}")

# Publish the test data DIRECTLY into the telemetry simulator's data folder
simulator_csv_path = "../telemetry-sim/data/simulation_test_data.csv"
os.makedirs("../telemetry-sim/data", exist_ok=True)
df_test.to_csv(simulator_csv_path, index=False)
print(f"  -> Deployed Test Data to: {simulator_csv_path}")


print("\nExport Complete! The model is ready for the Pi, and the data is ready for the simulator.")