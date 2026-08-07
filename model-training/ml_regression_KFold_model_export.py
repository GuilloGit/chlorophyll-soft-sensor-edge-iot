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
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "data/Playa_UPM_resampled_24H_1H.csv")
TEST_FILE = os.path.join(BASE_DIR, "data/Presa_UPM_resampled_24H_1H.csv")
MODELS_RANDOM_STATE = 22

FEATURES = ["EXO3(Temp_C)", "EXO3(spCond_uS_cm)", "EXO3(pH)", "SystemBattery"]
TARGET = "EXO3(Chlorophyll_ug_L)"

print("1. Loading Datasets...")
Training_Dataset = pd.read_csv(TRAIN_FILE)
Testing_Dataset = pd.read_csv(TEST_FILE)

# Clean up the unnecessary index column without losing numeric datatypes
Training_Dataset = Training_Dataset.drop(columns=["Unnamed: 0"]).dropna()
Testing_Dataset = Testing_Dataset.drop(columns=["Unnamed: 0"]).dropna()

print("2. Building Pipeline Template...")
# The pipeline encapsulates the RF model and both scalers.
# It is built as a template and cloned for each CV fold.
rf_model = RandomForestRegressor(
    n_estimators=100,
    n_jobs=-1,
    random_state=MODELS_RANDOM_STATE
)
target_regressor = TransformedTargetRegressor(
    regressor=rf_model,
    transformer=PowerTransformer()
)
pipeline_template = Pipeline([
    ('x_scaler', PowerTransformer()),
    ('rf_model_with_y_scaler', target_regressor)
])

# --- PHASE 1: PAPER-EQUIVALENT 10-FOLD CV EVALUATION ---
print("3. Running 10-Fold Cross-Validation Evaluation (paper-equivalent)...")
kf = KFold(n_splits=10, shuffle=False)
splits_b0 = list(kf.split(Training_Dataset))
splits_b1 = list(kf.split(Testing_Dataset))

cv_all_true, cv_all_pred = [], []
for fold_idx in range(10):
    train_idx_b0, test_idx_b0 = splits_b0[fold_idx]
    train_idx_b1, test_idx_b1 = splits_b1[fold_idx]

    df_fold_train = pd.concat([
        Training_Dataset.iloc[train_idx_b0],
        Testing_Dataset.iloc[train_idx_b1]
    ], ignore_index=True)
    df_fold_test = pd.concat([
        Training_Dataset.iloc[test_idx_b0],
        Testing_Dataset.iloc[test_idx_b1]
    ], ignore_index=True)

    fold_pipeline = clone(pipeline_template)
    fold_pipeline.fit(df_fold_train[FEATURES], df_fold_train[TARGET])
    fold_preds = fold_pipeline.predict(df_fold_test[FEATURES])

    fold_mae = mean_absolute_error(df_fold_test[TARGET], fold_preds)
    cv_all_true.append(df_fold_test[TARGET])
    cv_all_pred.append(fold_preds)
    print(f"  Fold {fold_idx:2d}: MAE={fold_mae:.3f} µg/L")

y_cv_true = pd.concat(cv_all_true)
y_cv_pred = np.concatenate(cv_all_pred)
cv_mae = mean_absolute_error(y_cv_true, y_cv_pred)
cv_r2  = r2_score(y_cv_true, y_cv_pred)
print(f"  -> CV MAE (all folds): {cv_mae:.3f} µg/L  |  R²: {cv_r2:.3f}")

# --- PHASE 2: TRAIN AND DEPLOY THE FINAL MODEL (LAST FOLD = 90/10 SPLIT) ---
print("4. Training the Final Deployment Model (last fold: 90% train / 10% holdout)...")
# The last fold's train set is the chronologically oldest 90% of the data.
# The last fold's test set is exported as the simulator data (the Pi's "live" stream).
boya_0_splits = splits_b0[-1]
boya_1_splits = splits_b1[-1]

df_train = pd.concat([
    Training_Dataset.iloc[boya_0_splits[0]],
    Testing_Dataset.iloc[boya_1_splits[0]]
], ignore_index=True)
df_test = pd.concat([
    Training_Dataset.iloc[boya_0_splits[1]],
    Testing_Dataset.iloc[boya_1_splits[1]]
], ignore_index=True)

X_train = df_train[FEATURES]
y_train = df_train[TARGET]

edge_pipeline = clone(pipeline_template)
edge_pipeline.fit(X_train, y_train)

print("4.5 Evaluating Holdout and Baseline...")
X_test = df_test[FEATURES]
y_test = df_test[TARGET]

dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)

rf_predictions    = edge_pipeline.predict(X_test)
dummy_predictions = dummy_model.predict(X_test)

holdout_mae = mean_absolute_error(y_test, rf_predictions)
holdout_r2  = r2_score(y_test, rf_predictions)

metrics = {
    "cross_validation": {
        "strategy": "KFold(n_splits=10, shuffle=False) — all folds concatenated",
        "mae": cv_mae,
        "r2": cv_r2,
        "description": "Paper-equivalent metric (Mozo et al. 2022 methodology)"
    },
    "deployed_model_holdout": {
        "description": "Last 10% of data — same split exported as sensor simulator data",
        "mae": holdout_mae,
        "r2": holdout_r2
    },
    "baseline_mean_holdout": {
        "mae": mean_absolute_error(y_test, dummy_predictions),
        "r2": r2_score(y_test, dummy_predictions)
    },
    "metadata": {
        "n_folds": 10,
        "holdout_samples": len(y_test),
        "train_samples": len(y_train),
        "target": TARGET
    }
}

metrics_path = os.path.join(BASE_DIR, "../edge-system/app/model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"  -> Saved Evaluation Metrics to: {metrics_path}")

print("5. Exporting Assets...")

edge_model_path = os.path.join(BASE_DIR, "../edge-system/app/model.joblib")
os.makedirs(os.path.join(BASE_DIR, "../edge-system/app"), exist_ok=True)
joblib.dump(edge_pipeline, edge_model_path, compress=3)
print(f"  -> Deployed Model to: {edge_model_path}")

# Compute SHA-256 for OTA verification
with open(edge_model_path, "rb") as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"  -> Model SHA-256: {sha}")

# Publish the test data into the on-Pi sensor simulator's data folder
simulator_csv_path = os.path.join(BASE_DIR, "../edge-system/sensor-sim/data/simulation_test_data.csv")
os.makedirs(os.path.join(BASE_DIR, "../edge-system/sensor-sim/data"), exist_ok=True)
df_test.to_csv(simulator_csv_path, index=False)
print(f"  -> Deployed Test Data to: {simulator_csv_path}")


print("\nExport Complete! The model is ready for the Pi, and the data is ready for the simulator.")