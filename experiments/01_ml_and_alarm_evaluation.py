"""
===============================================================================
Module Name:       01_ml_and_alarm_evaluation.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Empirical Validation Tier (Benchmark 01)

Description:       Evaluates the Random Forest soft-sensor against the Naive
                   Mean Predictor baseline on the unseen chronological holdout
                   dataset (21,713 samples). Computes continuous regression
                   metrics (MAE, MSE, RMSE, R²) and WHO Alert Level 1 (>= 10 µg/L)
                   binary classification metrics (Precision, Recall, F1, Specificity).
                   Generates publication-ready figures and summary tables.

Data Interfaces:
  - Upstream:      edge-system/app/model.joblib
                   edge-system/sensor-sim/data/simulation_test_data.csv
                   model-training/data/Playa_UPM_resampled_24H_1H.csv
  - Downstream:    experiments/ml_alarm_results.json
                   experiments/ml_alarm_summary_tables.md
                   experiments/figures/fig_5_1_regression_parity_and_residuals.png
                   experiments/figures/fig_5_2_who_alarm_confusion_matrix.png
                   experiments/figures/fig_5_3_holdout_timeseries_tracking.png
  - Storage / IPC: Local filesystem read/write.

References:        Mozo et al. (2022); WHO (2003).
===============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "edge-system/app/model.joblib")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "edge-system/sensor-sim/data/simulation_test_data.csv")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "model-training/data/Playa_UPM_resampled_24H_1H.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURES = ["EXO3(Temp_C)", "EXO3(spCond_uS_cm)", "EXO3(pH)", "SystemBattery"]
TARGET = "EXO3(Chlorophyll_ug_L)"
ALARM_THRESHOLD = 10.0

# Set plot aesthetics
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def run_evaluation():
    print("=" * 65)
    print("  01 — ML SOFT-SENSOR & WHO ALARM BENCHMARK EVALUATION")
    print("=" * 65)

    # 1. Load Model and Holdout Data
    print("\n1. Loading Deployed Model and Unseen Holdout Dataset...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Missing model ({MODEL_PATH}) or data ({TEST_DATA_PATH})")

    model = joblib.load(MODEL_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH).dropna()
    print(f"  -> Model loaded from: {MODEL_PATH}")
    print(f"  -> Holdout samples:  {len(df_test):,}")

    X_test = df_test[FEATURES]
    y_true = df_test[TARGET].values

    # 2. Compute Random Forest Predictions
    print("\n2. Executing Random Forest Inference on Holdout...")
    y_pred_rf = model.predict(X_test)
    y_pred_rf = np.maximum(0.0, y_pred_rf)  # Biological constraint: Chl-a cannot be negative

    # 3. Compute Naive Mean Predictor Baseline (from Training Data)
    print("\n3. Computing Naive Mean Baseline...")
    # Load training dataset to obtain the exact historical training mean (target column only)
    df_train_raw = pd.read_csv(TRAIN_DATA_PATH, usecols=[TARGET]).dropna()
    train_mean = float(df_train_raw[TARGET].mean())
    y_pred_mean = np.full_like(y_true, fill_value=train_mean)
    print(f"  -> Historical Training Mean: {train_mean:.3f} µg/L")

    # 4. Continuous Regression Metrics
    print("\n4. Calculating Regression Metrics...")
    mae_rf = float(mean_absolute_error(y_true, y_pred_rf))
    mse_rf = float(mean_squared_error(y_true, y_pred_rf))
    rmse_rf = float(np.sqrt(mse_rf))
    r2_rf = float(r2_score(y_true, y_pred_rf))

    mae_mean = float(mean_absolute_error(y_true, y_pred_mean))
    mse_mean = float(mean_squared_error(y_true, y_pred_mean))
    rmse_mean = float(np.sqrt(mse_mean))
    r2_mean = float(r2_score(y_true, y_pred_mean))

    print(f"  Random Forest Soft-Sensor:")
    print(f"    MAE:  {mae_rf:.3f} µg/L")
    print(f"    MSE:  {mse_rf:.3f} (µg/L)²")
    print(f"    RMSE: {rmse_rf:.3f} µg/L")
    print(f"    R²:   {r2_rf:.3f}")
    print(f"  Naive Mean Baseline:")
    print(f"    MAE:  {mae_mean:.3f} µg/L")
    print(f"    MSE:  {mse_mean:.3f} (µg/L)²")
    print(f"    RMSE: {rmse_mean:.3f} µg/L")
    print(f"    R²:   {r2_mean:.3f}")

    # 5. WHO Alert Level 1 Classification Metrics (>= 10 µg/L)
    print(f"\n5. Calculating WHO Level 1 Alarm Metrics (Threshold >= {ALARM_THRESHOLD} µg/L)...")
    alarm_true = (y_true >= ALARM_THRESHOLD).astype(int)
    alarm_pred_rf = (y_pred_rf >= ALARM_THRESHOLD).astype(int)
    alarm_pred_mean = (y_pred_mean >= ALARM_THRESHOLD).astype(int)

    def calc_alarm_metrics(y_t, y_p):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn))
        return {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "accuracy": round(accuracy, 4),
            "total_alarm_events_true": int(np.sum(y_t)),
            "total_alarm_events_pred": int(np.sum(y_p))
        }

    rf_alarm = calc_alarm_metrics(alarm_true, alarm_pred_rf)
    mean_alarm = calc_alarm_metrics(alarm_true, alarm_pred_mean)

    print(f"  RF Alarm Metrics:   Precision={rf_alarm['precision']:.3f}, Recall={rf_alarm['recall']:.3f}, F1={rf_alarm['f1_score']:.3f}")
    print(f"  Mean Alarm Metrics: Precision={mean_alarm['precision']:.3f}, Recall={mean_alarm['recall']:.3f}, F1={mean_alarm['f1_score']:.3f}")

    # 6. Save Structured JSON Results
    results = {
        "metadata": {
            "dataset": "As Conchas Reservoir (Playa & Presa buoys)",
            "holdout_samples": len(df_test),
            "target": TARGET,
            "features": FEATURES,
            "alarm_threshold_ug_L": ALARM_THRESHOLD,
            "literature_reference": "Mozo et al. (2022) Scientific Reports"
        },
        "regression_holdout": {
            "random_forest": {
                "mae_ug_L": round(mae_rf, 3),
                "mse": round(mse_rf, 3),
                "rmse_ug_L": round(rmse_rf, 3),
                "r2": round(r2_rf, 3)
            },
            "naive_mean_baseline": {
                "mae_ug_L": round(mae_mean, 3),
                "mse": round(mse_mean, 3),
                "rmse_ug_L": round(rmse_mean, 3),
                "r2": round(r2_mean, 3),
                "train_mean_value": round(train_mean, 3)
            }
        },
        "who_alarm_classification": {
            "random_forest": rf_alarm,
            "naive_mean_baseline": mean_alarm
        }
    }

    json_path = os.path.join(SCRIPT_DIR, "ml_alarm_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  -> Saved JSON results to: {json_path}")

    # 7. Generate Figures and Export CSV Data
    print("\n7. Generating Publication-Ready Figures and Excel-Ready CSV Data...")

    # Figure 5.1: Parity Plot & Residual Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Parity
    ax1.scatter(y_true, y_pred_rf, alpha=0.25, color="#1f77b4", edgecolors="none", s=18, label="RF Predictions", rasterized=True)
    max_val = max(np.max(y_true), np.max(y_pred_rf)) * 1.05
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect Agreement (1:1)")
    ax1.axhline(y=ALARM_THRESHOLD, color="#e94560", linestyle=":", label=f"WHO Alert Level 1 ({ALARM_THRESHOLD} µg/L)")
    ax1.axvline(x=ALARM_THRESHOLD, color="#e94560", linestyle=":")
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_xlabel("Actual Chlorophyll-a (µg/L)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Predicted Chlorophyll-a (µg/L)", fontsize=11, fontweight="bold")
    ax1.set_title("Regression Parity: Actual vs. Predicted", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Residual Error Histogram
    residuals = y_pred_rf - y_true
    sns.histplot(residuals, bins=60, kde=True, color="#0a3d62", ax=ax2)
    ax2.axvline(0, color="r", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Residual Error (Predicted - Actual, µg/L)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax2.set_title(f"Residual Distribution (MAE: {mae_rf:.2f} µg/L)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig51_path = os.path.join(FIGURES_DIR, "fig_5_1_regression_parity_and_residuals.png")
    plt.savefig(fig51_path)
    plt.close()
    print(f"  -> Saved {fig51_path}")

    # CSV for Fig 5.1
    csv51_path = os.path.join(FIGURES_DIR, "data_fig_5_1_regression_parity.csv")
    df_fig51 = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "actual_chlorophyll_ug_L": np.round(y_true, 3),
        "predicted_chlorophyll_ug_L": np.round(y_pred_rf, 3),
        "residual_error_ug_L": np.round(residuals, 3)
    })
    df_fig51.to_csv(csv51_path, index=False)
    print(f"  -> Saved CSV data to: {csv51_path}")

    # Figure 5.2: Confusion Matrix Heatmap (WHO Level 1)
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    cm = np.array([[rf_alarm["TN"], rf_alarm["FP"]],
                   [rf_alarm["FN"], rf_alarm["TP"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (<10 µg/L)", "Alarm (≥10 µg/L)"],
                yticklabels=["Normal (<10 µg/L)", "Alarm (≥10 µg/L)"],
                annot_kws={"size": 14, "fontweight": "bold"}, ax=ax)
    ax.set_xlabel("Predicted Condition", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(f"WHO Level 1 Alarm Confusion Matrix\nPrecision: {rf_alarm['precision']:.1%} | Recall: {rf_alarm['recall']:.1%} | F1: {rf_alarm['f1_score']:.3f}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig52_path = os.path.join(FIGURES_DIR, "fig_5_2_who_alarm_confusion_matrix.png")
    plt.savefig(fig52_path)
    plt.close()
    print(f"  -> Saved {fig52_path}")

    # CSV for Fig 5.2
    csv52_path = os.path.join(FIGURES_DIR, "data_fig_5_2_confusion_matrix.csv")
    df_fig52 = pd.DataFrame([
        {"Metric": "True Negatives (TN)", "Count": rf_alarm["TN"], "Description": "Correctly classified normal water"},
        {"Metric": "False Positives (FP)", "Count": rf_alarm["FP"], "Description": "False alarm triggered"},
        {"Metric": "False Negatives (FN)", "Count": rf_alarm["FN"], "Description": "Missed bloom event"},
        {"Metric": "True Positives (TP)", "Count": rf_alarm["TP"], "Description": "Correctly detected bloom alarm"},
        {"Metric": "Precision", "Count": rf_alarm["precision"], "Description": "Positive Predictive Value"},
        {"Metric": "Recall", "Count": rf_alarm["recall"], "Description": "Sensitivity / Hit Rate"},
        {"Metric": "F1-Score", "Count": rf_alarm["f1_score"], "Description": "Harmonic Mean of Precision & Recall"},
        {"Metric": "Specificity", "Count": rf_alarm["specificity"], "Description": "True Negative Rate"}
    ])
    df_fig52.to_csv(csv52_path, index=False)
    print(f"  -> Saved CSV data to: {csv52_path}")

    # Figure 5.3: Time-Series Snippet (7-day window / 672 readings)
    snippet_len = min(672, len(y_true))
    idx_slice = slice(1000, 1000 + snippet_len) if len(y_true) > 1700 else slice(0, snippet_len)
    
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    time_indices = np.arange(snippet_len) * 15 / 60  # convert 15-min intervals to hours
    ax.plot(time_indices, y_true[idx_slice], color="#16213e", linewidth=1.5, label="Actual Ground Truth (EXO3)", alpha=0.9)
    ax.plot(time_indices, y_pred_rf[idx_slice], color="#e94560", linewidth=1.5, linestyle="--", label="Random Forest Soft-Sensor", alpha=0.9)
    ax.axhline(y=ALARM_THRESHOLD, color="#ff9900", linestyle=":", linewidth=1.2, label=f"WHO Alert Level 1 ({ALARM_THRESHOLD} µg/L)")
    ax.set_xlabel("Elapsed Time (Hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Chlorophyll-a (µg/L)", fontsize=11, fontweight="bold")
    ax.set_title("Temporal Chlorophyll-a Tracking (7-Day Continuous Holdout Window)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig53_path = os.path.join(FIGURES_DIR, "fig_5_3_holdout_timeseries_tracking.png")
    plt.savefig(fig53_path)
    plt.close()
    print(f"  -> Saved {fig53_path}")

    # CSV for Fig 5.3
    csv53_path = os.path.join(FIGURES_DIR, "data_fig_5_3_timeseries_tracking.csv")
    df_fig53 = pd.DataFrame({
        "elapsed_hours": np.round(time_indices, 2),
        "actual_chlorophyll_ug_L": np.round(y_true[idx_slice], 3),
        "predicted_chlorophyll_ug_L": np.round(y_pred_rf[idx_slice], 3),
        "alarm_threshold_ug_L": ALARM_THRESHOLD
    })
    df_fig53.to_csv(csv53_path, index=False)
    print(f"  -> Saved CSV data to: {csv53_path}")

    # 8. Markdown Summary Table for Thesis
    md_summary = f"""# ML Evaluation and WHO Alarm Performance Summary

## 1. Continuous Regression Performance (Holdout Fold: {len(df_test):,} Samples)

| Model Architecture | MAE (µg/L) | MSE (µg/L)² | RMSE (µg/L) | R² Score | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest Soft-Sensor** | **{mae_rf:.3f}** | **{mse_rf:.3f}** | **{rmse_rf:.3f}** | **{r2_rf:.3f}** | Scikit-learn Pipeline with PowerTransformer (X and y) |
| **Naive Mean Baseline** | {mae_mean:.3f} | {mse_mean:.3f} | {rmse_mean:.3f} | {r2_mean:.3f} | Constant mean predictor ({train_mean:.2f} µg/L) |

## 2. WHO Alert Level 1 Alarm Classification (Threshold ≥ {ALARM_THRESHOLD} µg/L)

| Model Architecture | Precision | Recall (Sensitivity) | F1-Score | Specificity | True Positives | False Positives | False Negatives | True Negatives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **{rf_alarm['precision']:.3f}** | **{rf_alarm['recall']:.3f}** | **{rf_alarm['f1_score']:.3f}** | **{rf_alarm['specificity']:.3f}** | {rf_alarm['TP']:,} | {rf_alarm['FP']:,} | {rf_alarm['FN']:,} | {rf_alarm['TN']:,} |
| **Naive Mean Baseline** | {mean_alarm['precision']:.3f} | {mean_alarm['recall']:.3f} | {mean_alarm['f1_score']:.3f} | {mean_alarm['specificity']:.3f} | {mean_alarm['TP']:,} | {mean_alarm['FP']:,} | {mean_alarm['FN']:,} | {mean_alarm['TN']:,} |

## 3. Key Observations & Academic Justification
- **Non-Linear Biological Capture**: The Random Forest soft-sensor reduces the holdout MAE from {mae_mean:.3f} µg/L (Naive Mean) down to {mae_rf:.3f} µg/L.
- **Alarm Sensitivity**: The soft-sensor successfully captures {rf_alarm['recall']:.1%} of all real algal bloom alarm events with an F1-score of {rf_alarm['f1_score']:.3f}, validating its effectiveness as an automatic early warning trigger.
- **Literature Alignment**: Aligned with Mozo et al. (2022), demonstrating that physical proxies (pH, temperature, conductivity) are competent for edge soft-sensing.
"""
    md_path = os.path.join(SCRIPT_DIR, "ml_alarm_summary_tables.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_summary)
    print(f"  -> Saved Markdown summary to: {md_path}")
    print("\n[OK] ML & Alarm Evaluation Complete!")


if __name__ == "__main__":
    run_evaluation()
