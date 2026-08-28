# ML Training & Evaluation Scripts

Python scripts for data preparation, model training, OTA deployment, and empirical evaluation on the PC.

## Files

- `model-training/ml_regression_KFold_model_export.py` — Load datasets, sequential 10-Fold CV, train Random Forest pipeline with PowerTransformer, export `model.joblib` and `simulation_test_data.csv`.
- `model-training/model_test.py` — QA sanity check validating inference on exported pipeline.
- `model-training/publish_ota.py` — Publish Over-The-Air model update command via MQTT.
- `experiments/01_ml_and_alarm_evaluation.py` — Evaluate holdout regression (MAE, MSE, RMSE, R²) and WHO Alert Level 1 alarm classification metrics (Precision, Recall, F1, Confusion Matrix).
- `experiments/02_edge_hardware_profiling.py` — Profile SQLite telemetry data for latency distributions, RAM stability leak regression, and mathematical energy trade-off model.
- `experiments/03_system_robustness_ota_suite.py` — Automated 5-case test harness validating Test-Before-Swap OTA fail-safes and crash resilience.
- `experiments/run_clean_benchmark.py` — Automated clean container benchmark orchestrator (`docker compose down -v` -> fresh stream -> automated evaluation).

## Usage

```bash
# 1. Train and export model pipeline + test dataset
python model-training/ml_regression_KFold_model_export.py

# 2. Sanity check exported model
python model-training/model_test.py

# 3. Run automated empirical benchmark suites
python experiments/01_ml_and_alarm_evaluation.py
python experiments/02_edge_hardware_profiling.py
python experiments/03_system_robustness_ota_suite.py

# Or run the full clean-container benchmark orchestrator:
python experiments/run_clean_benchmark.py
```

## Outputs

- Model saved to `edge-system/app/model.joblib`
- Simulation dataset saved to `edge-system/sensor-sim/data/simulation_test_data.csv`
- Model metrics saved to `edge-system/app/model_metrics.json`
- Verification figures (.png) and CSV datasets exported to `experiments/figures/`
- Full documentation: see `model-training/README.md` and `experiments/README.md`
