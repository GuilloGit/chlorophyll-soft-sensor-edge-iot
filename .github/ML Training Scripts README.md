# ML Training Scripts

Python scripts for data preparation, model training, and MQTT simulation on the PC.

## Files

- `train.py` — Load dataset, feature engineering, sequential split, Random Forest training
- `feature_eng.py` — Rolling statistics, window functions for sensor data
- `mqtt_publisher.py` — Simulate live sensor data via MQTT

## Usage

```bash
# Train the model
python train.py

# Simulate MQTT publisher (test set)
python mqtt_publisher.py
```

## Outputs

- Model saved to `../models/model.joblib`
- Training metrics logged to console
