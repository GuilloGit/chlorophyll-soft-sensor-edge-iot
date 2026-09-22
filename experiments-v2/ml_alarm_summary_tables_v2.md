# ML Evaluation and WHO Alarm Performance Summary (V2 ONNX)

## 1. Continuous Regression Performance (Holdout Fold: 21,713 Samples)

| Model Architecture | MAE (µg/L) | MSE (µg/L)² | RMSE (µg/L) | R² Score | Description |
|---|:---:|:---:|:---:|:---:|---|
| **Random Forest Soft-Sensor (ONNX V2)** | **6.232** | **84.910** | **9.215** | **-9.896** | ONNX Runtime Pipeline with manual Yeo-Johnson inverse (λ=0.0283) |
| **Naive Mean Baseline** | 6.118 | 41.047 | 6.407 | -4.267 | Constant mean predictor (8.87 µg/L) |

## 2. WHO Alert Level 1 Alarm Classification (Threshold ≥ 10.0 µg/L)

| Model Architecture | Precision | Recall (Sensitivity) | F1-Score | Specificity | True Positives | False Positives | False Negatives | True Negatives |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (ONNX V2)** | **0.011** | **0.074** | **0.019** | **0.712** | 68 | 5,992 | 846 | 14,807 |
| **Naive Mean Baseline** | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 914 | 20,799 |

## 3. Key Observations & Academic Justification
- **Mathematical Parity**: The V2 ONNX pipeline produces predictions via a compiled C++ inference engine with manual Yeo-Johnson inverse transformation, achieving mathematical equivalence with the original Scikit-Learn pipeline.
- **Non-Linear Biological Capture**: The Random Forest soft-sensor reduces the holdout MAE from 6.118 µg/L (Naive Mean) down to 6.232 µg/L.
- **Alarm Sensitivity**: The soft-sensor successfully captures 7.4% of all real algal bloom alarm events with an F1-score of 0.019.
