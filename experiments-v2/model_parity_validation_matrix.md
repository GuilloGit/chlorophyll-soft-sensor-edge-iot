# Model Parity and WHO Alarm Validation Matrix

**Project:** Chlorophyll-a Soft-Sensor Edge-IoT System  
**Tier:** Empirical Validation Tier (Benchmark 01)  
**Holdout Partition:** As Conchas Reservoir Unseen Chronological Dataset ($N = 21,713$ samples)  
**Evaluation Target:** `EXO3(Chlorophyll_ug_L)`  
**WHO Alert Level 1 Threshold:** $\ge 10.0\,\mu\text{g/L}$  
**Date:** September 2026  

---

## 1. Executive Summary & Objective

This document tracks, maps, and validates the empirical equivalence and operational performance of the machine learning inference pipelines across three sequential revisions:
1. **V1 Baseline (Scikit-Learn Joblib):** The reference Scikit-Learn `TransformedTargetRegressor` containing a 100-estimator `RandomForestRegressor` with full built-in target unscaling.
2. **V2 Pre-Fix (Truncated Manual Inversion):** The initial ONNX export where the manual edge inverse transformation omitted the internal affine z-score de-standardization moments ($\mu, \sigma$), causing an output space collapse ($[-1.55, 9.67]\,\mu\text{g/L}$) and a 100% failure rate on WHO Level 1 alarm detection (0 TP, 914 FN).
3. **V2 Post-Fix (Corrected Affine Unscaling + Biological Floor):** The corrected ONNX Runtime inference pipeline incorporating full affine unscaling ($\hat{y}^{(\lambda)} = \hat{y}_{\text{norm}} \cdot \sigma + \mu$) prior to inverse Yeo-Johnson transformation and clamping to non-negative physical concentrations.

The primary objective is to prove **exact numerical parity** ($\Delta < 10^{-12}\,\mu\text{g/L}$) between V1 and V2 Post-Fix, resolving the critical alarm failure and restoring operational safety.

---

## 2. Mathematical Pipeline Formulation

```
                               FORWARD PIPELINE (Training)
Raw Chl-a y ──► [Yeo-Johnson ψ(λ, y)] ──► y^(λ) ──► [StandardScaler (· - μ)/σ] ──► y_norm ──► RF Trees Fit
                                                                                                 │
═════════════════════════════════════════════════════════════════════════════════════════════════╪══════════════════════
                               INVERSE PIPELINE (Inference)                                     ▼
Pipeline Revision          Inference Engine                Target De-standardization        Target Inversion      Output Clamping
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
V1 Baseline (Joblib)       Scikit-Learn (Python)           Automated via TransformedTarget   Automated ψ^(-1)      None (Linear)
V2 Pre-Fix (Buggy)         ONNX Runtime (C++ / Python)     OMITTED (u = y_pred_norm)         Manual ψ^(-1)(u, λ)   None (Unbounded)
V2 Post-Fix (Validated)    ONNX Runtime (C++ / Python)     RESTORED: u = y_norm * σ + μ      Manual ψ^(-1)(u, λ)   max(0.0, y)
```

### 2.1 Forward Normalization (Scikit-Learn)
Given the training target $y \ge 0$, Scikit-Learn fits a `PowerTransformer(method='yeo-johnson', standardize=True)`:
$$\psi(\lambda, y) = \begin{cases} \frac{(y + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \ne 0, y \ge 0 \\ \ln(y + 1) & \text{if } \lambda = 0, y \ge 0 \end{cases}$$
The transformed target is subsequently normalized via empirical moments:
$$y_{\text{norm}} = \frac{\psi(\lambda, y) - \mu_y}{\sigma_y}$$
For the As Conchas training dataset, the calibrated parameters are:
* $\lambda = 0.02829662709583453$
* $\mu_y = 2.1953387515664177$
* $\sigma_y = 0.7377154823101166$

### 2.2 Inversion Mechanics
* **V2 Pre-Fix Inversion (Erroneous):**
  $$\hat{y}_{\text{pre}} = \psi^{-1}(\lambda, \hat{y}_{\text{norm}})$$
  Because $\mu_y$ and $\sigma_y$ were omitted, the predicted normalized target $\hat{y}_{\text{norm}} \in [-1.2, 2.1]$ was fed directly into $\psi^{-1}$. Because $\psi^{-1}(\lambda, 0) = 0$, this artificially bound the predicted concentration range to $[-1.554, 9.671]\,\mu\text{g/L}$. Consequently, $\hat{y}_{\text{pre}}$ could never exceed $9.671\,\mu\text{g/L}$, rendering detection of the $10.0\,\mu\text{g/L}$ alarm threshold mathematically impossible.
* **V2 Post-Fix Inversion (Corrected):**
  $$\hat{y}^{(\lambda)} = \hat{y}_{\text{norm}} \cdot \sigma_y + \mu_y$$
  $$\hat{y}_{\text{post}} = \max\left(0.0, \; \psi^{-1}(\lambda, \hat{y}^{(\lambda)})\right)$$
  Where $\psi^{-1}(\lambda, u)$ for $\lambda \ne 0$ is:
  $$\psi^{-1}(\lambda, u) = \begin{cases} (u \lambda + 1)^{1/\lambda} - 1 & \text{if } u \ge 0 \\ 1 - (-(2 - \lambda) u + 1)^{1/(2 - \lambda)} & \text{if } u < 0 \end{cases}$$

---

## 3. 3-Way Model Parity & Performance Matrix

The following table presents the audited results across all 21,713 unseen chronological holdout samples:

| Evaluation Dimension | Metric / Parameter | V1 Baseline (Scikit-Learn) | V2 Pre-Fix (ONNX Runtime) | V2 Post-Fix (ONNX Runtime) | Parity Status / Delta (V1 vs. V2 Post-Fix) |
|---|---|:---:|:---:|:---:|:---:|
| **Pipeline Metadata** | Model Format | `.joblib` (Python Pickled) | `.onnx` (ONNX Opset 15) | `.onnx` (ONNX Opset 15) | Format migration complete |
| | Disk Footprint | 415.29 MB | 943.75 MB | 943.75 MB | Zero external scikit-learn dependency |
| | Target Unscaling | Internal (Automated) | Manual (Truncated, $\lambda$ only) | Manual (Complete: $\lambda, \mu, \sigma$) | Full mathematical restoration |
| | Biological Floor | None ($y \in \mathbb{R}$) | None ($y \in \mathbb{R}$) | Clamped: $\max(0.0, y)$ | Physically consistent |
| **Output Dynamic Range** | Min Predicted Chl-a | $0.0000\,\mu\text{g/L}$ | $-1.5542\,\mu\text{g/L}$ | $0.0000\,\mu\text{g/L}$ | Exact floor match |
| | Max Predicted Chl-a | $28.3411\,\mu\text{g/L}$ | $9.6713\,\mu\text{g/L}$ | $28.3411\,\mu\text{g/L}$ | $\Delta < 10^{-12}\,\mu\text{g/L}$ |
| | Mean Predicted Chl-a | $8.0735\,\mu\text{g/L}$ | $3.2982\,\mu\text{g/L}$ | $8.0735\,\mu\text{g/L}$ | $\Delta < 10^{-12}\,\mu\text{g/L}$ |
| **Continuous Regression** | Mean Absolute Error (MAE) | **6.232 µg/L** | 2.902 µg/L *(artifact)* | **6.232 µg/L** | **Exact match (0.0000 µg/L delta)** |
| | Mean Squared Error (MSE) | 84.922 (µg/L)² | 15.792 (µg/L)² | 84.922 (µg/L)² | Exact match |
| | Root Mean Squared Error (RMSE) | **9.215 µg/L** | 3.974 µg/L | **9.215 µg/L** | **Exact match (0.0000 µg/L delta)** |
| | Coefficient of Det. ($R^2$) | **-9.896** | -1.028 | **-9.896** | **Exact match** |
| **WHO Level 1 Classification** | Threshold | $\ge 10.0\,\mu\text{g/L}$ | $\ge 10.0\,\mu\text{g/L}$ | $\ge 10.0\,\mu\text{g/L}$ | Operational regulatory threshold |
| ($\ge 10\,\mu\text{g/L}$) | True Positives (TP) | **68** | **0** *(Critical Failure)* | **68** | **Restored (100% agreement)** |
| | False Negatives (FN) | **846** | 914 | **846** | Restored (68 bloom events recovered) |
| | False Positives (FP) | 6,009 | 0 | 6,009 | Exact match |
| | True Negatives (TN) | 14,790 | 20,799 | 14,790 | Exact match |
| | Recall (Sensitivity) | **7.44%** | **0.00%** | **7.44%** | **Fully recovered** |
| | Precision | **1.12%** | **0.00%** | **1.12%** | **Fully recovered** |
| | F1-Score | **0.019** | **0.000** | **0.019** | **Fully recovered** |
| | Specificity | 71.11% | 100.00% | 71.11% | Exact match |

---

## 4. Empirical Parity Audit & Validation Criteria

### 4.1 Numerical Equivalence
Direct element-wise comparison between the Scikit-Learn V1 predictions ($\mathbf{y}_{\text{V1}}$) and the corrected ONNX V2 predictions ($\mathbf{y}_{\text{V2}}$) across all 21,713 test samples yields:
* **Maximum Absolute Error:**
  $$\max_i |y_{\text{V1}, i} - y_{\text{V2}, i}| = 3.126 \times 10^{-13}\,\mu\text{g/L}$$
* **Mean Absolute Error:**
  $$\frac{1}{N}\sum_{i=1}^N |y_{\text{V1}, i} - y_{\text{V2}, i}| = 4.882 \times 10^{-14}\,\mu\text{g/L}$$
* **Pearson Correlation Coefficient ($r$):**
  $$r(\mathbf{y}_{\text{V1}}, \mathbf{y}_{\text{V2}}) = 1.0000000000000$$

> [!NOTE]
> The observed deviation ($\approx 3.12 \times 10^{-13}$) is 10 orders of magnitude smaller than the sensor measurement resolution ($0.01\,\mu\text{g/L}$) and is entirely attributable to IEEE 754 64-bit double-precision floating-point rounding between Python's native math library and the C++ ONNX Runtime vector intrinsics.

### 4.2 Deconstruction of the Pre-Fix "Low MAE" Paradox
A superficial reading of the pre-fix benchmark reported that V2 "improved" MAE from $6.232\,\mu\text{g/L}$ down to $2.902\,\mu\text{g/L}$. The mathematical investigation proves that this was an **accidental statistical illusion**:
1. The unseen chronological holdout partition (November–December winter regime) exhibits a depressed real chlorophyll-a concentration mean of $\bar{y}_{\text{true}} = 2.802\,\mu\text{g/L}$.
2. The missing affine unscaling compressed all predicted outputs into a narrow range around $3.298\,\mu\text{g/L}$.
3. By predicting values near $3.3\,\mu\text{g/L}$ for almost all samples, the pre-fix model accidentally acted as a near-constant predictor around the partition mean, artificially minimizing the linear $\ell_1$ distance $|y - \bar{y}|$ on low-algal samples.
4. However, this output compression totally blinded the model to variance and extreme events: the maximum output was $9.67\,\mu\text{g/L}$, missing every single one of the 914 algal bloom occurrences ($y \ge 10\,\mu\text{g/L}$).
5. The post-fix model correctly predicts the wide variance of the true physical process (up to $28.34\,\mu\text{g/L}$), successfully detecting extreme events at the expense of higher regression penalty, exactly replicating the V1 baseline.

---

## 5. Deployment Sign-Off Checklist

- [x] Target serialization format updated: `edge-system/app-v2/y_lambda.json` contains `y_lambda`, `y_mean`, and `y_scale`.
- [x] Exporter verified: `model-training/export_onnx.py` automatically extracts transformer moments from Scikit-Learn pipeline.
- [x] Edge runtime verified: `edge-system/app-v2/app.py` implements `_inverse_power_transform` with full affine unscaling and non-negative clamping.
- [x] Holdout benchmark script verified: `experiments-v2/01_ml_and_alarm_evaluation_v2.py` updated to use full unscaling.
- [x] Numerical parity confirmed: Maximum divergence $\le 3.13 \times 10^{-13}\,\mu\text{g/L}$.
- [x] Alarm detection restored: 68 True Positives captured.
