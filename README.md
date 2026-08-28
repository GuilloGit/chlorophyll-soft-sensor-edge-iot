# Chlorophyll-a Soft-Sensor Edge-IoT System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![MQTT](https://img.shields.io/badge/protocol-MQTT%20v5%2Fv3.1.1-orange.svg)](https://mqtt.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source, edge-native Internet of Things (IoT) system for real-time **Chlorophyll-a concentration soft-sensing** and **Harmful Algal Bloom (HAB)** early detection in freshwater reservoirs. 

Instead of relying on fragile, expensive, and bio-fouling optical fluorometers, this system deploys a lightweight machine learning pipeline on edge hardware (Raspberry Pi 4) to infer Chlorophyll-a concentrations in real time using low-cost, rugged physical surrogate variables: **Water Temperature**, **Specific Electrical Conductance**, **pH**, and **System Battery Voltage**.

```mermaid
flowchart LR
    subgraph SENSORS["Physical / In Situ Sonde"]
        EXO["Water Sonde\n(Temp, spCond, pH, Battery)"]
    end

    subgraph EDGE["Edge Processing Tier (Raspberry Pi 4 / Docker)"]
        SIM["Telemetry Reader\n(sensor-sim)"]
        BROKER["Mosquitto Broker\n(:1883)"]
        APP["Inference Engine\n(Random Forest + SQLite WAL)"]
        
        SIM -->|"sensor/water/raw\n(15-min interval)"| BROKER
        BROKER --> APP
    end

    subgraph CLOUD["Cloud Analytics Tier"]
        SUBSCRIBER["Cloud Analytics\n(analytics_subscriber.py)"]
    end

    APP -->|"sensor/water/daily_batch\n(24h smart batch, QoS 1)"| SUBSCRIBER
    APP -->|"sensor/water/alarm\n(WHO Level 1 >= 10 µg/L, QoS 2)"| SUBSCRIBER
```

---

## Key Technical Features

* **Edge Machine Learning Inference**: Deploys a serialized Scikit-Learn pipeline (Yeo-Johnson `PowerTransformer` + `RandomForestRegressor` + `TransformedTargetRegressor`) executing in **~27.7 ms** on ARM64 hardware (Raspberry Pi 4 Model B).
* **Communication Energy Conservation (Smart Batching)**: Aggregates regular operational readings into 24-hour daily arrays (96 samples at 15-minute intervals) published over MQTT QoS 1. Reduces cellular network attach cycles and radio energy consumption by **96.7%** compared to continuous cloud-streaming architectures.
* **Immediate Critical Hazard Alerts**: Bypasses the 24-hour batching queue when Chlorophyll-a exceeds the World Health Organization (WHO) Alert Level 1 threshold ($\ge 10.0\,\mu\text{g/L}$), dispatching an immediate event-driven alarm packet via MQTT QoS 2.
* **Reliability & Crash Resilience**: 
  - **SQLite with Write-Ahead Logging (WAL)**: Ensures zero database corruption during unexpected power cuts or hard resets.
  - **Atomic Checkpointing**: Sensor simulators utilize atomic temp-file replacement (`os.replace`) to maintain continuous tracking across power interruptions.
  - **Test-Before-Swap Remote OTA Updates**: Over-the-Air model updates are verified by SHA-256 hash, trial-loaded in memory, and hot-swapped under a concurrency lock before persistent disk replacement.

---

## System Architecture & Repository Structure

```
chlorophyll-soft-sensor-edge-iot/
├── edge-system/                    # Edge Processing Tier (Docker containerized)
│   ├── app/                        # Inference & persistence engine microservice
│   │   ├── app.py                  # Core inference, SQLite WAL, batching & OTA logic
│   │   ├── model.joblib            # Active Scikit-Learn Random Forest pipeline
│   │   ├── model_metrics.json      # Model validation & cross-validation metrics
│   │   └── Dockerfile              # Python 3.11-slim ARM64/AMD64 edge container
│   ├── sensor-sim/                 # Software-in-the-Loop (SIL) telemetry simulator
│   │   ├── sensor_reader.py        # Stream driver with atomic checkpoint state machine
│   │   ├── data/                   # Simulation test dataset (chronological holdout)
│   │   └── Dockerfile              # Telemetry simulator container
│   ├── mosquitto/                  # Edge MQTT broker configuration
│   └── docker/                     # Docker Compose deployment manifests
│       ├── docker-compose.yml      # Production stack (named persistent volumes)
│       └── docker-compose.override.yml # Development hot-reload configuration
├── cloud-system/                   # Cloud Analytics Tier
│   ├── analytics_subscriber.py     # Batch consumer, fidelity calculator & reporter
│   ├── edge_performance_report.json # Rolling edge KPI report (MAE, R², latency, RAM, CPU)
│   └── requirements.txt            # Cloud tier dependencies
├── model-training/                 # Machine Learning Workstation Tier
│   ├── ml_regression_KFold_model_export.py # Sequential 10-fold CV training pipeline
│   ├── model_test.py               # Local inference QA validation script
│   ├── publish_ota.py              # Remote OTA update command publisher
│   └── requirements.txt            # ML workstation dependencies
└── experiments/                    # Empirical Validation & Benchmarking Suite
    ├── run_clean_benchmark.py      # Automated clean-container orchestration driver
    ├── 01_ml_and_alarm_evaluation.py # Soft-sensor regression and alarm benchmark
    ├── 02_edge_hardware_profiling.py # Edge resource profiling & energy model
    ├── 03_system_robustness_ota_suite.py # 5-stage automated resilience & OTA testbench
    ├── figures/                    # Publication-grade PNG charts and raw CSV datasets
    └── requirements.txt            # Experiment suite dependencies
```

---

## Empirical Benchmark Highlights

The system was evaluated over **21,638 empirical samples** (representing ~7.5 continuous months of freshwater reservoir monitoring from the As Conchas reservoir, Galicia, Spain) on a physical **Raspberry Pi 4 Model B (8 GB RAM)**:

| Metric Category | Measured Parameter | Empirical Value | System Interpretation |
|---|---|:---:|---|
| **Inference Fidelity** | Mean Absolute Error (MAE) | **1.218 µg/L** | Captures non-linear biological dynamics (baseline naive error = 2.458 µg/L) |
| **Inference Fidelity** | Coefficient of Determination ($R^2$) | **0.584** | Substantial variance explained using only 4 low-cost physical variables |
| **Alarm Reliability** | WHO Level 1 Recall (Sensitivity) | **81.0%** | Detects over 8 out of 10 blooming events without optical instrumentation |
| **Alarm Reliability** | WHO Level 1 Specificity | **93.8%** | Extremely low false positive rate during non-bloom conditions |
| **Edge Compute Latency**| Random Forest Inference (Mean) | **27.7 ms** | Real-time slack margin > 99.996% of 15-minute sampling interval |
| **Edge Compute Latency**| 95th Percentile Latency (P95) | **33.3 ms** | Predictable execution with minimal tail latency |
| **Storage Overhead**   | SQLite WAL Insertion Time | **1.86 ms** | Negligible I/O burden on embedded SD/eMMC media |
| **Memory Stability**   | RAM Leak Linear Regression Slope | **-0.00229 MB/sample** | Absolute zero memory accumulation over 21,000+ continuous predictions |
| **Energy Conservation**| Modem Transmission Reduction | **-96.7%** | Cellular energy drops from 28.80 kJ/day (continuous) to 0.96 kJ/day (batched) |
| **Energy Conservation**| Total Daily System Energy Savings | **-34.6%** | Lowers daily buoy energy draw from 43.2 Wh/day to 28.2 Wh/day |
| **System Resilience**  | Fail-Safe & Crash Recovery | **100% Pass** | Zero corrupted database transactions and zero failed OTA rollouts |

---

## Quickstart Guide

### 1. Prerequisites
* [Docker](https://docs.docker.com/get-docker/) & Docker Compose
* Python 3.10+ (for cloud subscriber or workstation training)

### 2. Launch the Edge System (Production Mode)
To deploy the complete edge stack (Mosquitto MQTT broker, sensor simulator, and inference engine) with persistent named volumes:

```bash
cd edge-system/docker
docker compose -f docker-compose.yml up -d --build
```

Inspect operational container logs:
```bash
docker compose logs -f edge_inference_app
```

### 3. Launch Cloud Analytics Monitor
On a workstation or server with network access to the edge broker:

```bash
cd cloud-system
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the cloud listener (pointing to edge broker IP if remote)
export MQTT_BROKER="localhost"
python analytics_subscriber.py
```

The cloud monitor continuously generates updated fidelity and performance summaries in `cloud-system/edge_performance_report.json`.

### 4. Train or Retrain the Model
To re-run the 10-fold sequential cross-validation pipeline and export edge assets:

```bash
cd model-training
pip install -r requirements.txt
python ml_regression_KFold_model_export.py
```

Validate exported model integrity locally:
```bash
python model_test.py
```

### 5. Execute Automated Empirical Benchmarks
Run the automated test runner to reproduce all figures and validation metrics:

```bash
cd experiments
pip install -r requirements.txt

# Option A: Run complete clean-container benchmark from scratch
python run_clean_benchmark.py

# Option B: Run individual evaluation scripts
python 01_ml_and_alarm_evaluation.py       # Generates regression & alarm charts
python 02_edge_hardware_profiling.py       # Generates latency, RAM, and energy charts
python 03_system_robustness_ota_suite.py   # Runs the 5-stage resilience test suite
```

---

## MQTT Communication Contracts

The system utilizes 5 core MQTT topics:

| Topic | Publisher | Subscriber | QoS | Payload Summary |
|---|---|---|:---:|---|
| `sensor/water/raw` | `sensor-sim` | `edge-app` | 1 | Real-time JSON telemetry: Temp, spCond, pH, Battery, Ground Truth |
| `sensor/water/daily_batch` | `edge-app` | `cloud-system` | 1 | 96-element JSON array of readings, predictions, and hardware stats |
| `sensor/water/alarm` | `edge-app` | `cloud-system` | 2 | Immediate notification when Predicted Chlorophyll-a $\ge 10.0\,\mu\text{g/L}$ |
| `buoy/ota/update` | `publish_ota.py` | `edge-app` | 1 | Remote update command: `url`, `sha256`, and `version` |
| `buoy/ota/status` | `edge-app` | `publish_ota.py` | 1 | Execution confirmation: `version`, `status`, and `message` |

---

## Scientific Literature References

1. **Mozo, A., Morón-López, J., Vakaruk, S., Pompa-Pernía, A. G., González-Prieto, A., Aguilar, J. A. P., Gómez-Canaval, S., & Ortiz, J. M.** (2022). *Chlorophyll soft-sensor based on machine learning models for algal bloom predictions.* Scientific Reports, 12(1), 13529. [https://doi.org/10.1038/s41598-022-17299-5](https://doi.org/10.1038/s41598-022-17299-5)
2. **Martín-Suazo, S., Morón-López, J., Mozo, A., & Ortiz, J. M.** (2024). *Deep learning methods for multi-horizon long-term forecasting of Harmful Algal Blooms.* Knowledge-Based Systems, 301, 112279. [https://doi.org/10.1016/j.knosys.2024.112279](https://doi.org/10.1016/j.knosys.2024.112279)
3. **World Health Organization (WHO)**. (2003). *Guidelines for safe recreational water environments. Volume 1, Coastal and fresh waters.* World Health Organization.
4. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E.** (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
