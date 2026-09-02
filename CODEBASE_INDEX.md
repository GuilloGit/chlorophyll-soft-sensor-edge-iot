# Codebase Index & Architectural Reference (`CODEBASE_INDEX.md`)

> **System:** Edge-IoT Soft-Sensor for Real-Time Chlorophyll-a Estimation in Freshwater Reservoirs  
> **Repository:** `chlorophyll-soft-sensor-edge-iot`  
> **Target Audience:** Embedded Systems & Machine Learning Engineers  
> **Status:** Active & Maintained (V1 Scikit-Learn Baseline & V2 ONNX Runtime Engine)

---

## 1. System Architecture & Topology

The system comprises three physical and architectural tiers:
1. **Machine Learning Workstation Tier (`model-training/`)**: Offline data preprocessing, non-shuffled sequential 10-fold cross-validation, Scikit-Learn pipeline fitting, ONNX graph decomposition/chunking (Opset 15), and Over-The-Air (OTA) update publishing.
2. **Edge Processing Tier (`edge-system/`)**: Containerized edge services running on ARM64/AMD64 hardware (Raspberry Pi 4 Model B). Ingests multi-parameter telemetry, executes real-time inference (V1 Scikit-Learn or V2 ONNX Runtime), maintains crash-resilient SQLite WAL storage, aggregates 24-hour daily telemetry batches, and triggers immediate WHO Alert Level 1 hazard alarms.
3. **Cloud Analytics Tier (`cloud-system/`)**: Ingests daily batch telemetry and immediate alarm messages via MQTT to compute rolling soft-sensor fidelity metrics and edge hardware health KPIs.
4. **Empirical Validation Suites (`experiments/` and `experiments-v2/`)**: Automated containerized testbenches evaluating holdout regression, WHO alarm classification, hardware latency/RAM stability distributions, and 5-stage system robustness/OTA resilience.

---

## 2. Directory & Module Navigation Matrix

| Path | Language | Subsystem / Role | Target Runtime | Primary Entry Point / Execution |
|---|:---:|---|---|---|
| `edge-system/app/app.py` | Python 3.11 | V1 Edge Engine (Scikit-Learn + SQLite) | Docker / Pi | `python app.py` (via `docker-compose.yml`) |
| `edge-system/app-v2/app.py` | Python 3.11 | V2 Edge Engine (ONNX Runtime + NumPy) | Docker / Pi | `python app.py` (via `docker-compose-v2.yml`) |
| `edge-system/sensor-sim/sensor_reader.py` | Python 3.11 | SIL Telemetry Streamer & Checkpointer | Docker / Pi | `python sensor_reader.py` |
| `edge-system/docker/docker-compose.yml` | YAML | V1 Stack Orchestrator (Production) | Docker Compose | `docker compose -f docker-compose.yml up -d` |
| `edge-system/docker/docker-compose-v2.yml` | YAML | V2 Stack Orchestrator (Production ONNX) | Docker Compose | `docker compose -f docker-compose-v2.yml up -d` |
| `edge-system/docker/docker-compose.override.yml` | YAML | Development Stack (Code Mounts) | Docker Compose | `docker compose up -d` |
| `edge-system/mosquitto/mosquitto.conf` | Conf | MQTT Broker Configuration | Mosquitto | Loaded by `mqtt_broker` container |
| `cloud-system/analytics_subscriber.py` | Python 3.10+ | Telemetry Consumer & KPI Calculator | PC / Cloud VM | `python analytics_subscriber.py` |
| `model-training/ml_regression_KFold_model_export.py` | Python 3.10+ | 10-Fold CV Training & Model Exporter | ML Workstation | `python ml_regression_KFold_model_export.py` |
| `model-training/export_onnx.py` | Python 3.10+ | 10x10 Tree Chunking ONNX Converter | ML Workstation | `python export_onnx.py` |
| `model-training/model_test.py` | Python 3.10+ | Local Inference QA Sanity Check | ML Workstation | `python model_test.py` |
| `model-training/publish_ota.py` | Python 3.10+ | V1 OTA Update Dispatcher CLI | Workstation / LAN | `python publish_ota.py --url <URL>` |
| `model-training/publish_ota_v2.py` | Python 3.10+ | V2 ONNX OTA Update Dispatcher CLI | Workstation / LAN | `python publish_ota_v2.py --url <URL>` |
| `experiments/run_clean_benchmark.py` | Python 3.10+ | V1 Benchmark Orchestrator | Workstation / Pi | `python run_clean_benchmark.py` |
| `experiments/01_ml_and_alarm_evaluation.py` | Python 3.10+ | V1 Holdout Regression & Alarm Suite | Workstation / Pi | `python 01_ml_and_alarm_evaluation.py` |
| `experiments/02_edge_hardware_profiling.py` | Python 3.10+ | V1 Latency, RAM, & Energy Profiler | Workstation / Pi | `python 02_edge_hardware_profiling.py` |
| `experiments/03_system_robustness_ota_suite.py` | Python 3.10+ | V1 5-Case Automated Resilience Suite | Workstation / Pi | `python 03_system_robustness_ota_suite.py` |
| `experiments-v2/run_clean_benchmark_v2.py` | Python 3.10+ | V2 Benchmark Orchestrator | Workstation / Pi | `python run_clean_benchmark_v2.py` |
| `experiments-v2/01_ml_and_alarm_evaluation_v2.py` | Python 3.10+ | V2 Holdout Regression & Alarm Suite | Workstation / Pi | `python 01_ml_and_alarm_evaluation_v2.py` |
| `experiments-v2/02_edge_hardware_profiling_v2.py` | Python 3.10+ | V2 Latency, RAM, & Energy Profiler | Workstation / Pi | `python 02_edge_hardware_profiling_v2.py` |
| `experiments-v2/03_system_robustness_ota_suite_v2.py` | Python 3.10+ | V2 5-Case Automated Resilience Suite | Workstation / Pi | `python 03_system_robustness_ota_suite_v2.py` |

---

## 3. Subsystem Symbol & Core API Directory

### 3.1 Edge Inference Engines (`edge-system/app/app.py` & `edge-system/app-v2/app.py`)

* `init_database() -> None`: Initializes SQLite schema (`readings` table) with PRAGMA `journal_mode=WAL`, `synchronous=NORMAL`, and creates index `idx_readings_timestamp`.
* `load_initial_model() -> None`: Fallback bootloader. If persistent model (`/data/model.joblib` or `/data/model_v2.onnx`) is absent, copies baked image model. In V2, explicitly configures `rt.SessionOptions(intra_op_num_threads=1, inter_op_num_threads=1)` to prevent OpenMP active waiting CPU spin-locks.
* `_inverse_power_transform(y_transformed: np.ndarray, lmbda: float) -> np.ndarray`: *(V2 pure NumPy)* Inverts the Yeo-Johnson power transform (target unscaling) to project normalized predictions into physical Chlorophyll-a units ($\mu\text{g/L}$). Aliased as `_inverse_yeo_johnson`.
* `store_reading(cursor, conn, data: dict, inference_ms: float, db_write_ms: float) -> None`: Executes atomic parameterized SQL insertion into `readings`.
* `_handle_sensor_reading(payload_str: str) -> None`: Parses incoming `sensor/water/raw` JSON telemetry, extracts physical features, runs thread-safe model inference under `model_lock`, measures latency with `time.perf_counter()`, persists to SQLite, queues 24h batch under `batch_lock`, and evaluates WHO threshold ($\ge 10.0\,\mu\text{g/L}$) to dispatch QoS 2 alarms.
* `_handle_ota_update(client, payload: dict) -> None`: Implements the **Test-Before-Swap** fail-safe state machine:
  1. Downloads artifact to temporary file via HTTP `requests.get`.
  2. Verifies cryptographic SHA-256 hash.
  3. Trial-loads model into isolated variable (`joblib.load` in V1; `rt.InferenceSession` in V2).
  4. Acquires `model_lock` and updates in-memory reference.
  5. Atomically replaces disk file using `os.replace`.
  6. Emits status packet to `buoy/ota/status`.

### 3.2 Telemetry Simulator (`edge-system/sensor-sim/sensor_reader.py`)

* `load_checkpoint() -> int`: Reads last processed row index from `/data/checkpoint.json` with fallback recovery.
* `save_checkpoint(index: int) -> None`: Atomically saves progress using `tempfile.mkstemp`, `os.write`, `os.fsync`, and `os.replace`.
* `connect_mqtt() -> mqtt.Client`: Establishes resilient broker connection with non-blocking reconnection loops.
* `main() -> None`: Reads chronological CSV observations, compresses sampling intervals via `TIME_SCALE_FACTOR` ($T_{\text{effective}} = T_{\text{nominal}} / 900$), formats telemetry JSON payload, and publishes over MQTT QoS 1.

### 3.3 Model Training & Asset Export (`model-training/`)

* `ml_regression_KFold_model_export.py`:
  - Enforces sequential non-shuffled 10-fold cross-validation (`KFold(n_splits=10, shuffle=False)`).
  - Fits Scikit-Learn pipeline: `PowerTransformer` (Yeo-Johnson) for $X$, `RandomForestRegressor` (100 trees, random state 22), and `TransformedTargetRegressor` for $y$.
  - Exports `model.joblib` (compress=3), `model_metrics.json`, and `simulation_test_data.csv`.
* `export_onnx.py`:
  - Decomposes `model.joblib` into base scaler and 100-tree Random Forest.
  - Splits ensemble into 10 chunks of 10 trees, scales tree weights by $0.1$, and inserts an ONNX `Sum` node to satisfy ONNX Opset 15 and avoid 2GB serialization / protobuf limits.
  - Extracts learned target power transform parameter $\lambda$ to `y_lambda.json`.
* `publish_ota.py` / `publish_ota_v2.py`:
  - CLI utilities computing `hashlib.sha256()` on the target model binary and publishing `{url, sha256, version}` to `buoy/ota/update` (QoS 1).

### 3.4 Cloud Analytics (`cloud-system/analytics_subscriber.py`)

* `on_message(client, userdata, msg) -> None`: Demuxes `sensor/water/daily_batch` and `sensor/water/alarm`.
* `generate_report() -> None`: Computes cumulative MAE, $R^2$, latency percentiles, average/peak CPU %, average/peak RAM MB, and total alarm count, atomically saving to `edge_performance_report.json`.

---

## 4. Communication Contracts & Data Schemas

### 4.1 MQTT Message Contracts

| Topic | Direction | Publisher | Subscriber | QoS | Payload Schema & Format |
|---|---|---|---|:---:|---|
| `sensor/water/raw` | Internal Edge | `edge_sensor_sim` | `edge_inference_app` | 1 | JSON Object: `{"timestamp": str, "features": {"EXO3(Temp_C)": float, "EXO3(spCond_uS_cm)": float, "EXO3(pH)": float, "SystemBattery": float}, "ground_truth": float}` |
| `sensor/water/daily_batch` | Edge $\to$ WAN | `edge_inference_app` | `analytics_subscriber` | 1 | JSON Array of 96 objects: `[{"timestamp": str, "predicted_chlorophyll": float, "actual_chlorophyll": float, "alarm": int, "inference_ms": float, "db_write_ms": float, "cpu_percent": float, "ram_mb": float}, ...]` |
| `sensor/water/alarm` | Edge $\to$ WAN | `edge_inference_app` | `analytics_subscriber` | 2 | JSON Object: `{"timestamp": str, "predicted_chlorophyll": float, "threshold": 10.0, "level": "WHO Alert Level 1"}` |
| `buoy/ota/update` | WAN $\to$ Edge | `publish_ota.py` | `edge_inference_app` | 1 | JSON Object: `{"url": str, "sha256": str, "version": str}` |
| `buoy/ota/status` | Edge $\to$ WAN | `edge_inference_app` | Operator / Cloud | 1 | JSON Object: `{"version": str, "status": "success"|"failed", "message": str, "timestamp": str}` |

### 4.2 SQLite Database Schema (`/data/sensor_data.db`)

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    temp_c REAL NOT NULL,
    spcond_us_cm REAL NOT NULL,
    ph REAL NOT NULL,
    battery REAL NOT NULL,
    chlorophyll_predicted REAL NOT NULL,
    chlorophyll_actual REAL NOT NULL,
    alarm_triggered INTEGER NOT NULL,
    inference_ms REAL NOT NULL,
    db_write_ms REAL NOT NULL,
    processed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings (timestamp);
```

---

## 5. Configuration & Environment Variables

| Variable | Target Service | Default Value | Role & Architectural Impact |
|---|---|---|---|
| `MQTT_BROKER` | `sensor-sim`, `app`, `app-v2`, `analytics` | `mqtt_broker` (or `localhost`) | Hostname / IP address of Mosquitto broker |
| `MQTT_PORT` | All MQTT services | `1883` | TCP port for MQTT communications |
| `REAL_INTERVAL_SEC` | `sensor-sim` | `900` | Physical sensor sampling cadence (15 minutes = 900 s) |
| `TIME_SCALE_FACTOR` | `sensor-sim` | `900` | Compression factor ($900\times \implies 1.0\text{ s}$ per telemetry row) |
| `DATASET_PATH` | `sensor-sim` | `/app/data/simulation_test_data.csv` | Path to chronological holdout simulation dataset |
| `CHECKPOINT_PATH` | `sensor-sim` | `/data/checkpoint.json` | Persistent volume path for crash-safe resume index |
| `MODEL_PATH` | `app` / `app-v2` | `model.joblib` / `model_v2.onnx` | Active model path on persistent Docker volume (`/data`) |
| `BAKED_MODEL_PATH` | `app` / `app-v2` | `model.joblib` / `model_v2.onnx` | Fallback model image path inside container image (`/app`) |
| `LAMBDA_PATH` | `app-v2` | `y_lambda.json` | Learned power transform parameter path on persistent volume |
| `DB_PATH` | `app` / `app-v2` | `/data/sensor_data.db` | Persistent SQLite database file location |
| `BATCH_SIZE` | `app` / `app-v2` | `96` | Aggregation window size ($96 \times 15\text{ min} = 24\text{ hours}$) |

---

## 6. Developer Workflow Cheatsheet

```bash
# ==============================================================================
# 1. PC ML Workstation: Model Training & ONNX Export
# ==============================================================================
cd model-training
python ml_regression_KFold_model_export.py  # Trains RF, exports model.joblib & CSV
python export_onnx.py                      # Converts model.joblib -> model_v2.onnx
python model_test.py                       # Validates local inference execution

# ==============================================================================
# 2. Remote Over-The-Air (OTA) Model Deployment
# ==============================================================================
# Dispatch V1 Joblib model update
python publish_ota.py --url https://example.com/model.joblib --version 1.1.0 --broker <PI_IP>

# Dispatch V2 ONNX model update
python publish_ota_v2.py --url https://example.com/model_v2.onnx --version 2.0.0 --broker <PI_IP>

# ==============================================================================
# 3. Edge Execution on Raspberry Pi 4 (Docker Compose)
# ==============================================================================
cd edge-system/docker

# Run V1 Baseline Stack (Scikit-Learn)
docker compose -f docker-compose.yml up -d --build

# Run V2 Optimized Stack (Decoupled ONNX Runtime)
docker compose -f docker-compose-v2.yml up -d --build

# Inspect aggregated real-time logs
docker compose -f docker-compose-v2.yml logs -f

# Clean volume teardown (resets database and checkpoints)
docker compose -f docker-compose-v2.yml down -v

# ==============================================================================
# 4. Cloud Analytics Monitoring
# ==============================================================================
cd cloud-system
python analytics_subscriber.py             # Ingests daily_batch and alarm topics

# ==============================================================================
# 5. Automated Experimental Benchmarking Suites
# ==============================================================================
# Execute V1 Baseline Validation Suite
python experiments/01_ml_and_alarm_evaluation.py
python experiments/02_edge_hardware_profiling.py
python experiments/03_system_robustness_ota_suite.py

# Execute V2 ONNX Validation Suite
python experiments-v2/01_ml_and_alarm_evaluation_v2.py
python experiments-v2/02_edge_hardware_profiling_v2.py
python experiments-v2/03_system_robustness_ota_suite_v2.py
```
