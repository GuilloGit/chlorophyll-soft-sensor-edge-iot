# Edge Processing Tier (`edge-system`)

The **Edge Processing Tier** is the computational core of the Chlorophyll-a Soft-Sensor system. Designed for containerized deployment on ARM64 single-board computers (specifically the **Raspberry Pi 4 Model B**), this subsystem ingests real-time physicochemical telemetry, performs local machine learning inference, stores high-frequency observations in a crash-resilient SQLite database, aggregates daily batches to minimize cellular transmission energy, dispatches immediate hazard alarms, and manages remote Over-The-Air (OTA) model hot-swapping.

The tier supports two side-by-side execution architectures:
1. **V1 Baseline Architecture (`app/`, `docker-compose.yml`)**: Deploys a full Scikit-Learn pipeline serialized with Joblib. Serves as the baseline reference implementation.
2. **V2 Decoupled Architecture (`app-v2/`, `docker-compose-v2.yml`)**: Deploys an ultra-lightweight C++ ONNX Runtime engine with pure NumPy inverse power unscaling, eliminating heavy Scikit-Learn/Pandas dependencies, slashing RAM by ~97%, and accelerating inference by 27.2x.

```mermaid
graph TD
    subgraph "Docker Bridge Network (edge_net)"
        MOSQ["mosquitto_edge\n(Eclipse Mosquitto 2.x :1883)"]
        SIM["edge_sensor_sim\n(SIL Telemetry Streamer)"]
        APP1["V1: edge_inference_app\n(Scikit-Learn + SQLite WAL)"]
        APP2["V2: edge_inference_app_v2\n(ONNX Runtime + NumPy + SQLite WAL)"]

        SIM -->|"PUB sensor/water/raw (QoS 1)"| MOSQ
        MOSQ -->|"SUB sensor/water/raw (QoS 1)"| APP1
        MOSQ -->|"SUB sensor/water/raw (QoS 1)"| APP2
        MOSQ -->|"SUB buoy/ota/update (QoS 1)"| APP1
        MOSQ -->|"SUB buoy/ota/update (QoS 1)"| APP2
        APP1 -->|"PUB buoy/ota/status (QoS 1)"| MOSQ
        APP2 -->|"PUB buoy/ota/status (QoS 1)"| MOSQ
    end

    subgraph "Persistent Docker Volumes"
        V_DATA[("sensor_data\n/data/sensor_data.db\n/data/checkpoint.json\n/data/model.joblib or model_v2.onnx")]
        V_MOSQ[("mosquitto_data\nmosquitto_logs")]
    end

    APP1 --- V_DATA
    APP2 --- V_DATA
    SIM --- V_DATA
    MOSQ --- V_MOSQ

    APP1 -->|"WAN: sensor/water/daily_batch (QoS 1)"| CLOUD["Cloud Analytics Tier"]
    APP1 -->|"WAN: sensor/water/alarm (QoS 2)"| CLOUD
    APP2 -->|"WAN: sensor/water/daily_batch (QoS 1)"| CLOUD
    APP2 -->|"WAN: sensor/water/alarm (QoS 2)"| CLOUD
```

---

## Container Topology & Microservices

| Container Service | Base Image | Role & Responsibility | Volume Mounts | Architecture |
|---|---|---|---|:---:|
| `mosquitto_edge` | `eclipse-mosquitto:2` | Local MQTT message broker facilitating inter-process communication (IPC) between containers and WAN telemetry uplinks. | `mosquitto_data:/mosquitto/data`<br/>`mosquitto_logs:/mosquitto/log` | Shared (V1 & V2) |
| `edge_sensor_sim` | `python:3.11-slim` | Software-in-the-Loop (SIL) telemetry simulator feeding chronological historical observations into the broker at a configurable time-scaled interval. | `sensor_data:/data` | Shared (V1 & V2) |
| `edge_inference_app` | `python:3.11-slim` | **V1 Baseline Engine**: Scikit-Learn Random Forest pipeline (`model.joblib`), SQLite WAL persistence, 24h batching, and Joblib OTA updates. | `sensor_data:/data` | V1 Baseline |
| `edge_inference_app_v2` | `python:3.11-slim` | **V2 Optimized Engine**: C++ ONNX Runtime engine (`model_v2.onnx`), pure NumPy inverse power transform (`target_transform.json`), zero-spin single-threaded session options, and ONNX OTA updates. | `sensor_data:/data` | V2 Production |

---

## Architectural Comparison: V1 Baseline vs. V2 ONNX Runtime

The table below summarizes the empirical profile measured across 21,638 continuous observations on a physical Raspberry Pi 4 Model B (8 GB RAM):

| Performance Parameter | V1 Baseline (`app`) | V2 Optimized (`app-v2`) | Impact / Improvement |
|---|:---:|:---:|---|
| **Inference Engine** | Scikit-Learn 1.3+ / Joblib | ONNX Runtime 1.17+ / C++ Engine | Decoupled runtime architecture |
| **Dependencies on Edge** | `scikit-learn`, `pandas`, `joblib` | `onnxruntime`, `numpy` | Strips ~500 MB of Python wheels |
| **Target Unscaling** | `TransformedTargetRegressor` | `_inverse_power_transform` (NumPy) | Pure mathematical unscaling on edge |
| **Inference Latency (Mean)** | 27.74 ms | 1.02 ms | **27.2x faster execution** |
| **Inference Throughput** | 36.0 predictions/sec | 985.1 predictions/sec | **27.4x higher capacity** |
| **Real-Time Slack Margin** | 99.9966% (of 15 min) | 99.9998% (of 15 min) | Near-instantaneous execution |
| **Edge RAM Consumption** | ~2,415 MB (peak ~2,600 MB) | ~75–80 MB | **~97% reduction in memory footprint** |
| **Docker Image Size** | ~1,200 MB | ~280 MB | **76.7% smaller storage footprint** |
| **OTA Update Payload** | ~415 MB uncompressed | ~535 KB (`model_v2.onnx`) | **>99% smaller transmission payload** |
| **OpenMP CPU Idle Behavior**| Default (idle ~0.3%) | Single-threaded (idle ~0.0%) | **Eliminates 130% CPU spin-lock** |

---

## Deployment & Execution Workflows

All deployment configurations reside in `edge-system/docker/`.

### 1. V2 Production Mode (Recommended: ONNX Decoupled Stack)
To run the high-efficiency V2 production stack:
```bash
cd edge-system/docker

# Build and start V2 services in detached mode
docker compose -f docker-compose-v2.yml up -d --build

# Follow aggregated logs
docker compose -f docker-compose-v2.yml logs -f

# Follow inference engine logs specifically
docker compose -f docker-compose-v2.yml logs -f edge_inference_app_v2
```

### 2. V1 Production Mode (Baseline Scikit-Learn Stack)
To run the baseline V1 stack for comparative benchmarking:
```bash
cd edge-system/docker

# Build and start all services in the background
docker compose -f docker-compose.yml up -d --build

# Follow V1 logs
docker compose -f docker-compose.yml logs -f edge_inference_app
```

### 3. Development Mode (Hot-Reload with Local Code Mounts)
For rapid local iteration, run without specifying `-f`. Docker Compose merges `docker-compose.yml` and `docker-compose.override.yml`, bind-mounting source directories:
```bash
cd edge-system/docker
docker compose up -d --build
```

### 4. Container Teardown & Reset Commands
```bash
# Stop containers without removing persistent data
docker compose -f docker-compose-v2.yml stop

# Stop containers and remove virtual bridge networks (preserves volumes)
docker compose -f docker-compose-v2.yml down

# Clean reset: purges containers, networks, and named volumes (sensor_data)
docker compose -f docker-compose-v2.yml down -v
```

---

## MQTT Communication Contracts & Topic Schemas

### 1. Topic: `sensor/water/raw`
* **Publisher**: `edge_sensor_sim` (or physical water sonde logger)
* **Subscriber**: `edge_inference_app` / `edge_inference_app_v2`
* **QoS**: 1 (At-least-once delivery)
* **Description**: Raw streaming physicochemical observation from the water sonde.

```json
{
  "timestamp": "2026-09-03T10:15:00Z",
  "features": {
    "EXO3(Temp_C)": 21.86,
    "EXO3(spCond_uS_cm)": 71.66,
    "EXO3(pH)": 7.51,
    "SystemBattery": 13.32
  },
  "ground_truth": 0.81
}
```

| Field | Type | Units | Description |
|---|---|:---:|---|
| `timestamp` | ISO-8601 string | UTC | Observation timestamp |
| `features.EXO3(Temp_C)` | float | °C | Surface water temperature |
| `features.EXO3(spCond_uS_cm)` | float | µS/cm | Specific electrical conductivity |
| `features.EXO3(pH)` | float | pH units | Water pH (acidity / alkalinity) |
| `features.SystemBattery` | float | Volts (V) | Buoy battery voltage (proxy for solar irradiance) |
| `ground_truth` | float | µg/L | Reference optical measurement (used for real-time validation) |

---

### 2. Topic: `sensor/water/daily_batch`
* **Publisher**: `edge_inference_app` / `edge_inference_app_v2`
* **Subscriber**: `cloud-system/analytics_subscriber.py`
* **QoS**: 1 (At-least-once delivery)
* **Description**: Array of 96 consolidated readings collected over a 24-hour monitoring period ($96 \times 15\,\text{min} = 24\,\text{hours}$).

```json
[
  {
    "timestamp": "2026-09-03T10:15:00Z",
    "predicted_chlorophyll": 0.899,
    "actual_chlorophyll": 0.810,
    "alarm": false,
    "inference_ms": 1.05,
    "db_write_ms": 1.85,
    "cpu_percent": 0.0,
    "ram_mb": 76.4
  }
]
```

---

### 3. Topic: `sensor/water/alarm`
* **Publisher**: `edge_inference_app` / `edge_inference_app_v2`
* **Subscriber**: Cloud / Alert Management Systems
* **QoS**: 2 (Exactly-once delivery)
* **Description**: Immediate event-driven emergency notification published the instant inferred Chlorophyll-a exceeds the WHO Alert Level 1 threshold ($\ge 10.0\,\mu\text{g/L}$).

```json
{
  "timestamp": "2026-09-03T14:30:00Z",
  "predicted_chlorophyll": 14.235,
  "threshold": 10.0,
  "level": "WHO_LEVEL_1"
}
```

---

### 4. Topic: `buoy/ota/update`
* **Publisher**: `model-training/publish_ota.py` (V1) or `model-training/publish_ota_v2.py` (V2)
* **Subscriber**: `edge_inference_app` / `edge_inference_app_v2`
* **QoS**: 1 (At-least-once delivery)
* **Description**: Remote command instructing the edge device to retrieve and hot-swap an updated model.

```json
{
  "url": "https://github.com/org/repo/releases/download/v2.0.0/model_v2.onnx",
  "sha256": "3a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b",
  "version": "2.0.0"
}
```

---

### 5. Topic: `buoy/ota/status`
* **Publisher**: `edge_inference_app` / `edge_inference_app_v2`
* **Subscriber**: Monitoring Console / DevOps
* **QoS**: 1 (At-least-once delivery)
* **Description**: Confirmation emitted upon conclusion of the Test-Before-Swap update process.

```json
{
  "version": "2.0.0",
  "status": "success",
  "message": "Model updated",
  "timestamp": "2026-09-03T15:00:12"
}
```

---

## Embedded SQLite Database & WAL Mode

Edge observations and resource metrics are persisted in `/data/sensor_data.db`.

### Data Definition Language (DDL)

```sql
-- High-frequency telemetry observations and model predictions
CREATE TABLE IF NOT EXISTS readings (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    temp_c                  REAL,
    spcond_us_cm            REAL,
    ph                      REAL,
    battery                 REAL,
    chlorophyll_predicted   REAL,
    chlorophyll_actual      REAL,
    alarm_triggered         INTEGER DEFAULT 0,
    inference_ms            REAL,
    db_write_ms             REAL,
    processed_at            TEXT    DEFAULT (datetime('now'))
);

-- Edge computational footprint time-series
CREATE TABLE IF NOT EXISTS system_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    cpu_percent             REAL,
    ram_mb                  REAL
);

CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON readings (timestamp);
```

### Write-Ahead Logging (WAL) Configuration
On container initialization, the database executes:
```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```
* **WAL Mode**: Appends transactions to a `.db-wal` write log rather than rewriting database pages in place. Readers and writers operate concurrently without database locking contention.
* **Synchronous = NORMAL**: Commits data safely to disk at checkpoints without forcing immediate disk cache flushes on every single insert, dramatically reducing SD card / eMMC flash write wear.
* **Crash Resilience**: If power is cut mid-transaction, SQLite's recovery automatically rolls back incomplete WAL frames upon reboot, guaranteeing zero table corruption.

---

## Environment Variable Reference

### `edge_inference_app` (V1 Baseline) & `edge_inference_app_v2` (V2 Production)
| Variable | Default Value | Target Engine | Description |
|---|:---:|:---:|---|
| `MQTT_BROKER` | `mqtt_broker` | V1 & V2 | Hostname or IP of the MQTT broker |
| `MQTT_PORT` | `1883` | V1 & V2 | MQTT broker port |
| `MODEL_PATH` | `model.joblib` / `model_v2.onnx` | V1 / V2 | Active runtime model path on persistent volume |
| `BAKED_MODEL_PATH` | `model.joblib` / `model_v2.onnx` | V1 / V2 | Dockerfile-baked fallback model inside image |
| `TRANSFORM_PATH` | `target_transform.json` | V2 | JSON file containing learned target transform parameters ($\lambda, \mu, \sigma$; supports `UNSCALER_PATH` / `LAMBDA_PATH` alias) |
| `DB_PATH` | `/data/sensor_data.db` | V1 & V2 | Path to persistent SQLite database |
| `BATCH_SIZE` | `96` | V1 & V2 | Consolidated daily batch size ($96 \times 15\,\text{min} = 24\,\text{h}$) |

### `edge_sensor_sim`
| Variable | Default Value | Description |
|---|:---:|---|
| `MQTT_BROKER` | `mqtt_broker` | Hostname or IP of the MQTT broker |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `DATASET_PATH` | `/app/data/simulation_test_data.csv` | Chronological holdout CSV dataset |
| `CHECKPOINT_PATH` | `/data/checkpoint.json` | Persistent state file tracking streaming index |
| `REAL_INTERVAL_SEC` | `900` | Real-world sampling interval ($900\,\text{s} = 15\,\text{min}$) |
| `TIME_SCALE_FACTOR` | `900` | Acceleration factor ($900\times$ equates to $1.0\,\text{s}$ effective per row) |

---

## Test-Before-Swap Over-The-Air (OTA) Architecture

To eliminate the risk of deploying corrupted or incompatible machine learning models over remote cellular links, both inference engines implement the **Test-Before-Swap** fail-safe protocol:

```
1. Download Remote Asset     --> Saved to temporary staging path (/data/model.tmp)
2. Verify Cryptographic Hash --> SHA-256 computed on bytes; compared to expected hash
                                (If mismatch: abort, delete temp file, report failure)
3. In-Memory Trial-Load      --> Trial deserialization in isolated dummy variable:
                                - V1: joblib.load(temp_path)
                                - V2: rt.InferenceSession(temp_path, sess_options=...)
                                (Catches corrupt bytecode, schema mismatches, or missing symbols)
4. Thread-Safe Hot-Swap      --> Acquire model_lock; swap active in-memory reference
                                (Zero-downtime: ongoing inference queries proceed without restarts)
5. Atomic Disk Persistence   --> os.replace(temp_path, final_path)
                                (Guarantees reboot survival via atomic filesystem inode replacement)
```
