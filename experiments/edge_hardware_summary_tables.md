# Edge System Performance and Hardware Profiling Summary

## 1. Latency and Timing Performance (Evaluated over 21,638 samples)

| Performance Parameter | Mean | Median | 95th Percentile (P95) | 99th Percentile (P99) | Max | Real-Time Slack Margin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest Inference** | **27.74 ms** | 27.13 ms | **30.41 ms** | 40.83 ms | 721.72 ms | — |
| **SQLite WAL Write Latency** | 2.42 ms | 2.23 ms | 4.31 ms | — | 12.00 ms | — |
| **Total Edge Cycle Time** | **30.16 ms** | — | **34.72 ms** | — | — | **99.9966%** |

- **Throughput**: 36.0 predictions/sec.
- **Timing Constraint**: The 15-minute ($900,000$ ms) physical sampling window is exceeded by less than $0.02\%$ of compute time, confirming complete real-time edge feasibility.

## 2. Computational Overhead and RAM Stability

| Resource Metric | Average | Peak | Idle Baseline | Stability Analysis |
|---|:---:|:---:|:---:|---|
| **CPU Utilization** | **0.3%** | 11.5% | ~1.2% | Low overhead; CPU remains idle between bursts |
| **RAM Footprint** | **2415.1 MB** | 2491.3 MB | 2428.6 MB | **Slope: -0.002290 MB/sample** (No memory leak confirmed) |
| **Serialized Model Size** | **415.29 MB** | — | — | Lightweight compressed Joblib artifact |

## 3. Mathematical Energy Consumption Model (Edge vs. 4G/LTE Cloud)

| Subsystem / Operation | Edge Computing (Smart 24h Batch) | Cloud-Centric (Continuous 15-min 4G) | Energy Reduction |
|---|:---:|:---:|:---:|
| **Cellular Modem Energy** | **4.35 kJ/day** | 130.32 kJ/day | **-96.7%** |
| **Edge Compute Energy (Inference)** | 0.011 kJ/day | 0.00 kJ/day | — |
| **Total Daily Energy Consumption** | **66.01 Wh/day** | **101.00 Wh/day** | **-34.6%** |

- **Battery Autonomy**: Proves that edge inference and 24-hour batching extend remote solar buoy battery lifespan by avoiding repetitive cellular network attach cycles.
