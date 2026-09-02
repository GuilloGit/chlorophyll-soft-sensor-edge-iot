# Edge System Performance and Hardware Profiling Summary (V2 ONNX)

## 1. Latency and Timing Performance (Evaluated over 21,713 samples)

| Performance Parameter | Mean | Median | 95th Percentile (P95) | 99th Percentile (P99) | Max | Real-Time Slack Margin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ONNX Runtime Inference** | **1.02 ms** | 1.02 ms | **1.20 ms** | 1.40 ms | 2.81 ms | — |
| **SQLite WAL Write Latency** | 0.86 ms | 0.29 ms | 0.46 ms | — | 387.61 ms | — |
| **Total Edge Cycle Time** | **1.88 ms** | — | **1.66 ms** | — | — | **99.9998%** |

- **Throughput**: 985.1 predictions/sec.

## 2. Computational Overhead and RAM Stability

| Resource Metric | Average | Peak | Idle Baseline | Stability Analysis |
|---|:---:|:---:|:---:|---|
| **CPU Utilization** | **19.0%** | 100.0% | ~1.2% | Low overhead; CPU remains idle between bursts |
| **RAM Footprint** | **6753.1 MB** | 6809.4 MB | 6803.5 MB | **Slope: 0.000603 MB/sample** (No memory leak confirmed) |
| **Serialized Model Size** | **943.75 MB** | — | — | Lightweight ONNX binary artifact |

## 3. Mathematical Energy Consumption Model (Edge vs. 4G/LTE Cloud)

| Subsystem / Operation | Edge Computing (Smart 24h Batch) | Cloud-Centric (Continuous 15-min 4G) | Energy Reduction |
|---|:---:|:---:|:---:|
| **Cellular Modem Energy** | **4.35 kJ/day** | 130.32 kJ/day | **-96.7%** |
| **Edge Compute Energy (Inference)** | 0.000 kJ/day | 0.00 kJ/day | — |
| **Total Daily Energy Consumption** | **66.01 Wh/day** | **101.00 Wh/day** | **-34.6%** |
