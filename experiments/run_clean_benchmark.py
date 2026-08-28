"""
===============================================================================
Module Name:       run_clean_benchmark.py
Project:           Chlorophyll-a Soft-Sensor Edge-IoT System
Tier / Subsystem:  Empirical Validation Tier (Benchmark Orchestrator)

Description:       Automates the end-to-end clean benchmarking cycle:
                   1. Flushes Docker volumes and containers (docker compose down -v).
                   2. Builds and starts clean containers (docker compose up -d --build).
                   3. Monitors operational telemetry streaming until all holdout samples
                      are processed.
                   4. Extracts /data/sensor_data.db to experiments/clean_run_data.db.
                   5. Automatically triggers benchmark evaluation suites (01 and 02).

Data Interfaces:
  - Upstream:      Docker engine; edge-system/docker/docker-compose.yml
  - Downstream:    experiments/clean_run_data.db
  - Storage / IPC: Docker volume mount; local filesystem.

References:        Mozo et al. (2022).
===============================================================================
"""

import os
import sys
import time
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
COMPOSE_FILE = os.path.join(PROJECT_ROOT, "edge-system/docker/docker-compose.yml")
CLEAN_DB_DEST = os.path.join(SCRIPT_DIR, "clean_run_data.db")
PYTHON_EXE = sys.executable


def run_command(cmd_list, desc=""):
    if desc:
        print(f"\n>>> {desc}")
    print(f"Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[WARN/ERR] Return code: {res.returncode}")
        print(f"Stdout: {res.stdout}")
        print(f"Stderr: {res.stderr}")
    else:
        if res.stdout:
            print(res.stdout.strip())
    return res


def get_processed_count():
    """Queries SQLite count inside edge_inference_app container."""
    try:
        res = subprocess.run(
            ["docker", "exec", "edge_inference_app", "python", "-c",
             "import sqlite3; conn=sqlite3.connect('/data/sensor_data.db'); print(conn.execute('SELECT COUNT(*) FROM readings').fetchone()[0])"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            return int(res.stdout.strip())
    except Exception:
        pass
    return -1


def main():
    print("=" * 65)
    print("  EDGE-IOT CLEAN BENCHMARK ORCHESTRATOR")
    print("=" * 65)

    # 1. Reset Environment & Flush Volumes
    run_command(["docker", "compose", "-f", COMPOSE_FILE, "down", "-v"],
                "1. Flushing Docker volumes and stopping existing containers...")

    # 2. Build & Launch Clean Containers
    run_command(["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "--build"],
                "2. Building and launching fresh edge containers...")

    # 3. Wait for startup
    print("\n[INFO] [BenchmarkOrchestrator] Waiting 5 seconds for MQTT broker and services to initialize...")
    time.sleep(5)

    # 4. Monitor Stream Progress
    print("\n[INFO] [BenchmarkOrchestrator] 3. Monitoring Operational Stream Progress...")
    total_samples = 21713
    last_count = 0
    stagnant_cycles = 0

    while True:
        count = get_processed_count()
        if count >= 0:
            pct = (count / total_samples) * 100.0 if total_samples > 0 else 0
            print(f"  [INFO] [BenchmarkOrchestrator] [Progress] {count:,} / {total_samples:,} readings processed ({pct:.1f}%)")
            
            if count >= total_samples or (count > 0 and count == last_count and stagnant_cycles >= 6):
                if count == last_count:
                    stagnant_cycles += 1
                if count >= total_samples or stagnant_cycles >= 6:
                    print(f"\n[INFO] [BenchmarkOrchestrator] Stream complete. Processed total of {count:,} samples.")
                    break
            else:
                stagnant_cycles = 0
            last_count = count
        else:
            print("[INFO] [BenchmarkOrchestrator] Waiting for container database initialization...")

        time.sleep(4)

    # 5. Extract Clean SQLite Database
    print(f"\n[INFO] [BenchmarkOrchestrator] 4. Extracting SQLite Database from Container to {CLEAN_DB_DEST}...")
    run_command(["docker", "cp", "edge_inference_app:/data/sensor_data.db", CLEAN_DB_DEST])

    # 6. Trigger Evaluations
    print("\n[INFO] [BenchmarkOrchestrator] 5. Triggering Automated Profiling & Evaluation Suites...")
    script_01 = os.path.join(SCRIPT_DIR, "01_ml_and_alarm_evaluation.py")
    script_02 = os.path.join(SCRIPT_DIR, "02_edge_hardware_profiling.py")

    run_command([PYTHON_EXE, script_01], "Running ML & Alarm Evaluation...")
    run_command([PYTHON_EXE, script_02], "Running Edge Hardware Profiling...")

    print("\n" + "=" * 65)
    print("  [INFO] [BenchmarkOrchestrator] Clean Benchmark Run & Evaluation Complete")
    print("=" * 65)


if __name__ == "__main__":
    main()
