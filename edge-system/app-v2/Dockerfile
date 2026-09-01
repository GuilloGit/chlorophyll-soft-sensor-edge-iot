# --- Stage: Dependencies (cached, rarely changes) ---
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /data

# 1. Install dependencies (heaviest layer, cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code (changes occasionally)
COPY app.py .

# 3. Copy the ML model LAST (changes on every OTA/retrain)
# Balena's delta-updater will only transmit this layer's diff.
COPY model_v2.onnx .
COPY y_lambda.json .

CMD ["python", "-u", "app.py"]