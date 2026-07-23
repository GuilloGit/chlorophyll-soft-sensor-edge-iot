# --- Stage: Dependencies (cached, rarely changes) ---
FROM python:3.11-slim
WORKDIR /app

RUN mkdir -p /data

# 1. Install dependencies (heaviest layer, cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application code (changes occasionally)
COPY app.py .

# 3. Copy the ML model LAST (changes on every OTA/retrain)
# Balena's delta-updater will only transmit this layer's diff.
COPY model.joblib .

CMD ["python", "-u", "app.py"]