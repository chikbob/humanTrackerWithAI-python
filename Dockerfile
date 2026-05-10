FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV MONITORING_DB_PATH=/app/data/monitoring.db
ENV BOOTSTRAP_DEMO_DATA=1
ENV DEMO_SEED_EMPLOYEES=120
ENV DEMO_SEED_VISITS=900
ENV DEMO_SEED_VALUE=42
ENV STUN_URLS=stun:stun.l.google.com:19302

RUN mkdir -p /app/data /app/runtime_data

EXPOSE 8501

CMD ["bash", "scripts/start_streamlit.sh"]
