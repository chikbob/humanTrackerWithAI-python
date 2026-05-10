FROM node:22-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json frontend/tsconfig.json frontend/tsconfig.app.json frontend/vite.config.ts frontend/index.html ./
COPY frontend/src ./src

RUN npm ci
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

ENV PYTHONUNBUFFERED=1
ENV MONITORING_DB_PATH=/app/data/monitoring.db
ENV BOOTSTRAP_DEMO_DATA=1
ENV DEMO_SEED_EMPLOYEES=120
ENV DEMO_SEED_VISITS=900
ENV DEMO_SEED_VALUE=42
ENV STUN_URLS=stun:stun.l.google.com:19302

RUN mkdir -p /app/data /app/runtime_data

EXPOSE 8000

CMD ["bash", "scripts/start_api.sh"]
