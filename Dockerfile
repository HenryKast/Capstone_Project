# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

COPY env/requirements.txt env/requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
# Shipped artifacts so the API serves without retraining / ESPN cookies
COPY data/output/csv ./data/output/csv
COPY data/output/models ./data/output/models

RUN pip install --upgrade pip \
    && pip install -r env/requirements.txt \
    && pip install -e .

EXPOSE 8000
# Cloud hosts (Railway/Render/Fly) inject $PORT
CMD ["sh", "-c", "uvicorn rookie_ppr.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
