# Build context: repository root (needs both ml/ and backend/).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/srv:/srv/backend \
    TF_CPP_MIN_LOG_LEVEL=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/backend
COPY ml/requirements.txt /srv/ml/requirements.txt
COPY backend/requirements.txt backend/requirements-dev.txt /srv/backend/
RUN pip install -r requirements-dev.txt

COPY ml /srv/ml
COPY backend /srv/backend
COPY pyproject.toml /srv/pyproject.toml

RUN useradd --create-home --uid 1000 sagar && mkdir -p /srv/models /srv/data && chown -R sagar /srv
USER sagar

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
