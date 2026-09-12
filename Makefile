# SAGAR DRISHTI developer commands. On Windows without make, use ./dev.ps1 <target>.
PY ?= .venv/bin/python
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
endif
export PYTHONPATH := $(CURDIR)$(if $(filter Windows_NT,$(OS)),;,:)$(CURDIR)/backend
CLI := cd backend && ../$(PY) -m app.cli
COMPOSE := docker compose

.PHONY: help dev up down logs venv install backend frontend worker scheduler migrate \
        load-historical bootstrap ingest forecast evaluate train retrain benchmark reproduce-base \
        test test-ml test-backend test-frontend lint build

help:
	@echo "dev | up | down | logs | install | backend | frontend | worker | scheduler | migrate"
	@echo "load-historical | bootstrap | ingest | forecast | evaluate | train | benchmark VERSION=v1"
	@echo "test | test-ml | test-backend | test-frontend | lint | build"

# --- full stack in docker ------------------------------------------------------
dev:            ## build + start everything (db, redis, migrate, backend, worker, scheduler, frontend)
	$(COMPOSE) up --build
up:
	$(COMPOSE) up -d --build
down:
	$(COMPOSE) down
logs:
	$(COMPOSE) logs -f --tail=200 backend worker scheduler

# --- local processes (db + redis from docker) ----------------------------------
venv:
	python -m venv .venv
install: venv
	$(PY) -m pip install -r backend/requirements-dev.txt
	cd frontend && npm ci
infra:
	$(COMPOSE) up -d db redis
backend: infra
	cd backend && ../$(PY) -m uvicorn app.main:app --reload --port 8000
frontend:
	cd frontend && npm run dev
worker: infra
	cd backend && ../$(PY) -m celery -A app.workers.celery_app worker -Q ingestion,ml --concurrency 2 --loglevel INFO --pool solo
scheduler: infra
	cd backend && ../$(PY) -m celery -A app.workers.celery_app beat --loglevel INFO

migrate:
	cd backend && ../$(PY) -m alembic upgrade head

# --- pipeline operations (synchronous, via CLI) --------------------------------
load-historical:  ## BYU icebergs -> tracking.observations, plus official sea-ice history -> seaice.observations
	$(CLI) load-historical
bootstrap:        ## register models/base and create + deploy v1
	$(CLI) register-base && $(CLI) bootstrap-v1
ingest:           ## fetch the official USNIC CSV now (FILE=path.csv to import a local file)
	$(CLI) ingest $(if $(FILE),--file $(FILE),)
forecast:
	$(CLI) forecast
evaluate:
	$(CLI) evaluate
train retrain:    ## policy-gated retraining (FORCE=1 to ignore eligibility thresholds)
	$(CLI) retrain $(if $(FORCE),--force,)
benchmark:        ## all-horizon benchmark on the fixed historical test split
	$(CLI) benchmark --version $(VERSION)
env-status:       ## environmental sources, credentials configured (yes/no), trainable schemas
	$(CLI) env-status
env-align:
	$(CLI) env-align
env-overlay:
	$(CLI) env-overlay
env-prefetch:     ## pre-warm the environmental cache for a retraining experiment
	$(CLI) env-prefetch
reproduce-base:   ## ONLY if the original artifact is lost: retrain base from the notebook recipe
	$(PY) -m ml.training.reproduce_base --zip data/bootstrap/consolidated_database_v8.0.zip --out models/base-reproduced

# --- quality --------------------------------------------------------------------
test: test-ml test-backend test-frontend
test-ml:
	$(PY) -m pytest ml/tests -q
test-backend: infra
	$(PY) -m pytest backend/tests -q
test-frontend:
	cd frontend && npm test
lint:
	$(PY) -m ruff check ml backend/app backend/tests
	cd frontend && npm run typecheck
build:
	cd frontend && npm run build
