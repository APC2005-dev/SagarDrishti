# SAGAR DRISHTI

Antarctic iceberg monitoring and 7-day trajectory forecasting.

Official iceberg positions from the **U.S. National Ice Center (USNIC) Antarctic
Iceberg CSV** are ingested automatically, stored permanently in PostGIS, turned
into 14-entry sequences, and forecast with the **existing research GRU** (D+1 …
D+7 in one inference). Forecasts are scored against later official observations,
and those scores drive policy-gated retraining with champion/challenger promotion
and full lineage.

```
USNIC CSV ─► ingestion (validate, dedup on iceberg+Last Update) ─► tracking.* (PostGIS, permanent)
   ─► ProductionSequenceAdapter (14 chronological entries, km/day by elapsed time)
   ─► GRU vN (one pass → D+1…D+7) ─► ml.forecasts (append-only, model-versioned)
   ─► next official observation ─► ml.forecast_evaluations (official ground truth only)
   ─► retraining (eligibility policy) ─► candidate vN+1 ─► champion/challenger ─► promote | reject
   ─► React + Three.js console (1/3/7-day views filter the same run)
```

---

## Contents

1. [Repository layout](#repository-layout)
2. [Quick start](#quick-start)
3. [Database](#database)
4. [Ingestion](#ingestion)
5. [ML pipeline](#ml-pipeline)
6. [Model versioning & retraining](#model-versioning--retraining)
7. [API](#api)
8. [Frontend](#frontend)
9. [Testing](#testing)
10. [Deployment & configuration](#deployment--configuration)
11. [Design decisions and known limitations](#design-decisions-and-known-limitations)

---

## Repository layout

```
backend/            FastAPI app, SQLAlchemy/GeoAlchemy models, Alembic, Celery jobs, services, tests
  app/api/v1/       HTTP routers (icebergs, forecasts, models, operations, feeds, overview, health)
  app/services/     ingestion, forecast, evaluation, model registry, retraining, historical loader, USNIC client/parser
  app/repositories/ async read queries used by the API
  app/workers/      Celery app + beat schedule + jobs
  app/cli.py        operator CLI (same services, synchronous)
  alembic/          migrations (schemas, tables, indexes, constraints, append-only triggers)
ml/                 pure ML package — no DB or web imports
  adapters/         ProductionSequenceAdapter, BYU dataset loader, entry types
  features/         EPSG:4326↔3031 transforms, seasonal encoding, 14×6 feature matrix
  models/           architecture registry, bundle loader/validator, immutable artifact store
  training/         dataset builder + chronological splits, trainer, retrainer, reproduce_base
  evaluation/       metrics, evaluator, champion/challenger policy
  inference/        predictor, forecast generator
  versioning/       vN numbering
frontend/           React 19 + Vite + TypeScript + three/@react-three/fiber + Framer Motion + TanStack Query + Zustand
models/             base/ (original artifact), v1/ … vN/ (immutable)
data/               bootstrap/ (BYU dataset + MANIFEST), raw/ (archived USNIC fetches), processed/
notebooks/          01_gru_iceberg_trajectory_model.ipynb — research notebook for the base GRU (not copied per version)
```

Orchestration that touches the database lives in `backend/app/services`; `ml/`
stays pure so every ML step is unit-testable without infrastructure.

## Quick start

Prerequisites: Docker Desktop, Python 3.11+ (3.13 works locally), Node 20+.

```bash
cp .env.example .env                  # set POSTGRES_PASSWORD, SECRET_KEY, DATABASE_URL
make install                          # venv + pip + npm      (Windows: ./dev.ps1 install)
make infra migrate                    # PostGIS + Redis, then Alembic
make load-historical                  # BYU v8.0 -> tracking.observations (historical_training_dataset)
# copy global_gru_trajectory_model.keras + global_gru_trajectory_scalers.joblib into models/base/
make bootstrap                        # register base, create + deploy v1
make ingest                           # fetch the live USNIC CSV
make forecast                         # (also triggered automatically after every ingestion)
make backend    # :8000  (OpenAPI at http://localhost:8000/docs)
make frontend   # :5173
make worker scheduler                 # background jobs
```

Or run everything in containers: `make dev` (db, redis, migrate, backend, worker,
scheduler, frontend).

**Windows without `make`:** the same targets are available as `dev <target> [<target> ...]`
from Command Prompt (`dev.cmd`) or `.\dev.ps1 <target> ...` from PowerShell, e.g.
`dev infra migrate`, `dev backend`, `dev ingest -File x.csv`, `dev benchmark -Version v1`.
Run `dev` with no arguments for the list.

## Database

PostgreSQL 16 + PostGIS 3.4, two schemas:

| schema | tables | purpose |
| --- | --- | --- |
| `tracking` | `icebergs`, `observations`, `observation_revisions`, `ingestion_runs`, `ingestion_row_errors` | official + historical observed data, identity, ingestion audit |
| `ml` | `model_versions`, `model_status_events`, `model_metrics`, `forecast_runs`, `forecast_sets`, `forecasts`, `forecast_evaluations`, `retraining_runs` | registry, predictions, evaluations, lineage |

* **SRID strategy**: every stored geometry is `geometry(Point, 4326)` with a GIST
  index; lat/lon are also stored as plain columns. Model math is in EPSG:3031 and
  is done explicitly in `ml/features/coordinate_transform.py` (pyproj). Forecasts
  store both WGS84 and the EPSG:3031 metres they were computed in (`predicted_x_m/_y_m`).
* **Provenance** column on observations: `official_usnic`,
  `historical_training_dataset`, `derived`, `interpolated` (check-constrained;
  `predicted` is rejected — predictions only exist in `ml.forecasts`, whose
  provenance is constrained to `predicted`).
* **Uniqueness**: `(iceberg_id, observation_date, provenance)` on observations;
  `(iceberg_id, model_version, anchor_observation_id)` on forecast sets;
  `(forecast_set_id, horizon)` on forecasts; one `deployed` model (partial unique index).
* **Permanence enforced by triggers** (not just convention): UPDATE/DELETE on
  `ml.forecasts`, `ml.forecast_sets`, `ml.forecast_evaluations`,
  `ml.model_status_events`, `tracking.observation_revisions` and DELETE on
  `tracking.observations` / `ml.model_versions` raise an error. A model version's
  identity (version, parent, paths, sha256, cutoff) cannot change; only its status can.
* Source corrections (USNIC re-issues a row for the same iceberg + date with
  different values) update the observation in place **after** archiving the old
  values into `observation_revisions` and incrementing `revision`.

Migrations: `make migrate`; `alembic check` verifies the ORM and migrations agree.

## Ingestion

`app/services/ingestion_service.py`, scheduled every `INGESTION_INTERVAL_HOURS`
(default 72) by Celery beat, or `make ingest`.

1. **Source**: `USNIC_CSV_URL` (default `https://usicecenter.gov/File/DownloadCurrent?pId=134`).
   If it fails or returns a non-CSV, the product page (`USNIC_PRODUCT_PAGE_URL`)
   is scraped for its current CSV link; the discovery method is recorded.
2. **Response validation**: HTTP 200, size limit, content type, not HTML, CSV header.
3. **Raw archive**: every payload is saved to `data/raw/usnic/YYYY/MM/<ts>_<sha>.csv`; SHA-256 stored on the run.
4. **Schema validation**: columns are matched by alias (`Iceberg`, `Latitude`,
   `Longitude`, `Last Update` required; lengths/areas/remarks optional) — a
   missing required column fails the whole run.
5. **Row validation**: designator normalised (`A-23A` → `A23A`), Antarctic
   latitude range, longitude range, `Last Update` as `MM/DD/YYYY` and not in the
   future, non-negative numerics. Bad rows are rejected individually and stored
   in `ingestion_row_errors`; conflicting duplicates within a file are rejected.
6. **Deduplication**: key = `(iceberg_id, Last Update)`. Re-downloading an
   unchanged file inserts nothing (`duplicate_observations` counted, status
   `unchanged`). `observation_date` = USNIC *Last Update*; `fetched_at` = our download time — both stored.
7. **Icebergs absent** from the latest file become `not_in_latest_source`; their
   history is untouched and no reason is invented.
8. On success the chain `forecast_evaluation_job → forecast_generation_job` runs.

## ML pipeline

### Base model (`models/base`)

From `notebooks/01_gru_iceberg_trajectory_model.ipynb` (sections 2–8: cleaning,
sequences, chronological split, training, p90 evaluation for D+1…D+7, export).
Architecture as trained (**note:** includes a
`LayerNormalization` layer that the written brief omits — the artifact is
authoritative, see `ml/models/gru_architecture.py`; `build_model()` reproduces its
44,366 parameters exactly):

```
Input(14,6) → GRU(96, dropout .10) → LayerNormalization → Dense(128, relu) → Dropout(.15) → Dense(14)
```

Features: `relative_x_km, relative_y_km, vx_km_per_day, vy_km_per_day, season_sin, season_cos`
(EPSG:3031, relative to the last entry). Targets: km displacement from the last
entry for D+1…D+7 (x,y). Scalers: `StandardScaler` fitted on the chronological
training split only. Test (2022-03-10 → 2026-04-23, 59,732 windows): mean error
1.255 / 3.637 / 10.834 km at D+1/3/7 vs 1.620 / 5.715 / 17.038 km for constant
velocity.

### ProductionSequenceAdapter (`ml/adapters/production_adapter.py`)

The research model saw 14 **consecutive days**; USNIC is **weekly**. The adapter
keeps the tensor at 14 entries but defines them as *14 chronological
observation entries*:

* anchor (last entry) = latest official USNIC observation; forecasts are anchor date + h days;
* strategy `bootstrap_v1`: fill earlier slots with the most recent BYU historical
  positions dated before the earliest official entry used; as weekly
  observations accumulate they push historical entries out automatically;
* strategy `official_only`: require 14 official entries (for future versions);
* velocity = displacement / **actual elapsed days** (daily data → identical to the notebook);
* only `official_usnic` and `historical_training_dataset` entries are eligible —
  never predicted/interpolated; no interpolation fills gaps;
* insufficient history / gaps > `ADAPTER_MAX_GAP_DAYS` → iceberg skipped with a reason;
* every forecast set stores the 14 input observation ids, dates, provenance,
  elapsed days, the raw feature tensor and diagnostics (official vs historical
  count, gap stats, `daily_cadence`) — shown in the UI.

### Forecasting & evaluation

One GRU inference per iceberg → 7 rows in `ml.forecasts` (with model version,
generation time, anchor observation, p90 risk radius per horizon from the
model's recorded metrics — `null` where that horizon was never evaluated).
Evaluation joins forecasts to *official* observations on `(iceberg, forecast_date
= observation_date)` and stores great-circle error; aggregates (MAE, RMSE, median,
p90 per horizon) are computed in SQL. With a weekly source most matches are D+7.

## Model versioning & retraining

* `base` = research artifact (status `validated`, not served). `v1` = byte-identical
  copy + `bootstrap_v1` adapter, deployed on first bootstrap. `v2 … vN` come only
  from retraining; numbers are `max(registry ∪ models/ dirs) + 1`, unbounded.
* Artifacts are written atomically to `models/vN/` (`model.keras`, `scalers.joblib`,
  `metadata.json`, `dataset_manifest.json`), made read-only, and checksummed in
  both the metadata and the registry; a changed file refuses to load.
* **Eligibility** (`RETRAIN_MIN_NEW_EVALUATIONS`, `…_MIN_NEW_ICEBERGS`,
  `…_MIN_DAYS_SINCE_LAST_TRAINING`) is checked weekly; ineligible checks are
  recorded as `skipped` runs with reasons.
* **Data** (cutoff frozen at run start): historical windows split at the base
  model's fixed boundaries (≤2018-08-23 / ≤2022-03-09 / after) + operational
  samples (official anchor, official D+h labels, other horizons NaN → masked
  Huber loss; architecture unchanged) split chronologically into train /
  validation / holdout. Never shuffled across time.
* **Strategy** `fine_tune` (default; champion weights + champion scalers +
  historical replay) or `from_scratch` (refit scalers on the new training split).
* **Champion/challenger**: both evaluated on identical samples; operational
  holdout decides when it has ≥ `PROMOTION_MIN_EVALUATION_SAMPLES` D+7 labels,
  otherwise the historical test. Promote only if D+7 MAE improves by
  `PROMOTION_MIN_RELATIVE_IMPROVEMENT` and D+1/D+3 do not regress by more than
  `PROMOTION_MAX_SHORT_HORIZON_REGRESSION` (and, for operational decisions, the
  historical D+7 does not regress either). Rejected versions are kept.
* Every transition is an `ml.model_status_events` row; lineage is
  `GET /api/v1/models/lineage`.

## Environmental-feature models (wind · ocean current · sea ice)

Starting with versions after v1, retraining can create **environmental model versions** that add per-entry
wind (`wind_u/v`), surface ocean current (`current_u/v`, 0.494 m) and sea-ice concentration to the six trajectory
features. The base model and v1 are untouched. Sources: Copernicus Marine (NRT/analysis-forecast for live
forecasts; GLORYS and reprocessed wind for training) and ERA5 (historical wind). A strict as-of rule prevents any
future information entering inputs; missing values are never fabricated (complete-case training, recorded
trajectory-only fallback at inference). Each retraining run trains one candidate per feature schema on identical
samples and reports whether each variable improved D+1…D+7 against the base model. Full details, source table,
latency and interpolation methodology: [docs/ENVIRONMENTAL_PIPELINE.md](docs/ENVIRONMENTAL_PIPELINE.md).

## API

OpenAPI docs: `http://localhost:8000/docs`. All responses are camelCase and typed.

| endpoint | notes |
| --- | --- |
| `GET /api/v1/icebergs` | currently tracked (`status=current`), paginated, `q` search |
| `GET /api/v1/icebergs/near?lat&lon&radius_km` | PostGIS `ST_DWithin` on latest official fixes |
| `GET /api/v1/icebergs/{id}` · `/position` · `/history` · `/forecast?horizon=1\|3\|7` · `/evaluations` | |
| `GET /api/v1/forecasts/latest?horizon=` | map layer: newest set per iceberg |
| `GET /api/v1/forecasts` | full append-only history; filter by iceberg, model_version, horizon/max_horizon, date range |
| `GET /api/v1/models` · `/current` · `/lineage` · `/{version}` · `/{version}/evaluations` | |
| `GET /api/v1/operations/ingestion` · `/forecasting` · `/retraining` | run history, feed state, eligibility |
| `GET /api/v1/feeds` · `/overview` · `/health/live` · `/health/ready` | |

Every request carries an `X-Request-ID` (echoed, logged). Database outages
return `503 {"detail": "database unavailable"}`.

## Frontend

`frontend/` — dark operational console. Screens: Overview (widget registry in
`components/dashboard/widgets.tsx`), Icebergs (table + map + detail panel),
1-/3-/7-Day forecast (one route `/forecast/:horizon`, so the 3D scene stays
mounted and trajectories animate between horizons of the same run), Operations,
Feeds, Route planning (inputs + danger zones; no routing engine), Sea-ice (kept
separate from iceberg trajectories; no data until a source is configured).

3D scene (`components/globe`): Natural Earth 1:50m Antarctica (`world-atlas`)
projected to EPSG:3031 in the browser (`utils/projection.ts`, verified against
pyproj values), extruded as a slab whose top — together with the ocean — is
draped with **NASA Blue Marble shaded relief + bathymetry** imagery served in
native EPSG:3031 by NASA GIBS. Tiles are fetched through the backend proxy
`GET /api/v1/basemap/{layer}/{z}/{row}/{col}.jpeg` (allow-listed layers,
validated tile indices, cached under `data/processed/basemap/`), so the browser
still talks only to FastAPI and the map keeps working offline once cached. The
imagery is a static cloud-free composite, labelled as such on the map — it is a
backdrop, never an observation. Graticule, instanced official markers
(solid diamonds), forecast points (hollow rings), shader-revealed dashed
trajectories, p90 error rings sized with the projection scale factor, selected
iceberg history trail, fly-to camera. OFFICIAL vs FORECAST vs HISTORICAL vs
STALE are distinguished everywhere by chip, shape and colour.

Server state via TanStack Query (polling 30 s–5 min; the browser never contacts
USNIC). Errors render explicit backend-unavailable / not-found / empty states —
no placeholder numbers.

## Testing

```bash
make test            # ml + backend (needs PostGIS via docker) + frontend
```

* `ml/tests` — features (notebook EPSG:3031 values, elapsed-day velocity),
  adapter (bootstrap, provenance exclusion, gaps, anchors), dataset builder
  (window counts, chronological splits, NaN labels), BYU loader (notebook §2 semantics),
  model (44,366 params, masked loss ≡ Huber, immutability, checksum tamper
  detection, single-pass 7-day forecasts, fine-tune), policy + versioning.
* `backend/tests` — CSV parser (real USNIC file + malformed rows), source
  discovery/fallback, feed-state logic, PostGIS integration (dedup, corrections,
  disappearance, geometry SRID, row errors, failures), append-only triggers,
  registry/bootstrap, forecast→evaluation lineage, forced retraining, API endpoints.
  DB tests use `TEST_DATABASE_URL` (auto-created + migrated) and skip if unavailable.
* `frontend` — projection tests (`npm test`), `npm run typecheck`, `npm run build`.

## Deployment & configuration

All configuration is environment-driven (`.env.example`); secrets have no
defaults. Structured JSON logs (structlog) include `request_id` and events such as
`ingestion_started/completed` (checksum, rows, new, duplicates), `forecast_generation_completed`
(model version), `retraining_started`, `candidate_promoted`, `candidate_rejected`.
Containers: `db`, `redis`, `migrate` (one-shot), `backend`, `worker` (queues
`ingestion`, `ml`), `scheduler` (beat), `frontend`. The backend image runs as a
non-root user; `models/` and `data/` are volumes.

## Design decisions and known limitations

* **Base artifact not in repo.** The `.keras`/`.joblib` files from the Colab
  export are required in `models/base/`; until they are present the system runs
  with official positions and reports the model as unavailable. `make
  reproduce-base` exists only as a labelled fallback.
* **Weekly vs daily.** The base model has not been validated on weekly inputs;
  v1 forecasts are flagged (`daily_cadence = false`) and their real accuracy is
  measured by operational evaluation, which is what retraining optimises.
* **Evaluation cadence.** Only horizons that coincide with a later official
  observation date are scored (mostly D+7). Day-1/3 operational metrics appear
  only when the source cadence allows; offline benchmarks cover all horizons.
* **Risk radius** is the model's empirical p90 per horizon (notebook metrics for
  base/v1 give D+1/3/7 only; run `make benchmark VERSION=v1` to record all seven).
* **BYU sizes** (`size_1/size_2`) are kept as unit-less raw attributes because the
  notebook labels their unit inconsistently; they are not model inputs.
* The notebook filename says *xgboost*; its final production section is the GRU.
  The name is kept unchanged as the research record.
