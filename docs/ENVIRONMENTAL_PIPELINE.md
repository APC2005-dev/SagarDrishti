# Environmental-feature pipeline

Additive to the existing system. The **base model** (`models/base/global_gru_trajectory_model.keras`,
its scaler, its six features) and **v1** are never modified, retrained or fed environmental data.
Environmental inputs exist only in **new model versions** created by retraining, each recorded with an
explicit feature schema, architecture version, data sources and cutoffs.

```
official USNIC observation ─► 14 trajectory entries (ProductionSequenceAdapter, unchanged)
                               + per-entry wind / current / sea ice  (as-of rule, below)
                               ─► environmental GRU vN (one inference) ─► D+1…D+7
```

## 1. Sources (verified against the live Copernicus Marine catalogue, 2026-09-11)

| group | role | product / dataset | variables | level | native | latency | coverage |
|---|---|---|---|---|---|---|---|
| wind | operational | `WIND_GLO_PHY_L4_NRT_012_004` / `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | `eastward_wind`, `northward_wind` → `wind_u`, `wind_v` | 10 m | hourly, 0.125° | ~1 d | 2024-06-13 → |
| wind | historical | ERA5 `reanalysis-era5-single-levels` | `10m_u/v_component_of_wind` (`u10`,`v10`) | 10 m | hourly, 0.25° | ~5 d (ERA5T) | 1940 → |
| wind | historical (alt.) | `WIND_GLO_PHY_L4_MY_012_006` / `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | as NRT | 10 m | hourly, 0.125° | months | 2007-01-11 → |
| current | operational | `GLOBAL_ANALYSISFORECAST_PHY_001_024` / `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` | `uo`, `vo` → `current_u`, `current_v` | **uppermost level 0.494 m** | daily, 1/12° | ~1 d, **+9 d forecast** | 2022-06-01 → |
| current | historical | `GLOBAL_MULTIYEAR_PHY_001_030` / `cmems_mod_glo_phy_my_0.083deg_P1D-m` (GLORYS12) | `uo`, `vo` | 0.494 m | daily, 1/12° | ~2–3 months | 1993-01-01 → |
| sea ice | operational | `GLOBAL_ANALYSISFORECAST_PHY_001_024` / `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `siconc` (sea_ice_area_fraction) → `sea_ice_concentration` | surface | daily, 1/12° | ~1 d, +9 d forecast | 2022-06-01 → |
| sea ice | historical | `GLOBAL_MULTIYEAR_PHY_001_030` / `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `siconc` | surface | daily, 1/12° | months | 1993 → |

Global products are used (they cover the Southern Ocean; the ocean grids end at 80°S, south of every tracked
iceberg). No Arctic-only product is used. Current depth is requested as 0–1 m, which selects the 0.494 m level —
the layer relevant to iceberg drift; no deeper level is used.

Configuration (`.env`): `ENV_WIND_PROVIDER`, `ENV_CURRENT_PROVIDER`, `ENV_SEA_ICE_PROVIDER` (operational) and
`ENV_HISTORICAL_*_PROVIDER` (training) select source families; `ml/environment/providers/registry.py` is the only
place that maps a family to a dataset. Credentials: `COPERNICUS_MARINE_USERNAME/PASSWORD`, `CDSAPI_KEY`
(`CDSAPI_URL`). Without them every environmental feature is reported *not configured* and nothing is trained or
substituted.

## 2. Alignment and interpolation

For each trajectory entry `(iceberg, observation_date, lat, lon)`:

* **time** — the field for that UTC day (hourly sources are averaged over 00–24 UTC; daily sources are daily means).
* **space** — bilinear interpolation between the four surrounding grid nodes; nodes without data (land, ice shelf)
  are dropped and weights renormalised; `valid_neighbours` is stored; no valid node → missing. No extrapolation.
* **provenance per value** — provider, dataset, requested date, valid date used, staleness, interpolation,
  valid neighbours, quality flags (partial neighbourhood, out of physical range), cache file.

Today's conditions are never attached to historical observations: every value comes from the date of the entry
it belongs to (or the latest earlier date allowed by the as-of rule).

## 3. No leakage — the as-of rule

A forecast is issued at prediction time **T**. For an entry observed on **d ≤ T**, and an operational feed with
latency **L**:

```
valid_date = min(d, T − L)
```

Only fields that the operational feed would already have published at T are read. The same rule is applied to
*training* samples (T = anchor date), even when the value comes from a reanalysis. If `d − valid_date` exceeds
`ENV_MAX_STALENESS_DAYS` the value is missing. Future (D+1…D+7) observed or reanalysed fields are never inputs.

**Forecast-period environment.** The first environmental architecture does *not* consume D+1…D+7 forecast
fields: no archive of forecasts *as issued* exists for the training period, and using reanalysis in their place
would leak the future. Instead, each environmental forecast archives the current/sea-ice forecast fields that are
available at T (`environmental.forecast_snapshots`, lead 1–7 at the anchor). Once enough of that archive exists,
a future architecture (explicit new `architecture_version`) can learn from genuinely issued forecasts. ERA5 and
the scatterometer wind product have no forecast period, and none is pretended.

## 4. Missing data

Values are never fabricated. Strategy:

* **training / evaluation** — complete-case: a sample is used by a schema only if every entry has every variable
  of that schema. Candidates are compared on samples complete for *all* compared schemas (identical data).
  Missing rates per variable are recorded in the retraining run (`experiment.alignment`).
* **live forecasting** — if an environmental champion lacks any input for an iceberg, that iceberg is forecast by
  the trajectory-only fallback (`ENV_FALLBACK_MODEL_VERSION`, default: nearest trajectory ancestor, v1); the
  forecast set records `fallback = {champion, reason, missing}`.

## 5. Feature schemas and architectures

`ml/features/schemas.py`: `trajectory_v1` (base six) · `traj_wind_v1` · `traj_current_v1` ·
`traj_wind_current_v1` · `traj_wind_current_ice_v1`. The six trajectory features are identical in every schema and
come first. New variables = new `EnvVariable` + new schema; existing schemas are frozen.

`ml/models/gru_architecture.py`: `gru_entry14_to_day7_v1` (base, fixed at 6 inputs) and
`gru_env_concat_entry14_to_day7_v1` (same layer stack, `Input(14, N)`). A two-branch fusion architecture is an
explicit extension point, not implemented.

## 6. Retraining experiment

When at least one environmental schema is trainable (`ENV_CANDIDATE_SCHEMAS`, sources configured), a retraining
run:

1. freezes the cutoff; builds historical (BYU, fixed base-model split boundaries) and operational samples;
2. aligns environment to a chronologically-stratified subsample (`ENV_TRAINING_MAX_*`), with the training builder
   (historical source, then operational for later dates; as-of latency of the operational feed);
3. trains one candidate **per schema, from scratch, on the same samples** (`trajectory_v1` is the control);
4. selects the best on the **validation** period (common samples) — the test period is not used for selection;
5. scores base, champion and every candidate on identical test samples (operational holdout if it has enough D+7
   labels, else the historical test): MAE, RMSE, median, p90 for D+1…D+7, error distributions, paired bootstrap
   CIs vs base, and per-variable effects (wind, current, wind+current, sea ice | wind+current);
6. promotes the selected candidate only if it passes the promotion policy **against the champion and against the
   base**; every other candidate is kept as a `rejected` version with the reason.

Every candidate is an immutable `models/vN/` with `metadata.json` (feature schema, feature names, architecture,
parent, cutoffs, environmental sources and policy, validation/test metrics) and `dataset_manifest.json`.

## 7. Storage

* `environmental.cache_entries` — index of cached tile × month NetCDF subsets (bbox polygon, GIST); files live in
  `ENV_CACHE_DIR` (5°×5° tiles + 2-cell halo, one calendar month, sha256). Only needed subsets are downloaded.
* `environmental.ingestion_runs` — every tile fetch, alignment, overlay refresh, forecast snapshot.
* `environmental.observation_features` — values aligned to official observations (append-only).
* `environmental.forecast_snapshots` — forecast fields as issued at forecast time (append-only).
* `ml.model_versions` + `model_type`, `feature_schema_version`, `environmental_data_sources`,
  `environmental_data_cutoff` (immutable once written); `ml.forecast_sets` + `feature_schema_version`,
  `environment`, `environment_as_of`, `fallback`; `ml.retraining_runs.experiment`.

## 8. Answering the key questions

| question | where |
|---|---|
| Which environmental variables did vN use? | `GET /api/v1/models/vN` → `featureSchema`, `featureNames` |
| Where did they come from? | `environmentalDataSources` (dataset ids, levels, latency) |
| What were their timestamps? | forecast set `inputEntries[i].environment[group].valid_date`, `environmentAsOf` |
| What did the model predict / what did USNIC observe / error? | `/icebergs/{id}/forecast`, `/icebergs/{id}/evaluations` |
| Did wind / current / sea ice improve the forecast? | `/operations/retraining` → `metrics.test.effects` (verdict + CI per horizon) |
| Which model is deployed and why was a candidate promoted/rejected? | `/models/current`, `/models/vN` `statusHistory`, run `decision.reasons` |

## 9. Operator commands

```bash
dev env-status      # sources, credentials configured (yes/no), trainable schemas
dev env-align       # align environment to the latest official fixes
dev env-overlay     # refresh map overlay grids
dev env-prefetch    # pre-warm the cache for a retraining experiment (resumable)
dev retrain -Force  # run the experiment now (policy still decides promotion)
```
