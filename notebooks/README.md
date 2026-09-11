# Notebooks

## `01_gru_iceberg_trajectory_model.ipynb`

Research notebook that trains the platform's **base GRU** (14 daily positions →
next 7 daily positions) and measures the **p90 risk radius** for each forecast day
D+1…D+7 — the error distance used for forecast cones and, later, route-planning
danger zones.

**Run on Colab (recommended, GPU ≈ 10 min):** upload the notebook, choose a GPU
runtime, run all cells, upload `consolidated_database_v8.0.zip` when asked. The last
cell downloads `sagar_drishti_base_model.zip`.

**Run locally:** from this folder, with `pip install -r requirements.txt`; it reads
`../data/bootstrap/consolidated_database_v8.0.zip` and writes to
`../data/processed/notebook/` (override with `SAGAR_NOTEBOOK_WORKDIR`).

**Use the result:** copy `global_gru_trajectory_model.keras`,
`global_gru_trajectory_scalers.joblib` and `metadata.json` from the export into
`models/base/`, then run `dev bootstrap`.

The production pipeline does not import this notebook; the same steps are
implemented in `ml/` (and must stay consistent with it — architecture
`gru_entry14_to_day7_v1`, 44,366 parameters).

The original exploratory notebook (`01_iceberg_trajectory_xgboost.ipynb`, with the
constant-velocity, XGBoost, ERA5, ocean-current, Kalman and random-forest
experiments) is no longer kept here; its checksum is recorded in
`data/bootstrap/MANIFEST.json`.
