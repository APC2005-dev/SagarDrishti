"""SAGAR DRISHTI production ML package.

This package is deliberately free of database and web dependencies. It turns
chronological iceberg position entries into GRU input tensors, runs inference,
trains/evaluates model versions and manages immutable artifacts on disk. The
backend (``backend/app``) is responsible for moving data between PostGIS and
these pure functions.

The research notebook (``notebooks/01_gru_iceberg_trajectory_model.ipynb``) is the
source of truth for the base model; every function here that mirrors a notebook
step names its section (§2 cleaning, §3 sequences, §4 split & scalers,
§5 training, §6 evaluation & p90 risk radius, §7 example forecasts, §8 export).
"""
