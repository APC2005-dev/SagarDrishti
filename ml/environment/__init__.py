"""Environmental forcing (wind, ocean current, sea ice) for environmental model versions.

Nothing here is used by the base model or by trajectory-only versions.

Module map
----------
types.py              groups, provider specs, point samples
providers/            source adapters (Copernicus Marine, ERA5) returning canonical daily fields
cache.py              tile x month NetCDF cache — only the subsets actually needed are fetched
sampler.py            NaN-aware bilinear point sampling on a regular lat/lon grid
feature_builder.py    as-of (leakage-safe) per-entry feature construction
alignment.py          attach environmental columns to training/evaluation sequence datasets
"""
