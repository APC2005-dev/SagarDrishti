"""Convert a source dataset into the canonical daily grid (see providers/base.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

COORD_ALIASES = {"latitude": "lat", "longitude": "lon", "valid_time": "time"}


def canonicalize(ds: Any, native_to_canonical: Mapping[str, str], hourly: bool, attrs: Mapping[str, Any]) -> Any:
    """Rename, reduce to surface/daily, sort coordinates, float32, NaN for missing."""
    rename = {k: v for k, v in COORD_ALIASES.items() if k in ds.variables or k in ds.dims}
    ds = ds.rename(rename)
    keep = [n for n in native_to_canonical if n in ds.data_vars]
    missing = set(native_to_canonical) - set(keep)
    if missing:
        raise ValueError(f"source dataset lacks variables {sorted(missing)}")
    ds = ds[keep]
    if "depth" in ds.dims:
        # providers request only the uppermost level; keep it explicitly
        ds = ds.isel(depth=0, drop=True)
    for extra in [d for d in ds.dims if d not in ("time", "lat", "lon")]:
        ds = ds.isel({extra: 0}, drop=True)
    if hourly:
        ds = ds.resample(time="1D").mean(skipna=True)
    ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[D]").astype("datetime64[ns]"))
    lon = ds["lon"].values
    if lon.max() > 180:
        ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
    ds = ds.sortby("lat").sortby("lon")
    ds = ds.rename({n: native_to_canonical[n] for n in keep})
    for v in ds.data_vars:
        ds[v] = ds[v].astype(np.float32)
    ds.attrs.update({k: str(v) for k, v in attrs.items()})
    return ds.load()
