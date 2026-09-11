"""Loader for the historical training dataset (BYU consolidated database v8.0).

Reproduces notebook §2 exactly — the cleaning used to build the GRU
training sequences:

* one CSV per iceberg, iceberg id = upper-cased file stem (``a23a.csv`` -> ``A23A``)
* ``date`` is a YYYYDDD code
* position taken from the first sensor in ``SOURCE_PRIORITY`` whose lat is in
  [-90, -45] and lon in [-180, 180] (this discards the 0/0 "no data" placeholders)
* rows without date or position dropped; one row per (iceberg, date), first kept

``size_1``/``size_2`` are kept only as raw, unit-less attributes: the notebook
labelled them inconsistently (both nm and km appear) and they are not model
inputs, so no unit is asserted here.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.coordinate_transform import latlon_to_polar_m

SOURCE_PRIORITY: tuple[str, ...] = ("nic", "qscat", "ers", "ascat", "seawinds", "nscat", "oscat")
DATASET_NAME = "byu_mers_consolidated_antarctic_iceberg_database_v8.0"


def normalize_iceberg_id(raw: str) -> str:
    """Canonical id: upper-case, no whitespace, hyphens or underscores (``a-23a `` -> ``A23A``)."""
    return "".join(ch for ch in str(raw).strip().upper() if ch.isalnum())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_one(raw: pd.DataFrame, iceberg_id: str) -> pd.DataFrame:
    if "date" not in raw.columns:
        return pd.DataFrame()
    clean = pd.DataFrame(index=raw.index)
    clean["iceberg_id"] = iceberg_id
    date_code = pd.to_numeric(raw["date"], errors="coerce")
    year = (date_code // 1000).astype("Int64")
    doy = (date_code % 1000).astype("Int64")
    clean["date"] = pd.to_datetime(
        year.astype("string") + doy.astype("string").str.zfill(3), format="%Y%j", errors="coerce"
    )
    clean["latitude"] = np.nan
    clean["longitude"] = np.nan
    clean["position_source"] = pd.Series(pd.NA, index=raw.index, dtype="object")
    for source in SOURCE_PRIORITY:
        lat_col, lon_col = f"{source}_1", f"{source}_2"
        if lat_col not in raw.columns or lon_col not in raw.columns:
            continue
        lat = pd.to_numeric(raw[lat_col], errors="coerce")
        lon = pd.to_numeric(raw[lon_col], errors="coerce")
        choose = clean["latitude"].isna() & lat.between(-90, -45) & lon.between(-180, 180)
        clean.loc[choose, "latitude"] = lat[choose]
        clean.loc[choose, "longitude"] = lon[choose]
        clean.loc[choose, "position_source"] = source
    clean["size_1"] = pd.to_numeric(raw["size_1"], errors="coerce").replace(0, np.nan) if "size_1" in raw else np.nan
    clean["size_2"] = pd.to_numeric(raw["size_2"], errors="coerce").replace(0, np.nan) if "size_2" in raw else np.nan
    return clean.dropna(subset=["date", "latitude", "longitude"])


def load_byu_tracks(zip_path: Path) -> pd.DataFrame:
    """Load and clean every iceberg track from the BYU zip (notebook §2 semantics).

    Returns columns: iceberg_id, date, latitude, longitude, position_source,
    size_1, size_2, x_m, y_m, source_file — sorted by (iceberg_id, date).
    """
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for name in names:
            stem = Path(name).stem
            if stem.startswith("#"):
                continue
            with zf.open(name) as fh:
                raw = pd.read_csv(io.BytesIO(fh.read()))
            clean = _clean_one(raw, normalize_iceberg_id(stem))
            if not clean.empty:
                clean["source_file"] = name
                frames.append(clean)
    if not frames:
        raise ValueError(f"no usable iceberg tracks found in {zip_path}")
    tracks = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["iceberg_id", "date"], kind="mergesort")
        .drop_duplicates(["iceberg_id", "date"], keep="first")
        .reset_index(drop=True)
    )
    tracks["x_m"], tracks["y_m"] = latlon_to_polar_m(tracks["latitude"].to_numpy(), tracks["longitude"].to_numpy())
    return tracks
