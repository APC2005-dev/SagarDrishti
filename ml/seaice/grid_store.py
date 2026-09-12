"""On-disk store for sea-ice grids (observations and forecasts).

Grids are ~100x720 float32 pairs, too large and too numerous for the database,
so the arrays live here as compressed ``.npz`` files and the ``seaice`` tables
index them with a checksum. Observation files are written once and made
read-only: official data is never rewritten in place.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from ml.adapters.bootstrap_adapter import sha256_file


@dataclass(frozen=True)
class StoredGrid:
    path: Path
    sha256: str


def _atomic_savez(target: Path, read_only: bool, **arrays: np.ndarray) -> StoredGrid:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".npz", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        np.savez_compressed(tmp, **arrays)
        # numpy appends .npz when the name lacks it; mkstemp already gave us one.
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if read_only:
        os.chmod(target, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return StoredGrid(path=target, sha256=sha256_file(target))


def observation_path(root: Path, day: date) -> Path:
    """One file per official day, foldered by year so directories stay small."""
    return Path(root) / "observations" / f"{day.year:04d}" / f"{day.isoformat()}.npz"


def forecast_path(root: Path, version: str, anchor: date, horizon: int) -> Path:
    safe_version = version.replace("/", "_")
    return Path(root) / "forecasts" / safe_version / f"{anchor.isoformat()}_h{horizon}.npz"


def save_observation(root: Path, day: date, concentration: np.ndarray, mask: np.ndarray) -> StoredGrid:
    """Persist one official field. Refuses to overwrite an existing day."""
    target = observation_path(root, day)
    if target.exists():
        raise FileExistsError(f"sea-ice observation for {day} already stored at {target}")
    return _atomic_savez(
        target,
        read_only=True,
        concentration=concentration.astype(np.float32),
        mask=mask.astype(np.float32),
    )


def save_forecast(root: Path, version: str, anchor: date, horizon: int, prediction: np.ndarray) -> StoredGrid:
    return _atomic_savez(
        forecast_path(root, version, anchor, horizon),
        read_only=False,
        prediction=prediction.astype(np.float32),
    )


def load_observation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["concentration"].astype(np.float32), data["mask"].astype(np.float32)


def load_forecast(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["prediction"].astype(np.float32)
