"""Immutable on-disk model artifact store.

Layout::

    models/
        base/  global_gru_trajectory_model.keras, global_gru_trajectory_scalers.joblib, metadata.json
        v1/    model.keras, scalers.joblib, metadata.json[, dataset_manifest.json]
        vN/    ...

A version directory is written to a temporary sibling and atomically renamed
into place; an existing directory is never overwritten. Files are marked
read-only after writing. The database registry stores the same checksums so
tampering is detectable from either side.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ml.adapters.bootstrap_adapter import sha256_file
from ml.constants import FEATURE_NAMES, FORECAST_DAYS, SEQUENCE_LENGTH


class ArtifactExistsError(FileExistsError):
    pass


@dataclass(frozen=True)
class WrittenArtifact:
    directory: Path
    model_path: Path
    scaler_path: Path
    metadata_path: Path
    checksums: dict[str, str]


def _make_read_only(path: Path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


def _finalise(tmp: Path, target: Path, metadata: dict[str, Any], extra_files: dict[str, Any] | None) -> WrittenArtifact:
    model_path, scaler_path = tmp / "model.keras", tmp / "scalers.joblib"
    checksums = {"model.keras": sha256_file(model_path), "scalers.joblib": sha256_file(scaler_path)}
    metadata = {**metadata, "artifact_sha256": checksums}
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    for name, payload in (extra_files or {}).items():
        (tmp / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if target.exists():
        raise ArtifactExistsError(f"{target} already exists; model versions are immutable")
    tmp.rename(target)
    for child in target.iterdir():
        _make_read_only(child)
    return WrittenArtifact(
        directory=target,
        model_path=target / "model.keras",
        scaler_path=target / "scalers.joblib",
        metadata_path=target / "metadata.json",
        checksums=checksums,
    )


def scaler_payload(feature_scaler: Any, target_scaler: Any) -> dict[str, Any]:
    """Same dict layout the notebook exports (§8)."""
    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "history_days": SEQUENCE_LENGTH,
        "forecast_days": FORECAST_DAYS,
        "feature_names": list(FEATURE_NAMES),
    }


def write_version(
    models_root: Path,
    version: str,
    model: Any,
    feature_scaler: Any,
    target_scaler: Any,
    metadata: dict[str, Any],
    extra_files: dict[str, Any] | None = None,
) -> WrittenArtifact:
    target = Path(models_root) / version
    if target.exists():
        raise ArtifactExistsError(f"{target} already exists; model versions are immutable")
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=models_root))
    try:
        model.save(tmp / "model.keras")
        joblib.dump(scaler_payload(feature_scaler, target_scaler), tmp / "scalers.joblib")
        return _finalise(tmp, target, metadata, extra_files)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def copy_version(
    models_root: Path,
    source_dir: Path,
    version: str,
    metadata: dict[str, Any],
    extra_files: dict[str, Any] | None = None,
) -> WrittenArtifact:
    """Create a new version whose weights/scalers are a byte-identical copy (used for v1)."""
    from ml.models.model_loader import resolve_artifact_paths

    target = Path(models_root) / version
    if target.exists():
        raise ArtifactExistsError(f"{target} already exists; model versions are immutable")
    model_src, scaler_src, _ = resolve_artifact_paths(Path(source_dir))
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=models_root))
    try:
        shutil.copyfile(model_src, tmp / "model.keras")
        shutil.copyfile(scaler_src, tmp / "scalers.joblib")
        return _finalise(tmp, target, metadata, extra_files)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
