"""Immutable on-disk store for sea-ice model versions.

Layout under ``<models_dir>/sea_ice/``::

    base/  unet_residual_v4_best.pt, metadata.json     <- supplied base, never written again
    v1/    model.pt, metadata.json[, dataset_manifest.json]
    v2/    ...

Same guarantees as the trajectory store: versions are written atomically into a
temp directory and renamed into place, an existing directory is never
overwritten, files are made read-only, and every artifact is checksummed so a
later modification is detectable. The directory names come from the registry's
dynamic numbering — nothing here hard-codes a version.
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

import torch

from ml.adapters.bootstrap_adapter import sha256_file
from ml.seaice.constants import BASE_ARTIFACT_FILENAME, MODEL_FILENAME


class SeaIceArtifactExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class WrittenSeaIceArtifact:
    directory: Path
    model_path: Path
    metadata_path: Path
    checksums: dict[str, str]


def _make_read_only(path: Path) -> None:
    for f in path.rglob("*"):
        if f.is_file():
            os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _finalise(
    tmp: Path, target: Path, metadata: dict[str, Any], extra_files: dict[str, Any] | None
) -> WrittenSeaIceArtifact:
    model_path = tmp / MODEL_FILENAME
    checksums = {MODEL_FILENAME: sha256_file(model_path)}
    metadata = {**metadata, "artifact_sha256": checksums}
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    for name, payload in (extra_files or {}).items():
        (tmp / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _make_read_only(tmp)
    tmp.rename(target)
    return WrittenSeaIceArtifact(
        directory=target,
        model_path=target / MODEL_FILENAME,
        metadata_path=target / "metadata.json",
        checksums=checksums,
    )


def seaice_root(models_dir: Path) -> Path:
    """The sea-ice family's own subtree, kept separate from the trajectory versions."""
    return Path(models_dir) / "sea_ice"


def write_version(
    models_dir: Path,
    version: str,
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
    extra_files: dict[str, Any] | None = None,
) -> WrittenSeaIceArtifact:
    """Persist a newly trained sea-ice version. Refuses to touch an existing directory."""
    root = seaice_root(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / version
    if target.exists():
        raise SeaIceArtifactExistsError(f"{target} already exists; model versions are immutable")
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=root))
    try:
        torch.save(state_dict, tmp / MODEL_FILENAME)
        return _finalise(tmp, target, metadata, extra_files)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def copy_version(
    models_dir: Path, source_dir: Path, version: str, metadata: dict[str, Any]
) -> WrittenSeaIceArtifact:
    """Create a version whose weights are a byte-identical copy of ``source_dir``.

    Used to derive the first production version from the immutable base without
    retraining, and without ever writing into the base directory.
    """
    root = seaice_root(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / version
    if target.exists():
        raise SeaIceArtifactExistsError(f"{target} already exists; model versions are immutable")
    source_model = resolve_model_path(Path(source_dir))
    tmp = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=root))
    try:
        shutil.copyfile(source_model, tmp / MODEL_FILENAME)
        return _finalise(tmp, target, metadata, None)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def resolve_model_path(directory: Path) -> Path:
    """The ``.pt`` inside a version directory (``model.pt``, or the base's own filename)."""
    for name in (MODEL_FILENAME, BASE_ARTIFACT_FILENAME):
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no sea-ice model artifact in {directory}: expected {MODEL_FILENAME} or {BASE_ARTIFACT_FILENAME}"
    )


def resolve_model_path_or_none(directory: Path) -> Path | None:
    try:
        return resolve_model_path(directory)
    except FileNotFoundError:
        return None
