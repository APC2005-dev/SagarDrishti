"""Loading and validating model bundles (weights + scalers + metadata)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from ml.adapters.bootstrap_adapter import sha256_file
from ml.constants import FEATURE_NAMES, N_FEATURES, OUTPUT_SIZE, SEQUENCE_LENGTH

MODEL_FILENAMES = ("model.keras", "global_gru_trajectory_model.keras")
SCALER_FILENAMES = ("scalers.joblib", "global_gru_trajectory_scalers.joblib")


class ModelArtifactError(RuntimeError):
    """The artifact on disk is missing, corrupt, or violates the tensor contract."""


@dataclass
class ModelBundle:
    version: str
    model: Any  # keras.Model
    feature_scaler: Any  # sklearn StandardScaler fitted on (N*14, 6)
    target_scaler: Any  # sklearn StandardScaler fitted on (N, 14)
    metadata: dict[str, Any]
    directory: Path
    checksums: dict[str, str] = field(default_factory=dict)


def resolve_artifact_paths(directory: Path) -> tuple[Path, Path, Path]:
    model_path = next((directory / n for n in MODEL_FILENAMES if (directory / n).exists()), None)
    scaler_path = next((directory / n for n in SCALER_FILENAMES if (directory / n).exists()), None)
    if model_path is None or scaler_path is None:
        raise ModelArtifactError(
            f"artifact incomplete in {directory}: expected one of {MODEL_FILENAMES} and one of {SCALER_FILENAMES}"
        )
    return model_path, scaler_path, directory / "metadata.json"


def resolve_artifacts_or_none(directory: Path) -> tuple[Path | None, Path | None, Path]:
    """Like :func:`resolve_artifact_paths` but returns ``None`` for missing files."""
    model_path = next((directory / n for n in MODEL_FILENAMES if (directory / n).exists()), None)
    scaler_path = next((directory / n for n in SCALER_FILENAMES if (directory / n).exists()), None)
    return model_path, scaler_path, directory / "metadata.json"


def read_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "metadata.json"
    if not path.exists():
        raise ModelArtifactError(f"missing metadata.json in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checksums(directory: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Compare on-disk sha256 against ``metadata['artifact_sha256']`` when recorded."""
    model_path, scaler_path, _ = resolve_artifact_paths(directory)
    actual = {model_path.name: sha256_file(model_path), scaler_path.name: sha256_file(scaler_path)}
    expected: dict[str, str] = metadata.get("artifact_sha256") or {}
    for name, digest in expected.items():
        if name in actual and actual[name] != digest:
            raise ModelArtifactError(f"checksum mismatch for {directory / name}: artifact was modified")
    return actual


def validate_scalers(payload: Any) -> tuple[Any, Any]:
    if not isinstance(payload, dict) or "feature_scaler" not in payload or "target_scaler" not in payload:
        raise ModelArtifactError("scalers.joblib must be a dict with feature_scaler and target_scaler")
    fs, ts = payload["feature_scaler"], payload["target_scaler"]
    if getattr(fs, "n_features_in_", None) != N_FEATURES:
        raise ModelArtifactError(f"feature_scaler expects {getattr(fs, 'n_features_in_', None)} features, need {N_FEATURES}")
    if getattr(ts, "n_features_in_", None) != OUTPUT_SIZE:
        raise ModelArtifactError(f"target_scaler expects {getattr(ts, 'n_features_in_', None)} outputs, need {OUTPUT_SIZE}")
    names = payload.get("feature_names")
    if names is not None and tuple(names) != FEATURE_NAMES:
        raise ModelArtifactError(f"feature_names {names} differ from contract {FEATURE_NAMES}")
    if payload.get("history_days", SEQUENCE_LENGTH) != SEQUENCE_LENGTH:
        raise ModelArtifactError("scalers history_days does not match sequence length 14")
    return fs, ts


def validate_model_shapes(model: Any) -> None:
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    if in_shape[1:] != (SEQUENCE_LENGTH, N_FEATURES):
        raise ModelArtifactError(f"model input shape {in_shape} != (None, 14, 6)")
    if out_shape[1:] != (OUTPUT_SIZE,):
        raise ModelArtifactError(f"model output shape {out_shape} != (None, 14)")


def load_model_bundle(directory: Path, version: str | None = None, verify: bool = True) -> ModelBundle:
    import keras

    directory = Path(directory)
    metadata = read_metadata(directory)
    checksums = verify_checksums(directory, metadata) if verify else {}
    model_path, scaler_path, _ = resolve_artifact_paths(directory)
    try:
        # compile=False: inference never needs the optimiser, and it keeps the
        # original Colab artifact loadable without its training-time objects.
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001 - surface any deserialisation failure uniformly
        raise ModelArtifactError(f"cannot load {model_path}: {exc}") from exc
    validate_model_shapes(model)
    fs, ts = validate_scalers(joblib.load(scaler_path))
    return ModelBundle(
        version=version or metadata.get("version", directory.name),
        model=model,
        feature_scaler=fs,
        target_scaler=ts,
        metadata=metadata,
        directory=directory,
        checksums=checksums,
    )
