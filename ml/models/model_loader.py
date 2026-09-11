"""Loading and validating model bundles (weights + scalers + metadata).

The tensor contract is read from each artifact's own metadata
(``feature_names`` / ``feature_schema_version``). Artifacts written before
schemas existed (base, v1) carry only the six trajectory feature names and
resolve to ``trajectory_v1``; their validation is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from ml.adapters.bootstrap_adapter import sha256_file
from ml.constants import FEATURE_NAMES, OUTPUT_SIZE, SEQUENCE_LENGTH
from ml.features.schemas import TRAJECTORY_SCHEMA, get_schema, schema_for_features

MODEL_FILENAMES = ("model.keras", "global_gru_trajectory_model.keras")
SCALER_FILENAMES = ("scalers.joblib", "global_gru_trajectory_scalers.joblib")


class ModelArtifactError(RuntimeError):
    """The artifact on disk is missing, corrupt, or violates the tensor contract."""


@dataclass
class ModelBundle:
    version: str
    model: Any  # keras.Model
    feature_scaler: Any  # sklearn StandardScaler fitted on (N*14, n_features)
    target_scaler: Any  # sklearn StandardScaler fitted on (N, 14)
    metadata: dict[str, Any]
    directory: Path
    checksums: dict[str, str] = field(default_factory=dict)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_schema_version: str = TRAJECTORY_SCHEMA

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def is_environmental(self) -> bool:
        return get_schema(self.feature_schema_version).is_environmental


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


def feature_contract(metadata: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    """(feature_names, feature_schema_version) declared by an artifact's metadata."""
    names = tuple(metadata.get("feature_names") or FEATURE_NAMES)
    declared = metadata.get("feature_schema_version")
    try:
        schema = get_schema(declared) if declared else schema_for_features(names)
    except ValueError as exc:
        raise ModelArtifactError(str(exc)) from exc
    if schema.features != names:
        raise ModelArtifactError(f"feature_names {names} do not match schema {schema.version} {schema.features}")
    return names, schema.version


def verify_checksums(directory: Path, metadata: dict[str, Any]) -> dict[str, str]:
    """Compare on-disk sha256 against ``metadata['artifact_sha256']`` when recorded."""
    model_path, scaler_path, _ = resolve_artifact_paths(directory)
    actual = {model_path.name: sha256_file(model_path), scaler_path.name: sha256_file(scaler_path)}
    expected: dict[str, str] = metadata.get("artifact_sha256") or {}
    for name, digest in expected.items():
        if name in actual and actual[name] != digest:
            raise ModelArtifactError(f"checksum mismatch for {directory / name}: artifact was modified")
    return actual


def validate_scalers(payload: Any, feature_names: tuple[str, ...] = FEATURE_NAMES) -> tuple[Any, Any]:
    if not isinstance(payload, dict) or "feature_scaler" not in payload or "target_scaler" not in payload:
        raise ModelArtifactError("scalers.joblib must be a dict with feature_scaler and target_scaler")
    fs, ts = payload["feature_scaler"], payload["target_scaler"]
    n = len(feature_names)
    if getattr(fs, "n_features_in_", None) != n:
        raise ModelArtifactError(f"feature_scaler expects {getattr(fs, 'n_features_in_', None)} features, need {n}")
    if getattr(ts, "n_features_in_", None) != OUTPUT_SIZE:
        raise ModelArtifactError(f"target_scaler expects {getattr(ts, 'n_features_in_', None)} outputs, need {OUTPUT_SIZE}")
    names = payload.get("feature_names")
    if names is not None and tuple(names) != tuple(feature_names):
        raise ModelArtifactError(f"feature_names {names} differ from contract {feature_names}")
    if payload.get("history_days", SEQUENCE_LENGTH) != SEQUENCE_LENGTH:
        raise ModelArtifactError("scalers history_days does not match sequence length 14")
    return fs, ts


def validate_model_shapes(model: Any, n_features: int = len(FEATURE_NAMES)) -> None:
    in_shape = tuple(model.input_shape)
    out_shape = tuple(model.output_shape)
    if in_shape[1:] != (SEQUENCE_LENGTH, n_features):
        raise ModelArtifactError(f"model input shape {in_shape} != (None, {SEQUENCE_LENGTH}, {n_features})")
    if out_shape[1:] != (OUTPUT_SIZE,):
        raise ModelArtifactError(f"model output shape {out_shape} != (None, 14)")


def load_model_bundle(directory: Path, version: str | None = None, verify: bool = True) -> ModelBundle:
    import keras

    directory = Path(directory)
    metadata = read_metadata(directory)
    feature_names, schema_version = feature_contract(metadata)
    checksums = verify_checksums(directory, metadata) if verify else {}
    model_path, scaler_path, _ = resolve_artifact_paths(directory)
    try:
        # compile=False: inference never needs the optimiser, and it keeps the
        # original Colab artifact loadable without its training-time objects.
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001 - surface any deserialisation failure uniformly
        raise ModelArtifactError(f"cannot load {model_path}: {exc}") from exc
    validate_model_shapes(model, len(feature_names))
    fs, ts = validate_scalers(joblib.load(scaler_path), feature_names)
    return ModelBundle(
        version=version or metadata.get("version", directory.name),
        model=model,
        feature_scaler=fs,
        target_scaler=ts,
        metadata=metadata,
        directory=directory,
        checksums=checksums,
        feature_names=feature_names,
        feature_schema_version=schema_version,
    )
