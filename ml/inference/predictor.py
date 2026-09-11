"""Scaled GRU inference (notebook §6–§7)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ml.constants import FORECAST_DAYS, N_FEATURES, SEQUENCE_LENGTH
from ml.models.model_loader import ModelBundle


def scale_features(bundle: ModelBundle, X_raw: NDArray[np.float32]) -> NDArray[np.float32]:
    shape = X_raw.shape
    return bundle.feature_scaler.transform(X_raw.reshape(-1, N_FEATURES)).reshape(shape).astype(np.float32)


def predict_displacements_km(
    bundle: ModelBundle, X_raw: NDArray[np.float32], batch_size: int = 512
) -> NDArray[np.float64]:
    """One GRU pass -> all 7 horizons. Returns (n, 7, 2) displacement km (x, y)."""
    if X_raw.ndim != 3 or X_raw.shape[1:] != (SEQUENCE_LENGTH, N_FEATURES):
        raise ValueError(f"expected (n, {SEQUENCE_LENGTH}, {N_FEATURES}) got {X_raw.shape}")
    if len(X_raw) == 0:
        return np.zeros((0, FORECAST_DAYS, 2))
    scaled = bundle.model.predict(scale_features(bundle, X_raw), batch_size=batch_size, verbose=0)
    km = bundle.target_scaler.inverse_transform(np.asarray(scaled, dtype=np.float64))
    return km.reshape(-1, FORECAST_DAYS, 2)


def stack_features(features: Sequence[NDArray[np.float32]]) -> NDArray[np.float32]:
    if not features:
        return np.zeros((0, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
    return np.stack(features).astype(np.float32)
