"""Construction of the 14 x 6 GRU feature matrix and 14-value targets.

Mirrors notebook §3 with one deliberate generalisation: velocity divides by
the *actual elapsed days* between consecutive entries instead of assuming 1 day.
For the daily research sequences (elapsed = 1 everywhere) the output is
bit-for-bit what the notebook produced; for weekly USNIC entries the velocity
stays physically correct (km/day) instead of silently becoming km/entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from ml.constants import FORECAST_DAYS, N_FEATURES, SEQUENCE_LENGTH
from ml.features.seasonal_features import season_features


class FeatureConstructionError(ValueError):
    """Raised when entries cannot form a physically valid sequence."""


@dataclass(frozen=True)
class FeatureMatrix:
    features: NDArray[np.float32]  # (SEQUENCE_LENGTH, N_FEATURES)
    elapsed_days: NDArray[np.float64]  # (SEQUENCE_LENGTH,), element 0 is 0
    anchor_x_m: float
    anchor_y_m: float


def elapsed_days_between(dates: Sequence[date]) -> NDArray[np.float64]:
    """Elapsed days between consecutive entries; first element is 0."""
    ordinals = np.asarray([d.toordinal() for d in dates], dtype=np.float64)
    gaps = np.zeros(len(ordinals), dtype=np.float64)
    gaps[1:] = np.diff(ordinals)
    return gaps


def build_feature_matrix(
    x_m: Sequence[float] | NDArray[np.float64],
    y_m: Sequence[float] | NDArray[np.float64],
    dates: Sequence[date],
) -> FeatureMatrix:
    """Build one (14, 6) input from chronological EPSG:3031 positions.

    Features per entry i (anchor = last entry):
        relative_x_km, relative_y_km  = (p_i - p_anchor) / 1000
        vx_km_per_day, vy_km_per_day  = (p_i - p_{i-1}) / 1000 / elapsed_days_i
                                        (entry 0 copies entry 1, as in the notebook)
        season_sin, season_cos        = sin/cos(2*pi*doy_i / 365.25)
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if not (len(x) == len(y) == len(dates) == SEQUENCE_LENGTH):
        raise FeatureConstructionError(
            f"expected {SEQUENCE_LENGTH} entries, got x={len(x)} y={len(y)} dates={len(dates)}"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise FeatureConstructionError("non-finite projected coordinate in sequence")

    elapsed = elapsed_days_between(dates)
    if np.any(elapsed[1:] <= 0):
        raise FeatureConstructionError("entry dates must be strictly increasing")

    anchor_x, anchor_y = float(x[-1]), float(y[-1])
    vx = np.empty(SEQUENCE_LENGTH, dtype=np.float64)
    vy = np.empty(SEQUENCE_LENGTH, dtype=np.float64)
    vx[1:] = np.diff(x) / 1000.0 / elapsed[1:]
    vy[1:] = np.diff(y) / 1000.0 / elapsed[1:]
    vx[0] = vx[1]
    vy[0] = vy[1]
    sin, cos = season_features(dates)

    features = np.column_stack(
        [(x - anchor_x) / 1000.0, (y - anchor_y) / 1000.0, vx, vy, sin, cos]
    ).astype(np.float32)
    assert features.shape == (SEQUENCE_LENGTH, N_FEATURES)
    return FeatureMatrix(features=features, elapsed_days=elapsed, anchor_x_m=anchor_x, anchor_y_m=anchor_y)


def build_target_vector(
    anchor_x_m: float,
    anchor_y_m: float,
    future_x_m: Sequence[float | None],
    future_y_m: Sequence[float | None],
) -> NDArray[np.float32]:
    """Flatten future displacements (km) into [d1x, d1y, ..., d7x, d7y].

    Horizons without a known position are NaN; the masked training loss and the
    evaluator ignore them.
    """
    if len(future_x_m) != FORECAST_DAYS or len(future_y_m) != FORECAST_DAYS:
        raise FeatureConstructionError(f"expected {FORECAST_DAYS} future slots")
    fx = np.asarray([np.nan if v is None else v for v in future_x_m], dtype=np.float64)
    fy = np.asarray([np.nan if v is None else v for v in future_y_m], dtype=np.float64)
    return np.column_stack([(fx - anchor_x_m) / 1000.0, (fy - anchor_y_m) / 1000.0]).reshape(-1).astype(
        np.float32
    )
