"""Turn model input sequences into complete 7-day forecasts.

One inference per iceberg produces all horizons D+1..D+7; the 1/3/7-day views
are filters over this single output, never separate models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ml.adapters.sequence_builder import ModelInputSequence
from ml.constants import FORECAST_DAYS
from ml.features.coordinate_transform import polar_m_to_latlon
from ml.inference.predictor import predict_displacements_km, stack_features
from ml.models.model_loader import ModelBundle


@dataclass(frozen=True)
class ForecastPoint:
    horizon_days: int
    forecast_date: date
    predicted_x_m: float  # EPSG:3031
    predicted_y_m: float  # EPSG:3031
    predicted_latitude: float  # EPSG:4326
    predicted_longitude: float  # EPSG:4326
    risk_radius_km_p90: float | None


@dataclass(frozen=True)
class GeneratedForecast:
    sequence: ModelInputSequence
    model_version: str
    points: tuple[ForecastPoint, ...]


def generate_forecasts(
    bundle: ModelBundle,
    sequences: Sequence[ModelInputSequence],
    risk_radius_km_p90: Mapping[int, float | None] | None = None,
) -> list[GeneratedForecast]:
    if not sequences:
        return []
    displacement_km = predict_displacements_km(bundle, stack_features([s.features for s in sequences]))
    risk = dict(risk_radius_km_p90 or {})
    out: list[GeneratedForecast] = []
    for seq, disp in zip(sequences, displacement_km, strict=True):
        px = seq.anchor_x_m + disp[:, 0] * 1000.0
        py = seq.anchor_y_m + disp[:, 1] * 1000.0
        lat, lon = polar_m_to_latlon(px, py)
        anchor_date = seq.anchor.observation_date
        points = tuple(
            ForecastPoint(
                horizon_days=h,
                forecast_date=anchor_date + timedelta(days=h),
                predicted_x_m=float(px[h - 1]),
                predicted_y_m=float(py[h - 1]),
                predicted_latitude=float(lat[h - 1]),
                predicted_longitude=float(lon[h - 1]),
                risk_radius_km_p90=risk.get(h),
            )
            for h in range(1, FORECAST_DAYS + 1)
        )
        if not all(np.isfinite([p.predicted_latitude for p in points])):
            raise ValueError(f"non-finite forecast for {seq.iceberg_id}")
        out.append(GeneratedForecast(sequence=seq, model_version=bundle.version, points=points))
    return out


def filter_horizon(points: Sequence[ForecastPoint], horizon: int) -> list[ForecastPoint]:
    """1-day view -> [D+1]; 3-day -> [D+1..D+3]; 7-day -> all."""
    if not 1 <= horizon <= FORECAST_DAYS:
        raise ValueError(f"horizon must be 1..{FORECAST_DAYS}")
    return [p for p in points if p.horizon_days <= horizon]
