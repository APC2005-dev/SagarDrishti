"""Offline evaluation of a model bundle on a labelled sequence dataset.

Replicates notebook §6 (inverse-scale, add to anchor, project back to
WGS84, haversine) for all 7 horizons, skipping horizons whose target is NaN.
Also provides the constant-velocity benchmark, kept only as a research
reference — it is never used to produce production forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml.constants import FORECAST_DAYS
from ml.evaluation.metrics import ErrorSummary, summarize
from ml.features.coordinate_transform import haversine_km, polar_m_to_latlon
from ml.inference.predictor import predict_displacements_km
from ml.models.model_loader import ModelBundle
from ml.training.dataset_builder import SequenceDataset


@dataclass(frozen=True)
class EvaluationResult:
    protocol: str
    by_horizon: dict[int, ErrorSummary]
    overall: ErrorSummary

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "overall": self.overall.as_dict(),
            "by_horizon": {str(h): s.as_dict() for h, s in self.by_horizon.items()},
        }


def errors_from_displacements(
    dataset: SequenceDataset, predicted_km: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(n, 7) great-circle errors in km; NaN where the target is unknown."""
    actual_km = dataset.y.reshape(-1, FORECAST_DAYS, 2).astype(np.float64)
    ax = dataset.anchor_x_m[:, None]
    ay = dataset.anchor_y_m[:, None]
    act_lat, act_lon = polar_m_to_latlon(ax + actual_km[..., 0] * 1000, ay + actual_km[..., 1] * 1000)
    pred_lat, pred_lon = polar_m_to_latlon(ax + predicted_km[..., 0] * 1000, ay + predicted_km[..., 1] * 1000)
    err = haversine_km(act_lat, act_lon, pred_lat, pred_lon)
    err[~np.isfinite(actual_km[..., 0])] = np.nan
    return err


def _result(protocol: str, err: NDArray[np.float64]) -> EvaluationResult:
    by_h = {h: summarize(err[:, h - 1]) for h in range(1, FORECAST_DAYS + 1)}
    return EvaluationResult(protocol=protocol, by_horizon=by_h, overall=summarize(err.reshape(-1)))


def evaluate_bundle(bundle: ModelBundle, dataset: SequenceDataset, protocol: str) -> EvaluationResult:
    predicted = predict_displacements_km(bundle, dataset.X)
    return _result(protocol, errors_from_displacements(dataset, predicted))


def evaluate_constant_velocity(dataset: SequenceDataset, protocol: str) -> EvaluationResult:
    """Research benchmark: extrapolate the last entry's velocity (features 2, 3)."""
    horizons = np.arange(1, FORECAST_DAYS + 1, dtype=np.float64)
    vx = dataset.X[:, -1, 2].astype(np.float64)[:, None]
    vy = dataset.X[:, -1, 3].astype(np.float64)[:, None]
    predicted = np.stack([vx * horizons, vy * horizons], axis=-1)
    return _result(protocol, errors_from_displacements(dataset, predicted))
