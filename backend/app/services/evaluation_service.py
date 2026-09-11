"""Prediction-vs-actual evaluation against official USNIC observations only.

A forecast (iceberg, forecast_date) is matched to an official observation
(iceberg, observation_date == forecast_date). With a weekly source, most
matches land on D+7; other horizons are evaluated whenever the source cadence
allows. Evaluations are append-only; a source correction (new observation
revision) yields an additional evaluation row, keeping both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.ml import Forecast, ForecastEvaluation
from app.models.tracking import Observation
from ml.features.coordinate_transform import haversine_km

log = get_logger(__name__)
OFFICIAL = "official_usnic"


@dataclass
class EvaluationOutcome:
    evaluated: int
    by_horizon: dict[int, int]


class EvaluationService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def evaluate(self, observation_ids: list[int] | None = None) -> EvaluationOutcome:
        E = ForecastEvaluation
        q = (
            select(Forecast, Observation)
            .join(Observation, and_(Observation.iceberg_id == Forecast.iceberg_id,
                                    Observation.observation_date == Forecast.forecast_date,
                                    Observation.provenance == OFFICIAL))
            .outerjoin(E, and_(E.forecast_id == Forecast.id, E.actual_observation_id == Observation.id,
                               E.actual_observation_revision == Observation.revision))
            .where(E.id.is_(None))
        )
        if observation_ids is not None:
            if not observation_ids:
                return EvaluationOutcome(0, {})
            q = q.where(Observation.id.in_(observation_ids))
        pairs = self.session.execute(q).all()
        if not pairs:
            return EvaluationOutcome(0, {})
        err = haversine_km(
            np.array([f.predicted_latitude for f, _ in pairs]), np.array([f.predicted_longitude for f, _ in pairs]),
            np.array([o.latitude for _, o in pairs]), np.array([o.longitude for _, o in pairs]),
        )
        counts: dict[int, int] = {}
        for (f, o), e in zip(pairs, err, strict=True):
            self.session.add(
                ForecastEvaluation(
                    forecast_id=f.id, iceberg_id=f.iceberg_id, model_version=f.model_version,
                    forecast_horizon_days=f.forecast_horizon_days, forecast_date=f.forecast_date,
                    predicted_latitude=f.predicted_latitude, predicted_longitude=f.predicted_longitude,
                    actual_latitude=o.latitude, actual_longitude=o.longitude, error_km=float(e),
                    actual_observation_id=o.id, actual_observation_revision=o.revision, actual_source=o.provenance,
                    actual_observation_date=o.observation_date, evaluation_mode="operational",
                )
            )
            counts[f.forecast_horizon_days] = counts.get(f.forecast_horizon_days, 0) + 1
        self.session.commit()
        log.info("forecast_evaluation_completed", evaluated=len(pairs), by_horizon=counts)
        return EvaluationOutcome(len(pairs), counts)


def aggregate_query(model_version: str | None = None):  # type: ignore[no-untyped-def]
    """SQL aggregate (MAE, RMSE, median, p90) per horizon over operational evaluations."""
    E = ForecastEvaluation
    q = select(
        E.model_version,
        E.forecast_horizon_days,
        func.count().label("n"),
        func.avg(E.error_km).label("mae_km"),
        func.sqrt(func.avg(E.error_km * E.error_km)).label("rmse_km"),
        func.percentile_cont(0.5).within_group(E.error_km).label("median_km"),
        func.percentile_cont(0.9).within_group(E.error_km).label("p90_km"),
    ).where(E.evaluation_mode == "operational")
    if model_version:
        q = q.where(E.model_version == model_version)
    return q.group_by(E.model_version, E.forecast_horizon_days).order_by(E.model_version, E.forecast_horizon_days)
