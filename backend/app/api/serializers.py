"""ORM rows -> response schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from app.models.ml import Forecast, ForecastEvaluation, ForecastSet
from app.models.tracking import Iceberg, Observation
from app.schemas.ml import AnchorObservation, EvaluationOut, ForecastPoint, ForecastSetOut, InputEntry
from app.schemas.tracking import IcebergSummary
from app.services.status_service import is_stale


def iceberg_summary(iceberg: Iceberg, obs: Observation | None, fc_version: str | None, fc_generated, stale_days: int) -> dict:  # type: ignore[no-untyped-def]
    today = datetime.now(UTC).date()
    return dict(
        iceberg_id=iceberg.iceberg_id,
        status=iceberg.status,
        latitude=obs.latitude if obs else None,
        longitude=obs.longitude if obs else None,
        last_update=obs.observation_date if obs else None,
        length_nm=obs.length_nm if obs else None,
        width_nm=obs.width_nm if obs else None,
        area_sq_nm=obs.area_sq_nm if obs else None,
        area_sq_km=obs.area_sq_km if obs else None,
        source=obs.source if obs else None,
        provenance=obs.provenance if obs else None,
        latest_observation_id=obs.id if obs else None,
        is_stale=is_stale(obs.observation_date if obs else None, stale_days, today),
        days_since_update=(today - obs.observation_date).days if obs else None,
        first_seen=iceberg.first_seen,
        last_seen=iceberg.last_seen,
        has_forecast=fc_version is not None,
        latest_forecast_model_version=fc_version,
        latest_forecast_generated_at=fc_generated,
    )


def to_summary(row, stale_days: int) -> IcebergSummary:  # type: ignore[no-untyped-def]
    iceberg, obs, fc_version, fc_generated = row
    return IcebergSummary(**iceberg_summary(iceberg, obs, fc_version, fc_generated, stale_days))


def forecast_set_out(
    fs: ForecastSet, points: list[Forecast], anchor: Observation, horizon: int, champion: str | None, stale_days: int, with_inputs: bool = False
) -> ForecastSetOut:
    return ForecastSetOut(
        forecast_set_id=fs.id,
        iceberg_id=fs.iceberg_id,
        model_version=fs.model_version,
        is_champion=fs.model_version == champion,
        generated_at=fs.generated_at,
        latest_observation_date=fs.latest_observation_date,
        requested_horizon=horizon,
        is_stale=is_stale(fs.latest_observation_date, stale_days),
        adapter_strategy=fs.adapter_strategy,
        diagnostics=fs.diagnostics,
        anchor=AnchorObservation(
            observation_id=anchor.id, observation_date=anchor.observation_date, latitude=anchor.latitude,
            longitude=anchor.longitude, provenance=anchor.provenance,  # type: ignore[arg-type]
        ),
        points=[
            ForecastPoint(
                forecast_id=p.id, horizon_days=p.forecast_horizon_days, forecast_date=p.forecast_date,
                predicted_latitude=p.predicted_latitude, predicted_longitude=p.predicted_longitude,
                predicted_x_m=p.predicted_x_m, predicted_y_m=p.predicted_y_m, risk_radius_km_p90=p.risk_radius_km_p90,
            )
            for p in points
        ],
        input_entries=[InputEntry(**e) for e in fs.input_entries] if with_inputs else None,
        feature_schema_version=fs.feature_schema_version or "trajectory_v1",
        environment_as_of=fs.environment_as_of,
        anchor_environment=(fs.environment[-1] if fs.environment else None) if with_inputs or fs.environment is not None else None,
        fallback=fs.fallback,
    )


def evaluation_out(e: ForecastEvaluation) -> EvaluationOut:
    return EvaluationOut(
        evaluation_id=e.id, forecast_id=e.forecast_id, iceberg_id=e.iceberg_id, model_version=e.model_version,
        horizon_days=e.forecast_horizon_days, forecast_date=e.forecast_date, predicted_latitude=e.predicted_latitude,
        predicted_longitude=e.predicted_longitude, actual_latitude=e.actual_latitude, actual_longitude=e.actual_longitude,
        error_km=e.error_km, actual_source=e.actual_source, actual_observation_id=e.actual_observation_id,  # type: ignore[arg-type]
        actual_observation_date=e.actual_observation_date, evaluated_at=e.evaluated_at,
    )
