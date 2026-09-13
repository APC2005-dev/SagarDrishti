"""Sea-ice core product API.

Backed entirely by stored rows and stored grids: the champion comes from the
registry, observations from ``seaice.observations`` (official Copernicus Marine
data only), and fields from the ``.npz`` files those rows index. Nothing is
simulated. When the model or the data window is genuinely unavailable the
response says so with a reason instead of returning placeholder numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import SessionDep, SettingsDep
from app.db.session import sync_engine
from app.models.seaice import (
    SeaIceEvaluation,
    SeaIceForecast,
    SeaIceForecastSet,
    SeaIceObservation,
    SeaIceRun,
)
from app.schemas.common import ERROR_RESPONSES
from app.schemas.seaice import (
    SeaIceEvaluationOut,
    SeaIceField,
    SeaIceForecastOut,
    SeaIceModelOut,
    SeaIceObservationOut,
    SeaIceRunOut,
    SeaIceStatus,
    SeaIceWindowEntry,
)
from app.services.seaice_ingestion_service import SeaIceIngestionService
from app.services.seaice_registry import SeaIceRegistry
from ml.seaice.constants import AUTHORITY, GRID_RESOLUTION_DEG, HORIZONS, WINDOW
from ml.seaice.grid_geo import ICE_EDGE_THRESHOLD, to_points
from ml.seaice.grid_store import load_forecast, load_observation

router = APIRouter(prefix="/sea-ice", tags=["sea-ice"])

DEFAULT_HORIZON = max(HORIZONS)
Horizon = Annotated[int, Query(description=f"Forecast horizon in days; one of {list(HORIZONS)}")]
MinConcentration = Annotated[
    float, Query(ge=0.0, le=1.0, description="Only return cells at or above this concentration")
]


def _model_out(mv: Any) -> SeaIceModelOut:
    return SeaIceModelOut(
        version=mv.version,
        short_version=mv.short_version,
        family=mv.model_family,
        version_number=mv.version_number,
        parent_version=mv.parent_version,
        architecture=mv.architecture,
        architecture_version=mv.architecture_version,
        status=mv.status,
        status_reason=mv.status_reason,
        artifact_origin=mv.artifact_origin,
        input_window_entries=mv.input_sequence_length,
        forecast_horizons_days=list(HORIZONS),
        deployed_at=mv.deployed_at,
        training_data_cutoff=mv.training_data_cutoff,
        day1_rmse=mv.day1_error,
        day3_rmse=mv.day3_error,
        day7_rmse=mv.day7_error,
    )


@router.get("/status", response_model=SeaIceStatus, summary="Sea-ice model, data window and latest forecasts")
def status(settings: SettingsDep) -> SeaIceStatus:
    """Everything the sea-ice screen renders, with explicit reasons when empty."""
    with Session(sync_engine()) as session:
        registry = SeaIceRegistry(session, settings)
        ingestion = SeaIceIngestionService(session, settings)
        configured, reason = ingestion.source.is_configured()

        champion = registry.get_champion()
        model_reason = None if champion else "no sea-ice model is deployed (run: python -m app.cli seaice-bootstrap)"

        count = int(session.execute(select(func.count()).select_from(SeaIceObservation)).scalar_one())
        entries = list(
            session.execute(
                select(SeaIceObservation)
                .where(SeaIceObservation.n_valid_cells > 0)
                .order_by(SeaIceObservation.observation_date.desc())
                .limit(WINDOW)
            ).scalars()
        )[::-1]
        latest = entries[-1] if entries else None
        window_complete = len(entries) == WINDOW

        forecast_reason: str | None = None
        if champion is None:
            forecast_reason = model_reason
        elif not window_complete:
            forecast_reason = (
                f"only {len(entries)} official sea-ice entries stored; "
                f"{WINDOW} chronological entries are required to build an input sequence"
            )

        # Only forecast sets anchored on an observation that actually carries data:
        # a set built from an empty source field has no field to show.
        newest_set = session.execute(
            select(SeaIceForecastSet)
            .join(SeaIceObservation, SeaIceForecastSet.anchor_observation_id == SeaIceObservation.id)
            .where(SeaIceObservation.n_valid_cells > 0)
            .order_by(SeaIceForecastSet.anchor_date.desc(), SeaIceForecastSet.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        forecasts: list[SeaIceForecastOut] = []
        if newest_set is not None:
            rows = session.execute(
                select(SeaIceForecast)
                .where(SeaIceForecast.forecast_set_id == newest_set.id)
                .order_by(SeaIceForecast.horizon_days)
            ).scalars()
            forecasts = [
                SeaIceForecastOut(
                    model_version=newest_set.model_version,
                    anchor_date=newest_set.anchor_date,
                    generated_at=newest_set.generated_at,
                    horizon_days=f.horizon_days,
                    target_date=f.target_date,
                    mean_concentration=f.mean_concentration,
                    input_window_entries=newest_set.input_window_entries,
                    input_entry_dates=list(newest_set.input_entry_dates),
                    input_span_days=newest_set.input_span_days,
                    daily_cadence=newest_set.daily_cadence,
                )
                for f in rows
            ]
        elif forecast_reason is None:
            forecast_reason = "no sea-ice forecast has been generated yet (run: python -m app.cli seaice-forecast)"

        evaluations = list(
            session.execute(
                select(SeaIceEvaluation).order_by(SeaIceEvaluation.evaluated_at.desc(), SeaIceEvaluation.id.desc()).limit(12)
            ).scalars()
        )
        runs = list(
            session.execute(
                select(SeaIceRun).order_by(SeaIceRun.started_at.desc(), SeaIceRun.id.desc()).limit(10)
            ).scalars()
        )
        return SeaIceStatus(
            enabled=settings.seaice_enabled,
            source_configured=configured,
            source_reason=None if configured else reason,
            dataset_id=settings.seaice_dataset_id,
            authority=AUTHORITY,
            model=_model_out(champion) if champion else None,
            model_unavailable_reason=model_reason,
            latest_observation=SeaIceObservationOut.model_validate(latest) if latest else None,
            observation_count=count,
            window_entries_required=WINDOW,
            window=[
                SeaIceWindowEntry(
                    observation_date=e.observation_date,
                    n_valid_cells=e.n_valid_cells,
                    mean_concentration=e.mean_concentration,
                )
                for e in entries
            ],
            window_complete=window_complete,
            forecast_unavailable_reason=forecast_reason,
            latest_forecasts=forecasts,
            recent_evaluations=[SeaIceEvaluationOut.model_validate(e) for e in evaluations],
            recent_runs=[SeaIceRunOut.model_validate(r) for r in runs],
        )


@router.get("/latest", response_model=SeaIceField, responses=ERROR_RESPONSES,
            summary="Latest official sea-ice concentration field")
def latest(
    settings: SettingsDep,
    min_concentration: MinConcentration = ICE_EDGE_THRESHOLD,
    stride: Annotated[int, Query(ge=1, le=10, description="Sub-sample the grid for lighter payloads")] = 1,
) -> SeaIceField:
    with Session(sync_engine()) as session:
        observation = session.execute(
            select(SeaIceObservation)
            .where(SeaIceObservation.n_valid_cells > 0)
            .order_by(SeaIceObservation.observation_date.desc())
            .limit(1)
        ).scalar_one_or_none()
        if observation is None:
            raise HTTPException(404, "no official sea-ice observation stored (run: python -m app.cli seaice-ingest)")
        concentration, mask = load_observation(Path(observation.grid_path))
        return SeaIceField(
            kind="observation",
            valid_date=observation.observation_date,
            horizon_days=None,
            model_version=None,
            anchor_date=None,
            resolution_deg=observation.grid_resolution_deg,
            crs=observation.crs,
            min_concentration=min_concentration,
            mean_concentration=observation.mean_concentration,
            points=to_points(concentration, mask, min_concentration, stride),
        )


@router.get("/forecast", response_model=SeaIceField, responses=ERROR_RESPONSES,
            summary="Latest sea-ice concentration forecast from the deployed champion")
def forecast(
    settings: SettingsDep,
    horizon: Horizon = DEFAULT_HORIZON,
    min_concentration: MinConcentration = ICE_EDGE_THRESHOLD,
    stride: Annotated[int, Query(ge=1, le=10)] = 1,
) -> SeaIceField:
    if horizon not in HORIZONS:
        raise HTTPException(422, f"horizon must be one of {list(HORIZONS)}")
    with Session(sync_engine()) as session:
        row = session.execute(
            select(SeaIceForecast, SeaIceForecastSet)
            .join(SeaIceForecastSet, SeaIceForecast.forecast_set_id == SeaIceForecastSet.id)
            .join(SeaIceObservation, SeaIceForecastSet.anchor_observation_id == SeaIceObservation.id)
            .where(SeaIceForecast.horizon_days == horizon, SeaIceObservation.n_valid_cells > 0)
            .order_by(SeaIceForecastSet.anchor_date.desc(), SeaIceForecastSet.id.desc())
            .limit(1)
        ).first()
        if row is None:
            raise HTTPException(
                404,
                f"no stored sea-ice forecast at D+{horizon}; the model runs only when "
                f"{WINDOW} chronological official entries are available",
            )
        prediction, forecast_set = row[0], row[1]
        anchor = session.get(SeaIceObservation, forecast_set.anchor_observation_id)
        _, mask = load_observation(Path(anchor.grid_path))
        field = load_forecast(Path(prediction.grid_path))
        return SeaIceField(
            kind="forecast",
            valid_date=prediction.target_date,
            horizon_days=prediction.horizon_days,
            model_version=forecast_set.model_version,
            anchor_date=forecast_set.anchor_date,
            resolution_deg=GRID_RESOLUTION_DEG,
            crs=anchor.crs,
            min_concentration=min_concentration,
            mean_concentration=prediction.mean_concentration,
            points=to_points(field, mask, min_concentration, stride),
        )


@router.get("/model", response_model=SeaIceModelOut, responses=ERROR_RESPONSES,
            summary="Deployed sea-ice champion (never the highest version number)")
async def model(session: SessionDep, settings: SettingsDep) -> SeaIceModelOut:
    from app.repositories import queries
    from ml.versioning.version_manager import FAMILY_SEA_ICE

    champion = await queries.deployed_model(session, FAMILY_SEA_ICE)
    if champion is None:
        raise HTTPException(404, "no sea-ice model is deployed")
    return _model_out(champion)


@router.get("/runs", response_model=list[SeaIceRunOut], summary="Sea-ice ingestion / forecast / evaluation runs")
async def runs(session: SessionDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[SeaIceRunOut]:
    rows = (
        await session.execute(
            select(SeaIceRun).order_by(SeaIceRun.started_at.desc(), SeaIceRun.id.desc()).limit(limit)
        )
    ).scalars()
    return [SeaIceRunOut.model_validate(r) for r in rows]
