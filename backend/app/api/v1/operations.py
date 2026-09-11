from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import Limit, SessionDep, SettingsDep
from app.db.session import sync_engine
from app.models.ml import RetrainingRun
from app.repositories import queries
from app.schemas.operations import (
    ForecastRunOut,
    IngestionRunOut,
    IngestionStatus,
    RetrainingRunOut,
    RetrainingStatus,
    RowErrorOut,
)
from app.services.status_service import feed_state

router = APIRouter(prefix="/operations", tags=["operations"])


def _retraining_out(r: RetrainingRun) -> RetrainingRunOut:
    metrics = dict(r.metrics or {})
    metrics.pop("traceback", None)
    metrics.pop("history", None)
    return RetrainingRunOut(
        id=r.id, run_id=str(r.run_id), trigger=r.trigger, status=r.status, champion_version=r.champion_version,
        candidate_version=r.candidate_version, source_data_cutoff=r.source_data_cutoff, sample_count=r.sample_count,
        eligibility=r.eligibility, decision=r.decision, metrics=metrics, started_at=r.started_at,
        completed_at=r.completed_at, failure_reason=r.failure_reason, experiment=r.experiment,
    )


@router.get("/ingestion", response_model=IngestionStatus, summary="USNIC ingestion status and recent runs")
async def ingestion(session: SessionDep, settings: SettingsDep, limit: Limit = 20) -> IngestionStatus:
    runs = await queries.ingestion_runs(session, limit)
    latest = runs[0] if runs else None
    success = await queries.last_successful_ingestion(session)
    latest_date = await queries.latest_official_date(session)
    state, reasons = feed_state(
        latest.status if latest else None, latest.new_observations if latest else 0,
        success.fetched_at if success else None, latest_date, settings.feed_degraded_after_hours, settings.stale_after_days,
    )
    errors = await queries.row_errors(session, latest.id) if latest else []
    return IngestionStatus(
        feed_state=state, state_reasons=reasons,
        latest_run=IngestionRunOut.model_validate(latest) if latest else None,
        last_successful_run=IngestionRunOut.model_validate(success) if success else None,
        latest_official_observation_date=latest_date,
        recent_runs=[IngestionRunOut.model_validate(r) for r in runs],
        latest_row_errors=[RowErrorOut.model_validate(e) for e in errors],
    )


@router.get("/forecasting", response_model=list[ForecastRunOut], summary="Forecast generation runs")
async def forecasting(session: SessionDep, limit: Limit = 20) -> list[ForecastRunOut]:
    return [ForecastRunOut.model_validate(r) for r in await queries.forecast_runs(session, limit)]


@router.get("/retraining", response_model=RetrainingStatus, summary="Retraining policy, eligibility and history")
async def retraining(session: SessionDep, settings: SettingsDep, limit: Limit = 20) -> RetrainingStatus:
    from sqlalchemy.orm import Session

    from app.services.retraining_service import RetrainingService

    # Eligibility uses the same sync code path as the worker so the two can never disagree.
    with Session(sync_engine()) as sync_session:
        svc = RetrainingService(sync_session, settings)
        policy, eligibility = svc.policy_snapshot(), svc.eligibility()
    runs = await queries.retraining_runs(session, limit)
    latest = (await session.execute(select(RetrainingRun).order_by(RetrainingRun.started_at.desc()).limit(1))).scalar_one_or_none()
    return RetrainingStatus(
        policy=policy, eligibility=eligibility,
        latest_run=_retraining_out(latest) if latest else None,
        recent_runs=[_retraining_out(r) for r in runs],
    )
