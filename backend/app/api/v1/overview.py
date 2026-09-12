from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import SessionDep, SettingsDep
from app.models.environment import EnvIngestionRun
from app.repositories import queries
from app.schemas.ml import HorizonMetrics
from app.schemas.operations import ChampionSummary, EnvironmentSummary, ForecastRunOut, Overview
from app.services.status_service import feed_state
from ml.features.schemas import get_schema
from ml.versioning.version_manager import FAMILY_TRAJECTORY

router = APIRouter(prefix="/overview", tags=["overview"])


def _schema_description(version: str | None) -> str | None:
    try:
        return get_schema(version or "trajectory_v1").description
    except ValueError:
        return None


@router.get("", response_model=Overview, summary="Mission-control summary for the landing screen")
async def overview(session: SessionDep, settings: SettingsDep) -> Overview:
    now = datetime.now(UTC)
    counts = await queries.iceberg_status_counts(session)
    runs = await queries.ingestion_runs(session, 1)
    latest = runs[0] if runs else None
    success = await queries.last_successful_ingestion(session)
    latest_date = await queries.latest_official_date(session)
    state, reasons = feed_state(
        latest.status if latest else None, latest.new_observations if latest else 0,
        success.fetched_at if success else None, latest_date, settings.feed_degraded_after_hours, settings.stale_after_days, now,
    )
    # The overview's pipeline state describes the iceberg product.
    champion = await queries.deployed_model(session, FAMILY_TRAJECTORY)
    fruns = await queries.forecast_runs(session, 1)
    rruns = await queries.retraining_runs(session, 1)
    agg = await queries.evaluation_aggregates(session, champion.version if champion else None) if champion else []
    pipeline: str = "UNKNOWN"
    if champion is None:
        pipeline = "FAILED"
    elif fruns:
        pipeline = {"success": "LIVE", "partial": "SYNCED", "skipped": "DEGRADED", "failed": "FAILED"}.get(fruns[0].status, "UNKNOWN")
    return Overview(
        generated_at=now,
        tracked_icebergs=sum(counts.values()),
        active_icebergs=counts.get("active", 0),
        not_in_latest_source=counts.get("not_in_latest_source", 0),
        historical_only_icebergs=counts.get("historical_only", 0),
        stale_icebergs=await queries.stale_count(session, (now - timedelta(days=settings.stale_after_days)).date()),
        official_observations=await queries.official_observation_total(session),
        new_observations_last_run=latest.new_observations if latest else 0,
        latest_usnic_update=latest_date,
        last_sync_at=success.completed_at if success else None,
        feed_state=state,
        feed_state_reasons=reasons,
        champion=ChampionSummary(
            version=champion.version, architecture_version=champion.architecture_version,
            adapter_strategy=champion.adapter_strategy, deployed_at=champion.deployed_at,
            day1_error=champion.day1_error, day3_error=champion.day3_error, day7_error=champion.day7_error,
            model_type=champion.model_type, feature_schema_version=champion.feature_schema_version,
            feature_schema_description=_schema_description(champion.feature_schema_version),
            environmental_sources=sorted({
                src["dataset_id"] for g in (champion.environmental_data_sources or {}).values() for src in g.values() if src
            }),
        ) if champion else None,
        environment=EnvironmentSummary(
            enabled=settings.env_enabled,
            configured_groups=sorted({r.group for r in (await session.execute(
                select(EnvIngestionRun).where(EnvIngestionRun.status == "success", EnvIngestionRun.group.is_not(None)).limit(500)
            )).scalars() if r.group}),
            last_sync_at=await session.scalar(
                select(func.max(EnvIngestionRun.completed_at)).where(EnvIngestionRun.status.in_(("success", "partial")))
            ),
        ),
        model_versions=len(await queries.model_versions(session)),
        icebergs_with_active_forecasts=await queries.active_forecast_count(session, champion.version if champion else None),
        last_forecast_run=ForecastRunOut.model_validate(fruns[0]) if fruns else None,
        operational_errors=[
            HorizonMetrics(horizon_days=r.forecast_horizon_days, n=r.n, mae_km=r.mae_km, rmse_km=r.rmse_km,
                           median_km=r.median_km, p90_km=r.p90_km)
            for r in agg
        ],
        evaluations_total=await queries.evaluation_total(session),
        retraining_status=rruns[0].status if rruns else None,
        pipeline_state=pipeline,  # type: ignore[arg-type]
    )
