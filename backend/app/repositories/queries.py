"""Async read queries used by the API. Writes live in the (sync) services."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from sqlalchemy import Select, and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, defer

from app.models.ml import (
    Forecast,
    ForecastEvaluation,
    ForecastRun,
    ForecastSet,
    ModelMetric,
    ModelStatusEvent,
    ModelVersion,
    RetrainingRun,
)
from app.models.tracking import Iceberg, IngestionRowError, IngestionRun, Observation
from app.services.evaluation_service import aggregate_query

OFFICIAL = "official_usnic"


def latest_official_subquery():  # type: ignore[no-untyped-def]
    """DISTINCT ON: newest official observation per iceberg."""
    return (
        select(Observation)
        .options(defer(Observation.geom))
        .where(Observation.provenance == OFFICIAL)
        .distinct(Observation.iceberg_id)
        .order_by(Observation.iceberg_id, Observation.observation_date.desc(), Observation.id.desc())
        .subquery("latest_official")
    )


def latest_set_subquery(model_version: str | None = None):  # type: ignore[no-untyped-def]
    q = (
        select(ForecastSet.id, ForecastSet.iceberg_id, ForecastSet.model_version, ForecastSet.generated_at)
        .distinct(ForecastSet.iceberg_id)
        .order_by(ForecastSet.iceberg_id, ForecastSet.generated_at.desc(), ForecastSet.id.desc())
    )
    if model_version:
        q = q.where(ForecastSet.model_version == model_version)
    return q.subquery("latest_set")


def iceberg_listing(status: str | None, search: str | None) -> Select:  # type: ignore[type-arg]
    lo = aliased(Observation, latest_official_subquery())
    ls = latest_set_subquery()
    q = (
        select(Iceberg, lo, ls.c.model_version, ls.c.generated_at)
        .outerjoin(lo, lo.iceberg_id == Iceberg.iceberg_id)
        .outerjoin(ls, ls.c.iceberg_id == Iceberg.iceberg_id)
    )
    if status == "current":
        q = q.where(Iceberg.status == "active")
    elif status == "official":
        q = q.where(lo.id.is_not(None))
    elif status and status != "all":
        q = q.where(Iceberg.status == status)
    if search:
        q = q.where(Iceberg.iceberg_id.ilike(f"%{''.join(ch for ch in search.upper() if ch.isalnum())}%"))
    return q.order_by(lo.area_sq_km.desc().nulls_last(), Iceberg.iceberg_id)


async def count(session: AsyncSession, q: Select) -> int:  # type: ignore[type-arg]
    return int(await session.scalar(select(func.count()).select_from(q.order_by(None).subquery())) or 0)


async def get_iceberg_row(session: AsyncSession, iceberg_id: str):  # type: ignore[no-untyped-def]
    q = iceberg_listing("all", None).where(Iceberg.iceberg_id == iceberg_id)
    return (await session.execute(q)).first()


async def observation_counts(session: AsyncSession, iceberg_id: str) -> dict[str, int]:
    rows = await session.execute(
        select(Observation.provenance, func.count()).where(Observation.iceberg_id == iceberg_id).group_by(Observation.provenance)
    )
    return {p: int(n) for p, n in rows.all()}


async def forecast_sets_with_points(
    session: AsyncSession,
    horizon: int,
    iceberg_ids: Sequence[str] | None = None,
    model_version: str | None = None,
    with_inputs: bool = False,
) -> list[tuple[ForecastSet, list[Forecast], Observation]]:
    ls = latest_set_subquery(model_version)
    q = select(ForecastSet).join(ls, ls.c.id == ForecastSet.id)
    if not with_inputs:
        q = q.options(defer(ForecastSet.input_features), defer(ForecastSet.input_entries))
    if iceberg_ids is not None:
        q = q.where(ForecastSet.iceberg_id.in_(iceberg_ids))
    sets = list((await session.execute(q.order_by(ForecastSet.iceberg_id))).scalars())
    if not sets:
        return []
    points = (
        await session.execute(
            select(Forecast)
            .options(defer(Forecast.geom))
            .where(Forecast.forecast_set_id.in_([s.id for s in sets]), Forecast.forecast_horizon_days <= horizon)
            .order_by(Forecast.forecast_set_id, Forecast.forecast_horizon_days)
        )
    ).scalars()
    by_set: dict[int, list[Forecast]] = {}
    for p in points:
        by_set.setdefault(p.forecast_set_id, []).append(p)
    anchors = {
        o.id: o
        for o in (
            await session.execute(
                select(Observation).options(defer(Observation.geom)).where(Observation.id.in_([s.anchor_observation_id for s in sets]))
            )
        ).scalars()
    }
    return [(s, by_set.get(s.id, []), anchors[s.anchor_observation_id]) for s in sets]


def forecast_rows(
    iceberg_id: str | None, model_version: str | None, horizon: int | None, max_horizon: int | None,
    forecast_date_from: date | None, forecast_date_to: date | None,
) -> Select:  # type: ignore[type-arg]
    q = select(Forecast).options(defer(Forecast.geom))
    if iceberg_id:
        q = q.where(Forecast.iceberg_id == iceberg_id)
    if model_version:
        q = q.where(Forecast.model_version == model_version)
    if horizon:
        q = q.where(Forecast.forecast_horizon_days == horizon)
    if max_horizon:
        q = q.where(Forecast.forecast_horizon_days <= max_horizon)
    if forecast_date_from:
        q = q.where(Forecast.forecast_date >= forecast_date_from)
    if forecast_date_to:
        q = q.where(Forecast.forecast_date <= forecast_date_to)
    return q.order_by(Forecast.generated_at.desc(), Forecast.iceberg_id, Forecast.forecast_horizon_days)


async def model_versions(session: AsyncSession) -> list[ModelVersion]:
    return list((await session.execute(select(ModelVersion).order_by(ModelVersion.version_number))).scalars())


async def model_version(session: AsyncSession, version: str) -> ModelVersion | None:
    return (await session.execute(select(ModelVersion).where(ModelVersion.version == version))).scalar_one_or_none()


async def deployed_model(session: AsyncSession) -> ModelVersion | None:
    return (await session.execute(select(ModelVersion).where(ModelVersion.status == "deployed"))).scalar_one_or_none()


async def model_metrics(session: AsyncSession, version: str) -> list[ModelMetric]:
    return list(
        (await session.execute(
            select(ModelMetric).where(ModelMetric.model_version == version)
            .order_by(ModelMetric.protocol, ModelMetric.horizon_days.nulls_last(), ModelMetric.computed_at.desc())
        )).scalars()
    )


async def status_events(session: AsyncSession, version: str) -> list[ModelStatusEvent]:
    return list(
        (await session.execute(
            select(ModelStatusEvent).where(ModelStatusEvent.model_version == version).order_by(ModelStatusEvent.created_at)
        )).scalars()
    )


async def evaluation_aggregates(session: AsyncSession, version: str | None) -> list:  # type: ignore[type-arg]
    if version is None:
        E = ForecastEvaluation
        q = select(
            E.forecast_horizon_days, func.count().label("n"), func.avg(E.error_km).label("mae_km"),
            func.sqrt(func.avg(E.error_km * E.error_km)).label("rmse_km"),
            func.percentile_cont(0.5).within_group(E.error_km).label("median_km"),
            func.percentile_cont(0.9).within_group(E.error_km).label("p90_km"),
        ).where(E.evaluation_mode == "operational").group_by(E.forecast_horizon_days).order_by(E.forecast_horizon_days)
        return list((await session.execute(q)).all())
    return list((await session.execute(aggregate_query(version))).all())


async def recent_evaluations(session: AsyncSession, version: str | None = None, iceberg_id: str | None = None, limit: int = 50) -> list[ForecastEvaluation]:
    q = select(ForecastEvaluation).order_by(ForecastEvaluation.evaluated_at.desc(), ForecastEvaluation.id.desc()).limit(limit)
    if version:
        q = q.where(ForecastEvaluation.model_version == version)
    if iceberg_id:
        q = q.where(ForecastEvaluation.iceberg_id == iceberg_id)
    return list((await session.execute(q)).scalars())


async def ingestion_runs(session: AsyncSession, limit: int = 20) -> list[IngestionRun]:
    return list((await session.execute(select(IngestionRun).order_by(IngestionRun.fetched_at.desc(), IngestionRun.id.desc()).limit(limit))).scalars())


async def last_successful_ingestion(session: AsyncSession) -> IngestionRun | None:
    return (
        await session.execute(
            select(IngestionRun).where(IngestionRun.status.in_(("success", "partial", "unchanged")))
            .order_by(IngestionRun.fetched_at.desc()).limit(1)
        )
    ).scalar_one_or_none()


async def row_errors(session: AsyncSession, run_id: int, limit: int = 50) -> list[IngestionRowError]:
    return list(
        (await session.execute(
            select(IngestionRowError).where(IngestionRowError.ingestion_run_id == run_id).order_by(IngestionRowError.row_number).limit(limit)
        )).scalars()
    )


async def latest_official_date(session: AsyncSession) -> date | None:
    return await session.scalar(select(func.max(Observation.observation_date)).where(Observation.provenance == OFFICIAL))


async def forecast_runs(session: AsyncSession, limit: int = 20) -> list[ForecastRun]:
    return list((await session.execute(select(ForecastRun).order_by(ForecastRun.started_at.desc(), ForecastRun.id.desc()).limit(limit))).scalars())


async def retraining_runs(session: AsyncSession, limit: int = 20) -> list[RetrainingRun]:
    return list((await session.execute(select(RetrainingRun).order_by(RetrainingRun.started_at.desc(), RetrainingRun.id.desc()).limit(limit))).scalars())


async def iceberg_status_counts(session: AsyncSession) -> dict[str, int]:
    rows = await session.execute(select(Iceberg.status, func.count()).group_by(Iceberg.status))
    return {s: int(n) for s, n in rows.all()}


async def active_forecast_count(session: AsyncSession, model_version: str | None) -> int:
    if not model_version:
        return 0
    lo = latest_official_subquery()
    q = (
        select(func.count(func.distinct(ForecastSet.iceberg_id)))
        .join(Iceberg, Iceberg.iceberg_id == ForecastSet.iceberg_id)
        .join(lo, and_(lo.c.iceberg_id == ForecastSet.iceberg_id, lo.c.id == ForecastSet.anchor_observation_id))
        .where(Iceberg.status == "active", ForecastSet.model_version == model_version)
    )
    return int(await session.scalar(q) or 0)


async def stale_count(session: AsyncSession, before: date) -> int:
    lo = latest_official_subquery()
    q = select(func.count()).select_from(Iceberg).join(lo, lo.c.iceberg_id == Iceberg.iceberg_id).where(
        Iceberg.status == "active", lo.c.observation_date < before
    )
    return int(await session.scalar(q) or 0)


async def official_observation_total(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(Observation).where(Observation.provenance == OFFICIAL)) or 0)


async def evaluation_total(session: AsyncSession) -> int:
    return int(await session.scalar(select(func.count()).select_from(ForecastEvaluation)) or 0)


async def model_children(session: AsyncSession, version: str) -> list[str]:
    return list((await session.execute(select(ModelVersion.version).where(ModelVersion.parent_version == version))).scalars())
