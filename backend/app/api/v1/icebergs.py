from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import defer

from app.api.deps import Horizon, IcebergId, Limit, Offset, SessionDep, SettingsDep
from app.api.serializers import evaluation_out, forecast_set_out, iceberg_summary, to_summary
from app.models.tracking import Observation
from app.repositories import queries
from app.schemas.common import ERROR_RESPONSES, Page
from app.schemas.ml import EvaluationOut, ForecastSetOut
from app.schemas.tracking import IcebergDetail, IcebergSummary, NearbyObservation, ObservationOut

router = APIRouter(prefix="/icebergs", tags=["icebergs"])
StatusFilter = Literal["current", "official", "active", "not_in_latest_source", "historical_only", "all"]


@router.get("", response_model=Page[IcebergSummary], summary="Currently tracked icebergs with latest official position")
async def list_icebergs(
    session: SessionDep,
    settings: SettingsDep,
    limit: Limit = 100,
    offset: Offset = 0,
    status: Annotated[StatusFilter, Query(description="'current' = present in the latest USNIC file")] = "current",
    q: Annotated[str | None, Query(max_length=32, description="Designator search")] = None,
) -> Page[IcebergSummary]:
    stmt = queries.iceberg_listing(status, q)
    total = await queries.count(session, stmt)
    rows = (await session.execute(stmt.limit(limit).offset(offset))).all()
    return Page(items=[to_summary(r, settings.stale_after_days) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/near", response_model=list[NearbyObservation], summary="Latest official positions within a radius (PostGIS)")
async def near(
    session: SessionDep,
    lat: Annotated[float, Query(ge=-90, le=-30)],
    lon: Annotated[float, Query(ge=-180, le=180)],
    radius_km: Annotated[float, Query(gt=0, le=2000)] = 100,
) -> list[NearbyObservation]:
    lo = queries.latest_official_subquery()
    point = func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326)
    geom = select(Observation.id, Observation.geom).where(Observation.id.in_(select(lo.c.id))).subquery()
    dist = func.ST_Distance(func.geography(geom.c.geom), func.geography(point)) / 1000.0
    q = (
        select(Observation, dist.label("d"))
        .options(defer(Observation.geom))
        .join(geom, geom.c.id == Observation.id)
        .where(func.ST_DWithin(func.geography(geom.c.geom), func.geography(point), radius_km * 1000))
        .order_by("d")
    )
    rows = (await session.execute(q)).all()
    return [NearbyObservation(**ObservationOut.model_validate(o).model_dump(), distance_km=float(d)) for o, d in rows]


@router.get("/{iceberg_id}", response_model=IcebergDetail, responses=ERROR_RESPONSES, summary="Iceberg metadata and latest state")
async def get_iceberg(iceberg_id: IcebergId, session: SessionDep, settings: SettingsDep) -> IcebergDetail:
    row = await queries.get_iceberg_row(session, iceberg_id)
    if row is None:
        raise HTTPException(404, f"iceberg {iceberg_id} not found")
    iceberg = row[0]
    counts = await queries.observation_counts(session, iceberg_id)
    return IcebergDetail(
        **iceberg_summary(*row, settings.stale_after_days),
        first_official_seen=iceberg.first_official_seen,
        last_official_seen=iceberg.last_official_seen,
        official_observation_count=counts.get("official_usnic", 0),
        historical_observation_count=counts.get("historical_training_dataset", 0),
        status_changed_at=iceberg.status_changed_at,
    )


@router.get("/{iceberg_id}/position", response_model=ObservationOut, responses=ERROR_RESPONSES, summary="Latest official USNIC observation")
async def position(iceberg_id: IcebergId, session: SessionDep) -> ObservationOut:
    obs = (
        await session.execute(
            select(Observation).options(defer(Observation.geom))
            .where(Observation.iceberg_id == iceberg_id, Observation.provenance == "official_usnic")
            .order_by(Observation.observation_date.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if obs is None:
        raise HTTPException(404, f"no official observation for {iceberg_id}")
    return ObservationOut.model_validate(obs)


@router.get("/{iceberg_id}/history", response_model=Page[ObservationOut], responses=ERROR_RESPONSES, summary="Historical observations (official by default)")
async def history(
    iceberg_id: IcebergId,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    offset: Offset = 0,
    include_historical: Annotated[bool, Query(description="Also return historical-training-dataset positions")] = False,
) -> Page[ObservationOut]:
    provenances = ["official_usnic"] + (["historical_training_dataset"] if include_historical else [])
    stmt = (
        select(Observation).options(defer(Observation.geom))
        .where(Observation.iceberg_id == iceberg_id, Observation.provenance.in_(provenances))
        .order_by(Observation.observation_date.desc())
    )
    total = await queries.count(session, stmt)
    if total == 0 and await queries.get_iceberg_row(session, iceberg_id) is None:
        raise HTTPException(404, f"iceberg {iceberg_id} not found")
    rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars()
    return Page(items=[ObservationOut.model_validate(o) for o in rows], total=total, limit=limit, offset=offset)


@router.get("/{iceberg_id}/forecast", response_model=ForecastSetOut, responses=ERROR_RESPONSES, summary="Latest 7-day forecast, filtered to horizons 1..h")
async def forecast(
    iceberg_id: IcebergId,
    session: SessionDep,
    settings: SettingsDep,
    horizon: Horizon = 7,
    model_version: Annotated[str | None, Query(description="Latest forecast from a specific model")] = None,
) -> ForecastSetOut:
    rows = await queries.forecast_sets_with_points(session, horizon, [iceberg_id], model_version, with_inputs=True)
    if not rows:
        raise HTTPException(404, f"no forecast for {iceberg_id}")
    champion = await queries.deployed_model(session)
    fs, points, anchor = rows[0]
    return forecast_set_out(fs, points, anchor, horizon, champion.version if champion else None, settings.stale_after_days, with_inputs=True)


@router.get("/{iceberg_id}/evaluations", response_model=list[EvaluationOut], summary="Prediction-vs-actual errors for this iceberg")
async def evaluations(iceberg_id: IcebergId, session: SessionDep, limit: Limit = 50) -> list[EvaluationOut]:
    return [evaluation_out(e) for e in await queries.recent_evaluations(session, iceberg_id=iceberg_id, limit=limit)]
