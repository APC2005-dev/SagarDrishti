from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Query

from app.api.deps import Horizon, Limit, Offset, SessionDep, SettingsDep
from app.api.serializers import forecast_set_out
from app.repositories import queries
from app.schemas.common import Page
from app.schemas.ml import ForecastRow, ForecastSetOut
from ml.versioning.version_manager import FAMILY_TRAJECTORY

router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@router.get("/latest", response_model=list[ForecastSetOut], summary="Latest forecast set for every iceberg (map layer)")
async def latest(
    session: SessionDep,
    settings: SettingsDep,
    horizon: Horizon = 7,
    active_only: Annotated[bool, Query(description="Only icebergs present in the latest USNIC file")] = True,
    model_version: Annotated[str | None, Query()] = None,
) -> list[ForecastSetOut]:
    ids = None
    if active_only:
        ids = [r[0].iceberg_id for r in (await session.execute(queries.iceberg_listing("current", None))).all()]
    rows = await queries.forecast_sets_with_points(session, horizon, ids, model_version)
    # This endpoint serves iceberg trajectory forecasts: resolve the TRAJECTORY champion.
    champion = await queries.deployed_model(session, FAMILY_TRAJECTORY)
    cv = champion.version if champion else None
    return [forecast_set_out(fs, pts, anchor, horizon, cv, settings.stale_after_days) for fs, pts, anchor in rows]


@router.get("", response_model=Page[ForecastRow], summary="All stored forecasts (append-only history) with filters")
async def list_forecasts(
    session: SessionDep,
    limit: Limit = 100,
    offset: Offset = 0,
    iceberg: Annotated[str | None, Query(description="Iceberg designator")] = None,
    model_version: Annotated[str | None, Query()] = None,
    horizon: Annotated[int | None, Query(ge=1, le=7, description="Exact horizon")] = None,
    max_horizon: Annotated[int | None, Query(ge=1, le=7, description="Horizons 1..max")] = None,
    forecast_date_from: Annotated[date | None, Query()] = None,
    forecast_date_to: Annotated[date | None, Query()] = None,
) -> Page[ForecastRow]:
    iceberg_id = "".join(ch for ch in iceberg.upper() if ch.isalnum()) if iceberg else None
    stmt = queries.forecast_rows(iceberg_id, model_version, horizon, max_horizon, forecast_date_from, forecast_date_to)
    total = await queries.count(session, stmt)
    rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars()
    items = [
        ForecastRow(
            forecast_id=f.id, forecast_set_id=f.forecast_set_id, iceberg_id=f.iceberg_id, model_version=f.model_version,
            generated_at=f.generated_at, latest_observation_date=f.latest_observation_date, horizon_days=f.forecast_horizon_days,
            forecast_date=f.forecast_date, predicted_latitude=f.predicted_latitude, predicted_longitude=f.predicted_longitude,
            risk_radius_km_p90=f.risk_radius_km_p90,
        )
        for f in rows
    ]
    return Page(items=items, total=total, limit=limit, offset=offset)
