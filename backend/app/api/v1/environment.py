"""Environmental forcing: source status, fetch history, overlay grids.

Endpoints only read what the workers have already fetched and cached; no
request ever triggers a download from Copernicus or CDS.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import Limit, SessionDep, SettingsDep
from app.db.session import sync_engine
from app.models.environment import EnvCacheEntry, EnvIngestionRun
from app.schemas.common import ERROR_RESPONSES
from app.schemas.environment import EnvField, EnvRunOut, EnvSourceOut, EnvStatus, SchemaAvailability
from app.services.environment_service import EnvironmentService
from ml.environment.sampler import INTERPOLATION
from ml.features.schemas import get_schema

router = APIRouter(prefix="/environment", tags=["environment"])
Group = Literal["wind", "current", "sea_ice"]


@router.get("/status", response_model=EnvStatus, summary="Environmental sources, availability, latency and cache")
def status(settings: SettingsDep) -> EnvStatus:
    with Session(sync_engine()) as session:
        env = EnvironmentService(session, settings)
        sources = []
        for st in env.status():
            spec = st.spec or {}
            sources.append(EnvSourceOut(
                group=st.group, role=st.role, provider=st.provider, configured=st.configured, reason=st.reason,
                authority=spec.get("authority"), product_id=spec.get("product_id"), dataset_id=spec.get("dataset_id"),
                variables=spec.get("native_variables"), units=spec.get("units"),
                spatial_resolution_deg=spec.get("spatial_resolution_deg"), temporal_resolution=spec.get("temporal_resolution"),
                aggregation=spec.get("aggregation"), latency_days=spec.get("latency_days"),
                coverage_start=spec.get("coverage_start"), coverage_end=spec.get("coverage_end"), depth=spec.get("depth"),
                supports_forecast=bool(spec.get("supports_forecast")), forecast_lead_days=int(spec.get("forecast_lead_days") or 0),
                notes=spec.get("notes"),
            ))
        n, size, latest = session.execute(
            select(func.count(), func.coalesce(func.sum(EnvCacheEntry.bytes), 0), func.max(EnvCacheEntry.fetched_at))
        ).one()
        schemas = []
        for name in settings.env_candidate_schemas:
            try:
                schema = get_schema(name)
            except ValueError as exc:
                schemas.append(SchemaAvailability(version=name, description="unknown", features=[], groups=[], trainable=False, reason=str(exc)))
                continue
            ok, reason = env.schema_trainable(schema) if schema.is_environmental else (True, "trajectory only")
            schemas.append(SchemaAvailability(version=schema.version, description=schema.description, features=list(schema.features),
                                              groups=list(schema.groups), trainable=ok, reason=reason))
        return EnvStatus(
            enabled=settings.env_enabled, sources=sources, last_runs=env.last_runs(), cache_entries=int(n),
            cache_bytes=int(size), latest_cache_fetch=latest, schemas=schemas,
            policy={
                "as_of_rule": "entry observed on d at prediction time T uses the field valid on min(d, T - operational latency)",
                "interpolation": INTERPOLATION,
                "missing_data": "complete-case for training/evaluation; trajectory-only fallback (recorded) at inference",
                "forecast_period_environment": "not a model input; issued forecast fields archived for future architectures",
                "max_staleness_days": settings.env_max_staleness_days,
            },
        )


@router.get("/runs", response_model=list[EnvRunOut], summary="Environmental fetch / alignment / overlay runs")
async def runs(session: SessionDep, limit: Limit = 50, group: Annotated[Group | None, Query()] = None) -> list[EnvRunOut]:
    q = select(EnvIngestionRun).order_by(EnvIngestionRun.started_at.desc(), EnvIngestionRun.id.desc()).limit(limit)
    if group:
        q = q.where(EnvIngestionRun.group == group)
    return [EnvRunOut.model_validate(r) for r in (await session.execute(q)).scalars()]


@router.get("/field", response_model=EnvField, responses=ERROR_RESPONSES, summary="Coarse overlay grid (cached only)")
def field(settings: SettingsDep, group: Annotated[Group, Query()], day: Annotated[date | None, Query(alias="date")] = None) -> EnvField:
    with Session(sync_engine()) as session:
        data = EnvironmentService(session, settings).overlay_field(group, day)
    if data is None:
        raise HTTPException(404, f"no cached {group} overlay (run the environment overlay job; requires configured credentials)")
    return EnvField(**data)
