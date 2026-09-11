from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Response
from sqlalchemy import text

from app.api.deps import SessionDep, SettingsDep
from app.repositories import queries
from app.schemas.operations import HealthCheck

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthCheck, summary="Liveness probe (process is up)")
async def live(settings: SettingsDep) -> HealthCheck:
    return HealthCheck(status="ok", checks={}, version=settings.app_version)


@router.get("/ready", response_model=HealthCheck, summary="Readiness: database, PostGIS and deployed model artifact")
async def ready(session: SessionDep, settings: SettingsDep, response: Response) -> HealthCheck:
    checks: dict[str, dict] = {}
    status = "ok"
    try:
        version = await session.scalar(text("SELECT postgis_lib_version()"))
        checks["database"] = {"ok": True, "postgis": version}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": type(exc).__name__}
        response.status_code = 503
        return HealthCheck(status="unavailable", checks=checks, version=settings.app_version)
    champion = await queries.deployed_model(session)
    if champion is None:
        checks["model"] = {"ok": False, "error": "no deployed model"}
        status = "degraded"
    else:
        present = Path(champion.model_path).exists() and Path(champion.scaler_path).exists()
        checks["model"] = {"ok": present, "version": champion.version}
        if not present:
            status = "degraded"
    return HealthCheck(status=status, checks=checks, version=settings.app_version)  # type: ignore[arg-type]
