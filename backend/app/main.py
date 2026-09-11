from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError, OperationalError

from app.api.v1 import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger, new_request_id, request_id_var

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
log = get_logger("app")

DESCRIPTION = """
Antarctic iceberg monitoring and GRU trajectory forecasting.

* **Official** positions come only from the USNIC Antarctic iceberg CSV (`provenance = official_usnic`).
* **Forecasts** are produced by one GRU pass per iceberg (D+1..D+7); `horizon=1|3|7` filters that same run.
* Coordinates are WGS84 (EPSG:4326) degrees; projected values are labelled EPSG:3031.
* Responses use camelCase field names.
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    log.info("api_started", env=settings.app_env, version=settings.app_version)
    yield
    log.info("api_stopped")


app = FastAPI(
    title="SAGAR DRISHTI API",
    version=settings.app_version,
    description=DESCRIPTION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "overview", "description": "Mission-control summary"},
        {"name": "icebergs", "description": "Tracked icebergs and official observations"},
        {"name": "forecasts", "description": "Stored 7-day GRU forecasts (append-only)"},
        {"name": "models", "description": "Model registry, lineage and evaluation metrics"},
        {"name": "operations", "description": "Ingestion, forecasting and retraining runs"},
        {"name": "feeds", "description": "Data source status"},
        {"name": "environment", "description": "Wind / ocean current / sea ice sources, alignment and overlays"},
        {"name": "basemap", "description": "Static EPSG:3031 basemap tiles (cached proxy)"},
        {"name": "health", "description": "Probes"},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(OperationalError)
@app.exception_handler(DBAPIError)
async def database_unavailable(request: Request, exc: Exception) -> JSONResponse:
    log.error("database_error", path=request.url.path, error=type(exc).__name__)
    return JSONResponse(status_code=503, content={"detail": "database unavailable", "requestId": request_id_var.get()})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error", path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error", "requestId": request_id_var.get()})


app.include_router(api_router)
