from fastapi import APIRouter

from app.api.v1 import basemap, environment, feeds, forecasts, health, icebergs, models, operations, overview

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(overview.router)
api_router.include_router(icebergs.router)
api_router.include_router(forecasts.router)
api_router.include_router(models.router)
api_router.include_router(operations.router)
api_router.include_router(feeds.router)
api_router.include_router(basemap.router)
api_router.include_router(environment.router)
