"""Ports and route planning.

The browser talks only to these endpoints; the NGA World Port Index and
Copernicus Marine are reached from the backend. Every failure carries a
machine-readable ``code`` so the UI can explain it without exposing a traceback.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import SettingsDep
from app.core.logging import get_logger
from app.db.session import sync_engine
from app.models.routing import Port, Route
from app.schemas.routing import (
    PortOut,
    RouteConfidence,
    RouteEndpoint,
    RouteMetrics,
    RouteModelVersions,
    RouteOut,
    RouteRequest,
    RouteSummary,
)
from app.services.port_resolver import PortError, PortResolver
from app.services.route_service import RouteError, RouteService

log = get_logger(__name__)

port_router = APIRouter(prefix="/ports", tags=["ports"])
route_router = APIRouter(prefix="/routes", tags=["routes"])


def _camel(value: Any) -> Any:
    """Deep snake_case -> camelCase for raw JSONB payloads.

    ``curvature`` and ``environment_snapshot`` are free-form dicts, so pydantic's
    alias generator never reaches inside them. Converting here keeps the whole
    response in one convention instead of mixing cases on the wire.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            head, *rest = str(key).split("_")
            out[head + "".join(word.title() for word in rest)] = _camel(item)
        return out
    if isinstance(value, list):
        return [_camel(item) for item in value]
    return value


def _fail(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _port_out(port: Port, settings: Any) -> PortOut:
    inside = (
        settings.routing_domain_west <= port.longitude <= settings.routing_domain_east
        and settings.routing_domain_south <= port.latitude <= settings.routing_domain_north
    )
    return PortOut(
        id=port.id,
        identifier=port.identifier,
        name=port.name,
        country_name=port.country_name,
        region_name=port.region_name,
        unlocode=port.unlocode,
        latitude=float(port.latitude),
        longitude=float(port.longitude),
        source=port.source,
        in_routing_domain=inside,
    )


@port_router.get("/search", response_model=list[PortOut], summary="Search the NGA World Port Index")
def search_ports(
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, description="Port name fragment")],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
    domain_only: Annotated[bool, Query(description="Only ports inside the supported routing region")] = True,
) -> list[PortOut]:
    with Session(sync_engine()) as session:
        resolver = PortResolver(session, settings)
        try:
            matches = resolver.search(q, limit=limit, domain_only=domain_only)
        except PortError as exc:
            raise _fail(503, exc.code, exc.message) from exc
        rows = [session.get(Port, m.id) for m in matches]
        session.commit()
        return [_port_out(p, settings) for p in rows if p is not None]


def _route_out(session: Session, route: Route, settings: Any) -> RouteOut:
    departure = session.get(Port, route.departure_port_id)
    destination = session.get(Port, route.destination_port_id)
    geometry = (
        {"type": "LineString", "coordinates": [[w["longitude"], w["latitude"]] for w in route.waypoints]}
        if route.waypoints
        else None
    )
    return RouteOut(
        route_id=route.route_id,
        status=route.status,
        error_code=route.error_code,
        error_message=route.error_message,
        departure=RouteEndpoint(
            port=_port_out(departure, settings),
            requested_latitude=route.requested_departure_latitude,
            requested_longitude=route.requested_departure_longitude,
            resolved_latitude=route.resolved_departure_latitude,
            resolved_longitude=route.resolved_departure_longitude,
            connector_km=route.departure_connector_km,
        ),
        destination=RouteEndpoint(
            port=_port_out(destination, settings),
            requested_latitude=route.requested_destination_latitude,
            requested_longitude=route.requested_destination_longitude,
            resolved_latitude=route.resolved_destination_latitude,
            resolved_longitude=route.resolved_destination_longitude,
            connector_km=route.destination_connector_km,
        ),
        connectors_validated=route.connectors_validated,
        model_versions=RouteModelVersions(
            trajectory=route.trajectory_model_version,
            sea_ice=route.sea_ice_model_version,
            route_planner=route.route_planner_version,
        ),
        trajectory_forecast_run_id=route.trajectory_forecast_run_id,
        sea_ice_forecast_run_id=route.sea_ice_forecast_run_id,
        forecast_reference_time=route.forecast_reference_time,
        metrics=RouteMetrics(
            distance_km=route.distance_km,
            duration_hours=route.duration_hours,
            fuel_proxy=route.fuel_proxy,
            sic_exposure_hours=route.sic_exposure_hours,
            weighted_objective=route.weighted_objective,
        ),
        confidence=RouteConfidence(status=route.confidence_status),
        curvature=_camel(route.curvature),
        waypoints=route.waypoints or [],
        geometry=geometry,
        environment_snapshot=_camel(route.environment_snapshot),
        expansions=route.expansions,
        runtime_seconds=route.runtime_seconds,
        created_at=route.created_at,
    )


def _summary(session: Session, route: Route) -> RouteSummary:
    departure = session.get(Port, route.departure_port_id)
    destination = session.get(Port, route.destination_port_id)
    return RouteSummary(
        route_id=route.route_id,
        status=route.status,
        departure_name=departure.name if departure else "?",
        destination_name=destination.name if destination else "?",
        distance_km=route.distance_km,
        duration_hours=route.duration_hours,
        fuel_proxy=route.fuel_proxy,
        sic_exposure_hours=route.sic_exposure_hours,
        trajectory_model_version=route.trajectory_model_version,
        sea_ice_model_version=route.sea_ice_model_version,
        route_planner_version=route.route_planner_version,
        confidence_status=route.confidence_status,
        forecast_reference_time=route.forecast_reference_time,
        created_at=route.created_at,
    )


@route_router.post("", response_model=RouteOut, summary="Plan a route between two World Port Index ports")
def create_route(request: RouteRequest, settings: SettingsDep) -> RouteOut:
    """Validate the ports, assemble the forecast snapshot and run time-dependent A*.

    Runs synchronously: the search is bounded by ``ROUTE_MAX_RUNTIME_SECONDS``,
    so no background-job infrastructure is introduced for it.
    """
    if not settings.routing_enabled:
        raise _fail(503, "ROUTE_ENGINE_ERROR", "Route planning is disabled.")
    with Session(sync_engine()) as session:
        service = RouteService(session, settings)
        try:
            route = service.plan(
                request.departure_value(), request.destination_value(), request.max_sic
            )
        except PortError as exc:
            session.rollback()
            status = 503 if exc.code == "PORT_SERVICE_UNAVAILABLE" else 422
            raise _fail(status, exc.code, exc.message) from exc
        except RouteError as exc:  # pragma: no cover - plan() records these on the row
            session.rollback()
            raise _fail(422, exc.code, exc.message) from exc
        payload = _route_out(session, route, settings)
        # A failed computation is persisted for auditability but must never
        # become an active vessel.
        session.commit()
        return payload


@route_router.get("", response_model=list[RouteSummary], summary="Route history (newest first)")
def list_routes(settings: SettingsDep, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> list[RouteSummary]:
    with Session(sync_engine()) as session:
        rows = RouteService(session, settings).history(limit)
        return [_summary(session, r) for r in rows]


@route_router.get("/active", response_model=RouteSummary | None, summary="The current active vessel route")
def active_route(settings: SettingsDep) -> RouteSummary | None:
    """Backed by persisted state, so it survives a browser refresh."""
    with Session(sync_engine()) as session:
        route = RouteService(session, settings).active()
        return _summary(session, route) if route else None


@route_router.get("/{route_id}", response_model=RouteOut, summary="One stored route")
def get_route(route_id: str, settings: SettingsDep) -> RouteOut:
    with Session(sync_engine()) as session:
        route = RouteService(session, settings).get(route_id)
        if route is None:
            raise _fail(404, "ROUTE_NOT_FOUND", f"No route {route_id}")
        return _route_out(session, route, settings)
