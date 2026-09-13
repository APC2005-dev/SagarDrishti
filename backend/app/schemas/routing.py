"""Response models for ports and computed routes.

Metric names say what the numbers are. ``fuel_proxy`` is a dimensionless
relative proxy, not litres or tonnes; ``sic_exposure_hours`` is
concentration-weighted hours, not a collision probability; and confidence is a
*status*, never a percentage, because the route planner's confidence is not
calibrated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ApiModel


class PortOut(ApiModel):
    id: int
    identifier: str = Field(description="World Port Index port number")
    name: str
    country_name: str | None
    region_name: str | None = None
    unlocode: str | None = None
    latitude: float
    longitude: float
    source: str
    in_routing_domain: bool


class RouteModelVersions(ApiModel):
    trajectory: str | None
    sea_ice: str | None
    route_planner: str


class RouteMetrics(ApiModel):
    distance_km: float | None
    duration_hours: float | None
    fuel_proxy: float | None = Field(description="Dimensionless relative proxy, not litres or tonnes")
    sic_exposure_hours: float | None = Field(description="Sea-ice concentration-weighted hours along the route")
    weighted_objective: float | None


class RouteConfidence(ApiModel):
    status: str = Field(description="'not_calibrated' — no calibrated route-safety probability exists")
    percent: None = Field(default=None, description="Always null: no calibrated probability is claimed")


class RouteEndpoint(ApiModel):
    port: PortOut
    requested_latitude: float
    requested_longitude: float
    resolved_latitude: float | None
    resolved_longitude: float | None
    connector_km: float | None


class RouteWaypoint(ApiModel):
    waypoint: int
    elapsed_hours: int
    arrival_time: str
    latitude: float
    longitude: float
    action: str


class RouteOut(ApiModel):
    route_id: str
    status: str
    error_code: str | None
    error_message: str | None
    departure: RouteEndpoint
    destination: RouteEndpoint
    connectors_validated: bool | None
    model_versions: RouteModelVersions
    trajectory_forecast_run_id: int | None
    sea_ice_forecast_run_id: int | None
    forecast_reference_time: datetime | None
    metrics: RouteMetrics
    confidence: RouteConfidence
    curvature: dict[str, Any] | None
    waypoints: list[RouteWaypoint]
    geometry: dict[str, Any] | None = Field(description="GeoJSON LineString, [longitude, latitude] order")
    environment_snapshot: dict[str, Any] | None
    expansions: int | None
    runtime_seconds: float | None
    created_at: datetime


class RouteRequest(ApiModel):
    departure_port: str | None = Field(default=None, description="Port name or World Port Index identifier")
    destination_port: str | None = Field(default=None)
    departure_port_id: str | None = Field(default=None, description="Canonical WPI identifier (preferred)")
    destination_port_id: str | None = Field(default=None)
    max_sic: float | None = Field(
        default=None,
        ge=0.05,
        le=1.0,
        description=(
            "Vessel ice capability: the highest sea-ice concentration the vessel may transit. "
            "Defaults to the route notebook's illustrative 0.30, which is a software-test "
            "configuration rather than a certified ice class."
        ),
    )

    def departure_value(self) -> str | None:
        return self.departure_port_id or self.departure_port

    def destination_value(self) -> str | None:
        return self.destination_port_id or self.destination_port


class RouteSummary(ApiModel):
    """Compact form for the Feeds active-vessel card and route history."""

    route_id: str
    status: str
    departure_name: str
    destination_name: str
    distance_km: float | None
    duration_hours: float | None
    fuel_proxy: float | None
    sic_exposure_hours: float | None
    trajectory_model_version: str | None
    sea_ice_model_version: str | None
    route_planner_version: str
    confidence_status: str
    forecast_reference_time: datetime | None
    created_at: datetime
