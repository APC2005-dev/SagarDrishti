"""``routing`` schema — canonical ports and computed voyage routes.

Ports are a cached projection of the NGA World Port Index (the authoritative
source); the browser never contacts NGA directly.

A route row is the durable record of one A* computation. It stores the
authoritative geometry in PostGIS plus the exact model versions, forecast runs
and forecast reference time that produced it, so a historical route stays
reproducible and auditable after newer models are promoted. Route rows are never
rewritten when a model is promoted later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import ML_SCHEMA, ROUTING_SCHEMA, Base
from app.models.tracking import _in

PORT_SOURCES = ("nga_world_port_index",)
# Computation lifecycle, then the product's "voyage" lifecycle. Route
# computation succeeding is NOT the same event as a vessel moving; the two are
# modelled explicitly rather than conflated.
ROUTE_STATUSES = (
    "VALIDATING",
    "RUNNING",
    "ROUTE_FOUND",
    "NO_FEASIBLE_ROUTE",
    "FAILED",
    "ACTIVE",
    "COMPLETED",
)


class Port(Base):
    """One canonical maritime port from the NGA World Port Index."""

    __tablename__ = "ports"
    __table_args__ = (
        UniqueConstraint("source", "identifier", name="uq_routing_ports_source_identifier"),
        CheckConstraint(_in("source", PORT_SOURCES), name="source"),
        CheckConstraint("latitude BETWEEN -90 AND 90", name="latitude"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="longitude"),
        Index("ix_routing_ports_normalized_name", "normalized_name"),
        {"schema": ROUTING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(32), server_default="nga_world_port_index")
    identifier: Mapped[str] = mapped_column(String(32))  # WPI port number
    name: Mapped[str] = mapped_column(String(128))
    normalized_name: Mapped[str] = mapped_column(String(128))
    unlocode: Mapped[str | None] = mapped_column(String(16))
    country_code: Mapped[str | None] = mapped_column(String(8))
    country_name: Mapped[str | None] = mapped_column(String(96))
    region_name: Mapped[str | None] = mapped_column(String(96))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Route(Base):
    """One computed voyage route and the environment snapshot that produced it."""

    __tablename__ = "routes"
    __table_args__ = (
        CheckConstraint(_in("status", ROUTE_STATUSES), name="status"),
        CheckConstraint("departure_port_id <> destination_port_id", name="distinct_ports"),
        Index("ix_routing_routes_status_created", "status", "created_at"),
        {"schema": ROUTING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    route_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(24), server_default="VALIDATING")
    error_code: Mapped[str | None] = mapped_column(String(48))
    error_message: Mapped[str | None] = mapped_column(Text)

    departure_port_id: Mapped[int] = mapped_column(ForeignKey(f"{ROUTING_SCHEMA}.ports.id"))
    destination_port_id: Mapped[int] = mapped_column(ForeignKey(f"{ROUTING_SCHEMA}.ports.id"))
    # Requested port position vs the grid cell actually used, both kept so the
    # endpoint mapping is auditable rather than a silent relocation.
    requested_departure_latitude: Mapped[float] = mapped_column(Float)
    requested_departure_longitude: Mapped[float] = mapped_column(Float)
    requested_destination_latitude: Mapped[float] = mapped_column(Float)
    requested_destination_longitude: Mapped[float] = mapped_column(Float)
    resolved_departure_latitude: Mapped[float | None] = mapped_column(Float)
    resolved_departure_longitude: Mapped[float | None] = mapped_column(Float)
    resolved_destination_latitude: Mapped[float | None] = mapped_column(Float)
    resolved_destination_longitude: Mapped[float | None] = mapped_column(Float)
    departure_connector_km: Mapped[float | None] = mapped_column(Float)
    destination_connector_km: Mapped[float | None] = mapped_column(Float)
    connectors_validated: Mapped[bool | None] = mapped_column()

    # --- provenance: which models and which forecasts produced this route ---
    trajectory_model_version: Mapped[str | None] = mapped_column(
        ForeignKey(f"{ML_SCHEMA}.model_versions.version")
    )
    trajectory_forecast_run_id: Mapped[int | None] = mapped_column(BigInteger)
    sea_ice_model_version: Mapped[str | None] = mapped_column(
        ForeignKey(f"{ML_SCHEMA}.model_versions.version")
    )
    sea_ice_forecast_run_id: Mapped[int | None] = mapped_column(BigInteger)
    route_planner_version: Mapped[str] = mapped_column(String(32))
    forecast_reference_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    environment_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # --- metrics, all accumulated from the legs the search actually chose ---
    distance_km: Mapped[float | None] = mapped_column(Float)
    duration_hours: Mapped[float | None] = mapped_column(Float)
    fuel_proxy: Mapped[float | None] = mapped_column(Float)
    sic_exposure_hours: Mapped[float | None] = mapped_column(Float)
    weighted_objective: Mapped[float | None] = mapped_column(Float)
    expansions: Mapped[int | None] = mapped_column(Integer)
    runtime_seconds: Mapped[float | None] = mapped_column(Float)
    # Deliberately a status, not a percentage: the notebook's confidence is not
    # a calibrated probability.
    confidence_status: Mapped[str] = mapped_column(String(24), server_default="not_calibrated")
    curvature: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # --- authoritative geometry ---
    geom: Mapped[Any | None] = mapped_column(Geometry("LINESTRING", srid=4326, spatial_index=True))
    waypoints: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
