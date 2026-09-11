from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import Field

from app.schemas.common import ApiModel


class EnvSourceOut(ApiModel):
    group: str
    role: str = Field(description="operational (live forecasts) | historical (training features)")
    provider: str | None
    configured: bool = Field(description="Client installed and credentials present (values are never exposed)")
    reason: str
    authority: str | None = None
    product_id: str | None = None
    dataset_id: str | None = None
    variables: dict[str, str] | None = None
    units: dict[str, str] | None = None
    spatial_resolution_deg: float | None = None
    temporal_resolution: str | None = None
    aggregation: str | None = None
    latency_days: float | None = None
    coverage_start: date | None = None
    coverage_end: date | None = None
    depth: str | None = None
    supports_forecast: bool = False
    forecast_lead_days: int = 0
    notes: str | None = None


class SchemaAvailability(ApiModel):
    version: str
    description: str
    features: list[str]
    groups: list[str]
    trainable: bool
    reason: str


class EnvStatus(ApiModel):
    enabled: bool
    sources: list[EnvSourceOut]
    last_runs: dict[str, dict[str, Any]]
    cache_entries: int
    cache_bytes: int
    latest_cache_fetch: datetime | None
    schemas: list[SchemaAvailability]
    policy: dict[str, Any]


class EnvRunOut(ApiModel):
    id: int
    kind: str
    group: str | None
    provider: str | None
    dataset_id: str | None
    role: str | None
    status: str
    records: int
    bytes: int | None
    duration_ms: int | None
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class VectorSummary(ApiModel):
    speed_m_s: float
    direction_deg: float
    direction_convention: str = Field(description="'from' (meteorological, wind) or 'towards' (oceanographic, current)")


class EnvGroupValue(ApiModel):
    group: str
    provider: str
    dataset_id: str
    values: dict[str, float | None]
    units: dict[str, str]
    valid_date: date | None
    as_of: date
    staleness_days: int | None
    interpolation: str
    valid_neighbours: int
    missing: bool
    reason: str | None
    quality_flags: list[str]
    vector: VectorSummary | None = None


class IcebergEnvironment(ApiModel):
    iceberg_id: str
    observation_id: int | None
    observation_date: date | None
    aligned: bool
    reason: str | None = None
    groups: list[EnvGroupValue] = []


class EnvField(ApiModel):
    group: str
    variables: list[str]
    valid_date: str
    provider: str | None
    dataset_id: str | None
    resolution_deg: float
    points: list[list[float]] = Field(description="[lat, lon, value_1, value_2?] per grid cell (WGS84)")
