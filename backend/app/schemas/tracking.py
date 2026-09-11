from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from app.schemas.common import ApiModel, Provenance


class ObservationOut(ApiModel):
    id: int
    iceberg_id: str
    observation_date: date = Field(description="Authoritative observation date (USNIC 'Last Update')")
    latitude: float = Field(description="WGS84 / EPSG:4326 degrees")
    longitude: float = Field(description="WGS84 / EPSG:4326 degrees")
    length_nm: float | None = None
    width_nm: float | None = None
    area_sq_nm: float | None = None
    area_sq_km: float | None = None
    source: str
    source_file: str | None = None
    fetched_at: datetime | None = Field(None, description="When our backend downloaded the source file")
    provenance: Provenance
    revision: int
    ingestion_run_id: int | None = None


class IcebergSummary(ApiModel):
    iceberg_id: str
    status: str = Field(description="active | not_in_latest_source | historical_only")
    latitude: float | None = None
    longitude: float | None = None
    last_update: date | None = Field(None, description="Date of the latest official USNIC observation")
    length_nm: float | None = None
    width_nm: float | None = None
    area_sq_nm: float | None = None
    area_sq_km: float | None = None
    source: str | None = None
    provenance: Provenance | None = None
    latest_observation_id: int | None = None
    is_stale: bool
    days_since_update: int | None = None
    first_seen: date | None = None
    last_seen: date | None = None
    has_forecast: bool = False
    latest_forecast_model_version: str | None = None
    latest_forecast_generated_at: datetime | None = None


class IcebergDetail(IcebergSummary):
    first_official_seen: date | None = None
    last_official_seen: date | None = None
    official_observation_count: int
    historical_observation_count: int
    status_changed_at: datetime | None = None


class NearbyObservation(ObservationOut):
    distance_km: float
