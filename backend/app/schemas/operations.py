from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from app.schemas.common import ApiModel
from app.schemas.ml import HorizonMetrics

FeedStateLiteral = Literal["LIVE", "SYNCED", "DEGRADED", "FAILED", "UNKNOWN"]


class IngestionRunOut(ApiModel):
    id: int
    source: str
    source_url: str | None
    discovery_method: str | None
    trigger: str
    fetched_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    http_status: int | None
    checksum_sha256: str | None
    row_count: int
    valid_rows: int
    new_observations: int
    updated_observations: int
    duplicate_observations: int
    failed_rows: int
    missing_from_source: int
    source_latest_update: date | None
    status: str
    error_message: str | None


class RowErrorOut(ApiModel):
    row_number: int
    raw_row: dict[str, Any]
    errors: list[str]


class IngestionStatus(ApiModel):
    feed_state: FeedStateLiteral
    state_reasons: list[str]
    latest_run: IngestionRunOut | None
    last_successful_run: IngestionRunOut | None
    latest_official_observation_date: date | None
    recent_runs: list[IngestionRunOut]
    latest_row_errors: list[RowErrorOut]


class ForecastRunOut(ApiModel):
    id: int
    model_version: str | None
    trigger: str
    ingestion_run_id: int | None
    status: str
    icebergs_considered: int
    forecast_sets_created: int
    already_forecast: int
    skipped: dict[str, int] | None
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class RetrainingRunOut(ApiModel):
    id: int
    run_id: str
    trigger: str
    status: str
    champion_version: str | None
    candidate_version: str | None
    source_data_cutoff: datetime | None
    sample_count: int | None
    eligibility: dict[str, Any] | None
    decision: dict[str, Any] | None
    metrics: dict[str, Any] | None
    started_at: datetime
    completed_at: datetime | None
    failure_reason: str | None
    experiment: dict[str, Any] | None = None


class RetrainingStatus(ApiModel):
    policy: dict[str, Any]
    eligibility: dict[str, Any] | None
    latest_run: RetrainingRunOut | None
    recent_runs: list[RetrainingRunOut]


class FeedOut(ApiModel):
    id: str
    name: str
    provider: str
    description: str
    product_url: str
    source_url: str | None
    cadence: str
    poll_interval_hours: float
    state: FeedStateLiteral
    state_reasons: list[str]
    last_fetch_at: datetime | None
    last_success_at: datetime | None
    latest_official_observation_date: date | None
    fetch_duration_ms: int | None
    checksum_sha256: str | None
    record_count: int | None
    discovery_method: str | None
    error_message: str | None
    # iceberg and sea_ice are CORE model feeds; environmental covers the
    # future feature sources (wind, current, ...).
    category: str = "iceberg"  # iceberg | sea_ice | environmental
    configured: bool = True


class ChampionSummary(ApiModel):
    version: str
    architecture_version: str
    adapter_strategy: str | None
    deployed_at: datetime | None
    day1_error: float | None
    day3_error: float | None
    day7_error: float | None
    model_type: str = "trajectory"
    feature_schema_version: str = "trajectory_v1"
    feature_schema_description: str | None = None
    environmental_sources: list[str] = []


class EnvironmentSummary(ApiModel):
    enabled: bool
    configured_groups: list[str]
    last_sync_at: datetime | None


class Overview(ApiModel):
    generated_at: datetime
    tracked_icebergs: int
    active_icebergs: int
    not_in_latest_source: int
    historical_only_icebergs: int
    stale_icebergs: int
    official_observations: int
    new_observations_last_run: int
    latest_usnic_update: date | None
    last_sync_at: datetime | None
    feed_state: FeedStateLiteral
    feed_state_reasons: list[str]
    champion: ChampionSummary | None
    model_versions: int
    icebergs_with_active_forecasts: int
    last_forecast_run: ForecastRunOut | None
    operational_errors: list[HorizonMetrics]
    evaluations_total: int
    retraining_status: str | None
    pipeline_state: FeedStateLiteral
    environment: EnvironmentSummary | None = None


class HealthCheck(ApiModel):
    status: Literal["ok", "degraded", "unavailable"]
    checks: dict[str, dict[str, Any]]
    version: str
