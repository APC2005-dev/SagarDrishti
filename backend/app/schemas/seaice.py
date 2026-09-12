"""Response models for the sea-ice core product.

Every field here is read from stored rows or computed from stored grids. When
data or a model is genuinely unavailable the endpoint says so explicitly rather
than returning placeholder numbers.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import Field

from app.schemas.common import ApiModel


class SeaIceModelOut(ApiModel):
    """The deployed sea-ice champion, straight from the registry."""

    version: str
    short_version: str
    family: str
    version_number: int
    parent_version: str | None
    architecture: str
    architecture_version: str
    status: str
    status_reason: str | None
    artifact_origin: str
    input_window_entries: int = Field(description="Chronological database entries per input sequence (not calendar days)")
    forecast_horizons_days: list[int]
    deployed_at: datetime | None
    training_data_cutoff: datetime | None
    day1_rmse: float | None
    day3_rmse: float | None
    day7_rmse: float | None


class SeaIceObservationOut(ApiModel):
    observation_date: date
    provenance: str
    authority: str
    dataset_id: str
    variable: str
    preprocessing_version: str
    grid_shape: list[int]
    grid_resolution_deg: float
    crs: str
    n_valid_cells: int
    mean_concentration: float | None
    fetched_at: datetime
    source_time: datetime | None


class SeaIceField(ApiModel):
    """A concentration grid rendered as sparse points for the map."""

    kind: str = Field(description="'observation' or 'forecast'")
    valid_date: date
    horizon_days: int | None
    model_version: str | None
    anchor_date: date | None
    resolution_deg: float
    crs: str
    min_concentration: float
    mean_concentration: float | None
    points: list[list[float]] = Field(description="[lat, lon, concentration] per ice-covered cell (WGS84)")


class SeaIceWindowEntry(ApiModel):
    observation_date: date
    n_valid_cells: int
    mean_concentration: float | None


class SeaIceForecastOut(ApiModel):
    model_version: str
    anchor_date: date
    generated_at: datetime
    horizon_days: int
    target_date: date
    mean_concentration: float | None
    input_window_entries: int
    input_entry_dates: list[str]
    input_span_days: int | None
    daily_cadence: bool | None


class SeaIceEvaluationOut(ApiModel):
    model_version: str
    horizon_days: int
    anchor_date: date
    target_date: date
    rmse: float
    mae: float
    persistence_rmse: float | None
    persistence_mae: float | None
    n_valid_cells: int
    evaluated_at: datetime


class SeaIceRunOut(ApiModel):
    id: int
    kind: str
    trigger: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    duration_ms: int | None
    entries_new: int
    entries_duplicate: int
    source_latest_date: date | None
    error_message: str | None


class SeaIceStatus(ApiModel):
    """Everything the sea-ice screen needs, with explicit unavailability."""

    enabled: bool
    source_configured: bool
    source_reason: str | None
    dataset_id: str
    authority: str
    model: SeaIceModelOut | None
    model_unavailable_reason: str | None
    latest_observation: SeaIceObservationOut | None
    observation_count: int
    window_entries_required: int
    window: list[SeaIceWindowEntry]
    window_complete: bool
    forecast_unavailable_reason: str | None
    latest_forecasts: list[SeaIceForecastOut]
    recent_evaluations: list[SeaIceEvaluationOut]
    recent_runs: list[SeaIceRunOut]
