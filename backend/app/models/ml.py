"""``ml`` schema — registry, forecasts, evaluations, retraining, lineage.

Forecast rows are append-only. A forecast references the model version that
produced it and (through its forecast set) the exact observation ids and
feature tensor used as input.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import ML_SCHEMA, TRACKING_SCHEMA, Base
from app.models.tracking import _in

MODEL_STATUSES = ("candidate", "validated", "deployed", "rejected", "archived")
RETRAINING_STATUSES = ("running", "skipped", "promoted", "rejected", "failed")
FORECAST_RUN_STATUSES = ("running", "success", "partial", "failed", "skipped")


class RetrainingRun(Base):
    __tablename__ = "retraining_runs"
    __table_args__ = (CheckConstraint(_in("status", RETRAINING_STATUSES), name="status"), {"schema": ML_SCHEMA})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    trigger: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONB)
    eligibility: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    champion_version: Mapped[str | None] = mapped_column(String(32))
    candidate_version: Mapped[str | None] = mapped_column(String(32))
    source_data_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sample_count: Mapped[int | None] = mapped_column(Integer)
    dataset_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    # Feature-schema experiment: candidates, selection, ablation, alignment quality.
    experiment: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


MODEL_TYPES = ("base", "trajectory", "environmental", "sea_ice")
# Model families have INDEPENDENT version lineages: trajectory v4 and sea-ice v4
# are unrelated models that merely share a number.
MODEL_FAMILIES = ("trajectory", "sea_ice")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        CheckConstraint(_in("status", MODEL_STATUSES), name="status"),
        CheckConstraint("version_number >= 0", name="version_number"),
        CheckConstraint(_in("model_type", MODEL_TYPES), name="model_type"),
        CheckConstraint(_in("model_family", MODEL_FAMILIES), name="model_family"),
        # Version numbers are scoped to a family, so each lineage counts from 1
        # independently.
        UniqueConstraint("model_family", "version_number", name="uq_model_versions_family_version_number"),
        # At most one champion per family at any time.
        Index("uq_model_versions_single_deployed", "model_family", unique=True,
              postgresql_where=text("status = 'deployed'")),
        {"schema": ML_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # Globally unique storage key. Trajectory keeps its bare identifiers
    # ("base", "v1", ...); other families are qualified ("sea_ice/v1"), which
    # keeps every existing foreign key to this column valid while letting each
    # family number its own lineage. Use ``short_version`` for display.
    version: Mapped[str] = mapped_column(String(32), unique=True)
    model_family: Mapped[str] = mapped_column(String(16), server_default="trajectory", index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    parent_version: Mapped[str | None] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"), index=True)
    architecture: Mapped[str] = mapped_column(String(32))
    architecture_version: Mapped[str] = mapped_column(String(64))
    input_sequence_length: Mapped[int] = mapped_column(Integer)
    input_semantics: Mapped[str] = mapped_column(String(64))
    forecast_horizon_days: Mapped[int] = mapped_column(Integer)
    feature_names: Mapped[list[str]] = mapped_column(JSONB)
    adapter_strategy: Mapped[str | None] = mapped_column(String(32))
    model_path: Mapped[str] = mapped_column(Text)
    scaler_path: Mapped[str] = mapped_column(Text)
    metadata_path: Mapped[str] = mapped_column(Text)
    artifact_sha256: Mapped[dict[str, str]] = mapped_column(JSONB)
    artifact_origin: Mapped[str] = mapped_column(String(64))
    training_data_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_sample_count: Mapped[int | None] = mapped_column(Integer)
    training_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    dataset_manifest: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    validation_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    test_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    day1_error: Mapped[float | None] = mapped_column(Float)
    day3_error: Mapped[float | None] = mapped_column(Float)
    day7_error: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), index=True)
    status_reason: Mapped[str | None] = mapped_column(Text)
    retraining_run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{ML_SCHEMA}.retraining_runs.id"))
    deployed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_type: Mapped[str] = mapped_column(String(16), server_default="trajectory")
    feature_schema_version: Mapped[str] = mapped_column(String(48), server_default="trajectory_v1", index=True)
    environmental_data_sources: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    environmental_data_cutoff: Mapped[date | None] = mapped_column(Date)

    @property
    def short_version(self) -> str:
        """The identifier as its own family numbers it: ``sea_ice/v3`` -> ``v3``."""
        return self.version.rpartition("/")[2]


class ModelStatusEvent(Base):
    __tablename__ = "model_status_events"
    __table_args__ = ({"schema": ML_SCHEMA},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"), index=True)
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(64))
    retraining_run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{ML_SCHEMA}.retraining_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelMetric(Base):
    """Metric records per (model, protocol, horizon). Append-only; latest wins."""

    __tablename__ = "model_metrics"
    __table_args__ = (
        Index("ix_model_metrics_lookup", "model_version", "protocol", "horizon_days", "computed_at"),
        {"schema": ML_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"))
    protocol: Mapped[str] = mapped_column(String(96))
    horizon_days: Mapped[int | None] = mapped_column(Integer)  # NULL = all horizons pooled
    n: Mapped[int] = mapped_column(Integer)
    mae_km: Mapped[float | None] = mapped_column(Float)
    rmse_km: Mapped[float | None] = mapped_column(Float)
    median_km: Mapped[float | None] = mapped_column(Float)
    p90_km: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))
    retraining_run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{ML_SCHEMA}.retraining_runs.id"))
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ForecastRun(Base):
    __tablename__ = "forecast_runs"
    __table_args__ = (CheckConstraint(_in("status", FORECAST_RUN_STATUSES), name="status"), {"schema": ML_SCHEMA})

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version: Mapped[str | None] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"), index=True)
    trigger: Mapped[str] = mapped_column(String(32))
    ingestion_run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.ingestion_runs.id"))
    status: Mapped[str] = mapped_column(String(16))
    icebergs_considered: Mapped[int] = mapped_column(Integer, server_default="0")
    forecast_sets_created: Mapped[int] = mapped_column(Integer, server_default="0")
    already_forecast: Mapped[int] = mapped_column(Integer, server_default="0")
    skipped: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ForecastSet(Base):
    """One GRU inference for one iceberg: the 14 input entries and their features."""

    __tablename__ = "forecast_sets"
    __table_args__ = (
        UniqueConstraint("iceberg_id", "model_version", "anchor_observation_id", name="uq_forecast_sets_iceberg_model_anchor"),
        Index("ix_forecast_sets_iceberg_generated", "iceberg_id", "generated_at"),
        {"schema": ML_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_run_id: Mapped[int] = mapped_column(ForeignKey(f"{ML_SCHEMA}.forecast_runs.id"), index=True)
    iceberg_id: Mapped[str] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.icebergs.iceberg_id"))
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"), index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    anchor_observation_id: Mapped[int] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.observations.id"))
    latest_observation_date: Mapped[date] = mapped_column(Date)
    adapter_version: Mapped[str] = mapped_column(String(64))
    adapter_strategy: Mapped[str] = mapped_column(String(32))
    input_observation_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    input_entries: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    input_features: Mapped[list[list[float]]] = mapped_column(JSONB)
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), server_default="issued")
    feature_schema_version: Mapped[str] = mapped_column(String(48), server_default="trajectory_v1")
    # Per-entry environmental values + provenance (source, valid date, interpolation, quality).
    environment: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    environment_as_of: Mapped[date | None] = mapped_column(Date)
    # Set when an environmental champion could not be used and a trajectory model was.
    fallback: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("forecast_set_id", "forecast_horizon_days", name="uq_forecasts_set_horizon"),
        CheckConstraint("forecast_horizon_days BETWEEN 1 AND 7", name="horizon"),
        CheckConstraint("provenance = 'predicted'", name="provenance"),
        Index("ix_forecasts_iceberg_forecast_date", "iceberg_id", "forecast_date"),
        Index("ix_forecasts_model_horizon", "model_version", "forecast_horizon_days"),
        Index("ix_forecasts_geom", "geom", postgresql_using="gist"),
        {"schema": ML_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_set_id: Mapped[int] = mapped_column(ForeignKey(f"{ML_SCHEMA}.forecast_sets.id"))
    iceberg_id: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    latest_observation_date: Mapped[date] = mapped_column(Date)
    forecast_horizon_days: Mapped[int] = mapped_column(Integer)
    forecast_date: Mapped[date] = mapped_column(Date, index=True)
    predicted_latitude: Mapped[float] = mapped_column(Float)
    predicted_longitude: Mapped[float] = mapped_column(Float)
    predicted_x_m: Mapped[float] = mapped_column(Float)  # EPSG:3031
    predicted_y_m: Mapped[float] = mapped_column(Float)  # EPSG:3031
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    risk_radius_km_p90: Mapped[float | None] = mapped_column(Float)
    provenance: Mapped[str] = mapped_column(String(16), server_default="predicted")
    status: Mapped[str] = mapped_column(String(16), server_default="issued")


class ForecastEvaluation(Base):
    __tablename__ = "forecast_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "forecast_id", "actual_observation_id", "actual_observation_revision", name="uq_forecast_evaluations_forecast_actual"
        ),
        CheckConstraint(
            "evaluation_mode = 'research' OR actual_source = 'official_usnic'", name="official_ground_truth"
        ),
        Index("ix_forecast_evaluations_model_horizon", "model_version", "forecast_horizon_days"),
        Index("ix_forecast_evaluations_iceberg", "iceberg_id", "forecast_date"),
        {"schema": ML_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(ForeignKey(f"{ML_SCHEMA}.forecasts.id"), index=True)
    iceberg_id: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"))
    forecast_horizon_days: Mapped[int] = mapped_column(Integer)
    forecast_date: Mapped[date] = mapped_column(Date)
    predicted_latitude: Mapped[float] = mapped_column(Float)
    predicted_longitude: Mapped[float] = mapped_column(Float)
    actual_latitude: Mapped[float] = mapped_column(Float)
    actual_longitude: Mapped[float] = mapped_column(Float)
    error_km: Mapped[float] = mapped_column(Float)
    actual_observation_id: Mapped[int] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.observations.id"))
    actual_observation_revision: Mapped[int] = mapped_column(Integer)
    actual_source: Mapped[str] = mapped_column(String(32))
    actual_observation_date: Mapped[date] = mapped_column(Date)
    evaluation_mode: Mapped[str] = mapped_column(String(16), server_default="operational")
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
