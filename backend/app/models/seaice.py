"""``seaice`` schema — official Copernicus Marine sea-ice fields and their forecasts.

The sea-ice model is a **core model in its own right**, not an environmental
feature: the ``environmental`` schema stores point values sampled for individual
icebergs, whereas this schema stores the full gridded state the U-Net consumes
and predicts.

Grids are large, so the arrays live on disk as compressed ``.npz`` files under
``SEAICE_DATA_DIR`` and these tables index them with full provenance and a
checksum. Nothing here is ever deleted when new data arrives: history is
append-only so past windows stay reproducible.

``observations`` holds **official data only**. Model output goes to
``forecasts``; the two never mix, so a prediction can never be mistaken for
ground truth.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import ML_SCHEMA, SEAICE_SCHEMA, Base
from app.models.tracking import _in

SEAICE_RUN_KINDS = ("ingestion", "forecast", "evaluation", "retraining")
SEAICE_RUN_STATUSES = ("running", "success", "unchanged", "partial", "failed", "skipped")
SEAICE_PROVENANCE = ("official_cmems_osisaf",)


class SeaIceRun(Base):
    """One execution of a sea-ice job: fetch, forecast, evaluation or retraining."""

    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(_in("kind", SEAICE_RUN_KINDS), name="kind"),
        CheckConstraint(_in("status", SEAICE_RUN_STATUSES), name="status"),
        Index("ix_seaice_runs_kind_started", "kind", "started_at"),
        {"schema": SEAICE_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    trigger: Mapped[str] = mapped_column(String(32), server_default="scheduled")
    dataset_id: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    source_latest_date: Mapped[date | None] = mapped_column(Date)
    entries_seen: Mapped[int] = mapped_column(Integer, server_default="0")
    entries_new: Mapped[int] = mapped_column(Integer, server_default="0")
    entries_duplicate: Mapped[int] = mapped_column(Integer, server_default="0")
    status: Mapped[str] = mapped_column(String(16), server_default="running")
    error_message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class SeaIceObservation(Base):
    """One official daily sea-ice concentration field, stored permanently.

    ``observation_date`` is unique: re-fetching a date already stored is a
    duplicate and is skipped, never written twice and never overwritten.
    """

    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("observation_date", name="uq_seaice_observations_date"),
        CheckConstraint(_in("provenance", SEAICE_PROVENANCE), name="provenance"),
        CheckConstraint("n_valid_cells >= 0", name="n_valid_cells"),
        Index("ix_seaice_observations_date", "observation_date"),
        {"schema": SEAICE_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    observation_date: Mapped[date] = mapped_column(Date)
    source_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provenance: Mapped[str] = mapped_column(String(32), server_default="official_cmems_osisaf")
    authority: Mapped[str] = mapped_column(String(128))
    dataset_id: Mapped[str] = mapped_column(String(128))
    variable: Mapped[str] = mapped_column(String(32))
    # Preprocessing actually applied, so a stored grid stays interpretable even
    # if the pipeline's defaults later change.
    preprocessing_version: Mapped[str] = mapped_column(String(32))
    grid_shape: Mapped[list[int]] = mapped_column(JSONB)
    grid_resolution_deg: Mapped[float] = mapped_column(Float)
    crs: Mapped[str] = mapped_column(String(32))
    grid_path: Mapped[str] = mapped_column(Text)
    grid_sha256: Mapped[str] = mapped_column(String(64))
    n_valid_cells: Mapped[int] = mapped_column(Integer)
    mean_concentration: Mapped[float | None] = mapped_column(Float)
    ice_area_km2: Mapped[float | None] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ingestion_run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SeaIceForecastSet(Base):
    """One forecast produced from a 7-entry window by one model version."""

    __tablename__ = "forecast_sets"
    __table_args__ = (
        UniqueConstraint("model_version", "anchor_observation_id", name="uq_seaice_forecast_sets_model_anchor"),
        {"schema": SEAICE_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"), index=True)
    anchor_observation_id: Mapped[int] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.observations.id"))
    anchor_date: Mapped[date] = mapped_column(Date, index=True)
    # The exact 7 entries used, oldest first — the audit trail for the window rule.
    input_observation_ids: Mapped[list[int]] = mapped_column(JSONB)
    input_entry_dates: Mapped[list[str]] = mapped_column(JSONB)
    input_window_entries: Mapped[int] = mapped_column(Integer)
    # Elapsed calendar span of those entries; may exceed the entry count when the
    # source has gaps. Recorded, never used to reshape the window.
    input_span_days: Mapped[int | None] = mapped_column(Integer)
    daily_cadence: Mapped[bool | None] = mapped_column()
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.runs.id"))
    diagnostics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class SeaIceForecast(Base):
    """One predicted concentration grid at one horizon."""

    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint("forecast_set_id", "horizon_days", name="uq_seaice_forecasts_set_horizon"),
        CheckConstraint("horizon_days > 0", name="horizon_days"),
        {"schema": SEAICE_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_set_id: Mapped[int] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.forecast_sets.id"), index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    grid_path: Mapped[str] = mapped_column(Text)
    grid_sha256: Mapped[str] = mapped_column(String(64))
    mean_concentration: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SeaIceEvaluation(Base):
    """A stored forecast scored against the official observation that later arrived.

    Ground truth is official data only — the FK points at ``observations``,
    which model output can never enter.
    """

    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint("forecast_id", "actual_observation_id", name="uq_seaice_evaluations_forecast_actual"),
        Index("ix_seaice_evaluations_model_horizon", "model_version", "horizon_days"),
        {"schema": SEAICE_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_id: Mapped[int] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.forecasts.id"), index=True)
    model_version: Mapped[str] = mapped_column(ForeignKey(f"{ML_SCHEMA}.model_versions.version"))
    horizon_days: Mapped[int] = mapped_column(Integer)
    anchor_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date)
    actual_observation_id: Mapped[int] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.observations.id"))
    rmse: Mapped[float] = mapped_column(Float)
    mae: Mapped[float] = mapped_column(Float)
    persistence_rmse: Mapped[float | None] = mapped_column(Float)
    persistence_mae: Mapped[float | None] = mapped_column(Float)
    n_valid_cells: Mapped[int] = mapped_column(Integer)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey(f"{SEAICE_SCHEMA}.runs.id"))
