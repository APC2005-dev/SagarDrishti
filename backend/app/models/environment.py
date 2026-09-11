"""``environmental`` schema — environmental forcing, kept apart from official
iceberg observations and from predictions.

Large gridded data are **not** stored here: raw tile × month NetCDF subsets
live in the file cache (``ENV_CACHE_DIR``) and this schema indexes them
(``cache_entries``). Point values actually used for an observation or a
forecast are stored with full provenance.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
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

from app.db.base import ENV_SCHEMA, ML_SCHEMA, TRACKING_SCHEMA, Base
from app.models.tracking import _in

ENV_GROUPS = ("wind", "current", "sea_ice")
ENV_RUN_KINDS = ("tile_fetch", "alignment", "overlay", "forecast_snapshot", "backfill")
ENV_RUN_STATUSES = ("running", "success", "partial", "failed", "skipped")


class EnvIngestionRun(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(_in("kind", ENV_RUN_KINDS), name="kind"),
        CheckConstraint(_in("status", ENV_RUN_STATUSES), name="status"),
        Index("ix_env_ingestion_runs_group_started", "group", "started_at"),
        {"schema": ENV_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    kind: Mapped[str] = mapped_column(String(24))
    group: Mapped[str | None] = mapped_column(String(16))
    provider: Mapped[str | None] = mapped_column(String(64))
    dataset_id: Mapped[str | None] = mapped_column(String(128))
    role: Mapped[str | None] = mapped_column(String(16))
    request: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16))
    records: Mapped[int] = mapped_column(Integer, server_default="0")
    bytes: Mapped[int | None] = mapped_column(BigInteger)
    cache_path: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class EnvCacheEntry(Base):
    """Index of one cached tile × month NetCDF subset (the file itself is on disk)."""

    __tablename__ = "cache_entries"
    __table_args__ = (
        UniqueConstraint("provider", "dataset_id", "tile_key", "period_start", name="uq_env_cache_entries_tile_period"),
        Index("ix_env_cache_entries_geom", "geom", postgresql_using="gist"),
        {"schema": ENV_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(String(128))
    group: Mapped[str] = mapped_column(String(16))
    variables: Mapped[list[str]] = mapped_column(JSONB)
    tile_key: Mapped[str] = mapped_column(String(16))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    bbox: Mapped[dict[str, float]] = mapped_column(JSONB)
    geom: Mapped[Any] = mapped_column(Geometry("POLYGON", srid=4326, spatial_index=False))
    spatial_resolution_deg: Mapped[float] = mapped_column(Float)
    file_path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    bytes: Mapped[int] = mapped_column(BigInteger)
    max_valid_date: Mapped[date | None] = mapped_column(Date)
    complete: Mapped[bool] = mapped_column(Boolean)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ObservationEnvironment(Base):
    """Environmental state aligned to one official observation, as known at ``as_of``."""

    __tablename__ = "observation_features"
    __table_args__ = (
        UniqueConstraint("observation_id", "group", "provider", "as_of", name="uq_env_observation_features"),
        CheckConstraint(_in('"group"', ENV_GROUPS), name="group"),  # "group" is a reserved word
        Index("ix_env_observation_features_geom", "geom", postgresql_using="gist"),
        {"schema": ENV_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.observations.id"), index=True)
    iceberg_id: Mapped[str] = mapped_column(String(32), index=True)
    group: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(String(128))
    values: Mapped[dict[str, float | None]] = mapped_column(JSONB)
    units: Mapped[dict[str, str]] = mapped_column(JSONB)
    observation_date: Mapped[date] = mapped_column(Date)
    as_of: Mapped[date] = mapped_column(Date)
    valid_date: Mapped[date | None] = mapped_column(Date)
    staleness_days: Mapped[int | None] = mapped_column(Integer)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    interpolation: Mapped[str] = mapped_column(String(48))
    valid_neighbours: Mapped[int] = mapped_column(Integer)
    missing: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(Text)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB)
    cache_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ForecastEnvironmentSnapshot(Base):
    """Forecast fields *as issued* at forecast time (leakage-free archive for future models)."""

    __tablename__ = "forecast_snapshots"
    __table_args__ = (
        UniqueConstraint("forecast_set_id", "group", "lead_day", name="uq_env_forecast_snapshots"),
        {"schema": ENV_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    forecast_set_id: Mapped[int] = mapped_column(ForeignKey(f"{ML_SCHEMA}.forecast_sets.id"), index=True)
    group: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(64))
    dataset_id: Mapped[str] = mapped_column(String(128))
    issued_on: Mapped[date] = mapped_column(Date)
    lead_day: Mapped[int] = mapped_column(Integer)
    valid_date: Mapped[date] = mapped_column(Date)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    location_basis: Mapped[str] = mapped_column(String(24))
    values: Mapped[dict[str, float | None]] = mapped_column(JSONB)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
