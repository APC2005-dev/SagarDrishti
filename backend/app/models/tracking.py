"""``tracking`` schema — official operational data, identity and ingestion audit.

Nothing predicted ever lands here. Historical rows are never deleted; source
corrections are applied in place only after the previous values are archived
in ``observation_revisions``.
"""

from __future__ import annotations

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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import TRACKING_SCHEMA, Base

PROVENANCE_VALUES = ("official_usnic", "historical_training_dataset", "derived", "interpolated", "predicted")
ICEBERG_STATUSES = ("active", "not_in_latest_source", "historical_only")
INGESTION_STATUSES = ("running", "success", "partial", "unchanged", "failed")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint(_in("status", INGESTION_STATUSES), name="status"),
        Index("ix_ingestion_runs_source_fetched_at", "source", "fetched_at"),
        {"schema": TRACKING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(64))
    source_url: Mapped[str | None] = mapped_column(Text)
    discovery_method: Mapped[str | None] = mapped_column(String(32))
    trigger: Mapped[str] = mapped_column(String(32), server_default="scheduled")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    http_status: Mapped[int | None] = mapped_column(Integer)
    content_type: Mapped[str | None] = mapped_column(String(128))
    content_length: Mapped[int | None] = mapped_column(Integer)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    raw_file_path: Mapped[str | None] = mapped_column(Text)
    header: Mapped[list[str] | None] = mapped_column(JSONB)
    row_count: Mapped[int] = mapped_column(Integer, server_default="0")
    valid_rows: Mapped[int] = mapped_column(Integer, server_default="0")
    new_observations: Mapped[int] = mapped_column(Integer, server_default="0")
    updated_observations: Mapped[int] = mapped_column(Integer, server_default="0")
    duplicate_observations: Mapped[int] = mapped_column(Integer, server_default="0")
    failed_rows: Mapped[int] = mapped_column(Integer, server_default="0")
    missing_from_source: Mapped[int] = mapped_column(Integer, server_default="0")
    source_latest_update: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16), server_default="running")
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class IngestionRowError(Base):
    __tablename__ = "ingestion_row_errors"
    __table_args__ = ({"schema": TRACKING_SCHEMA},)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ingestion_run_id: Mapped[int] = mapped_column(
        ForeignKey(f"{TRACKING_SCHEMA}.ingestion_runs.id", ondelete="CASCADE"), index=True
    )
    row_number: Mapped[int] = mapped_column(Integer)
    raw_row: Mapped[dict[str, Any]] = mapped_column(JSONB)
    errors: Mapped[list[str]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Iceberg(Base):
    __tablename__ = "icebergs"
    __table_args__ = (
        CheckConstraint(_in("status", ICEBERG_STATUSES), name="status"),
        {"schema": TRACKING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    iceberg_id: Mapped[str] = mapped_column(String(32), unique=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    first_seen: Mapped[date | None] = mapped_column(Date)
    last_seen: Mapped[date | None] = mapped_column(Date)
    first_official_seen: Mapped[date | None] = mapped_column(Date)
    last_official_seen: Mapped[date | None] = mapped_column(Date)
    last_seen_in_source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{TRACKING_SCHEMA}.ingestion_runs.id", ondelete="SET NULL")
    )
    status_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("iceberg_id", "observation_date", "provenance", name="uq_observations_iceberg_date_provenance"),
        CheckConstraint(_in("provenance", PROVENANCE_VALUES), name="provenance"),
        # Observations are *observed* data only: model output lives in ml.forecasts.
        CheckConstraint("provenance NOT IN ('predicted')", name="not_predicted"),
        CheckConstraint("latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180", name="coordinates"),
        CheckConstraint("revision >= 1", name="revision"),
        Index("ix_observations_iceberg_date_desc", "iceberg_id", "observation_date"),
        Index("ix_observations_official_date", "observation_date", postgresql_where="provenance = 'official_usnic'"),
        Index("ix_observations_geom", "geom", postgresql_using="gist"),
        {"schema": TRACKING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    iceberg_id: Mapped[str] = mapped_column(
        String(32), ForeignKey(f"{TRACKING_SCHEMA}.icebergs.iceberg_id", onupdate="CASCADE")
    )
    observation_date: Mapped[date] = mapped_column(Date)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326, spatial_index=False))
    length_nm: Mapped[float | None] = mapped_column(Float)
    width_nm: Mapped[float | None] = mapped_column(Float)
    area_sq_nm: Mapped[float | None] = mapped_column(Float)
    area_sq_km: Mapped[float | None] = mapped_column(Float)
    area_sq_mi: Mapped[float | None] = mapped_column(Float)
    remarks: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64))
    source_file: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{TRACKING_SCHEMA}.ingestion_runs.id", ondelete="SET NULL"), index=True
    )
    provenance: Mapped[str] = mapped_column(String(32), index=True)
    position_sensor: Mapped[str | None] = mapped_column(String(16))
    raw_attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    content_hash: Mapped[str] = mapped_column(String(64))
    revision: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ObservationRevision(Base):
    """Previous values of an official observation that the source later corrected."""

    __tablename__ = "observation_revisions"
    __table_args__ = (
        UniqueConstraint("observation_id", "revision", name="uq_observation_revisions_obs_revision"),
        {"schema": TRACKING_SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey(f"{TRACKING_SCHEMA}.observations.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    previous_values: Mapped[dict[str, Any]] = mapped_column(JSONB)
    replaced_by_ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{TRACKING_SCHEMA}.ingestion_runs.id", ondelete="SET NULL")
    )
    replaced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
