"""Ingestion against a real PostGIS database."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select, text

from app.models.tracking import Iceberg, IngestionRowError, IngestionRun, Observation, ObservationRevision
from app.services.ingestion_service import IngestionService
from app.services.usnic.source import SourceUnavailableError
from tests.helpers import csv_rows, document

pytestmark = pytest.mark.db


def _count(db, provenance: str = "official_usnic") -> int:  # type: ignore[no-untyped-def]
    return db.scalar(select(func.count()).select_from(Observation).where(Observation.provenance == provenance))


def test_first_ingestion_persists_everything(db, settings, fixture_csv) -> None:  # type: ignore[no-untyped-def]
    out = IngestionService(db, settings).run(document=document(fixture_csv))
    assert out.status == "success"
    assert len(out.new_observation_ids) == 33 and out.duplicates == 0
    assert _count(db) == 33
    run = db.get(IngestionRun, out.run_id)
    assert run.checksum_sha256 and run.row_count == 33 and run.source_latest_update == date(2026, 9, 10)
    a81 = db.execute(select(Observation).where(Observation.iceberg_id == "A81")).scalar_one()
    assert a81.observation_date == date(2026, 9, 10)  # Last Update, not fetch date
    assert a81.fetched_at.date() == date(2026, 9, 11)
    assert a81.provenance == "official_usnic"
    # PostGIS geometry is WGS84 (lon, lat)
    x, y, srid = db.execute(text("SELECT ST_X(geom), ST_Y(geom), ST_SRID(geom) FROM tracking.observations WHERE id = :i"), {"i": a81.id}).one()
    assert (x, y, srid) == (pytest.approx(-47.22), pytest.approx(-57.36), 4326)
    assert db.scalar(select(func.count()).select_from(Iceberg).where(Iceberg.status == "active")) == 33


def test_refetching_same_csv_creates_no_duplicates(db, settings, fixture_csv) -> None:  # type: ignore[no-untyped-def]
    IngestionService(db, settings).run(document=document(fixture_csv))
    again = IngestionService(db, settings).run(document=document(fixture_csv))
    assert again.status == "unchanged"
    assert again.new_observation_ids == [] and again.duplicates == 33
    assert _count(db) == 33
    assert db.scalar(select(func.count()).select_from(IngestionRun)) == 2  # fetch events kept for audit


def test_source_correction_is_archived_not_overwritten_silently(db, settings) -> None:  # type: ignore[no-untyped-def]
    IngestionService(db, settings).run(document=document(csv_rows("A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026")))
    out = IngestionService(db, settings).run(document=document(csv_rows("A81,28,25,-57.40,-47.30,518.07,391.20,1341.79,09/10/2026")))
    assert out.updated_observation_ids and not out.new_observation_ids
    obs = db.execute(select(Observation).where(Observation.iceberg_id == "A81")).scalar_one()
    assert (obs.latitude, obs.revision) == (-57.40, 2)
    rev = db.execute(select(ObservationRevision)).scalar_one()
    assert rev.previous_values["latitude"] == -57.36 and rev.revision == 1


def test_disappearing_iceberg_keeps_history(db, settings) -> None:  # type: ignore[no-untyped-def]
    IngestionService(db, settings).run(document=document(csv_rows(
        "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/03/2026",
        "D37,30,7,-69.21,36.36,184.47,139.29,477.77,09/03/2026",
    )))
    out = IngestionService(db, settings).run(document=document(csv_rows("A81,28,25,-57.50,-47.10,518.07,391.20,1341.79,09/10/2026")))
    assert out.missing_from_source == 1
    d37 = db.execute(select(Iceberg).where(Iceberg.iceberg_id == "D37")).scalar_one()
    assert d37.status == "not_in_latest_source"
    assert db.scalar(select(func.count()).select_from(Observation).where(Observation.iceberg_id == "D37")) == 1
    assert _count(db) == 3


def test_malformed_rows_recorded_valid_rows_kept(db, settings) -> None:  # type: ignore[no-untyped-def]
    out = IngestionService(db, settings).run(document=document(csv_rows(
        "A81,28,25,-57.36,-47.22,518.07,391.20,1341.79,09/10/2026",
        "A83,12,7,oops,-50.61,73.29,55.34,189.82,09/10/2026",
    )))
    assert out.status == "partial" and out.failed_rows == 1
    err = db.execute(select(IngestionRowError)).scalar_one()
    assert err.row_number == 3 and err.raw_row["Iceberg"] == "A83"
    assert _count(db) == 1


def test_schema_failure_marks_run_failed(db, settings) -> None:  # type: ignore[no-untyped-def]
    out = IngestionService(db, settings).run(document=document("Name,Where\nfoo,bar\n"))
    assert out.status == "failed" and "missing required column" in out.error
    assert _count(db) == 0


def test_source_unavailable_marks_run_failed(db, settings) -> None:  # type: ignore[no-untyped-def]
    class Down:
        def fetch(self):  # type: ignore[no-untyped-def]
            raise SourceUnavailableError("HTTP 503")

    out = IngestionService(db, settings, client=Down()).run()  # type: ignore[arg-type]
    assert out.status == "failed"
    assert db.get(IngestionRun, out.run_id).error_message == "HTTP 503"


def test_database_rejects_predicted_rows_in_tracking(db) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy.exc import IntegrityError

    db.add(Iceberg(iceberg_id="X1", status="active"))
    db.flush()
    db.add(Observation(iceberg_id="X1", observation_date=date(2026, 1, 1), latitude=-60, longitude=0, geom="SRID=4326;POINT(0 -60)",
                       source="x", provenance="predicted", content_hash="h", revision=1))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
