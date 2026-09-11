"""USNIC ingestion: fetch -> validate -> parse -> deduplicate -> persist.

Deduplication key for official observations is (iceberg_id, observation_date)
where observation_date = USNIC ``Last Update``. Re-downloading an unchanged CSV
therefore inserts nothing; a changed row for an existing key is treated as a
source correction (previous values archived, revision incremented).
Observations are never deleted; icebergs absent from the latest file are only
flagged ``not_in_latest_source`` (no reason is inferred).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, text, tuple_
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.tracking import Iceberg, IngestionRowError, IngestionRun, Observation, ObservationRevision
from app.services.geo import ewkt_point
from app.services.usnic.parser import ParsedObservation, ParseResult, SchemaError, parse_usnic_csv, row_error_payload
from app.services.usnic.source import FetchedDocument, SourceUnavailableError, SourceValidationError, UsnicSourceClient

log = get_logger(__name__)

SOURCE_NAME = "usnic_antarctic_icebergs"
PROVENANCE_OFFICIAL = "official_usnic"
_LOCK_KEY = 0x5A6D_0001  # advisory lock: one ingestion at a time


@dataclass
class IngestionOutcome:
    run_id: int
    status: str
    checksum: str | None = None
    row_count: int = 0
    new_observation_ids: list[int] = field(default_factory=list)
    updated_observation_ids: list[int] = field(default_factory=list)
    duplicates: int = 0
    failed_rows: int = 0
    missing_from_source: int = 0
    error: str | None = None

    @property
    def changed_observation_ids(self) -> list[int]:
        return self.new_observation_ids + self.updated_observation_ids


class IngestionService:
    def __init__(self, session: Session, settings: Settings, client: UsnicSourceClient | None = None) -> None:
        self.session = session
        self.settings = settings
        self.client = client

    # ------------------------------------------------------------------ public
    def run(self, trigger: str = "scheduled", document: FetchedDocument | None = None) -> IngestionOutcome:
        """Run one ingestion. ``document`` injects a pre-fetched payload (CLI file import, tests)."""
        now = datetime.now(UTC)
        run = IngestionRun(source=SOURCE_NAME, trigger=trigger, fetched_at=now, status="running")
        self.session.add(run)
        self.session.commit()
        log.info("ingestion_started", run_id=run.id, trigger=trigger)

        if not self.session.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": _LOCK_KEY}).scalar():
            return self._fail(run, "another ingestion is running")
        try:
            if document is None:
                client = self.client or UsnicSourceClient(self.settings)
                try:
                    document = client.fetch()
                finally:
                    if self.client is None:
                        client.close()
            return self._process(run, document)
        except (SourceUnavailableError, SourceValidationError, SchemaError) as exc:
            self.session.rollback()
            return self._fail(run, str(exc))
        except Exception as exc:
            self.session.rollback()
            log.exception("ingestion_crashed", run_id=run.id)
            return self._fail(run, f"{type(exc).__name__}: {exc}")
        finally:
            self.session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": _LOCK_KEY})
            self.session.commit()

    # ----------------------------------------------------------------- helpers
    def _fail(self, run: IngestionRun, message: str) -> IngestionOutcome:
        run = self.session.merge(run)
        run.status = "failed"
        run.error_message = message
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        log.error("ingestion_failed", run_id=run.id, error=message)
        return IngestionOutcome(run_id=run.id, status="failed", error=message)

    def _archive_raw(self, doc: FetchedDocument, checksum: str) -> str:
        ts = doc.fetched_at
        folder = Path(self.settings.raw_data_dir) / "usnic" / f"{ts:%Y}" / f"{ts:%m}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ts:%Y%m%dT%H%M%SZ}_{checksum[:12]}.csv"
        if not path.exists():
            path.write_bytes(doc.content)
        return str(path)

    def _process(self, run: IngestionRun, doc: FetchedDocument) -> IngestionOutcome:
        checksum = hashlib.sha256(doc.content).hexdigest()
        run.source_url = doc.url
        run.discovery_method = doc.discovery_method
        run.fetched_at = doc.fetched_at
        run.http_status = doc.http_status
        run.content_type = doc.content_type
        run.content_length = len(doc.content)
        run.duration_ms = doc.duration_ms
        run.checksum_sha256 = checksum
        run.raw_file_path = self._archive_raw(doc, checksum)
        log.info("ingestion_fetched", run_id=run.id, url=doc.url, checksum=checksum, bytes=len(doc.content))

        parsed = parse_usnic_csv(doc.content, fetched_on=doc.fetched_at.date())
        run.header = parsed.header
        run.row_count = parsed.row_count
        run.valid_rows = len(parsed.observations)
        run.failed_rows = len(parsed.errors)
        run.source_latest_update = parsed.latest_update
        for err in parsed.errors:
            payload = row_error_payload(err)
            self.session.add(
                IngestionRowError(
                    ingestion_run_id=run.id,
                    row_number=err.row_number,
                    raw_row=payload["raw_row"],
                    errors=list(err.errors),
                )
            )
        if not parsed.observations:
            raise SchemaError(f"no valid rows in file ({len(parsed.errors)} rejected)")

        outcome = IngestionOutcome(run_id=run.id, status="success", checksum=checksum, row_count=parsed.row_count)
        self._upsert_observations(run, doc, parsed, outcome)
        outcome.missing_from_source = self._update_icebergs(run, parsed)

        previous = self.session.execute(
            select(IngestionRun.checksum_sha256)
            .where(IngestionRun.id != run.id, IngestionRun.status.in_(("success", "partial", "unchanged")))
            .order_by(IngestionRun.fetched_at.desc())
            .limit(1)
        ).scalar()
        if parsed.errors:
            outcome.status = "partial"
        elif not outcome.changed_observation_ids and previous == checksum:
            outcome.status = "unchanged"
        outcome.failed_rows = len(parsed.errors)

        run.new_observations = len(outcome.new_observation_ids)
        run.updated_observations = len(outcome.updated_observation_ids)
        run.duplicate_observations = outcome.duplicates
        run.missing_from_source = outcome.missing_from_source
        run.status = outcome.status
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        log.info(
            "ingestion_completed",
            run_id=run.id,
            status=run.status,
            checksum=checksum,
            rows=run.row_count,
            new=run.new_observations,
            updated=run.updated_observations,
            duplicates=run.duplicate_observations,
            failed=run.failed_rows,
            missing_from_source=run.missing_from_source,
            latest_update=str(parsed.latest_update),
        )
        return outcome

    def _ensure_icebergs(self, ids: set[str]) -> None:
        existing = set(self.session.execute(select(Iceberg.iceberg_id).where(Iceberg.iceberg_id.in_(ids))).scalars())
        for iceberg_id in sorted(ids - existing):
            self.session.add(Iceberg(iceberg_id=iceberg_id, status="active", status_changed_at=datetime.now(UTC)))
        self.session.flush()

    def _upsert_observations(
        self, run: IngestionRun, doc: FetchedDocument, parsed: ParseResult, outcome: IngestionOutcome
    ) -> None:
        self._ensure_icebergs({o.iceberg_id for o in parsed.observations})
        keys = [(o.iceberg_id, o.observation_date) for o in parsed.observations]
        existing = {
            (row.iceberg_id, row.observation_date): row
            for row in self.session.execute(
                select(Observation)
                .where(Observation.provenance == PROVENANCE_OFFICIAL)
                .where(tuple_(Observation.iceberg_id, Observation.observation_date).in_(keys))
                .with_for_update()
            ).scalars()
        }
        new_rows: list[Observation] = []
        for obs in parsed.observations:
            current = existing.get((obs.iceberg_id, obs.observation_date))
            if current is None:
                row = self._new_observation(run, doc, obs)
                self.session.add(row)
                new_rows.append(row)
            elif current.content_hash == obs.content_hash:
                outcome.duplicates += 1
            else:
                self._apply_correction(run, doc, current, obs)
                outcome.updated_observation_ids.append(current.id)
        self.session.flush()
        outcome.new_observation_ids = [r.id for r in new_rows]

    @staticmethod
    def _values(obs: ParsedObservation) -> dict[str, object]:
        return {
            "latitude": obs.latitude,
            "longitude": obs.longitude,
            "length_nm": obs.length_nm,
            "width_nm": obs.width_nm,
            "area_sq_nm": obs.area_sq_nm,
            "area_sq_km": obs.area_sq_km,
            "area_sq_mi": obs.area_sq_mi,
            "remarks": obs.remarks,
        }

    def _new_observation(self, run: IngestionRun, doc: FetchedDocument, obs: ParsedObservation) -> Observation:
        return Observation(
            iceberg_id=obs.iceberg_id,
            observation_date=obs.observation_date,
            geom=ewkt_point(obs.latitude, obs.longitude),
            source=SOURCE_NAME,
            source_file=doc.url,
            fetched_at=doc.fetched_at,
            ingestion_run_id=run.id,
            provenance=PROVENANCE_OFFICIAL,
            content_hash=obs.content_hash,
            revision=1,
            raw_attributes={"raw_iceberg": obs.raw_iceberg, "row_number": obs.row_number},
            **self._values(obs),
        )

    def _apply_correction(
        self, run: IngestionRun, doc: FetchedDocument, current: Observation, obs: ParsedObservation
    ) -> None:
        previous = {k: getattr(current, k) for k in self._values(obs)}
        previous.update(
            content_hash=current.content_hash,
            source_file=current.source_file,
            fetched_at=current.fetched_at.isoformat() if current.fetched_at else None,
            ingestion_run_id=current.ingestion_run_id,
        )
        self.session.add(
            ObservationRevision(
                observation_id=current.id,
                revision=current.revision,
                previous_values=previous,
                replaced_by_ingestion_run_id=run.id,
            )
        )
        for key, value in self._values(obs).items():
            setattr(current, key, value)
        current.geom = ewkt_point(obs.latitude, obs.longitude)
        current.content_hash = obs.content_hash
        current.revision += 1
        current.source_file = doc.url
        current.fetched_at = doc.fetched_at
        current.ingestion_run_id = run.id
        log.warning("observation_corrected_by_source", iceberg_id=obs.iceberg_id, date=str(obs.observation_date))

    def _update_icebergs(self, run: IngestionRun, parsed: ParseResult) -> int:
        in_file: dict[str, list[ParsedObservation]] = {}
        for o in parsed.observations:
            in_file.setdefault(o.iceberg_id, []).append(o)
        now = datetime.now(UTC)
        for iceberg in self.session.execute(select(Iceberg).where(Iceberg.iceberg_id.in_(in_file))).scalars():
            dates = [o.observation_date for o in in_file[iceberg.iceberg_id]]
            lo, hi = min(dates), max(dates)
            iceberg.first_seen = min(filter(None, [iceberg.first_seen, lo]))
            iceberg.last_seen = max(filter(None, [iceberg.last_seen, hi]))
            iceberg.first_official_seen = min(filter(None, [iceberg.first_official_seen, lo]))
            iceberg.last_official_seen = max(filter(None, [iceberg.last_official_seen, hi]))
            iceberg.last_seen_in_source_run_id = run.id
            if iceberg.status != "active":
                iceberg.status = "active"
                iceberg.status_changed_at = now
        missing = 0
        for iceberg in self.session.execute(
            select(Iceberg).where(Iceberg.status == "active", Iceberg.iceberg_id.not_in(in_file))
        ).scalars():
            iceberg.status = "not_in_latest_source"
            iceberg.status_changed_at = now
            missing += 1
            log.info("iceberg_not_in_latest_source", iceberg_id=iceberg.iceberg_id, run_id=run.id)
        return missing
