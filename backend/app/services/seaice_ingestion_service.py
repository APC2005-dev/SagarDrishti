"""Poll Copernicus Marine for new official sea-ice fields and store them permanently.

Contract, per the intended lifecycle:

* the scheduler CHECKS on a cadence (``SEAICE_POLL_INTERVAL_HOURS``, ~3 days);
  that says nothing about how often the source actually publishes, and it never
  by itself creates a model version;
* a date already stored is a duplicate and is skipped — never written twice,
  never overwritten (the database has a unique constraint on
  ``observation_date`` and an append-only trigger behind it);
* existing history is never deleted, so past 7-entry windows stay reproducible;
* only official data enters ``seaice.observations``. Model output goes to
  ``seaice.forecasts`` and can never be mistaken for an observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.seaice import SeaIceObservation, SeaIceRun
from ml.seaice.constants import (
    ARCHITECTURE_VERSION,
    COARSEN_FACTOR,
    CRS,
    GRID_RESOLUTION_DEG,
    GRID_SHAPE,
    VARIABLE,
)
from ml.seaice.grid_store import save_observation
from ml.seaice.preprocessing import coarsen_ice_conc
from ml.seaice.source import SeaIceSource, SeaIceSourceUnavailableError

log = get_logger(__name__)

# Bumped only if the preprocessing in ml.seaice.preprocessing changes, so stored
# grids always say which pipeline produced them.
PREPROCESSING_VERSION = f"coarsen{COARSEN_FACTOR}_v1"

# Distinguishes "caller said nothing, use the poll cap" from an explicit
# ``None``, which means "no cap" (a one-off historical backfill).
_USE_POLL_CAP = object()


@dataclass
class IngestionOutcome:
    run_id: int | None = None
    status: str = "running"
    dataset_id: str = ""
    source_latest_date: date | None = None
    seen: int = 0
    new: int = 0
    duplicates: int = 0
    stored_dates: list[str] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "dataset_id": self.dataset_id,
            "source_latest_date": self.source_latest_date.isoformat() if self.source_latest_date else None,
            "entries_seen": self.seen,
            "entries_new": self.new,
            "entries_duplicate": self.duplicates,
            "stored_dates": self.stored_dates,
            "error": self.error,
        }


class SeaIceIngestionService:
    def __init__(self, session: Session, settings: Settings, source: SeaIceSource | None = None) -> None:
        self.session = session
        self.settings = settings
        self.source = source or SeaIceSource(
            dataset_id=settings.seaice_dataset_id,
            username=_secret(settings.copernicus_marine_username),
            password=_secret(settings.copernicus_marine_password),
        )

    # ------------------------------------------------------------------ query
    def stored_dates(self) -> set[date]:
        return set(self.session.execute(select(SeaIceObservation.observation_date)).scalars())

    def latest_stored_date(self) -> date | None:
        return self.session.execute(
            select(SeaIceObservation.observation_date).order_by(SeaIceObservation.observation_date.desc()).limit(1)
        ).scalars().first()

    def _wanted(
        self,
        available: list[date],
        since: date | None = None,
        until: date | None = None,
        max_entries: Any = _USE_POLL_CAP,
    ) -> list[date]:
        """Which source dates this run should fetch: the missing ones, oldest first.

        Without an explicit range this is the incremental poller: the first run
        backfills ``SEAICE_INITIAL_BACKFILL_DAYS`` so windows can be built at
        all, later runs pick up only what appeared since. With ``since``/``until``
        it is a historical backfill over that window, which is what
        ``load-historical`` uses.
        """
        stored = self.stored_dates()
        if since is not None or until is not None:
            candidates = [
                d for d in available if (since is None or d >= since) and (until is None or d <= until)
            ]
        else:
            latest = self.latest_stored_date()
            if latest is None and available:
                earliest = available[-1] - timedelta(days=self.settings.seaice_initial_backfill_days)
                candidates = [d for d in available if d >= earliest]
            else:
                candidates = [d for d in available if latest is None or d > latest]
        missing = sorted(d for d in candidates if d not in stored)
        cap = self.settings.seaice_max_entries_per_run if max_entries is _USE_POLL_CAP else max_entries
        return missing if cap is None else missing[:cap]

    # -------------------------------------------------------------------- run
    def backfill(self, since: date | None = None, until: date | None = None) -> IngestionOutcome:
        """Load official history in bulk (used by ``load-historical``).

        Defaults to ``SEAICE_HISTORICAL_START`` and is not limited by the
        per-poll cap: this is a one-off catch-up, not the 3-day poller. Days
        already stored are skipped, so it is safe to re-run and it never
        rewrites or deletes existing history.
        """
        return self.run(
            trigger="historical",
            since=since or self.settings.seaice_historical_start,
            until=until,
            max_entries=self.settings.seaice_historical_max_entries,
        )

    def run(
        self,
        trigger: str = "scheduled",
        since: date | None = None,
        until: date | None = None,
        max_entries: Any = _USE_POLL_CAP,
    ) -> IngestionOutcome:
        started = datetime.now(UTC)
        run = SeaIceRun(kind="ingestion", trigger=trigger, dataset_id=self.settings.seaice_dataset_id, started_at=started)
        self.session.add(run)
        self.session.flush()
        outcome = IngestionOutcome(run_id=run.id, dataset_id=self.settings.seaice_dataset_id)
        try:
            with self.source.open() as dataset:
                available = self.source.available_dates(dataset)
                outcome.source_latest_date = available[-1] if available else None
                wanted = self._wanted(available, since=since, until=until, max_entries=max_entries)
                outcome.seen = len(wanted)
                stored_now = self.stored_dates()
                for entry in self.source.iter_entries(
                    dataset, wanted, chunk_size=self.settings.seaice_fetch_chunk_size
                ):
                    day = entry.observation_date
                    if day in stored_now:  # a concurrent run stored it first
                        outcome.duplicates += 1
                        continue
                    self._store(entry, run)
                    stored_now.add(day)
                    outcome.new += 1
                    outcome.stored_dates.append(day.isoformat())
                # Everything the source offers that we already hold.
                outcome.duplicates += len([d for d in available if d in stored_now]) - outcome.new
        except SeaIceSourceUnavailableError as exc:
            outcome.status, outcome.error = "skipped", str(exc)
            log.warning("seaice_ingestion_unavailable", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - recorded on the run and re-raised to the caller's policy
            outcome.status, outcome.error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("seaice_ingestion_failed", error=outcome.error)
        else:
            outcome.status = "success" if outcome.new else "unchanged"
        run.status = outcome.status
        run.entries_seen = outcome.seen
        run.entries_new = outcome.new
        run.entries_duplicate = max(outcome.duplicates, 0)
        run.source_latest_date = outcome.source_latest_date
        run.error_message = outcome.error
        run.completed_at = datetime.now(UTC)
        run.duration_ms = int((run.completed_at - started).total_seconds() * 1000)
        run.details = {"stored_dates": outcome.stored_dates}
        self.session.flush()
        log.info("seaice_ingestion_completed", **outcome.as_dict())
        return outcome

    def _store(self, entry: Any, run: SeaIceRun) -> SeaIceObservation:
        day = entry.observation_date
        concentration, mask = coarsen_ice_conc(entry.data_array)
        if tuple(concentration.shape) != GRID_SHAPE:
            raise ValueError(
                f"{day}: coarsened grid is {concentration.shape}, expected {GRID_SHAPE}; "
                "the source grid changed and the base model's weights assume the trained shape"
            )
        stored = save_observation(Path(self.settings.seaice_data_dir), day, concentration, mask)
        valid = int(mask.sum())
        observation = SeaIceObservation(
            observation_date=day,
            source_time=entry.source_time,
            provenance="official_cmems_osisaf",
            authority=self.source.authority,
            dataset_id=self.settings.seaice_dataset_id,
            variable=VARIABLE,
            preprocessing_version=PREPROCESSING_VERSION,
            grid_shape=list(GRID_SHAPE),
            grid_resolution_deg=GRID_RESOLUTION_DEG,
            crs=CRS,
            grid_path=str(stored.path),
            grid_sha256=stored.sha256,
            n_valid_cells=valid,
            mean_concentration=float(concentration[mask > 0].mean()) if valid else None,
            ice_area_km2=None,
            ingestion_run_id=run.id,
        )
        self.session.add(observation)
        self.session.flush()
        log.info(
            "seaice_observation_stored",
            date=day.isoformat(), valid_cells=valid, architecture=ARCHITECTURE_VERSION, sha256=stored.sha256[:12],
        )
        return observation


def _secret(value: Any) -> str | None:
    return value.get_secret_value() if value is not None else None


def load_recent_entries(session: Session, window: int) -> list[SeaIceObservation]:
    """The last ``window`` chronological sea-ice entries, oldest first.

    ENTRY-BASED, not calendar-based: this returns the most recent ``window``
    rows that exist, whatever dates they carry. Gaps in the official source are
    preserved as-is and never filled with fabricated days.
    """
    rows = list(
        session.execute(
            select(SeaIceObservation).order_by(SeaIceObservation.observation_date.desc()).limit(window)
        ).scalars()
    )
    return sorted(rows, key=lambda o: o.observation_date)


def stack_entries(entries: list[SeaIceObservation]) -> tuple[np.ndarray, np.ndarray]:
    """Load the stored grids for ``entries`` into ``(n, H, W)`` concentration and mask arrays."""
    from ml.seaice.grid_store import load_observation

    grids = [load_observation(Path(e.grid_path)) for e in entries]
    return (
        np.stack([g[0] for g in grids]).astype(np.float32),
        np.stack([g[1] for g in grids]).astype(np.float32),
    )
