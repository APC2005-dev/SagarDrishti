"""Generate sea-ice forecasts from the current champion.

The model loaded is always the version the registry records as ``deployed`` —
never the highest-numbered one, never a literal. The input is the last 7
chronological entries in the database; if fewer than 7 exist the run is skipped
with a reason rather than padded with invented days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.seaice import SeaIceForecast, SeaIceForecastSet, SeaIceRun
from app.services.seaice_ingestion_service import load_recent_entries, stack_entries
from app.services.seaice_registry import SeaIceRegistry
from ml.seaice.constants import HORIZONS, WINDOW
from ml.seaice.grid_store import save_forecast
from ml.seaice.inference import forecast_numpy
from ml.seaice.preprocessing import build_input_tensor, first_target_date

log = get_logger(__name__)


@dataclass
class ForecastOutcome:
    run_id: int | None = None
    status: str = "running"
    model_version: str | None = None
    anchor_date: date | None = None
    created: int = 0
    already: int = 0
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "model_version": self.model_version,
            "anchor_date": self.anchor_date.isoformat() if self.anchor_date else None,
            "created_forecasts": self.created,
            "already_forecast": self.already,
            "error": self.error,
            "diagnostics": self.diagnostics,
        }


class SeaIceForecastService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.registry = SeaIceRegistry(session, settings)

    def run(self, trigger: str = "scheduled") -> ForecastOutcome:
        started = datetime.now(UTC)
        run = SeaIceRun(kind="forecast", trigger=trigger, dataset_id=self.settings.seaice_dataset_id, started_at=started)
        self.session.add(run)
        self.session.flush()
        outcome = ForecastOutcome(run_id=run.id)
        try:
            champion = self.registry.get_champion()
            if champion is None:
                outcome.status, outcome.error = "skipped", "no deployed sea-ice model version"
                return self._finish(run, outcome, started)
            outcome.model_version = champion.version

            entries = load_recent_entries(self.session, WINDOW)
            if len(entries) < WINDOW:
                outcome.status = "skipped"
                outcome.error = f"only {len(entries)} sea-ice entries stored; {WINDOW} chronological entries are required"
                return self._finish(run, outcome, started)

            anchor = entries[-1]
            outcome.anchor_date = anchor.observation_date
            existing = self.session.execute(
                select(SeaIceForecastSet).where(
                    SeaIceForecastSet.model_version == champion.version,
                    SeaIceForecastSet.anchor_observation_id == anchor.id,
                )
            ).scalar_one_or_none()
            if existing is not None:
                outcome.status, outcome.already = "unchanged", len(HORIZONS)
                return self._finish(run, outcome, started)

            concentration, mask = stack_entries(entries)
            tensor = build_input_tensor(concentration, mask, first_target_date(anchor.observation_date))
            bundle = self.registry.load_bundle(champion)
            prediction = forecast_numpy(bundle.model, tensor)

            span = (entries[-1].observation_date - entries[0].observation_date).days
            outcome.diagnostics = {
                "input_entry_dates": [e.observation_date.isoformat() for e in entries],
                "span_days": span,
                "daily_cadence": span == WINDOW - 1,
                "window_rule": f"last {WINDOW} chronological database entries",
            }
            forecast_set = SeaIceForecastSet(
                model_version=champion.version,
                anchor_observation_id=anchor.id,
                anchor_date=anchor.observation_date,
                input_observation_ids=[e.id for e in entries],
                input_entry_dates=[e.observation_date.isoformat() for e in entries],
                input_window_entries=WINDOW,
                input_span_days=span,
                daily_cadence=span == WINDOW - 1,
                run_id=run.id,
                diagnostics=outcome.diagnostics,
            )
            self.session.add(forecast_set)
            self.session.flush()

            for index, horizon in enumerate(HORIZONS):
                grid = prediction[index]
                stored = save_forecast(
                    Path(self.settings.seaice_data_dir), champion.version, anchor.observation_date, horizon, grid
                )
                self.session.add(
                    SeaIceForecast(
                        forecast_set_id=forecast_set.id,
                        horizon_days=horizon,
                        target_date=anchor.observation_date + timedelta(days=horizon),
                        grid_path=str(stored.path),
                        grid_sha256=stored.sha256,
                        mean_concentration=float(grid[mask[-1] > 0].mean()) if mask[-1].any() else None,
                    )
                )
                outcome.created += 1
            self.session.flush()
            outcome.status = "success"
        except Exception as exc:  # noqa: BLE001 - surfaced on the run row
            outcome.status, outcome.error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("seaice_forecast_failed", error=outcome.error)
        return self._finish(run, outcome, started)

    def _finish(self, run: SeaIceRun, outcome: ForecastOutcome, started: datetime) -> ForecastOutcome:
        run.status = outcome.status
        run.error_message = outcome.error
        run.completed_at = datetime.now(UTC)
        run.duration_ms = int((run.completed_at - started).total_seconds() * 1000)
        run.details = outcome.as_dict()
        self.session.flush()
        log.info("seaice_forecast_completed", **{k: v for k, v in outcome.as_dict().items() if k != "diagnostics"})
        return outcome
