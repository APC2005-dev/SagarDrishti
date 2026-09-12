"""Score stored sea-ice forecasts against the official observations that arrived later.

Ground truth is official Copernicus Marine data only: an evaluation is written
only once the observation for the forecast's target date exists in
``seaice.observations``. Model output is never used as truth.

Each evaluation also records the persistence baseline on the identical cells, so
"is the model still adding skill?" can be answered directly from stored rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.seaice import (
    SeaIceEvaluation,
    SeaIceForecast,
    SeaIceForecastSet,
    SeaIceObservation,
    SeaIceRun,
)
from ml.seaice.evaluation import masked_mae, masked_rmse
from ml.seaice.grid_store import load_forecast, load_observation

log = get_logger(__name__)


@dataclass
class EvaluationOutcome:
    run_id: int | None = None
    status: str = "running"
    evaluated: int = 0
    pending: int = 0
    error: str | None = None
    versions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "evaluated": self.evaluated,
            "pending": self.pending,
            "versions": self.versions,
            "error": self.error,
        }


class SeaIceEvaluationService:
    def __init__(self, session: Session, settings: Any) -> None:
        self.session = session
        self.settings = settings

    def run(self, trigger: str = "scheduled") -> EvaluationOutcome:
        started = datetime.now(UTC)
        run = SeaIceRun(kind="evaluation", trigger=trigger, started_at=started)
        self.session.add(run)
        self.session.flush()
        outcome = EvaluationOutcome(run_id=run.id)
        try:
            scored = set(self.session.execute(select(SeaIceEvaluation.forecast_id)).scalars())
            rows = self.session.execute(
                select(SeaIceForecast, SeaIceForecastSet).join(
                    SeaIceForecastSet, SeaIceForecast.forecast_set_id == SeaIceForecastSet.id
                )
            ).all()
            versions: set[str] = set()
            for forecast, forecast_set in rows:
                if forecast.id in scored:
                    continue
                actual = self.session.execute(
                    select(SeaIceObservation).where(SeaIceObservation.observation_date == forecast.target_date)
                ).scalar_one_or_none()
                if actual is None:
                    outcome.pending += 1  # target date has not been published yet
                    continue
                anchor = self.session.get(SeaIceObservation, forecast_set.anchor_observation_id)
                truth, mask = load_observation(Path(actual.grid_path))
                prediction = load_forecast(Path(forecast.grid_path))
                persistence, _ = load_observation(Path(anchor.grid_path))
                self.session.add(
                    SeaIceEvaluation(
                        forecast_id=forecast.id,
                        model_version=forecast_set.model_version,
                        horizon_days=forecast.horizon_days,
                        anchor_date=forecast_set.anchor_date,
                        target_date=forecast.target_date,
                        actual_observation_id=actual.id,
                        rmse=masked_rmse(prediction, truth, mask),
                        mae=masked_mae(prediction, truth, mask),
                        persistence_rmse=masked_rmse(persistence, truth, mask),
                        persistence_mae=masked_mae(persistence, truth, mask),
                        n_valid_cells=int(mask.sum()),
                        run_id=run.id,
                    )
                )
                versions.add(forecast_set.model_version)
                outcome.evaluated += 1
            self.session.flush()
            outcome.versions = sorted(versions)
            outcome.status = "success" if outcome.evaluated else "unchanged"
        except Exception as exc:  # noqa: BLE001 - surfaced on the run row
            outcome.status, outcome.error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("seaice_evaluation_failed", error=outcome.error)
        run.status = outcome.status
        run.error_message = outcome.error
        run.completed_at = datetime.now(UTC)
        run.duration_ms = int((run.completed_at - started).total_seconds() * 1000)
        run.details = outcome.as_dict()
        self.session.flush()
        log.info("seaice_evaluation_completed", **outcome.as_dict())
        return outcome
