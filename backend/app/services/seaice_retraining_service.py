"""Sea-ice retraining: eligibility -> candidate -> comparison -> promote or reject.

A new version is created only when the policy says retraining is warranted —
never because the poller ran, and never on a fixed "every 3 days" cadence. The
3-day scheduler fetches data; this service decides whether that data justifies a
new model.

Champion and challenger are always scored on the SAME held-out samples with the
same masked metrics, so the comparison is like-for-like. A candidate that does
not clear the configured margin is registered and kept as ``rejected`` for
lineage and rollback; the champion is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import ModelStatusEvent, ModelVersion
from app.models.seaice import SeaIceEvaluation, SeaIceObservation, SeaIceRun
from app.services.seaice_ingestion_service import stack_entries
from app.services.seaice_registry import SeaIceRegistry
from ml.seaice.artifact_store import write_version
from ml.seaice.constants import ARCHITECTURE_VERSION, HORIZONS, WINDOW
from ml.seaice.dataset import build_samples, chronological_split
from ml.seaice.evaluation import HorizonAccumulator
from ml.seaice.inference import forecast_numpy, persistence_forecast_numpy
from ml.seaice.training import RetrainConfig, train_challenger
from ml.versioning.version_manager import short_version

log = get_logger(__name__)


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"promote": self.promote, "reasons": self.reasons}


@dataclass
class RetrainingOutcome:
    run_id: int | None = None
    status: str = "running"
    champion_version: str | None = None
    candidate_version: str | None = None
    decision: str | None = None
    eligibility: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "champion_version": self.champion_version,
            "candidate_version": self.candidate_version,
            "decision": self.decision,
            "eligibility": self.eligibility,
            "comparison": self.comparison,
            "error": self.error,
        }


class SeaIceRetrainingService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.registry = SeaIceRegistry(session, settings)

    # ------------------------------------------------------------- policy
    def policy_snapshot(self) -> dict[str, Any]:
        s = self.settings
        return {
            "window_entries": WINDOW,
            "window_rule": "last 7 chronological database entries (not 7 calendar days)",
            "horizons_days": list(HORIZONS),
            "eligibility": {
                "min_new_observations": s.seaice_retrain_min_new_observations,
                "min_days_since_last_training": s.seaice_retrain_min_days_since_last_training,
                "degradation": {
                    "horizon_days": s.seaice_degradation_horizon_days,
                    "max_rmse": s.seaice_degradation_max_rmse,
                    "relative_tolerance": s.seaice_degradation_relative_tolerance,
                    "min_samples": s.seaice_degradation_min_samples,
                },
            },
            "promotion": {
                "primary_horizon": s.seaice_promotion_primary_horizon,
                "min_relative_improvement": s.seaice_promotion_min_relative_improvement,
                "max_short_horizon_regression": s.seaice_promotion_max_short_horizon_regression,
                "min_evaluation_samples": s.seaice_promotion_min_evaluation_samples,
            },
            "training": RetrainConfig(
                epochs=s.seaice_retrain_epochs, learning_rate=s.seaice_retrain_learning_rate,
                batch_size=s.seaice_retrain_batch_size, patience=s.seaice_retrain_patience, seed=s.seaice_retrain_seed,
            ).as_dict(),
        }

    def _degradation(self, champion: ModelVersion | None, since: datetime | None) -> dict[str, Any]:
        """Has the champion's measured sea-ice error crossed the configured threshold?

        Uses stored operational evaluations (official ground truth). Both
        thresholds unset means the trigger is disabled and the check passes.
        """
        s = self.settings
        horizon = int(s.seaice_degradation_horizon_days)
        out: dict[str, Any] = {
            "horizon_days": horizon, "max_rmse": s.seaice_degradation_max_rmse,
            "relative_tolerance": s.seaice_degradation_relative_tolerance,
            "min_samples": s.seaice_degradation_min_samples,
        }
        if s.seaice_degradation_max_rmse is None and s.seaice_degradation_relative_tolerance is None:
            return {**out, "enabled": False, "value": None, "met": True, "reason": "degradation trigger disabled"}
        if champion is None:
            return {**out, "enabled": True, "value": None, "n": 0, "met": False, "reason": "no champion deployed"}
        query = select(func.count(), func.avg(SeaIceEvaluation.rmse)).where(
            SeaIceEvaluation.model_version == champion.version, SeaIceEvaluation.horizon_days == horizon
        )
        if since:
            query = query.where(SeaIceEvaluation.evaluated_at > since)
        count, mean_rmse = self.session.execute(query).one()
        baseline = {1: champion.day1_error, 3: champion.day3_error, 7: champion.day7_error}.get(horizon)
        baseline = float(baseline) if baseline is not None else None
        out |= {"enabled": True, "n": int(count), "baseline_rmse": baseline,
                "value": round(float(mean_rmse), 5) if mean_rmse is not None else None}
        if mean_rmse is None or count < s.seaice_degradation_min_samples:
            return {**out, "met": False, "reason": f"only {int(count)} evaluation(s) at D+{horizon}"}
        recent, triggers = float(mean_rmse), []
        if s.seaice_degradation_max_rmse is not None and recent >= s.seaice_degradation_max_rmse:
            triggers.append(f"mean D+{horizon} RMSE {recent:.5f} >= ceiling {s.seaice_degradation_max_rmse}")
        if s.seaice_degradation_relative_tolerance is not None and baseline:
            limit = baseline * (1.0 + s.seaice_degradation_relative_tolerance)
            if recent >= limit:
                triggers.append(
                    f"mean D+{horizon} RMSE {recent:.5f} >= {limit:.5f} "
                    f"(registered {baseline:.5f} +{s.seaice_degradation_relative_tolerance:.0%})"
                )
        return {**out, "met": bool(triggers), "reason": "; ".join(triggers) or "within configured threshold"}

    def eligibility(self) -> dict[str, Any]:
        s = self.settings
        champion = self.registry.get_champion()
        since = champion.training_data_cutoff if champion else None
        query = select(func.count()).select_from(SeaIceObservation)
        if since:
            query = query.where(SeaIceObservation.observation_date > since.date())
        new_observations = int(self.session.execute(query).scalar_one())
        days = (datetime.now(UTC) - since).total_seconds() / 86400 if since else None
        checks = {
            "champion_deployed": {"value": champion.version if champion else None, "met": champion is not None},
            "new_observations": {
                "value": new_observations, "threshold": s.seaice_retrain_min_new_observations,
                "met": new_observations >= s.seaice_retrain_min_new_observations,
            },
            "days_since_last_training": {
                "value": round(days, 2) if days is not None else None,
                "threshold": s.seaice_retrain_min_days_since_last_training,
                "met": days is None or days >= s.seaice_retrain_min_days_since_last_training,
            },
            "performance_degradation": self._degradation(champion, since),
        }
        return {
            "since": since.isoformat() if since else None,
            "checks": checks,
            "eligible": all(c["met"] for c in checks.values()),
        }

    # ---------------------------------------------------------------- data
    def _load_samples(self) -> tuple[list[Any], dict[str, Any]]:
        entries = list(
            self.session.execute(
                select(SeaIceObservation).order_by(SeaIceObservation.observation_date)
            ).scalars()
        )
        if len(entries) < WINDOW + max(HORIZONS):
            return [], {"entries": len(entries), "reason": "not enough stored entries to build a single sample"}
        concentration, mask = stack_entries(entries)
        samples, report = build_samples([e.observation_date for e in entries], concentration, mask)
        return samples, {"entries": len(entries), **report.as_dict()}

    def _score(self, model: Any, samples: list[Any]) -> dict[str, Any]:
        accumulator = HorizonAccumulator()
        for sample in samples:
            accumulator.add(forecast_numpy(model, sample.x), sample.y, sample.mask)
        return accumulator.as_dict()

    def _score_persistence(self, samples: list[Any]) -> dict[str, Any]:
        accumulator = HorizonAccumulator()
        for sample in samples:
            accumulator.add(persistence_forecast_numpy(sample.x), sample.y, sample.mask)
        return accumulator.as_dict()

    def decide(self, champion_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> PromotionDecision:
        s = self.settings
        primary = str(s.seaice_promotion_primary_horizon)
        reasons: list[str] = []
        samples = candidate_metrics.get("samples", 0)
        if samples < s.seaice_promotion_min_evaluation_samples:
            return PromotionDecision(False, [f"only {samples} evaluation samples (< {s.seaice_promotion_min_evaluation_samples})"])
        champion_rmse = (champion_metrics["by_horizon"].get(primary) or {}).get("rmse")
        candidate_rmse = (candidate_metrics["by_horizon"].get(primary) or {}).get("rmse")
        if champion_rmse is None or candidate_rmse is None or not np.isfinite(champion_rmse):
            return PromotionDecision(False, [f"no comparable D+{primary} metric"])
        improvement = (champion_rmse - candidate_rmse) / champion_rmse if champion_rmse else 0.0
        reasons.append(
            f"D+{primary} RMSE {candidate_rmse:.5f} vs champion {champion_rmse:.5f} ({improvement:+.2%})"
        )
        if improvement < s.seaice_promotion_min_relative_improvement:
            reasons.append(f"below the required {s.seaice_promotion_min_relative_improvement:.2%} improvement")
            return PromotionDecision(False, reasons)
        short = str(min(HORIZONS))
        champion_short = (champion_metrics["by_horizon"].get(short) or {}).get("rmse")
        candidate_short = (candidate_metrics["by_horizon"].get(short) or {}).get("rmse")
        if champion_short and candidate_short is not None:
            regression = (candidate_short - champion_short) / champion_short
            reasons.append(f"D+{short} RMSE change {regression:+.2%}")
            if regression > s.seaice_promotion_max_short_horizon_regression:
                reasons.append(
                    f"short-horizon regression exceeds {s.seaice_promotion_max_short_horizon_regression:.2%}"
                )
                return PromotionDecision(False, reasons)
        return PromotionDecision(True, reasons)

    # ----------------------------------------------------------------- run
    def run(self, trigger: str = "scheduled", force: bool = False) -> RetrainingOutcome:
        started = datetime.now(UTC)
        run = SeaIceRun(kind="retraining", trigger=trigger, started_at=started)
        self.session.add(run)
        self.session.flush()
        outcome = RetrainingOutcome(run_id=run.id)
        try:
            outcome.eligibility = self.eligibility()
            champion = self.registry.get_champion()
            outcome.champion_version = champion.version if champion else None
            if champion is None:
                outcome.status, outcome.error = "skipped", "no deployed sea-ice model version"
                return self._finish(run, outcome, started)
            if not outcome.eligibility["eligible"] and not force:
                outcome.status, outcome.decision = "skipped", "not eligible"
                return self._finish(run, outcome, started)

            samples, data_report = self._load_samples()
            if not samples:
                outcome.status, outcome.error = "skipped", data_report.get("reason", "no samples")
                return self._finish(run, outcome, started)
            train, validation = chronological_split(samples, self.settings.seaice_retrain_validation_fraction)
            if not train or not validation:
                outcome.status, outcome.error = "skipped", "not enough samples for a chronological train/validation split"
                return self._finish(run, outcome, started)

            config = RetrainConfig(
                epochs=self.settings.seaice_retrain_epochs,
                learning_rate=self.settings.seaice_retrain_learning_rate,
                batch_size=self.settings.seaice_retrain_batch_size,
                patience=self.settings.seaice_retrain_patience,
                seed=self.settings.seaice_retrain_seed,
            )
            champion_bundle = self.registry.load_bundle(champion)
            champion_state = {k: v.detach().clone() for k, v in champion_bundle.model.state_dict().items()}
            trained = train_challenger(train, validation, config, initial_state=champion_state)

            # Identical samples, identical metrics, for both models.
            champion_metrics = self._score(champion_bundle.model, validation)
            candidate_metrics = self._score(trained.model, validation)
            persistence_metrics = self._score_persistence(validation)
            decision = self.decide(champion_metrics, candidate_metrics)
            cutoff = max(s.anchor_date for s in train)

            version = self.registry.allocate_next_version()
            outcome.candidate_version = version
            metadata = {
                **{k: v for k, v in champion_bundle.metadata.items()
                   if k not in ("artifact_sha256", "version", "parent_version", "metrics", "training", "description")},
                "version": version,
                "model_family": "sea_ice",
                "parent_version": champion.version,
                "architecture_version": ARCHITECTURE_VERSION,
                "artifact_origin": f"retraining:{config.initialisation}",
                "training_data_cutoff": cutoff.isoformat(),
                "training": {
                    "config": config.as_dict(),
                    **trained.as_dict(),
                    "data": data_report,
                    "window_rule": f"last {WINDOW} chronological database entries",
                },
                "metrics": {"protocol": "seaice_validation_holdout", **candidate_metrics},
                "comparison": {
                    "protocol": "seaice_validation_holdout",
                    "samples": candidate_metrics["samples"],
                    "champion": {"version": champion.version, **champion_metrics},
                    "candidate": {"version": version, **candidate_metrics},
                    "persistence_baseline": persistence_metrics,
                },
                "decision": decision.as_dict(),
                "created_at": datetime.now(UTC).isoformat(),
            }
            written = write_version(
                Path(self.settings.models_dir), short_version(version), trained.model.state_dict(), metadata
            )
            outcome.comparison = metadata["comparison"]

            mv = self.registry._row_from_metadata(version, written.directory, metadata, written.checksums, "candidate")
            self.session.add(
                ModelStatusEvent(
                    model_version=version, from_status=None, to_status="candidate",
                    reason=f"trained by sea-ice retraining run {run.id}", actor="retraining",
                )
            )
            self.session.flush()
            self.registry.record_metrics(version, "seaice_validation_holdout", candidate_metrics, "retraining_comparison")
            self.registry.record_metrics(
                champion.version, "seaice_validation_holdout", champion_metrics, "retraining_comparison"
            )
            self.registry.transition(mv, "validated", "evaluated against the champion on identical samples", "retraining")

            reason = "; ".join(decision.reasons)
            if decision.promote:
                self.registry.promote(mv, reason, actor="retraining")
                outcome.decision, outcome.status = "promoted", "success"
            else:
                self.registry.transition(mv, "rejected", reason, "retraining")
                outcome.decision, outcome.status = "rejected", "success"
            log.info("seaice_retraining_decided", candidate=version, champion=champion.version, decision=outcome.decision)
        except Exception as exc:  # noqa: BLE001 - surfaced on the run row
            outcome.status, outcome.error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("seaice_retraining_failed", error=outcome.error)
        return self._finish(run, outcome, started)

    def _finish(self, run: SeaIceRun, outcome: RetrainingOutcome, started: datetime) -> RetrainingOutcome:
        run.status = "success" if outcome.status == "success" else outcome.status
        run.error_message = outcome.error
        run.completed_at = datetime.now(UTC)
        run.duration_ms = int((run.completed_at - started).total_seconds() * 1000)
        run.details = outcome.as_dict()
        self.session.flush()
        return outcome
