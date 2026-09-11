"""Retraining: eligibility policy -> dataset -> challenger -> comparison -> promote/reject.

Data sources (with the cutoff frozen at run start, so nothing after it leaks in):
* historical training dataset rows in ``tracking.observations`` -> daily
  sequences, split with the base model's fixed boundaries (train <= 2018-08-23,
  validation <= 2022-03-09, test after) so every challenger is compared on the
  same historical protocol as the base model;
* official USNIC observations -> operational samples (exactly the
  forecast/actual pairs that the evaluation pipeline scores), split
  chronologically by anchor date into train / validation / *holdout*.

Champion and challenger are evaluated on identical samples. The operational
holdout decides when it has enough labels; otherwise the historical test does.
Every trained challenger is written as an immutable version — rejected ones are
kept for lineage and reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import ForecastEvaluation, RetrainingRun
from app.models.tracking import Observation
from app.services.forecast_service import load_recent_entries
from app.services.model_registry import ModelRegistry
from ml.adapters.production_adapter import AdapterConfig, ProductionSequenceAdapter
from ml.constants import FEATURE_NAMES, INPUT_SEMANTICS_PRODUCTION
from ml.evaluation.champion_challenger import PromotionPolicy, decide
from ml.evaluation.evaluator import EvaluationResult
from ml.evaluation.metrics import headline_errors
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.models.artifact_store import write_version
from ml.training.dataset_builder import (
    SequenceDataset,
    build_historical_sequences,
    build_operational_samples,
    split_by_boundaries,
    split_by_fraction,
)
from ml.training.retrainer import RetrainConfig, evaluate_pair, train_challenger
from ml.training.trainer import TrainingConfig
from ml.versioning.version_manager import next_version

log = get_logger(__name__)
FINISHED = ("promoted", "rejected")


class RetrainingService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self.registry = ModelRegistry(session, settings)

    # ------------------------------------------------------------------ policy
    def promotion_policy(self) -> PromotionPolicy:
        s = self.settings
        return PromotionPolicy(
            primary_horizon=s.promotion_primary_horizon,
            min_relative_improvement=s.promotion_min_relative_improvement,
            max_short_horizon_regression=s.promotion_max_short_horizon_regression,
            min_evaluation_samples=s.promotion_min_evaluation_samples,
        )

    def retrain_config(self) -> RetrainConfig:
        s = self.settings
        return RetrainConfig(
            strategy=s.retrain_strategy,
            historical_replay_samples=s.retrain_historical_replay_samples,
            training=TrainingConfig(learning_rate=s.retrain_learning_rate, epochs=s.retrain_epochs),
        )

    def policy_snapshot(self) -> dict[str, Any]:
        s = self.settings
        return {
            "eligibility": {
                "min_new_evaluations": s.retrain_min_new_evaluations,
                "min_new_icebergs": s.retrain_min_new_icebergs,
                "min_days_since_last_training": s.retrain_min_days_since_last_training,
            },
            "promotion": self.promotion_policy().as_dict(),
            "training": self.retrain_config().as_dict(),
            "historical_protocol": {"train_end": str(s.historical_train_end), "validation_end": str(s.historical_validation_end)},
        }

    def eligibility(self) -> dict[str, Any]:
        last = self.session.execute(
            select(RetrainingRun).where(RetrainingRun.status.in_(FINISHED)).order_by(RetrainingRun.started_at.desc()).limit(1)
        ).scalar_one_or_none()
        champion = self.registry.get_deployed()
        since = last.source_data_cutoff if last and last.source_data_cutoff else (champion.deployed_at if champion else None)
        E = ForecastEvaluation
        q = select(func.count(), func.count(func.distinct(E.iceberg_id))).where(E.evaluation_mode == "operational")
        if since:
            q = q.where(E.evaluated_at > since)
        n_eval, n_bergs = self.session.execute(q).one()
        days = (datetime.now(UTC) - since).total_seconds() / 86400 if since else None
        s = self.settings
        checks = {
            "champion_deployed": {"value": champion.version if champion else None, "met": champion is not None},
            "new_evaluations": {"value": int(n_eval), "threshold": s.retrain_min_new_evaluations, "met": n_eval >= s.retrain_min_new_evaluations},
            "new_icebergs": {"value": int(n_bergs), "threshold": s.retrain_min_new_icebergs, "met": n_bergs >= s.retrain_min_new_icebergs},
            "days_since_last_training": {
                "value": round(days, 2) if days is not None else None,
                "threshold": s.retrain_min_days_since_last_training,
                "met": days is None or days >= s.retrain_min_days_since_last_training,
            },
        }
        return {"since": since.isoformat() if since else None, "checks": checks, "eligible": all(c["met"] for c in checks.values())}

    # --------------------------------------------------------------------- data
    def _historical_tracks(self) -> pd.DataFrame:
        q = select(Observation.iceberg_id, Observation.observation_date, Observation.latitude, Observation.longitude).where(
            Observation.provenance == "historical_training_dataset"
        )
        df = pd.DataFrame(self.session.execute(q).all(), columns=["iceberg_id", "date", "latitude", "longitude"])
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df["x_m"], df["y_m"] = latlon_to_polar_m(df["latitude"].to_numpy(), df["longitude"].to_numpy())
        return df

    def _operational(self, cutoff: datetime, adapter: ProductionSequenceAdapter) -> SequenceDataset:
        ids = list(
            self.session.execute(
                select(func.distinct(Observation.iceberg_id)).where(Observation.provenance == "official_usnic")
            ).scalars()
        )
        entries = load_recent_entries(self.session, ids, per_iceberg=10_000)
        return build_operational_samples(entries, adapter, cutoff=cutoff.date())

    # ---------------------------------------------------------------------- run
    def run(self, trigger: str = "scheduled", force: bool = False) -> RetrainingRun:
        run = RetrainingRun(run_id=uuid.uuid4(), trigger=trigger, status="running", policy=self.policy_snapshot())
        self.session.add(run)
        self.session.commit()
        elig = self.eligibility()
        run.eligibility = elig
        if not elig["eligible"] and not force:
            run.status = "skipped"
            run.completed_at = datetime.now(UTC)
            run.failure_reason = "; ".join(k for k, c in elig["checks"].items() if not c["met"]) + " not met"
            self.session.commit()
            log.info("retraining_skipped", run_id=str(run.run_id), eligibility=elig)
            return run
        try:
            return self._train_and_decide(run)
        except Exception as exc:
            self.session.rollback()
            run = self.session.merge(run)
            run.status = "failed"
            run.failure_reason = f"{type(exc).__name__}: {exc}"
            run.metrics = {**(run.metrics or {}), "traceback": traceback.format_exc()[-4000:]}
            run.completed_at = datetime.now(UTC)
            self.session.commit()
            log.exception("retraining_failed", run_id=str(run.run_id))
            return run

    def _train_and_decide(self, run: RetrainingRun) -> RetrainingRun:
        s = self.settings
        champion_row = self.registry.get_deployed()
        if champion_row is None:
            raise RuntimeError("no deployed champion to retrain from")
        champion = self.registry.load_bundle(champion_row)
        cutoff = datetime.now(UTC)
        run.champion_version = champion_row.version
        run.source_data_cutoff = cutoff
        log.info("retraining_started", run_id=str(run.run_id), champion=champion_row.version, cutoff=cutoff.isoformat())

        adapter = ProductionSequenceAdapter(AdapterConfig(strategy=champion_row.adapter_strategy or s.adapter_strategy,
                                                          max_gap_days=s.adapter_max_gap_days))
        hist = build_historical_sequences(self._historical_tracks())
        hsplit = split_by_boundaries(hist, np.datetime64(s.historical_train_end), np.datetime64(s.historical_validation_end))
        ops = self._operational(cutoff, adapter)
        if len(ops) >= 3:
            val_frac, test_frac = s.retrain_operational_validation_fraction, s.retrain_operational_test_fraction
            osplit = split_by_fraction(ops, 1 - val_frac - test_frac, val_frac)
        else:
            empty = SequenceDataset.empty()
            osplit = split_by_fraction(empty) if len(ops) == 0 else None  # type: ignore[assignment]
            if osplit is None:  # too few to split: keep them out of training, use as holdout only
                from ml.training.dataset_builder import ChronologicalSplit
                osplit = ChronologicalSplit(empty, empty, ops, None, None)

        candidate = next_version([m.version for m in self.registry.list_versions()], Path(s.models_dir))
        run.candidate_version = candidate
        self.session.commit()

        cfg = self.retrain_config()
        challenger = train_challenger(champion, hsplit.train, hsplit.validation, osplit.train, osplit.validation, cfg, candidate)

        protocols: dict[str, tuple[EvaluationResult, EvaluationResult]] = {}
        hist_protocol = f"historical_test_{s.historical_validation_end}"
        if len(hsplit.test):
            protocols[hist_protocol] = evaluate_pair(champion, challenger.bundle, hsplit.test, hist_protocol)
        op_protocol = f"operational_holdout_{str(run.run_id)[:8]}"
        if len(osplit.test):
            protocols[op_protocol] = evaluate_pair(champion, challenger.bundle, osplit.test, op_protocol)
        if not protocols:
            raise RuntimeError("no evaluation data for either protocol")

        policy = self.promotion_policy()
        op_n = protocols[op_protocol][1].by_horizon[policy.primary_horizon].n if op_protocol in protocols else 0
        primary = op_protocol if op_n >= policy.min_evaluation_samples else hist_protocol
        if primary not in protocols:
            primary = next(iter(protocols))
        decision = decide(*protocols[primary], policy)
        if primary != hist_protocol and hist_protocol in protocols:
            c, n = (r.by_horizon[policy.primary_horizon].mae_km for r in protocols[hist_protocol])
            if c is not None and n is not None and n > c * (1 + policy.max_short_horizon_regression):
                decision.promote = False
                decision.reasons.append(f"historical day-{policy.primary_horizon} regression {n:.3f} km > {c * (1 + policy.max_short_horizon_regression):.3f} km")

        cand_primary = protocols[primary][1]
        manifest = {
            "cutoff": cutoff.isoformat(),
            "historical": {"source": "tracking.observations provenance=historical_training_dataset", **hsplit.summary()},
            "operational": {
                "source": "tracking.observations provenance=official_usnic",
                **osplit.summary(),
                "anchor_observation_ids_sha256": hashlib.sha256(
                    json.dumps(sorted(int(i) for i in ops.anchor_observation_ids if i is not None)).encode()
                ).hexdigest(),
            },
            "train": challenger.train_summary,
            "validation": challenger.validation_summary,
        }
        metrics = {
            "primary_protocol": primary,
            "protocols": {p: {"champion": c.as_dict(), "challenger": n.as_dict()} for p, (c, n) in protocols.items()},
            "history": challenger.history,
        }
        metadata = {
            "version": candidate,
            "parent_version": champion_row.version,
            "architecture": champion_row.architecture,
            "architecture_version": champion_row.architecture_version,
            "input_sequence_length": champion_row.input_sequence_length,
            "input_semantics": INPUT_SEMANTICS_PRODUCTION,
            "forecast_horizon_days": champion_row.forecast_horizon_days,
            "feature_names": list(FEATURE_NAMES),
            "adapter_strategy": champion_row.adapter_strategy,
            "artifact_origin": f"retraining:{cfg.strategy}",
            "retraining_run_id": str(run.run_id),
            "training_data_cutoff": cutoff.isoformat(),
            "training": {"config": cfg.as_dict(), "started_at": datetime.fromtimestamp(challenger.started_at, UTC).isoformat(),
                         "completed_at": datetime.fromtimestamp(challenger.completed_at, UTC).isoformat()},
            "metrics": {"protocol": primary, **cand_primary.as_dict()},
            "comparison": {p: {"champion": c.as_dict(), "challenger": n.as_dict()} for p, (c, n) in protocols.items()},
            "decision": decision.as_dict(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        written = write_version(Path(s.models_dir), candidate, challenger.bundle.model, challenger.bundle.feature_scaler,
                                challenger.bundle.target_scaler, metadata,
                                extra_files={"dataset_manifest.json": {**manifest,
                                             "operational_anchor_observation_ids": [int(i) for i in ops.anchor_observation_ids if i is not None]}})

        from app.models.ml import ModelStatusEvent, ModelVersion  # local import keeps module import light

        head = headline_errors(cand_primary.by_horizon)
        mv = ModelVersion(
            version=candidate, version_number=int(candidate[1:]), parent_version=champion_row.version,
            architecture=champion_row.architecture, architecture_version=champion_row.architecture_version,
            input_sequence_length=champion_row.input_sequence_length, input_semantics=INPUT_SEMANTICS_PRODUCTION,
            forecast_horizon_days=champion_row.forecast_horizon_days, feature_names=list(FEATURE_NAMES),
            adapter_strategy=champion_row.adapter_strategy, model_path=str(written.model_path),
            scaler_path=str(written.scaler_path), metadata_path=str(written.metadata_path),
            artifact_sha256=written.checksums, artifact_origin=metadata["artifact_origin"],
            training_data_cutoff=cutoff, training_started_at=datetime.fromtimestamp(challenger.started_at, UTC),
            training_completed_at=datetime.fromtimestamp(challenger.completed_at, UTC),
            training_sample_count=int(challenger.train_summary["samples"]), training_config=cfg.as_dict(),
            dataset_manifest=manifest, validation_metrics={"history": {k: v[-1:] for k, v in challenger.history.items()}},
            test_metrics=metrics["protocols"], day1_error=head["day1_error"], day3_error=head["day3_error"],
            day7_error=head["day7_error"], status="candidate", status_reason="trained", retraining_run_id=run.id,
        )
        self.session.add(mv)
        self.session.flush()  # FK target must exist before its status event (no ORM relationship orders them)
        self.session.add(ModelStatusEvent(model_version=candidate, from_status=None, to_status="candidate",
                                          reason=f"trained by retraining run {run.run_id}", actor="retraining", retraining_run_id=run.id))
        self.session.flush()
        for p, (c, n) in protocols.items():
            self.registry.record_metrics(champion_row.version, p, {str(h): s_.as_dict() for h, s_ in c.by_horizon.items()},
                                         "retraining_comparison", c.overall.as_dict(), run.id)
            self.registry.record_metrics(candidate, p, {str(h): s_.as_dict() for h, s_ in n.by_horizon.items()},
                                         "retraining_comparison", n.overall.as_dict(), run.id)

        self.registry.transition(mv, "validated", f"evaluated on {', '.join(protocols)}", "retraining", run.id)
        reason = "; ".join(decision.reasons)
        if decision.promote:
            self.registry.promote(mv, f"challenger beat champion {champion_row.version}: {reason}", "retraining", run.id)
            run.status = "promoted"
            log.info("candidate_promoted", version=candidate, champion=champion_row.version, reasons=decision.reasons)
        else:
            self.registry.transition(mv, "rejected", reason, "retraining", run.id)
            run.status = "rejected"
            log.info("candidate_rejected", version=candidate, champion=champion_row.version, reasons=decision.reasons)
        run.sample_count = int(challenger.train_summary["samples"])
        run.dataset_manifest = manifest
        run.metrics = metrics
        run.decision = decision.as_dict()
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        return run


def benchmark_version(session: Session, settings: Settings, version: str) -> dict[str, Any]:
    """Evaluate one version on the fixed historical test split (all 7 horizons) and record metrics."""
    from ml.evaluation.evaluator import evaluate_bundle, evaluate_constant_velocity

    registry = ModelRegistry(session, settings)
    mv = registry.get(version)
    if mv is None:
        raise LookupError(f"unknown model version {version}")
    bundle = registry.load_bundle(mv)
    svc = RetrainingService(session, settings)
    hist = build_historical_sequences(svc._historical_tracks())
    split = split_by_boundaries(hist, np.datetime64(settings.historical_train_end), np.datetime64(settings.historical_validation_end))
    protocol = f"historical_test_{settings.historical_validation_end}"
    result = evaluate_bundle(bundle, split.test, protocol)
    cv = evaluate_constant_velocity(split.test, protocol)
    registry.record_metrics(version, protocol, {str(h): s.as_dict() for h, s in result.by_horizon.items()}, "benchmark", result.overall.as_dict())
    session.commit()
    return {"version": version, "protocol": protocol, "model": result.as_dict(), "constant_velocity_benchmark": cv.as_dict(),
            "test_samples": len(split.test), "computed_at": (datetime.now(UTC) + timedelta()).isoformat()}
