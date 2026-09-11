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
from app.models.ml import ForecastEvaluation, ModelStatusEvent, ModelVersion, RetrainingRun
from app.models.tracking import Observation
from app.services.environment_service import EnvironmentService
from app.services.forecast_service import load_recent_entries
from app.services.model_registry import ModelRegistry
from ml.adapters.production_adapter import AdapterConfig, ProductionSequenceAdapter
from ml.constants import FEATURE_NAMES, FORECAST_DAYS, INPUT_SEMANTICS_PRODUCTION
from ml.environment.alignment import attach_environment
from ml.environment.sampler import INTERPOLATION
from ml.evaluation.champion_challenger import PromotionPolicy, decide
from ml.evaluation.evaluator import EvaluationResult, evaluate_bundle
from ml.evaluation.metrics import headline_errors
from ml.features.coordinate_transform import latlon_to_polar_m
from ml.features.schemas import ALL_FEATURES, ENV_VARIABLES, TRAJECTORY_SCHEMA, FeatureSchema, get_schema
from ml.models.artifact_store import write_version
from ml.training.dataset_builder import (
    ChronologicalSplit,
    SequenceDataset,
    build_historical_sequences,
    build_operational_samples,
    split_by_boundaries,
    split_by_fraction,
    stratified_cap,
)
from ml.training.environmental import ablation_report, complete_subset, train_schema_candidate
from ml.training.retrainer import RetrainConfig, evaluate_pair, train_challenger
from ml.training.trainer import TrainingConfig
from ml.versioning.version_manager import BASE_VERSION, next_version

log = get_logger(__name__)
FINISHED = ("promoted", "rejected")


class RetrainingService:
    def __init__(self, session: Session, settings: Settings, environment: EnvironmentService | None = None) -> None:
        self.session = session
        self.settings = settings
        self.registry = ModelRegistry(session, settings)
        self._environment = environment

    def environment(self) -> EnvironmentService:
        if self._environment is None:
            self._environment = EnvironmentService(self.session, self.settings)
        return self._environment

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
            env_schemas, unavailable = self.trainable_environmental_schemas()
            champion_row = self.registry.get_deployed()
            if env_schemas or (champion_row is not None and champion_row.model_type == "environmental"):
                return self._run_experiment(run, env_schemas, unavailable)
            if unavailable:
                run.experiment = {"environmental_schemas_unavailable": unavailable}
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
            model_type="trajectory", feature_schema_version=champion_row.feature_schema_version,
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

    # ------------------------------------------------------ feature-schema experiment
    def trainable_environmental_schemas(self) -> tuple[list[FeatureSchema], dict[str, str]]:
        """Environmental schemas from ENV_CANDIDATE_SCHEMAS whose sources are configured."""
        if not self.settings.env_enabled:
            return [], {}
        schemas: list[FeatureSchema] = []
        unavailable: dict[str, str] = {}
        for name in self.settings.env_candidate_schemas:
            try:
                schema = get_schema(name)
            except ValueError as exc:
                unavailable[name] = str(exc)
                continue
            if not schema.is_environmental:
                continue
            ok, reason = self.environment().schema_trainable(schema)
            if ok:
                schemas.append(schema)
            else:
                unavailable[name] = reason
        return schemas, unavailable

    def prefetch_environment(self, max_samples: int | None = None) -> dict[str, Any]:
        """Fill the environmental cache for the samples a retraining experiment will use
        (same subsampling and as-of rule), without training. Resumable: cached tiles are reused."""
        s = self.settings
        if not self.trainable_environmental_schemas()[0]:
            return {"status": "skipped", "reason": "no environmental schema is trainable (sources not configured)"}
        cap = max_samples or s.env_training_max_historical_samples
        tracks = self._historical_tracks()
        hist = build_historical_sequences(tracks, with_entries=True) if len(tracks) else SequenceDataset.empty()
        hsplit = split_by_boundaries(hist, np.datetime64(s.historical_train_end), np.datetime64(s.historical_validation_end))
        builder = self.environment().training_builder()
        reports = {}
        for name, ds in (("historical_train", stratified_cap(hsplit.train, cap)),
                         ("historical_validation", stratified_cap(hsplit.validation, max(cap // 5, 1))),
                         ("historical_test", stratified_cap(hsplit.test, s.env_training_max_test_samples))):
            if len(ds):
                _, rep = attach_environment(ds, builder)
                reports[name] = rep.as_dict()
                self.session.commit()
        return {"status": "done", "tiles_fetched": builder.cache.fetch_count, "alignment": reports}

    def _split_operational(self, ops: SequenceDataset) -> ChronologicalSplit:
        s = self.settings
        if len(ops) >= 3:
            v, t = s.retrain_operational_validation_fraction, s.retrain_operational_test_fraction
            return split_by_fraction(ops, 1 - v - t, v)
        empty = SequenceDataset.empty(ops.feature_names)
        return ChronologicalSplit(empty, empty, ops, None, None)

    def _run_experiment(self, run: RetrainingRun, env_schemas: list[FeatureSchema], unavailable: dict[str, str]) -> RetrainingRun:
        """Train one candidate per feature schema on identical data, select on validation,
        promote only if the selected candidate beats BOTH the champion and the base."""
        s = self.settings
        champion_row = self.registry.get_deployed()
        if champion_row is None:
            raise RuntimeError("no deployed champion to compare against")
        champion = self.registry.load_bundle(champion_row)
        base_row = self.registry.get(BASE_VERSION)
        base = self.registry.load_bundle(base_row) if base_row else None
        env = self.environment()
        cutoff = datetime.now(UTC)
        run.champion_version = champion_row.version
        run.source_data_cutoff = cutoff
        self.session.commit()
        log.info("retraining_experiment_started", run_id=str(run.run_id), champion=champion_row.version,
                 schemas=[x.version for x in env_schemas], unavailable=unavailable)

        adapter = ProductionSequenceAdapter(AdapterConfig(strategy=champion_row.adapter_strategy or s.adapter_strategy,
                                                          max_gap_days=s.adapter_max_gap_days))
        tracks = self._historical_tracks()
        hist = build_historical_sequences(tracks, with_entries=True) if len(tracks) else SequenceDataset.empty()
        hsplit = split_by_boundaries(hist, np.datetime64(s.historical_train_end), np.datetime64(s.historical_validation_end))
        ops = self._operational(cutoff, adapter)
        osplit = self._split_operational(ops)
        parts = {
            "historical_train": stratified_cap(hsplit.train, s.env_training_max_historical_samples),
            "historical_validation": stratified_cap(hsplit.validation, max(s.env_training_max_historical_samples // 5, 1)),
            "historical_test": stratified_cap(hsplit.test, s.env_training_max_test_samples),
            "operational_train": osplit.train,
            "operational_validation": osplit.validation,
            "operational_test": osplit.test,
        }
        builder = env.training_builder()
        aligned: dict[str, SequenceDataset] = {}
        alignment: dict[str, Any] = {}
        for name, ds in parts.items():
            if len(ds):
                aligned[name], report = attach_environment(ds, builder)
                alignment[name] = report.as_dict()
            else:
                aligned[name] = SequenceDataset.empty(ALL_FEATURES)
        self.session.commit()  # persist cache-index rows written during alignment

        train = SequenceDataset.concat([aligned["historical_train"], aligned["operational_train"]])
        validation = SequenceDataset.concat([aligned["historical_validation"], aligned["operational_validation"]])
        schemas = ([get_schema(TRAJECTORY_SCHEMA)] if TRAJECTORY_SCHEMA in s.env_candidate_schemas else []) + env_schemas
        champion_schema = get_schema(champion_row.feature_schema_version or TRAJECTORY_SCHEMA)
        union_env = [v for v in ENV_VARIABLES if any(v in sc.env_variables for sc in [*schemas, champion_schema])]
        union = FEATURE_NAMES + tuple(union_env)
        common_val = complete_subset(validation, union)
        tests = {k: complete_subset(aligned[f"{k}_test"], union) for k in ("historical", "operational")}

        cfg = TrainingConfig(learning_rate=5e-4, epochs=s.retrain_epochs)
        existing = [m.version for m in self.registry.list_versions()]
        allocated: list[str] = []
        outcomes = []
        failures: dict[str, str] = {}
        for schema in schemas:
            version = next_version(existing + allocated, Path(s.models_dir))
            try:
                outcomes.append(train_schema_candidate(schema, train, validation, cfg, version))
                allocated.append(version)
            except ValueError as exc:
                failures[schema.version] = str(exc)
        if not outcomes:
            raise RuntimeError(f"no candidate could be trained: {failures}")

        P = s.promotion_primary_horizon
        val_results = {
            oc.schema.version: (evaluate_bundle(oc.bundle, common_val, "validation_common") if len(common_val) else oc.validation)
            for oc in outcomes
        }

        def val_score(oc) -> float:  # type: ignore[no-untyped-def]
            m = val_results[oc.schema.version].by_horizon[P].mae_km
            return float("inf") if m is None else float(m)

        best = min(outcomes, key=val_score)

        policy = self.promotion_policy()

        def labels(ds: SequenceDataset) -> int:
            return int(np.isfinite(ds.y.reshape(-1, FORECAST_DAYS, 2)[:, P - 1, 0]).sum()) if len(ds) else 0

        if labels(tests["operational"]) >= policy.min_evaluation_samples:
            key = "operational"
        elif len(tests["historical"]):
            key = "historical"
        elif len(tests["operational"]):
            key = "operational"
        else:
            raise RuntimeError("no common test samples with complete environmental inputs — cannot compare candidates")
        protocol = f"{key}_common_{str(run.run_id)[:8]}"
        test_ds = tests[key]

        champion_label = f"champion:{champion_row.version}"
        bundles = {"base": base} if base is not None else {}
        bundles[champion_label] = champion
        for oc in outcomes:
            bundles[oc.schema.version] = oc.bundle
        reference = "base" if base is not None else champion_label
        report_test = ablation_report(bundles, test_ds, protocol, reference=reference)
        report_val = ablation_report(bundles, common_val, "validation_common", reference=reference) if len(common_val) else None
        results = {label: evaluate_bundle(b, test_ds, protocol) for label, b in bundles.items()}

        best_label = best.schema.version
        dec_champion = decide(results[champion_label], results[best_label], policy)
        dec_base = decide(results["base"], results[best_label], policy) if base is not None else None
        promote = dec_champion.promote and (dec_base.promote if dec_base is not None else True)
        reasons = [f"vs champion {champion_row.version}: " + "; ".join(dec_champion.reasons)]
        if dec_base is not None:
            reasons.append("vs base: " + "; ".join(dec_base.reasons))

        manifest = {
            "cutoff": cutoff.isoformat(),
            "parts": {k: v.summary() for k, v in aligned.items()},
            "common_samples": {"validation": len(common_val), "historical_test": len(tests["historical"]),
                               "operational_test": len(tests["operational"])},
            "union_features": list(union),
            "operational_anchor_observation_ids_sha256": hashlib.sha256(
                json.dumps(sorted(int(i) for i in ops.anchor_observation_ids if i is not None)).encode()
            ).hexdigest(),
        }
        env_policy = {
            "as_of_rule": "entry observed on d, prediction time T: field valid on min(d, T - operational latency); nothing published after T is read",
            "interpolation": INTERPOLATION,
            "missing_data": "complete-case: samples with any missing required value are excluded from training and evaluation; "
                            "at inference the iceberg falls back to a trajectory-only model and the fallback is recorded",
            "forecast_period_environment": "not a model input: no leakage-free archive of issued forecasts exists for the training "
                                           "period; issued forecast fields are archived from now on (environmental.forecast_snapshots)",
            "max_staleness_days": s.env_max_staleness_days,
        }
        groups_used = tuple(sorted({g for oc in outcomes for g in oc.schema.groups}))
        sources = env.data_sources(groups_used) if groups_used else {}

        rows: dict[str, ModelVersion] = {}
        for oc in outcomes:
            schema, version = oc.schema, oc.bundle.version
            test_r = results[schema.version]
            head = headline_errors(test_r.by_horizon)
            is_env = schema.is_environmental
            meta = {
                "version": version, "parent_version": champion_row.version,
                "model_type": "environmental" if is_env else "trajectory",
                "architecture": "GRU", "architecture_version": oc.architecture_version,
                "input_sequence_length": 14, "input_semantics": INPUT_SEMANTICS_PRODUCTION, "forecast_horizon_days": FORECAST_DAYS,
                "feature_schema_version": schema.version, "feature_schema": schema.as_dict(), "feature_names": list(schema.features),
                "adapter_strategy": champion_row.adapter_strategy, "artifact_origin": "retraining_experiment:scratch",
                "initialisation": "scratch", "training_data_cutoff": cutoff.isoformat(),
                "environmental_data_sources": {g: sources[g] for g in schema.groups} if is_env else None,
                "environmental_data_cutoff": cutoff.date().isoformat() if is_env else None,
                "environmental_policy": env_policy if is_env else None,
                "retraining_run_id": str(run.run_id),
                "training": {"config": cfg.as_dict(), "train_samples": oc.train_samples, "validation_samples": oc.validation_samples,
                             "started_at": datetime.fromtimestamp(oc.started_at, UTC).isoformat(),
                             "completed_at": datetime.fromtimestamp(oc.completed_at, UTC).isoformat(),
                             "final_epoch": {k: v[-1] for k, v in oc.history.items() if v}},
                "metrics": {"protocol": "validation_common", **val_results[schema.version].as_dict()},
                "test_metrics": test_r.as_dict(),
                "created_at": datetime.now(UTC).isoformat(),
            }
            written = write_version(Path(s.models_dir), version, oc.bundle.model, oc.bundle.feature_scaler, oc.bundle.target_scaler,
                                    meta, extra_files={"dataset_manifest.json": manifest}, feature_names=schema.features)
            mv = ModelVersion(
                version=version, version_number=int(version[1:]), parent_version=champion_row.version, architecture="GRU",
                architecture_version=oc.architecture_version, input_sequence_length=14, input_semantics=INPUT_SEMANTICS_PRODUCTION,
                forecast_horizon_days=FORECAST_DAYS, feature_names=list(schema.features), adapter_strategy=champion_row.adapter_strategy,
                model_path=str(written.model_path), scaler_path=str(written.scaler_path), metadata_path=str(written.metadata_path),
                artifact_sha256=written.checksums, artifact_origin="retraining_experiment:scratch", training_data_cutoff=cutoff,
                training_started_at=datetime.fromtimestamp(oc.started_at, UTC), training_completed_at=datetime.fromtimestamp(oc.completed_at, UTC),
                training_sample_count=oc.train_samples, training_config=cfg.as_dict(), dataset_manifest=manifest,
                validation_metrics=val_results[schema.version].as_dict(), test_metrics=test_r.as_dict(),
                day1_error=head["day1_error"], day3_error=head["day3_error"], day7_error=head["day7_error"],
                status="candidate", status_reason="trained (feature-schema experiment)", retraining_run_id=run.id,
                model_type="environmental" if is_env else "trajectory", feature_schema_version=schema.version,
                environmental_data_sources=meta["environmental_data_sources"],
                environmental_data_cutoff=cutoff.date() if is_env else None,
            )
            self.session.add(mv)
            self.session.flush()
            self.session.add(ModelStatusEvent(model_version=version, from_status=None, to_status="candidate",
                                              reason=f"trained by retraining run {run.run_id} (schema {schema.version})",
                                              actor="retraining", retraining_run_id=run.id))
            self.session.flush()
            self.registry.record_metrics(version, protocol, {str(h): x.as_dict() for h, x in test_r.by_horizon.items()},
                                         "retraining_experiment", test_r.overall.as_dict(), run.id)
            self.registry.transition(mv, "validated", f"evaluated on {protocol} ({len(test_ds)} common samples)", "retraining", run.id)
            rows[schema.version] = mv
        for label, row in (("base", base_row), (champion_label, champion_row)):
            if row is not None and label in results:
                r = results[label]
                self.registry.record_metrics(row.version, protocol, {str(h): x.as_dict() for h, x in r.by_horizon.items()},
                                             "retraining_comparison", r.overall.as_dict(), run.id)

        for oc in outcomes:
            mv = rows[oc.schema.version]
            if oc is best:
                if promote:
                    self.registry.promote(mv, "; ".join(reasons), "retraining", run.id)
                else:
                    self.registry.transition(mv, "rejected", "; ".join(reasons), "retraining", run.id)
            else:
                fmt = lambda x: "n/a" if x == float("inf") else f"{x:.3f} km"  # noqa: E731
                self.registry.transition(
                    mv, "rejected",
                    f"not selected: validation D+{P} MAE {fmt(val_score(oc))} vs {best_label} {fmt(val_score(best))}",
                    "retraining", run.id,
                )

        run.status = "promoted" if promote else "rejected"
        run.candidate_version = rows[best_label].version
        run.sample_count = best.train_samples
        run.dataset_manifest = manifest
        run.metrics = {"primary_protocol": protocol, "test": report_test, "validation": report_val}
        run.decision = {
            "promote": promote, "protocol": protocol, "selected_schema": best_label, "selected_version": rows[best_label].version,
            "reasons": reasons, "vs_champion": dec_champion.as_dict(), "vs_base": dec_base.as_dict() if dec_base else None,
        }
        run.experiment = {
            "schemas": [x.version for x in schemas],
            "candidates": [
                {"version": rows[oc.schema.version].version, "schema": oc.schema.version, "architecture_version": oc.architecture_version,
                 "train_samples": oc.train_samples, "validation_samples": oc.validation_samples,
                 "validation_primary_mae_km": None if val_score(oc) == float("inf") else val_score(oc),
                 "status": rows[oc.schema.version].status}
                for oc in outcomes
            ],
            "failures": failures,
            "unavailable": unavailable,
            "selection": {"criterion": f"lowest D+{P} MAE on common validation samples", "selected": best_label,
                          "common_validation_samples": len(common_val)},
            "alignment": alignment,
            "union_features": list(union),
        }
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        log.info("candidate_promoted" if promote else "candidate_rejected", version=rows[best_label].version, schema=best_label,
                 champion=champion_row.version, reasons=reasons)
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
