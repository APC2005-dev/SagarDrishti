"""Forecast generation with the deployed (champion) model.

For each active iceberg whose latest official observation has not yet been
forecast by the current champion: build the 14-entry sequence, run one GRU
pass, persist a forecast set (inputs + diagnostics) and its 7 daily forecasts.
Existing forecasts are never modified.

Environmental champions additionally receive per-entry wind / current / sea-ice
values built with the as-of rule (nothing published after the forecast time
is used). If any required value is missing for an iceberg, that iceberg is
forecast by the trajectory-only fallback model instead and the forecast set
records why — values are never fabricated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import Forecast, ForecastRun, ForecastSet, ModelVersion
from app.models.tracking import Iceberg, Observation
from app.services.geo import ewkt_point
from app.services.model_registry import ModelRegistry
from ml.adapters.production_adapter import ADAPTER_VERSION, AdapterConfig, ProductionSequenceAdapter
from ml.adapters.sequence_builder import ModelInputSequence, TrajectoryEntry
from ml.environment.feature_builder import EnvFeatureBlock
from ml.features.schemas import TRAJECTORY_SCHEMA, get_schema
from ml.inference.forecast_generator import GeneratedForecast, generate_forecasts
from ml.models.model_loader import ModelArtifactError
from ml.provenance import Provenance

log = get_logger(__name__)
HISTORY_ROWS_PER_ICEBERG = 64  # >> 14; bounds the query regardless of archive size


@dataclass
class ForecastOutcome:
    run_id: int
    status: str
    model_version: str | None
    created_sets: int = 0
    already_forecast: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    fallback_sets: int = 0


def load_recent_entries(session: Session, iceberg_ids: list[str], per_iceberg: int = HISTORY_ROWS_PER_ICEBERG) -> dict[str, list[TrajectoryEntry]]:
    """Latest N model-eligible observations per iceberg (window function)."""
    if not iceberg_ids:
        return {}
    rn = func.row_number().over(partition_by=Observation.iceberg_id, order_by=Observation.observation_date.desc()).label("rn")
    inner = (
        select(Observation.id, Observation.iceberg_id, Observation.observation_date, Observation.latitude,
               Observation.longitude, Observation.provenance, Observation.source, rn)
        .where(Observation.iceberg_id.in_(iceberg_ids),
               Observation.provenance.in_([p.value for p in (Provenance.OFFICIAL_USNIC, Provenance.HISTORICAL_TRAINING_DATASET)]))
        .subquery()
    )
    rows = session.execute(select(inner).where(inner.c.rn <= per_iceberg).order_by(inner.c.iceberg_id, inner.c.observation_date)).all()
    grouped: dict[str, list[TrajectoryEntry]] = {i: [] for i in iceberg_ids}
    for r in rows:
        grouped[r.iceberg_id].append(
            TrajectoryEntry(r.iceberg_id, r.observation_date, r.latitude, r.longitude, Provenance(r.provenance), r.id, r.source)
        )
    return grouped


def _entry_environment(block: EnvFeatureBlock | None, i: int) -> dict[str, Any] | None:
    if block is None:
        return None
    return {
        g.value: {"values": s.values, "valid_date": s.valid_date.isoformat() if s.valid_date else None,
                  "provider": s.provider, "dataset_id": s.dataset_id, "staleness_days": s.staleness_days,
                  "missing": s.missing}
        for g, s in block.samples[i].items()
    }


class ForecastService:
    def __init__(self, session: Session, settings: Settings, environment: Any = None) -> None:
        self.session = session
        self.settings = settings
        self.registry = ModelRegistry(session, settings)
        self._environment = environment  # injectable EnvironmentService (tests)

    def environment(self) -> Any:
        if self._environment is None:
            from app.services.environment_service import EnvironmentService

            self._environment = EnvironmentService(self.session, self.settings)
        return self._environment

    def _existing(self, version: str, iceberg_ids: list[str]) -> set[tuple[str, int]]:
        return set(
            self.session.execute(
                select(ForecastSet.iceberg_id, ForecastSet.anchor_observation_id).where(
                    ForecastSet.model_version == version, ForecastSet.iceberg_id.in_(iceberg_ids or [""])
                )
            ).tuples()
        )

    def run(self, trigger: str = "scheduled", ingestion_run_id: int | None = None, iceberg_ids: list[str] | None = None) -> ForecastOutcome:
        champion = self.registry.get_deployed()
        run = ForecastRun(model_version=champion.version if champion else None, trigger=trigger,
                          ingestion_run_id=ingestion_run_id, status="running")
        self.session.add(run)
        self.session.commit()
        if champion is None:
            return self._finish(run, "skipped", error="no deployed model version")
        try:
            bundle = self.registry.load_bundle(champion)
        except ModelArtifactError as exc:
            log.error("forecast_model_unavailable", version=champion.version, error=str(exc))
            return self._finish(run, "failed", error=f"model {champion.version} unavailable: {exc}")
        schema = get_schema(champion.feature_schema_version or TRAJECTORY_SCHEMA)

        query = select(Iceberg.iceberg_id).where(Iceberg.status == "active")
        if iceberg_ids:
            query = query.where(Iceberg.iceberg_id.in_(iceberg_ids))
        targets = list(self.session.execute(query).scalars())
        run.icebergs_considered = len(targets)

        entries = load_recent_entries(self.session, targets)
        adapter = ProductionSequenceAdapter(
            AdapterConfig(strategy=champion.adapter_strategy or self.settings.adapter_strategy,
                          max_gap_days=self.settings.adapter_max_gap_days)
        )
        result = adapter.build_many(entries)
        skipped = Counter(s.reason.value for s in result.skipped)
        existing = self._existing(champion.version, [s.iceberg_id for s in result.sequences])
        fresh = [s for s in result.sequences if (s.iceberg_id, s.anchor.observation_id) not in existing]
        run.already_forecast = len(result.sequences) - len(fresh)

        generated_at = datetime.now(UTC)
        as_of = generated_at.date()
        created = 0
        fallback_created = 0
        try:
            if not schema.is_environmental:
                generated = generate_forecasts(bundle, fresh, self.registry.risk_radii(champion.version))
                for fc in generated:
                    self._persist(run, fc, champion, schema.version, generated_at)
                created = len(generated)
            else:
                ready: list[tuple[ModelInputSequence, EnvFeatureBlock]] = []
                lacking: list[tuple[ModelInputSequence, EnvFeatureBlock]] = []
                builder = self.environment().operational_builder()
                for seq in fresh:
                    block = builder.build([(e.latitude, e.longitude, e.observation_date) for e in seq.entries], as_of, schema.env_variables)
                    (ready if block.complete else lacking).append((seq, block))
                generated = generate_forecasts(
                    bundle, [s for s, _ in ready], self.registry.risk_radii(champion.version),
                    features=[np.concatenate([s.features, b.values], axis=1) for s, b in ready],
                )
                for fc, (_, block) in zip(generated, ready, strict=True):
                    fs = self._persist(run, fc, champion, schema.version, generated_at, block=block, as_of=as_of)
                    self._archive(fs, fc.sequence, as_of)
                created = len(generated)
                if lacking:
                    fallback_created = self._forecast_fallback(run, champion, lacking, generated_at, as_of, skipped)
        except Exception as exc:  # inference failure must not leave a half-written run
            self.session.rollback()
            log.exception("forecast_inference_failed", version=champion.version)
            return self._finish(run, "failed", skipped=dict(skipped), error=f"inference failed: {exc}")

        run.forecast_sets_created = created + fallback_created
        status = "success" if not skipped else "partial"
        if not run.forecast_sets_created and not run.already_forecast:
            status = "skipped" if skipped else "success"
        return self._finish(run, status, skipped=dict(skipped), fallback_sets=fallback_created)

    def _forecast_fallback(self, run: ForecastRun, champion: ModelVersion, lacking: list[tuple[ModelInputSequence, EnvFeatureBlock]],
                           generated_at: datetime, as_of: date, skipped: Counter) -> int:
        fb = self.registry.trajectory_fallback(champion)
        if fb is None:
            skipped["environment_unavailable"] += len(lacking)
            return 0
        fb_bundle = self.registry.load_bundle(fb)
        done = self._existing(fb.version, [s.iceberg_id for s, _ in lacking])
        todo = [(s, b) for s, b in lacking if (s.iceberg_id, s.anchor.observation_id) not in done]
        generated = generate_forecasts(fb_bundle, [s for s, _ in todo], self.registry.risk_radii(fb.version))
        for fc, (_, block) in zip(generated, todo, strict=True):
            reasons = block.missing_reasons()
            self._persist(
                run, fc, fb, fb.feature_schema_version, generated_at, block=block, as_of=as_of,
                fallback={"champion": champion.version, "champion_schema": champion.feature_schema_version,
                          "used_model": fb.version, "reason": "environmental inputs incomplete",
                          "missing": reasons[:20], "missing_count": len(reasons)},
                include_env_features=False,
            )
        log.warning("forecast_environment_fallback", champion=champion.version, fallback=fb.version, icebergs=len(generated))
        return len(generated)

    def _persist(self, run: ForecastRun, fc: GeneratedForecast, model: ModelVersion, schema_version: str, generated_at: datetime,
                 block: EnvFeatureBlock | None = None, as_of: date | None = None, fallback: dict[str, Any] | None = None,
                 include_env_features: bool = True) -> ForecastSet:
        seq = fc.sequence
        features = seq.features
        if block is not None and include_env_features:
            features = np.concatenate([seq.features, block.values], axis=1)
        fs = ForecastSet(
            forecast_run_id=run.id,
            iceberg_id=seq.iceberg_id,
            model_version=model.version,
            generated_at=generated_at,
            anchor_observation_id=seq.anchor.observation_id,
            latest_observation_date=seq.anchor.observation_date,
            adapter_version=ADAPTER_VERSION,
            adapter_strategy=seq.adapter_strategy,
            input_observation_ids=[e.observation_id for e in seq.entries],
            input_entries=[
                {"observation_id": e.observation_id, "date": e.observation_date.isoformat(), "latitude": e.latitude,
                 "longitude": e.longitude, "provenance": e.provenance.value, "elapsed_days": float(seq.elapsed_days[i]),
                 **({"environment": _entry_environment(block, i)} if block is not None else {})}
                for i, e in enumerate(seq.entries)
            ],
            input_features=np.where(np.isfinite(features), features, np.nan).astype(float).round(6).tolist(),
            diagnostics=seq.diagnostics.as_dict(),
            feature_schema_version=schema_version,
            environment=block.provenance() if block is not None else None,
            environment_as_of=as_of if block is not None else None,
            fallback=fallback,
        )
        self.session.add(fs)
        self.session.flush()
        for p in fc.points:
            self.session.add(
                Forecast(
                    forecast_set_id=fs.id,
                    iceberg_id=seq.iceberg_id,
                    model_version=model.version,
                    generated_at=generated_at,
                    latest_observation_date=seq.anchor.observation_date,
                    forecast_horizon_days=p.horizon_days,
                    forecast_date=p.forecast_date,
                    predicted_latitude=p.predicted_latitude,
                    predicted_longitude=p.predicted_longitude,
                    predicted_x_m=p.predicted_x_m,
                    predicted_y_m=p.predicted_y_m,
                    geom=ewkt_point(p.predicted_latitude, p.predicted_longitude),
                    risk_radius_km_p90=p.risk_radius_km_p90,
                )
            )
        return fs

    def _archive(self, fs: ForecastSet, seq: ModelInputSequence, as_of: date) -> None:
        if not self.settings.env_archive_forecast_fields:
            return
        try:
            with self.session.begin_nested():
                self.environment().archive_forecast_fields(fs.id, seq.anchor.latitude, seq.anchor.longitude, as_of)
        except Exception as exc:  # archiving is best-effort and never blocks a forecast
            log.warning("forecast_field_archive_failed", forecast_set_id=fs.id, error=str(exc))

    def _finish(self, run: ForecastRun, status: str, skipped: dict[str, int] | None = None, error: str | None = None,
                fallback_sets: int = 0) -> ForecastOutcome:
        run.status = status
        run.skipped = skipped or {}
        run.error_message = error
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        log.info("forecast_generation_completed", run_id=run.id, model_version=run.model_version, status=status,
                 created=run.forecast_sets_created, fallback=fallback_sets, already=run.already_forecast, skipped=skipped, error=error)
        return ForecastOutcome(run.id, status, run.model_version, run.forecast_sets_created, run.already_forecast, skipped or {}, error,
                               fallback_sets)


def latest_forecast_set_ids(session: Session, model_version: str | None = None) -> select:  # type: ignore[valid-type]
    """Subquery: newest forecast set per iceberg (optionally for one model)."""
    q = select(ForecastSet.iceberg_id, func.max(ForecastSet.generated_at).label("g"))
    if model_version:
        q = q.where(ForecastSet.model_version == model_version)
    latest = q.group_by(ForecastSet.iceberg_id).subquery()
    return select(ForecastSet.id).join(latest, and_(latest.c.iceberg_id == ForecastSet.iceberg_id, latest.c.g == ForecastSet.generated_at))
