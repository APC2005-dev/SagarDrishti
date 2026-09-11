"""Model registry: what exists, what is deployed, lineage, and status changes.

Status lifecycle::

    candidate -> validated -> deployed -> archived
         \\            \\
          -> rejected   -> rejected / archived

Only one version may be ``deployed`` (enforced by a partial unique index).
Every transition writes a ``model_status_events`` row with a reason.
Artifacts on disk are immutable; the registry records their sha256 and a
bundle is refused if the files no longer match.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import ModelMetric, ModelStatusEvent, ModelVersion
from ml.adapters.production_adapter import STRATEGY_BOOTSTRAP_V1
from ml.constants import INPUT_SEMANTICS_PRODUCTION
from ml.features.schemas import get_schema
from ml.models.artifact_store import WrittenArtifact, copy_version
from ml.models.model_loader import (
    ModelArtifactError,
    ModelBundle,
    feature_contract,
    load_model_bundle,
    read_metadata,
    resolve_artifacts_or_none,
)
from ml.versioning.version_manager import BASE_VERSION, version_number

log = get_logger(__name__)

ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"candidate", "validated"},
    "candidate": {"validated", "rejected"},
    "validated": {"deployed", "rejected", "archived"},
    "deployed": {"archived"},
    "rejected": set(),
    "archived": {"deployed"},  # explicit, audited rollback
}
_REGISTRY_LOCK = 0x5A6D_0002


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class LineageNode:
    version: str
    parent_version: str | None
    status: str
    status_reason: str | None
    day7_error: float | None
    created_at: datetime


_bundle_cache: dict[tuple[str, str], ModelBundle] = {}
_bundle_lock = threading.Lock()


class ModelRegistry:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    # ------------------------------------------------------------------ queries
    def get(self, version: str) -> ModelVersion | None:
        return self.session.execute(select(ModelVersion).where(ModelVersion.version == version)).scalar_one_or_none()

    def get_deployed(self) -> ModelVersion | None:
        return self.session.execute(select(ModelVersion).where(ModelVersion.status == "deployed")).scalar_one_or_none()

    def list_versions(self) -> list[ModelVersion]:
        return list(self.session.execute(select(ModelVersion).order_by(ModelVersion.version_number)).scalars())

    def lineage(self) -> list[LineageNode]:
        return [
            LineageNode(m.version, m.parent_version, m.status, m.status_reason, m.day7_error, m.created_at)
            for m in self.list_versions()
        ]

    def risk_radii(self, version: str) -> dict[int, float]:
        """p90 error per horizon: operational evaluations preferred, then offline benchmarks."""
        rows = self.session.execute(
            select(ModelMetric)
            .where(ModelMetric.model_version == version, ModelMetric.horizon_days.is_not(None), ModelMetric.p90_km.is_not(None))
            .order_by(ModelMetric.computed_at.desc())
        ).scalars()
        chosen: dict[int, tuple[int, float]] = {}
        for m in rows:
            rank = 0 if m.protocol.startswith("operational") else 1
            h = int(m.horizon_days)  # type: ignore[arg-type]
            if h not in chosen or rank < chosen[h][0]:
                chosen[h] = (rank, float(m.p90_km))  # type: ignore[arg-type]
        return {h: v for h, (_, v) in chosen.items()}

    def load_bundle(self, mv: ModelVersion) -> ModelBundle:
        key = (mv.version, json.dumps(mv.artifact_sha256, sort_keys=True))
        with _bundle_lock:
            if key not in _bundle_cache:
                bundle = load_model_bundle(Path(mv.model_path).parent, version=mv.version)
                if bundle.checksums and mv.artifact_sha256:
                    for name, digest in mv.artifact_sha256.items():
                        if bundle.checksums.get(name) not in (None, digest):
                            raise ModelArtifactError(f"{mv.version}: {name} differs from registered checksum")
                _bundle_cache[key] = bundle
            return _bundle_cache[key]

    # ---------------------------------------------------------------- mutation
    def _lock(self) -> None:
        self.session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _REGISTRY_LOCK})

    def transition(
        self, mv: ModelVersion, to_status: str, reason: str, actor: str = "system", retraining_run_id: int | None = None
    ) -> None:
        if to_status not in ALLOWED_TRANSITIONS.get(mv.status, set()):
            raise RegistryError(f"{mv.version}: transition {mv.status} -> {to_status} not allowed")
        self.session.add(
            ModelStatusEvent(
                model_version=mv.version,
                from_status=mv.status,
                to_status=to_status,
                reason=reason,
                actor=actor,
                retraining_run_id=retraining_run_id,
            )
        )
        log.info("model_status_changed", version=mv.version, from_status=mv.status, to_status=to_status, reason=reason)
        mv.status = to_status
        mv.status_reason = reason
        if to_status == "deployed":
            mv.deployed_at = datetime.now(UTC)
        self.session.flush()

    def promote(self, candidate: ModelVersion, reason: str, actor: str = "system", retraining_run_id: int | None = None) -> None:
        """Atomically replace the champion. The previous champion is archived, never deleted."""
        self._lock()
        current = self.get_deployed()
        if current is not None and current.version != candidate.version:
            self.transition(current, "archived", f"superseded by {candidate.version}", actor, retraining_run_id)
        self.transition(candidate, "deployed", reason, actor, retraining_run_id)

    def record_metrics(
        self, version: str, protocol: str, by_horizon: dict[str, dict[str, Any]], source: str,
        overall: dict[str, Any] | None = None, retraining_run_id: int | None = None,
    ) -> None:
        for h, s in by_horizon.items():
            if not s or s.get("n") in (None, 0) and s.get("mae_km") is None:
                continue
            self.session.add(
                ModelMetric(
                    model_version=version, protocol=protocol, horizon_days=int(h), n=int(s.get("n") or 0),
                    mae_km=s.get("mae_km"), rmse_km=s.get("rmse_km"), median_km=s.get("median_km"), p90_km=s.get("p90_km"),
                    source=source, retraining_run_id=retraining_run_id,
                )
            )
        if overall and overall.get("n"):
            self.session.add(
                ModelMetric(
                    model_version=version, protocol=protocol, horizon_days=None, n=int(overall["n"]),
                    mae_km=overall.get("mae_km"), rmse_km=overall.get("rmse_km"), median_km=overall.get("median_km"),
                    p90_km=overall.get("p90_km"), source=source, retraining_run_id=retraining_run_id,
                )
            )
        self.session.flush()

    def trajectory_fallback(self, champion: ModelVersion) -> ModelVersion | None:
        """Trajectory-only model to use when an environmental champion lacks inputs.

        ``ENV_FALLBACK_MODEL_VERSION`` if set, else the nearest trajectory-only
        ancestor of the champion, else v1.
        """
        if self.settings.env_fallback_model_version:
            return self.get(self.settings.env_fallback_model_version)
        node: ModelVersion | None = champion
        while node is not None:
            if node.model_type == "trajectory" and node.status in ("deployed", "validated", "archived"):
                return node
            node = self.get(node.parent_version) if node.parent_version else None
        return self.get("v1")

    def _row_from_metadata(self, version: str, directory: Path, meta: dict[str, Any], checksums: dict[str, str], status: str) -> ModelVersion:
        model_path, scaler_path, metadata_path = resolve_artifacts_or_none(directory)
        by_h = (meta.get("metrics") or {}).get("by_horizon") or {}
        cutoff = meta.get("training_data_cutoff")
        training = meta.get("training") or {}
        _, schema_version = feature_contract(meta)
        model_type = "base" if version == BASE_VERSION else ("environmental" if get_schema(schema_version).is_environmental else "trajectory")
        env_cutoff = meta.get("environmental_data_cutoff")
        mv = ModelVersion(
            model_type=model_type,
            feature_schema_version=schema_version,
            environmental_data_sources=meta.get("environmental_data_sources"),
            environmental_data_cutoff=date.fromisoformat(str(env_cutoff)[:10]) if env_cutoff else None,
            version=version,
            version_number=version_number(version),
            parent_version=meta.get("parent_version"),
            architecture=meta.get("architecture", "GRU"),
            architecture_version=meta["architecture_version"],
            input_sequence_length=int(meta.get("input_sequence_length", 14)),
            input_semantics=meta.get("input_semantics", INPUT_SEMANTICS_PRODUCTION),
            forecast_horizon_days=int(meta.get("forecast_horizon_days", 7)),
            feature_names=list(meta.get("feature_names", [])),
            adapter_strategy=meta.get("adapter_strategy"),
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            metadata_path=str(metadata_path),
            artifact_sha256=checksums,
            artifact_origin=meta.get("artifact_origin", "trained"),
            training_data_cutoff=datetime.fromisoformat(str(cutoff)).replace(tzinfo=UTC) if cutoff else None,
            training_sample_count=(training.get("split") or {}).get("train", {}).get("examples") or meta.get("training_sample_count"),
            training_config=training or None,
            test_metrics=meta.get("metrics"),
            day1_error=(by_h.get("1") or {}).get("mae_km"),
            day3_error=(by_h.get("3") or {}).get("mae_km"),
            day7_error=(by_h.get("7") or {}).get("mae_km"),
            status=status,
        )
        self.session.add(mv)
        self.session.flush()
        return mv

    def register_base(self) -> ModelVersion:
        """Register ``models/base`` (the original research artifact) if not already registered."""
        self._lock()
        existing = self.get(BASE_VERSION)
        if existing is not None:
            return existing
        directory = Path(self.settings.models_dir) / BASE_VERSION
        meta = read_metadata(directory)
        if resolve_artifacts_or_none(directory)[0] is None:
            raise ModelArtifactError(
                f"base artifact files are missing from {directory}: place global_gru_trajectory_model.keras and "
                "global_gru_trajectory_scalers.joblib there (see models/base/README.md)"
            )
        bundle = load_model_bundle(directory, version=BASE_VERSION)  # validates tensor contract
        mv = self._row_from_metadata(BASE_VERSION, directory, meta, bundle.checksums, status="validated")
        mv.status_reason = "research reference artifact (not served directly)"
        self.session.add(ModelStatusEvent(model_version=BASE_VERSION, from_status=None, to_status="validated",
                                          reason="registered original research artifact", actor="system"))
        metrics = meta.get("metrics") or {}
        if metrics.get("by_horizon"):
            self.record_metrics(BASE_VERSION, metrics.get("protocol", "historical_test"), metrics["by_horizon"], "notebook_record")
        log.info("model_registered", version=BASE_VERSION, sha256=bundle.checksums)
        return mv

    def ensure_v1_bootstrap(self) -> ModelVersion:
        """Create v1 (base weights + bootstrap adapter) and deploy it if no numbered version exists."""
        self._lock()
        existing = self.get("v1")
        if existing is not None:
            return existing
        base = self.get(BASE_VERSION) or self.register_base()
        models_root = Path(self.settings.models_dir)
        target = models_root / "v1"
        base_meta = read_metadata(models_root / BASE_VERSION)
        if target.exists():
            meta = read_metadata(target)
            bundle = load_model_bundle(target, version="v1")
            checksums = bundle.checksums
        else:
            meta = {
                **{k: base_meta[k] for k in ("architecture", "architecture_version", "input_sequence_length",
                                              "forecast_horizon_days", "feature_names") if k in base_meta},
                "version": "v1",
                "parent_version": BASE_VERSION,
                "input_semantics": INPUT_SEMANTICS_PRODUCTION,
                "adapter_strategy": STRATEGY_BOOTSTRAP_V1,
                "artifact_origin": "copy_of_base",
                "training_data_cutoff": base_meta.get("training_data_cutoff"),
                "training": {"note": "no additional training; weights and scalers are byte-identical to base",
                             "parent_artifact_sha256": base.artifact_sha256},
                "metrics": base_meta.get("metrics"),
                "description": (
                    "First production version. Entry 14 (anchor) = latest official USNIC observation; earlier entries "
                    "= most recent historical-training-dataset positions (ProductionSequenceAdapter bootstrap_v1)."
                ),
                "created_at": datetime.now(UTC).isoformat(),
            }
            written: WrittenArtifact = copy_version(models_root, models_root / BASE_VERSION, "v1", meta)
            checksums = written.checksums
        mv = self._row_from_metadata("v1", target, meta, checksums, status="candidate")
        self.session.add(ModelStatusEvent(model_version="v1", from_status=None, to_status="candidate",
                                          reason="bootstrap version created from base", actor="system"))
        metrics = meta.get("metrics") or {}
        if metrics.get("by_horizon"):
            self.record_metrics("v1", metrics.get("protocol", "historical_test"), metrics["by_horizon"], "inherited_from_base")
        self.transition(mv, "validated", "weights identical to base; base historical test metrics apply")
        if self.get_deployed() is None:
            self.promote(mv, "initial production deployment (bootstrap)")
        log.info("model_v1_bootstrapped", status=mv.status)
        return mv
