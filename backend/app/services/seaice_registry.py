"""Registry for the sea-ice model family.

Shares ``ml.model_versions``, ``ml.model_status_events`` and ``ml.model_metrics``
with the trajectory family — and reuses that family's status lifecycle rules
verbatim (:data:`ALLOWED_TRANSITIONS`) rather than inventing a second status
system — but every query is scoped to ``model_family = 'sea_ice'``.

Consequences of that scoping:

* the sea-ice lineage numbers itself from v1 independently of the trajectory
  lineage, so trajectory v4 and sea-ice v4 are unrelated models;
* there is exactly one deployed sea-ice version at a time, enforced by the
  per-family partial unique index, and it is the champion regardless of whether
  a higher-numbered version exists;
* nothing here ever names a version literally: the champion comes from the
  database and new numbers come from :func:`next_version_for_family`.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.ml import ModelMetric, ModelStatusEvent, ModelVersion
from app.services.model_registry import _REGISTRY_LOCK, ALLOWED_TRANSITIONS, RegistryError
from ml.seaice.artifact_store import resolve_model_path_or_none, seaice_root
from ml.seaice.constants import (
    ARCHITECTURE,
    ARCHITECTURE_VERSION,
    HORIZONS,
    IN_CHANNELS,
    MODEL_FAMILY,
    WINDOW,
)
from ml.seaice.model_loader import SeaIceArtifactError, SeaIceBundle, load_seaice_bundle, read_metadata
from ml.versioning.version_manager import (
    BASE_VERSION,
    FAMILY_SEA_ICE,
    is_numbered,
    next_version_for_family,
    qualify,
    short_version,
    version_number,
)

log = get_logger(__name__)

_bundle_cache: dict[tuple[str, str], SeaIceBundle] = {}
_bundle_lock = threading.Lock()

BASE_KEY = qualify(FAMILY_SEA_ICE, BASE_VERSION)  # storage key of the immutable base


class SeaIceRegistry:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    # ------------------------------------------------------------------ paths
    @property
    def family_root(self) -> Path:
        return seaice_root(Path(self.settings.models_dir))

    def directory_for(self, version: str) -> Path:
        """``sea_ice/v3`` -> ``<models_dir>/sea_ice/v3``."""
        return self.family_root / short_version(version)

    # ---------------------------------------------------------------- queries
    def get(self, version: str) -> ModelVersion | None:
        return self.session.execute(
            select(ModelVersion).where(
                ModelVersion.version == version, ModelVersion.model_family == MODEL_FAMILY
            )
        ).scalar_one_or_none()

    def get_champion(self) -> ModelVersion | None:
        """The deployed sea-ice version — the current working model, whatever its number."""
        return self.session.execute(
            select(ModelVersion).where(
                ModelVersion.model_family == MODEL_FAMILY, ModelVersion.status == "deployed"
            )
        ).scalar_one_or_none()

    def list_versions(self) -> list[ModelVersion]:
        return list(
            self.session.execute(
                select(ModelVersion)
                .where(ModelVersion.model_family == MODEL_FAMILY)
                .order_by(ModelVersion.version_number)
            ).scalars()
        )

    def latest_version(self) -> ModelVersion | None:
        """Highest-numbered version. Deliberately separate from :meth:`get_champion`."""
        versions = [m for m in self.list_versions() if is_numbered(short_version(m.version))]
        return max(versions, key=lambda m: m.version_number, default=None)

    def allocate_next_version(self) -> str:
        """Next free identifier in THIS family, from registry + filesystem state."""
        root = self.family_root
        return qualify(
            FAMILY_SEA_ICE,
            next_version_for_family(
                FAMILY_SEA_ICE, [m.version for m in self.list_versions()], root if root.exists() else None
            ),
        )

    def load_bundle(self, mv: ModelVersion) -> SeaIceBundle:
        key = (mv.version, json.dumps(mv.artifact_sha256, sort_keys=True))
        with _bundle_lock:
            if key not in _bundle_cache:
                bundle = load_seaice_bundle(Path(mv.model_path).parent, version=mv.version)
                for name, digest in (mv.artifact_sha256 or {}).items():
                    if bundle.checksums.get(name) not in (None, digest):
                        raise SeaIceArtifactError(f"{mv.version}: {name} differs from the registered checksum")
                _bundle_cache[key] = bundle
            return _bundle_cache[key]

    # --------------------------------------------------------------- mutation
    def _lock(self) -> None:
        from sqlalchemy import text

        self.session.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _REGISTRY_LOCK})

    def transition(
        self, mv: ModelVersion, to_status: str, reason: str, actor: str = "system", retraining_run_id: int | None = None
    ) -> None:
        if to_status not in ALLOWED_TRANSITIONS.get(mv.status, set()):
            raise RegistryError(f"{mv.version}: transition {mv.status} -> {to_status} not allowed")
        self.session.add(
            ModelStatusEvent(
                model_version=mv.version, from_status=mv.status, to_status=to_status, reason=reason,
                actor=actor, retraining_run_id=retraining_run_id,
            )
        )
        log.info("seaice_model_status_changed", version=mv.version, from_status=mv.status, to_status=to_status, reason=reason)
        mv.status = to_status
        mv.status_reason = reason
        if to_status == "deployed":
            mv.deployed_at = datetime.now(UTC)
        self.session.flush()

    def promote(self, candidate: ModelVersion, reason: str, actor: str = "system") -> None:
        """Make ``candidate`` the sea-ice champion. The previous one is archived, never deleted."""
        self._lock()
        current = self.get_champion()
        if current is not None and current.version != candidate.version:
            self.transition(current, "archived", f"superseded by {candidate.version}", actor)
        self.transition(candidate, "deployed", reason, actor)

    def record_metrics(self, version: str, protocol: str, results: dict[str, Any], source: str) -> None:
        """Store per-horizon RMSE/MAE. ``mae_km`` carries MAE and ``p90_km`` carries RMSE.

        The shared ``ml.model_metrics`` table is named for the trajectory model's
        kilometre errors; sea-ice values are dimensionless concentration
        fractions, so ``protocol`` and ``source`` record what the numbers mean.
        """
        for horizon, stats in (results.get("by_horizon") or {}).items():
            self.session.add(
                ModelMetric(
                    model_version=version, protocol=protocol, horizon_days=int(horizon),
                    n=int(stats.get("n_cells") or 0), mae_km=stats.get("mae"), p90_km=stats.get("rmse"),
                    source=source,
                )
            )
        self.session.flush()

    # -------------------------------------------------------------- lifecycle
    def _row_from_metadata(
        self, version: str, directory: Path, meta: dict[str, Any], checksums: dict[str, str], status: str
    ) -> ModelVersion:
        model_path = resolve_model_path_or_none(directory)
        metrics = (meta.get("metrics") or {}).get("by_horizon") or {}
        cutoff = meta.get("training_data_cutoff")
        mv = ModelVersion(
            model_family=MODEL_FAMILY,
            model_type=MODEL_FAMILY,
            feature_schema_version=f"{ARCHITECTURE_VERSION}_entry{WINDOW}",
            version=version,
            version_number=version_number(short_version(version)),
            parent_version=meta.get("parent_version"),
            architecture=ARCHITECTURE,
            architecture_version=meta.get("architecture_version", ARCHITECTURE_VERSION),
            input_sequence_length=WINDOW,
            input_semantics="chronological_observation_entries",
            forecast_horizon_days=max(HORIZONS),
            feature_names=list(meta.get("channel_layout") or []),
            adapter_strategy=None,
            model_path=str(model_path) if model_path else str(directory),
            scaler_path="",  # the .pt carries no separate scaler; preprocessing is fixed and in metadata
            metadata_path=str(directory / "metadata.json"),
            artifact_sha256=checksums,
            artifact_origin=meta.get("artifact_origin", "trained"),
            training_data_cutoff=datetime.fromisoformat(str(cutoff)).replace(tzinfo=UTC) if cutoff else None,
            training_config=meta.get("training"),
            test_metrics=meta.get("metrics"),
            day1_error=(metrics.get("1") or {}).get("rmse"),
            day3_error=(metrics.get("3") or {}).get("rmse"),
            day7_error=(metrics.get("7") or {}).get("rmse"),
            status=status,
        )
        if mv.input_sequence_length != WINDOW or len(mv.feature_names) not in (0, IN_CHANNELS):
            raise RegistryError(f"{version}: input contract does not match the sea-ice model")
        self.session.add(mv)
        self.session.flush()
        return mv

    def register_base(self) -> ModelVersion:
        """Register the supplied U-Net Residual v4 artifact. Read-only: never written to."""
        self._lock()
        existing = self.get(BASE_KEY)
        if existing is not None:
            return existing
        directory = self.family_root / BASE_VERSION
        if resolve_model_path_or_none(directory) is None:
            raise SeaIceArtifactError(
                f"base sea-ice artifact is missing from {directory}: place unet_residual_v4_best.pt there"
            )
        meta = read_metadata(directory)
        bundle = load_seaice_bundle(directory, version=BASE_KEY)  # validates the tensor contract
        mv = self._row_from_metadata(BASE_KEY, directory, meta, bundle.checksums, status="validated")
        mv.status_reason = "supplied base artifact (immutable; served until a retrained version is promoted)"
        self.session.add(
            ModelStatusEvent(
                model_version=BASE_KEY, from_status=None, to_status="validated",
                reason="registered supplied sea-ice base model", actor="system",
            )
        )
        metrics = meta.get("metrics") or {}
        if metrics.get("by_horizon"):
            self.record_metrics(BASE_KEY, metrics.get("protocol", "historical_test"), metrics, "notebook_record")
        log.info("seaice_base_registered", version=BASE_KEY, sha256=bundle.checksums)
        return mv

    def ensure_champion(self) -> ModelVersion:
        """Make sure the sea-ice family has a champion, without inventing a version.

        The supplied U-Net Residual v4 **is** the starting production model: it
        consumes exactly the same 7-entry input in production as it did in
        training, so there is no adapter step and nothing to derive. No ``v1`` is
        manufactured here. Numbered versions appear only when a retraining run
        actually produces and promotes one:

            base -> (new official data -> retraining) -> v1 -> v2 -> ...

        If a champion already exists it is returned unchanged, whatever its
        number. Deploying the base never writes to its artifact.
        """
        self._lock()
        champion = self.get_champion()
        if champion is not None:
            return champion
        base = self.get(BASE_KEY) or self.register_base()
        if base.status == "validated":
            self.promote(base, "initial sea-ice champion: supplied base model, no retraining yet")
        log.info("seaice_champion_ready", version=base.version, status=base.status)
        return base
