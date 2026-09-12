"""Background jobs. Each opens its own transaction scope."""

from __future__ import annotations

from celery import chain

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.services.evaluation_service import EvaluationService
from app.services.forecast_service import ForecastService
from app.services.ingestion_service import IngestionService
from app.services.model_registry import ModelRegistry, RegistryError
from app.services.retraining_service import RetrainingService
from app.workers.celery_app import celery_app
from ml.models.model_loader import ModelArtifactError

log = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=1800)
def usnic_ingestion_job(self, trigger: str = "scheduled") -> dict:  # type: ignore[no-untyped-def]
    with session_scope() as session:
        outcome = IngestionService(session, get_settings()).run(trigger=trigger)
    if outcome.status == "failed":
        log.warning("ingestion_job_failed", error=outcome.error, retry=self.request.retries)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=RuntimeError(outcome.error))
        return {"status": outcome.status, "error": outcome.error}
    # Evaluate old forecasts against any new/corrected official observations,
    # align environmental state to the new observations (no-op when disabled),
    # then (idempotently) forecast from the latest official observations.
    chain(
        forecast_evaluation_job.si(outcome.changed_observation_ids),
        environment_alignment_job.si(outcome.changed_observation_ids or None),
        forecast_generation_job.si("post_ingestion", outcome.run_id),
    ).apply_async()
    return {"status": outcome.status, "run_id": outcome.run_id, "new": len(outcome.new_observation_ids),
            "updated": len(outcome.updated_observation_ids), "duplicates": outcome.duplicates}


@celery_app.task
def forecast_evaluation_job(observation_ids: list[int] | None = None) -> dict:
    with session_scope() as session:
        outcome = EvaluationService(session).evaluate(observation_ids)
    return {"evaluated": outcome.evaluated, "by_horizon": outcome.by_horizon}


@celery_app.task(soft_time_limit=1800)
def environment_alignment_job(observation_ids: list[int] | None = None) -> dict:
    """Environmental state at official observations (only what was available at run time)."""
    from app.services.environment_service import EnvironmentService

    with session_scope() as session:
        return EnvironmentService(session, get_settings()).align_observations(observation_ids)


@celery_app.task(soft_time_limit=1800)
def environment_overlay_job() -> dict:
    """Coarse Southern Ocean wind / current / sea-ice grids for the map overlay."""
    from app.services.environment_service import EnvironmentService

    with session_scope() as session:
        return EnvironmentService(session, get_settings()).refresh_overlay()


@celery_app.task
def forecast_generation_job(trigger: str = "scheduled", ingestion_run_id: int | None = None) -> dict:
    with session_scope() as session:
        outcome = ForecastService(session, get_settings()).run(trigger=trigger, ingestion_run_id=ingestion_run_id)
    return {"status": outcome.status, "model_version": outcome.model_version, "created": outcome.created_sets,
            "already": outcome.already_forecast, "skipped": outcome.skipped, "error": outcome.error}


@celery_app.task
def retraining_job(force: bool = False, trigger: str = "scheduled") -> dict:
    with session_scope() as session:
        run = RetrainingService(session, get_settings()).run(trigger=trigger, force=force)
        result = {"status": run.status, "candidate": run.candidate_version, "champion": run.champion_version,
                  "reason": run.failure_reason}
    if result["status"] == "promoted":
        forecast_generation_job.delay("post_promotion")
    return result


@celery_app.task
def model_validation_job() -> dict:
    """Verify the champion's artifacts; bootstrap base -> v1 on a fresh deployment."""
    settings = get_settings()
    with session_scope() as session:
        registry = ModelRegistry(session, settings)
        champion = registry.get_deployed()
        if champion is None:
            try:
                champion = registry.ensure_v1_bootstrap()
            except (ModelArtifactError, FileNotFoundError) as exc:
                log.error("model_bootstrap_unavailable", error=str(exc))
                return {"ok": False, "error": str(exc)}
            forecast_generation_job.delay("post_bootstrap")
        try:
            registry.load_bundle(champion)
        except ModelArtifactError as exc:
            log.error("champion_artifact_invalid", version=champion.version, error=str(exc))
            return {"ok": False, "version": champion.version, "error": str(exc)}
        return {"ok": True, "version": champion.version}


# --------------------------------------------------------------------- sea ice
# The sea-ice model is an independent core model with its own lineage. Polling
# CHECKS for new official data on a cadence; it never creates a model version by
# itself. Retraining is a separate, policy-gated decision.
@celery_app.task(bind=True, max_retries=3, default_retry_delay=1800)
def seaice_ingestion_job(self, trigger: str = "scheduled") -> dict:  # type: ignore[no-untyped-def]
    from app.services.seaice_ingestion_service import SeaIceIngestionService

    settings = get_settings()
    if not settings.seaice_enabled:
        return {"status": "skipped", "reason": "sea-ice pipeline disabled"}
    with session_scope() as session:
        outcome = SeaIceIngestionService(session, settings).run(trigger=trigger)
    if outcome.status == "failed":
        log.warning("seaice_ingestion_job_failed", error=outcome.error, retry=self.request.retries)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=RuntimeError(outcome.error))
        return outcome.as_dict()
    if outcome.new:
        # Score outstanding forecasts against the newly arrived truth, then
        # forecast from the latest window. Neither step creates a version.
        chain(seaice_evaluation_job.si("post_ingestion"), seaice_forecast_job.si("post_ingestion")).apply_async()
    return {"status": outcome.status, "run_id": outcome.run_id, "new": outcome.new, "duplicates": outcome.duplicates}


@celery_app.task
def seaice_evaluation_job(trigger: str = "scheduled") -> dict:
    from app.services.seaice_evaluation_service import SeaIceEvaluationService

    with session_scope() as session:
        return SeaIceEvaluationService(session, get_settings()).run(trigger=trigger).as_dict()


@celery_app.task
def seaice_forecast_job(trigger: str = "scheduled") -> dict:
    from app.services.seaice_forecast_service import SeaIceForecastService

    with session_scope() as session:
        return SeaIceForecastService(session, get_settings()).run(trigger=trigger).as_dict()


@celery_app.task
def seaice_retraining_job(force: bool = False, trigger: str = "scheduled") -> dict:
    """Policy-gated. A new sea-ice version appears only when this decides one is warranted."""
    from app.services.seaice_retraining_service import SeaIceRetrainingService

    settings = get_settings()
    if not settings.seaice_enabled:
        return {"status": "skipped", "reason": "sea-ice pipeline disabled"}
    with session_scope() as session:
        outcome = SeaIceRetrainingService(session, settings).run(trigger=trigger, force=force)
    result = outcome.as_dict()
    if outcome.decision == "promoted":
        seaice_forecast_job.delay("post_promotion")
    return result


@celery_app.task
def seaice_model_validation_job() -> dict:
    """Verify the sea-ice champion's artifact; bootstrap the first version if there is none."""
    from app.services.seaice_registry import SeaIceRegistry
    from ml.seaice.model_loader import SeaIceArtifactError

    settings = get_settings()
    if not settings.seaice_enabled:
        return {"ok": True, "skipped": "sea-ice pipeline disabled"}
    with session_scope() as session:
        registry = SeaIceRegistry(session, settings)
        champion = registry.get_champion()
        if champion is None:
            try:
                champion = registry.ensure_champion()
            except (SeaIceArtifactError, FileNotFoundError, RegistryError) as exc:
                log.error("seaice_bootstrap_unavailable", error=str(exc))
                return {"ok": False, "error": str(exc)}
        try:
            registry.load_bundle(champion)
        except SeaIceArtifactError as exc:
            log.error("seaice_champion_artifact_invalid", version=champion.version, error=str(exc))
            return {"ok": False, "version": champion.version, "error": str(exc)}
        return {"ok": True, "version": champion.version}
