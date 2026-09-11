"""Background jobs. Each opens its own transaction scope."""

from __future__ import annotations

from celery import chain

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import session_scope
from app.services.evaluation_service import EvaluationService
from app.services.forecast_service import ForecastService
from app.services.ingestion_service import IngestionService
from app.services.model_registry import ModelRegistry
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
    # then (idempotently) forecast from the latest official observations.
    chain(
        forecast_evaluation_job.si(outcome.changed_observation_ids),
        forecast_generation_job.si("post_ingestion", outcome.run_id),
    ).apply_async()
    return {"status": outcome.status, "run_id": outcome.run_id, "new": len(outcome.new_observation_ids),
            "updated": len(outcome.updated_observation_ids), "duplicates": outcome.duplicates}


@celery_app.task
def forecast_evaluation_job(observation_ids: list[int] | None = None) -> dict:
    with session_scope() as session:
        outcome = EvaluationService(session).evaluate(observation_ids)
    return {"evaluated": outcome.evaluated, "by_horizon": outcome.by_horizon}


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
