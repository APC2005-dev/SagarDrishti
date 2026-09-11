"""Celery application and beat schedule.

Queues: ``ingestion`` (network I/O, fast) and ``ml`` (inference/training, CPU).
Long ML jobs never run inside the API process.
"""

from __future__ import annotations

from datetime import timedelta

from celery import Celery
from celery.signals import setup_logging

from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
redis_url = settings.redis_url.get_secret_value()

celery_app = Celery("sagar_drishti", broker=redis_url, backend=redis_url, include=["app.workers.tasks"])
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    result_expires=7 * 24 * 3600,
    task_routes={
        "app.workers.tasks.usnic_ingestion_job": {"queue": "ingestion"},
        "app.workers.tasks.*": {"queue": "ml"},
    },
    task_annotations={"app.workers.tasks.retraining_job": {"time_limit": 6 * 3600, "soft_time_limit": 5.5 * 3600}},
    beat_schedule={
        "usnic-ingestion": {"task": "app.workers.tasks.usnic_ingestion_job", "schedule": timedelta(hours=settings.ingestion_interval_hours)},
        "retraining-check": {"task": "app.workers.tasks.retraining_job", "schedule": timedelta(hours=settings.retrain_check_interval_hours)},
        "model-validation": {"task": "app.workers.tasks.model_validation_job", "schedule": timedelta(hours=24)},
        "environment-overlay": {"task": "app.workers.tasks.environment_overlay_job", "schedule": timedelta(hours=24)},
    },
)


@setup_logging.connect
def _configure(**_: object) -> None:
    configure_logging(settings.log_level, settings.log_json)
