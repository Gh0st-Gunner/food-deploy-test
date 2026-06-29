import sys
import os

# Add back-end path to sys.path so imports remain unbroken
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from celery import Celery

celery = Celery(
    "vnfood",
    include=[
        "workers.classification_worker",
        "workers.nutrition_worker",
        "workers.detection_worker",
        "workers.portion_worker",
        "workers.aggregator_worker",
    ],
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    result_expires=3600,
)

celery.conf.task_routes = {
    "workers.classification_worker.*": {"queue": "classification"},
    "workers.nutrition_worker.*": {"queue": "nutrition"},
    "workers.detection_worker.*": {"queue": "detection"},
    "workers.portion_worker.*": {"queue": "detection"},
    "workers.aggregator_worker.*": {"queue": "default"},
}


def configure_celery():
    """Configure broker and backend from settings. Called lazily."""
    from core.settings import get_settings
    settings = get_settings()
    celery.conf.broker_url = settings.celery_broker_url
    celery.conf.result_backend = settings.celery_result_backend


def preload_models_at_startup(**kwargs):
    """Preload models at worker startup based on VNFOOD_WORKER_PRELOAD_MODELS env var."""
    import os
    preload = os.environ.get("VNFOOD_WORKER_PRELOAD_MODELS", "")
    if not preload:
        return

    from core.model_registry import ModelRegistry
    registry = ModelRegistry()
    groups = [g.strip() for g in preload.split(",") if g.strip()]
    if groups:
        registry.preload(*groups)


from celery.signals import worker_process_init

worker_process_init.connect(preload_models_at_startup)

configure_celery()
