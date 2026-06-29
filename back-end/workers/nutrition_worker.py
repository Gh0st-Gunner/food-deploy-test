from datetime import datetime

from workers.celery_app import celery
from core.settings import get_settings
from core.database import update_job
from core.cache import (
    get_usda_cached, set_usda_cached,
    get_class_mapping_cached, set_class_mapping_cached,
)
from nutrition.usda_client import USDAClient
from nutrition.fatsecret_client import FatSecretClient
from nutrition.nutrition_provider import lookup_nutrition, lookup_ingredient_nutrition
from nutrition.food_mapping import USDA_SEARCH_TERMS, INGREDIENT_PROMPTS

settings = get_settings()


@celery.task(queue="nutrition")
def lookup_nutrition_task(job_id: str, class_name: str = None):
    """Look up nutrition data for the classified food."""
    if not class_name:
        from core.database import get_job
        job = get_job(job_id)
        if job:
            class_name = job.class_name

    update_job(job_id, progress={**_get_progress(job_id), "nutrition": "running"})

    usda_key = settings.usda_api_key
    fatsecret_id = settings.fatsecret_client_id
    fatsecret_secret = settings.fatsecret_client_secret

    result = lookup_nutrition(
        class_name,
        usda_key=usda_key,
        fatsecret_id=fatsecret_id,
        fatsecret_secret=fatsecret_secret,
    )

    update_job(
        job_id,
        nutrition=result.get("nutrients", {}),
        nutrition_source=result.get("source", ""),
        progress={**_get_progress(job_id), "nutrition": "completed"},
    )

    return {"nutrition": "done", "source": result.get("source", "")}


@celery.task(queue="nutrition")
def lookup_ingredient_nutrition_task(job_id: str, ingredients: list):
    """Look up per-ingredient nutrition data."""
    usda_key = settings.usda_api_key
    fatsecret_id = settings.fatsecret_client_id
    fatsecret_secret = settings.fatsecret_client_secret

    from core.cache import get_ingredient_nutrition_cached

    results = []
    for ing in ingredients:
        label = ing.get("label", "").rstrip(".")
        cached = get_ingredient_nutrition_cached(label)
        if cached:
            results.append({"label": label, **cached})
            continue

        result = lookup_ingredient_nutrition(
            label,
            usda_key=usda_key,
            fatsecret_id=fatsecret_id,
            fatsecret_secret=fatsecret_secret,
        )
        if result:
            results.append({"label": label, **result})
        else:
            results.append({"label": label, "nutrients": {}, "source": None})

    progress = _get_progress(job_id)
    update_job(job_id, progress={**progress, "ingredient_nutrition": "completed"})

    return {"ingredient_nutrition": results}


def _get_progress(job_id: str) -> dict:
    """Get current progress dict for a job."""
    from core.database import get_job
    job = get_job(job_id)
    return job.progress if job and job.progress else {}