import io
import numpy as np
from datetime import datetime

from workers.celery_app import celery
from core.model_registry import ModelRegistry
from core.database import update_job, get_job
from core.storage import download_bytes, upload_result_image
from segmentation.ingredient_detector import detect_ingredients
from segmentation.visualize import overlay_ingredients
from PIL import Image

registry = ModelRegistry()


def _serialize_ingredients(ingredients):
    """Serialize ingredient results for JSON storage (masks -> S3 keys)."""
    from nutrition.nutrition_provider import lookup_ingredient_nutrition
    from core.settings import get_settings
    settings = get_settings()
    usda_key = settings.usda_api_key
    fatsecret_id = settings.fatsecret_client_id
    fatsecret_secret = settings.fatsecret_client_secret

    serialized = []
    for ing in ingredients:
        label = ing.get("label", "").rstrip(".")
        entry = {
            "label": label,
            "confidence": ing.get("confidence", 0),
            "bbox": ing.get("bbox", []),
            "mask_pixel_count": ing.get("mask_pixel_count", 0),
        }
        
        # Query nutrition for this ingredient
        try:
            nut_res = lookup_ingredient_nutrition(
                label,
                usda_key=usda_key,
                fatsecret_id=fatsecret_id,
                fatsecret_secret=fatsecret_secret
            )
            if nut_res and "nutrients" in nut_res:
                entry["nutrition"] = nut_res["nutrients"]
        except Exception as e:
            print(f"Failed to lookup ingredient nutrition for {label}: {e}")

        # Masks are numpy arrays — we compute area stats but don't store the raw mask in JSON
        mask = ing.get("mask")
        if mask is not None:
            entry["mask_shape"] = list(mask.shape)
            entry["mask_area_ratio"] = float(np.sum(mask)) / mask.size if mask.size > 0 else 0
        serialized.append(entry)
    return serialized


@celery.task(bind=True, queue="detection")
def detect_ingredients_task(self, job_id: str, class_name: str, image_s3_key: str, params: dict = None):
    """Detect ingredients using Grounding DINO + SAM 2."""
    params = params or {}
    if not class_name:
        job = get_job(job_id)
        if job:
            class_name = job.class_name

    update_job(job_id, progress={**_get_progress(job_id), "detection": "running"})

    try:
        image_bytes = download_bytes(image_s3_key)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        update_job(job_id, status="failed", error=f"Image load failed: {e}",
                   progress={**_get_progress(job_id), "detection": "failed"})
        return {"error": str(e)}

    box_threshold = params.get("box_threshold", 0.3)

    grounding_model, grounding_processor = registry.get_grounding_dino()
    sam_model, sam_processor = registry.get_sam2()

    ingredient_results = detect_ingredients(
        image=image,
        class_name=class_name,
        grounding_model=grounding_model,
        grounding_processor=grounding_processor,
        sam_model=sam_model,
        sam_processor=sam_processor,
        box_threshold=box_threshold,
    )

    serialized = _serialize_ingredients(ingredient_results)

    # Generate and upload overlay image
    overlay_s3_key = None
    if ingredient_results:
        overlay = overlay_ingredients(image, ingredient_results)
        overlay_bytes = io.BytesIO()
        overlay.save(overlay_bytes, format="PNG")
        overlay_s3_key = upload_result_image(job_id, "overlay", overlay_bytes.getvalue())

    update_job(
        job_id,
        ingredients=serialized,
        overlay_s3_key=overlay_s3_key,
        progress={**_get_progress(job_id), "detection": "completed"},
    )

    return {
        "ingredients": serialized,
        "overlay_s3_key": overlay_s3_key,
        "ingredient_count": len(ingredient_results),
    }


def _get_progress(job_id: str) -> dict:
    job = get_job(job_id)
    return job.progress if job and job.progress else {}