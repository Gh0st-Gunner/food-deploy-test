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
    serialized = []
    for ing in ingredients:
        label = ing.get("label", "").rstrip(".")
        entry = {
            "label": label,
            "confidence": ing.get("confidence", 0),
            "bbox": ing.get("bbox", []),
            "mask_pixel_count": ing.get("mask_pixel_count", 0),
            "nutrition": {},  # Will be populated by CPU aggregator task or on-demand cache
        }

        # Masks are numpy arrays — we compute area stats but don't store the raw mask in JSON
        mask = ing.get("mask")
        if mask is not None:
            entry["mask_shape"] = list(mask.shape)
            entry["mask_area_ratio"] = float(np.sum(mask)) / mask.size if mask.size > 0 else 0
        serialized.append(entry)
    return serialized


@celery.task(bind=True, queue="detection", time_limit=180, soft_time_limit=150)
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
        
        # Downscale image to max 1024px to optimize memory and processing speed
        MAX_SIZE = 1024
        w, h = image.size
        if w > MAX_SIZE or h > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            try:
                resample_filter = Image.Resampling.BILINEAR
            except AttributeError:
                resample_filter = Image.BILINEAR
            image = image.resize((new_w, new_h), resample_filter)
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

    # 1. Compute and upload combined mask (union of all ingredient masks) to S3
    combined_mask_s3_key = None
    if ingredient_results:
        first_mask = ingredient_results[0].get("mask")
        if first_mask is not None:
            combined_mask = np.zeros(first_mask.shape, dtype=bool)
            for ing in ingredient_results:
                m = ing.get("mask")
                if m is not None:
                    combined_mask = combined_mask | m
            
            if np.sum(combined_mask) > 0:
                mask_uint8 = (combined_mask.astype(np.uint8)) * 255
                mask_image = Image.fromarray(mask_uint8)
                buf = io.BytesIO()
                mask_image.save(buf, format="PNG")
                combined_mask_s3_key = upload_result_image(job_id, "combined_mask", buf.getvalue())

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
        "combined_mask_s3_key": combined_mask_s3_key,
        "ingredient_count": len(ingredient_results),
    }


def _get_progress(job_id: str) -> dict:
    job = get_job(job_id)
    return job.progress if job and job.progress else {}