import io
import numpy as np
from datetime import datetime

from workers.celery_app import celery
from core.model_registry import ModelRegistry
from core.database import update_job, get_job
from core.storage import download_bytes, upload_result_image
from depth.portion_estimator import estimate_portion
from PIL import Image

registry = ModelRegistry()


@celery.task(queue="detection", time_limit=180, soft_time_limit=150)
def estimate_portion_task(job_id: str, class_name: str, image_s3_key: str,
                          ingredient_data: dict = None, params: dict = None):
    """Estimate portion size using area ratio + depth visualization."""
    params = params or {}
    if not class_name:
        job = get_job(job_id)
        if job:
            class_name = job.class_name

    progress = _get_progress(job_id)
    update_job(job_id, progress={**progress, "portion": "running"})

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
        update_job(job_id, progress={**progress, "portion": "failed"})
        return {"error": str(e)}

    # Try to load combined mask from S3 to avoid redundant SAM 2 runs
    combined_mask = None
    try:
        mask_key = f"results/{job_id}/combined_mask.png"
        mask_bytes = download_bytes(mask_key)
        mask_img = Image.open(io.BytesIO(mask_bytes))
        if mask_img.size != image.size:
            mask_img = mask_img.resize(image.size, Image.NEAREST)
        combined_mask = np.array(mask_img) > 128
        print(f"Portion: Pre-segmented combined mask loaded successfully from S3 for job {job_id}.")
    except Exception as e:
        print(f"Portion: Pre-segmented combined mask not found or failed to load: {e}. Falling back to SAM 2 segmentation.")

    depth_pipeline = registry.get_depth_model()
    sam_model, sam_processor = registry.get_sam2()
    reference_height_cm = params.get("reference_height_cm")

    result = estimate_portion(
        image=image,
        class_name=class_name,
        depth_pipeline=depth_pipeline,
        ingredient_masks=[{"mask": combined_mask}] if combined_mask is not None else None,
        reference_height_cm=reference_height_cm,
        sam_model=sam_model,
        sam_processor=sam_processor,
    )

    # Store depth map as PNG in S3 (can't be JSON-serialized)
    depth_map_s3_key = None
    depth_map = result.pop("depth_map", None)
    if depth_map is not None:
        depth_colored = (depth_map * 255).astype(np.uint8)
        depth_image = Image.fromarray(np.stack([depth_colored] * 3, axis=-1))
        buf = io.BytesIO()
        depth_image.save(buf, format="PNG")
        depth_map_s3_key = upload_result_image(job_id, "depth_map", buf.getvalue())
        result["depth_map_s3_key"] = depth_map_s3_key

    # Remove non-serializable ingredient_volumes if present
    result.pop("ingredient_volumes", None)

    update_job(
        job_id,
        portion=result,
        progress={**progress, "portion": "completed"},
    )

    return result


def _get_progress(job_id: str) -> dict:
    job = get_job(job_id)
    return job.progress if job and job.progress else {}