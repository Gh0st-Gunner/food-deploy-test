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


@celery.task(queue="detection")
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
    except Exception as e:
        update_job(job_id, progress={**progress, "portion": "failed"})
        return {"error": str(e)}

    depth_pipeline = registry.get_depth_model()
    sam_model, sam_processor = registry.get_sam2()
    reference_height_cm = params.get("reference_height_cm")

    result = estimate_portion(
        image=image,
        class_name=class_name,
        depth_pipeline=depth_pipeline,
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