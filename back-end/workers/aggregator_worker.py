from datetime import datetime

from workers.celery_app import celery
from core.database import update_job, get_job


@celery.task(queue="default")
def aggregate_results(job_id: str, *args, **kwargs):
    """Finalize a job after all sub-tasks complete."""
    job = get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}

    update_job(
        job_id,
        status="completed",
        completed_at=datetime.utcnow(),
        progress=_merge_progress(job, {"aggregation": "completed"}),
    )

    return {"job_id": job_id, "status": "completed"}


def _merge_progress(job, updates: dict) -> dict:
    progress = job.progress if job.progress else {}
    progress.update(updates)
    return progress