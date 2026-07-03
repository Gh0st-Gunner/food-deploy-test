import io
import json
import base64
from datetime import datetime

from workers.celery_app import celery
from core.model_registry import ModelRegistry
from core.database import update_job
from core.storage import download_bytes
from classification.predict import predict, predict_onnx
from nutrition.food_mapping import USDA_SEARCH_TERMS, INGREDIENT_PROMPTS, TYPICAL_PORTION_GRAMS
from PIL import Image

registry = ModelRegistry()


@celery.task(bind=True, queue="classification")
def classify_food(self, job_id: str, image_s3_key: str, model_names: list = None):
    """Classify a food image and return the consensus class name."""
    update_job(job_id, status="running", started_at=datetime.utcnow(),
               progress={"classification": "running"})

    try:
        image_bytes = download_bytes(image_s3_key)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        update_job(job_id, status="failed", error=f"Image load failed: {e}",
                   progress={"classification": "failed"})
        raise

    available = registry.list_classification_models()
    models_to_run = model_names if model_names else list(available.keys())

    # Run local PyTorch and ONNX models only
    predictions_data = {}
    for model_name in models_to_run:
        try:
            model_path = available[model_name]
            if model_name.endswith(".onnx") or model_path.endswith(".onnx"):
                session, input_name, class_names = registry.get_onnx_model(model_name)
                preds = predict_onnx(image, session, input_name, class_names, top_k=3)
                predictions_data[model_name] = {
                    "predictions": preds,
                    "accuracy": "N/A",
                    "detected_name": "ONNX",
                }
            else:
                model, class_names, device, detected_name, accuracy = registry.get_classifier(model_name)
                preds = predict(image, model, class_names, device, top_k=3)
                predictions_data[model_name] = {
                    "predictions": preds,
                    "accuracy": accuracy if isinstance(accuracy, (int, float)) else str(accuracy),
                    "detected_name": detected_name,
                }
        except Exception as e:
            predictions_data[model_name] = {"error": str(e)}

    if not predictions_data:
        update_job(job_id, status="failed", error="All models failed to load",
                   progress={"classification": "failed"})
        return {"class_name": None, "error": "All models failed"}

    from collections import Counter
    all_top = [d["predictions"][0]["class"] for d in predictions_data.values() if "predictions" in d]
    if not all_top:
        update_job(job_id, status="failed", error="No predictions produced",
                   progress={"classification": "failed"})
        return {"class_name": None, "error": "No predictions"}

    consensus = Counter(all_top).most_common(1)[0]
    class_name = consensus[0]
    top_confidence = max(
        d["predictions"][0]["probability"]
        for d in predictions_data.values()
        if "predictions" in d
    )

    update_job(
        job_id,
        class_name=class_name,
        confidence=top_confidence,
        predictions=predictions_data,
        progress={"classification": "completed"},
    )

    return {
        "class_name": class_name,
        "confidence": top_confidence,
        "predictions": predictions_data,
    }