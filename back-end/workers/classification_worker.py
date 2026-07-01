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

    # Check if we should use Gemini vision model
    is_gemini = False
    gemini_model_name = "gemini-1.5-flash"
    for m in models_to_run:
        if m.startswith("gemini:"):
            is_gemini = True
            gemini_model_name = m.split(":", 1)[1]
            break

    if is_gemini:
        try:
            import requests
            from core.settings import get_settings
            settings = get_settings()
            
            if not settings.gemini_api_key:
                raise ValueError("Google Gemini API Key (VNFOOD_GEMINI_API_KEY) is not set in environment or .env!")
                
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            prompt = (
                "Analyze this food image. Identify the primary food dish name in Vietnamese and English (e.g. 'Phở Bò (Beef Pho)'). "
                "Provide the estimated portion size description, total calories (kcal), and macros (protein, carbs, fat in grams). "
                "Respond ONLY in valid JSON format matching this schema: "
                "{\"class_name\": \"Dish Name\", \"confidence\": 0.95, \"calories\": 450, \"protein\": 25, \"carbs\": 55, \"fat\": 12}."
            )
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model_name}:generateContent?key={settings.gemini_api_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_b64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
            
            print(f"Calling Google Gemini API model '{gemini_model_name}'...")
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            
            res_json = r.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise ValueError(f"Gemini API returned no candidates. Full response: {res_json}")
                
            text_response = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            print(f"Gemini response: {text_response}")
            
            res_data = json.loads(text_response)
            class_name = res_data.get("class_name", "Unknown Food")
            confidence = float(res_data.get("confidence", 0.95))
            calories = int(res_data.get("calories", 350))
            protein = int(res_data.get("protein", 15))
            carbs = int(res_data.get("carbs", 40))
            fat = int(res_data.get("fat", 10))
            
            predictions_data = {
                f"gemini:{gemini_model_name}": {
                    "predictions": [{"class": class_name, "probability": confidence}],
                    "accuracy": "N/A",
                    "detected_name": f"Google Gemini ({gemini_model_name.upper()})"
                }
            }
            
            update_job(
                job_id,
                class_name=class_name,
                confidence=confidence,
                predictions=predictions_data,
                nutrition={
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat
                },
                nutrition_source=f"Google Gemini ({gemini_model_name})",
                progress={"classification": "completed"}
            )
            
            return {
                "class_name": class_name,
                "confidence": confidence,
                "predictions": predictions_data
            }
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            print("Falling back to local models...")
            models_to_run = [m for m in list(available.keys()) if not m.startswith("gemini:")]

    # Check if we should use Ollama vision model
    is_ollama = False
    ollama_model_name = "llava"
    for m in models_to_run:
        if m == "ollama_vision" or m.startswith("ollama:"):
            is_ollama = True
            if m.startswith("ollama:"):
                ollama_model_name = m.split(":", 1)[1]
            break

    if is_ollama:
        try:
            import requests
            from core.settings import get_settings
            settings = get_settings()
            
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            prompt = (
                "Analyze this food image. Identify the primary food dish name in Vietnamese and English (e.g. 'Phở Bò (Beef Pho)'). "
                "Provide the estimated portion size description, total calories (kcal), and macros (protein, carbs, fat in grams). "
                "Respond ONLY in valid JSON format matching this schema: "
                "{\"class_name\": \"Dish Name\", \"confidence\": 0.95, \"calories\": 450, \"protein\": 25, \"carbs\": 55, \"fat\": 12} "
                "without any markdown formatting (do not wrap in ```json ... ```)."
            )
            
            payload = {
                "model": ollama_model_name,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "format": "json"
            }
            
            print(f"Calling Ollama local vision model '{ollama_model_name}' at {settings.ollama_host}...")
            r = requests.post(f"{settings.ollama_host}/api/generate", json=payload, timeout=30)
            r.raise_for_status()
            
            res_json = r.json()
            response_text = res_json.get("response", "").strip()
            print(f"Ollama response: {response_text}")
            
            res_data = json.loads(response_text)
            class_name = res_data.get("class_name", "Unknown Food")
            confidence = float(res_data.get("confidence", 0.90))
            calories = int(res_data.get("calories", 350))
            protein = int(res_data.get("protein", 15))
            carbs = int(res_data.get("carbs", 40))
            fat = int(res_data.get("fat", 10))
            
            predictions_data = {
                f"ollama:{ollama_model_name}": {
                    "predictions": [{"class": class_name, "probability": confidence}],
                    "accuracy": "N/A",
                    "detected_name": f"Ollama Vision ({ollama_model_name.upper()})"
                }
            }
            
            update_job(
                job_id,
                class_name=class_name,
                confidence=confidence,
                predictions=predictions_data,
                nutrition={
                    "calories": calories,
                    "protein": protein,
                    "carbs": carbs,
                    "fat": fat
                },
                nutrition_source=f"Ollama Vision ({ollama_model_name})",
                progress={"classification": "completed"}
            )
            
            return {
                "class_name": class_name,
                "confidence": confidence,
                "predictions": predictions_data
            }
        except Exception as e:
            print(f"Error calling Ollama vision: {e}")
            print("Falling back to local models...")
            models_to_run = [m for m in list(available.keys()) if not m.startswith("ollama:")]

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