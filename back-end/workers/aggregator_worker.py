from datetime import datetime
from workers.celery_app import celery
from core.database import update_job, get_job
from nutrition.food_mapping import TYPICAL_PORTION_GRAMS

# Typical weight ratios for common dishes
TYPICAL_INGREDIENT_RATIOS = {
    "pho": {
        "rice noodle": 0.35,
        "broth": 0.50,
        "beef slice": 0.12,
        "bean sprout": 0.02,
        "basil": 0.005,
        "lime": 0.003,
        "hoisin sauce": 0.002
    },
    "bun-bo-hue": {
        "thick noodle": 0.40,
        "broth": 0.35,
        "beef shank": 0.10,
        "pork knuckle": 0.10,
        "lemongrass": 0.02,
        "chili": 0.01,
        "herb": 0.01,
        "lime": 0.01
    },
    "com-tam": {
        "broken rice": 0.50,
        "grilled pork chop": 0.35,
        "egg meatloaf": 0.10,
        "shredded pork": 0.03,
        "fish sauce": 0.01,
        "vegetables": 0.01
    },
    "banh-mi": {
        "bread": 0.45,
        "pork": 0.25,
        "pate": 0.15,
        "butter": 0.05,
        "herb": 0.05,
        "cucumber": 0.03,
        "pickled vegetable": 0.02
    },
    "banh-xeo": {
        "rice crepe": 0.45,
        "shrimp": 0.15,
        "pork belly": 0.15,
        "bean sprout": 0.15,
        "mushroom": 0.05,
        "herb": 0.05
    },
    "goi-cuon": {
        "rice paper": 0.15,
        "shrimp": 0.25,
        "pork": 0.25,
        "herb": 0.15,
        "vermicelli": 0.20
    },
    "hu-tieu": {
        "rice noodle": 0.35,
        "broth": 0.45,
        "pork": 0.10,
        "shrimp": 0.05,
        "liver": 0.03,
        "herb": 0.02
    }
}

def _get_nutrient_value(nutrition: dict, key: str, fallback: float = 0.0) -> float:
    nut = nutrition.get(key)
    if not nut:
        return fallback
    if isinstance(nut, dict):
        return float(nut.get("value", fallback))
    try:
        return float(nut)
    except (ValueError, TypeError):
        return fallback

def normalize_and_split_ingredients(class_name: str, raw_ingredients: list) -> list:
    """
    Normalizes labels and splits combined labels (e.g. 'broth hoisin sauce')
    into individual clean ingredient entries before weight distribution.
    """
    if not raw_ingredients:
        return []

    # Get the known ingredient list for the current dish
    typical_set = set()
    if class_name:
        clean_class = class_name.lower().replace("_", "-")
        ratio_template = TYPICAL_INGREDIENT_RATIOS.get(clean_class)
        if ratio_template:
            typical_set = set(ratio_template.keys())

    processed_ingredients = []

    for ing in raw_ingredients:
        label = ing.get("label", "").rstrip(".").strip().lower()
        if not label:
            continue

        # Find which typical ingredients are present in this label
        matched_ingredients = []
        for typical in typical_set:
            if typical in label:
                matched_ingredients.append(typical)

        # Sort matches by length descending so longer matching phrases win first
        matched_ingredients.sort(key=len, reverse=True)

        # Filter out substrings (e.g. if 'hoisin sauce' matches, 'sauce' is also in template, we don't want duplicate 'sauce')
        unique_matches = []
        for m in matched_ingredients:
            if not any(m in other for other in unique_matches):
                unique_matches.append(m)

        if len(unique_matches) > 1:
            num_matches = len(unique_matches)
            for matched in unique_matches:
                new_ing = ing.copy()
                new_ing["label"] = matched
                new_ing["confidence"] = ing.get("confidence", 0.0) / num_matches
                new_ing["mask_pixel_count"] = int(ing.get("mask_pixel_count", 0) / num_matches)
                new_ing["mask_area_ratio"] = ing.get("mask_area_ratio", 0.0) / num_matches
                processed_ingredients.append(new_ing)
        elif len(unique_matches) == 1:
            new_ing = ing.copy()
            new_ing["label"] = unique_matches[0]
            processed_ingredients.append(new_ing)
        else:
            processed_ingredients.append(ing)

    return processed_ingredients

def distribute_ingredient_weights(class_name: str, raw_ingredients: list, total_weight: float) -> list:
    """
    Consolidates raw ingredient detections and distributes the total portion weight
    among them using typical Vietnamese dish ratios to avoid duplicate items and inaccurate weights.
    """
    if not raw_ingredients:
        return []

    # Pre-process labels to normalize and split merged categories
    normalized_raw = normalize_and_split_ingredients(class_name, raw_ingredients)

    # 1. Group and deduplicate raw detections by matching labels
    grouped = {}
    for ing in normalized_raw:
        label = ing.get("label", "").rstrip(".").strip().lower()
        if not label:
            continue
            
        if label not in grouped:
            grouped[label] = {
                "label": label,
                "confidence": ing.get("confidence", 0.0),
                "bbox": ing.get("bbox", []),
                "mask_pixel_count": ing.get("mask_pixel_count", 0),
                "mask_area_ratio": ing.get("mask_area_ratio", 0.0),
                "nutrition": ing.get("nutrition", {})
            }
        else:
            grouped[label]["confidence"] = max(grouped[label]["confidence"], ing.get("confidence", 0.0))
            grouped[label]["mask_pixel_count"] += ing.get("mask_pixel_count", 0)
            grouped[label]["mask_area_ratio"] += ing.get("mask_area_ratio", 0.0)

    # 2. Get the ratio template for the current class name
    ratio_template = None
    if class_name:
        clean_class = class_name.lower().replace("_", "-")
        ratio_template = TYPICAL_INGREDIENT_RATIOS.get(clean_class)

    final_ingredients = []

    # Case A: Typical ratio template exists
    if ratio_template:
        all_labels = set(ratio_template.keys()) | set(grouped.keys())
        
        raw_ratios = {}
        for label in all_labels:
            ratio = ratio_template.get(label, 0.0)
            if ratio == 0.0:
                ratio = 0.05
            raw_ratios[label] = ratio

        total_ratio = sum(raw_ratios.values()) or 1.0
        normalized_ratios = {l: r / total_ratio for l, r in raw_ratios.items()}

        for label in all_labels:
            ratio = normalized_ratios[label]
            weight_g = max(1.0, round(ratio * total_weight, 1))
            
            detected = grouped.get(label, {})
            confidence = detected.get("confidence", 1.0)
            bbox = detected.get("bbox", [])
            mask_area_ratio = detected.get("mask_area_ratio", ratio)
            
            nutrition = detected.get("nutrition")
            if not nutrition:
                from nutrition.nutrition_provider import lookup_ingredient_nutrition
                try:
                    nut_res = lookup_ingredient_nutrition(label)
                    nutrition = nut_res.get("nutrients", {}) if nut_res else {}
                except Exception:
                    nutrition = {}

            calories = 0
            if nutrition:
                cal_100g = _get_nutrient_value(nutrition, "calories") or _get_nutrient_value(nutrition, "energy")
                calories = round((cal_100g * weight_g) / 100.0, 1)

            final_ingredients.append({
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "mask_area_ratio": mask_area_ratio,
                "weight_g": weight_g,
                "calories": calories,
                "nutrition": nutrition
            })

    # Case B: No ratio template exists (distribute proportionally to mask area ratios)
    else:
        total_area = sum(ing["mask_area_ratio"] for ing in grouped.values())
        for label, ing in grouped.items():
            ratio = ing["mask_area_ratio"] / total_area if total_area > 0 else (1.0 / len(grouped))
            weight_g = max(1.0, round(ratio * total_weight, 1))
            
            nutrition = ing.get("nutrition", {})
            calories = 0
            if nutrition:
                cal_100g = _get_nutrient_value(nutrition, "calories") or _get_nutrient_value(nutrition, "energy")
                calories = round((cal_100g * weight_g) / 100.0, 1)

            final_ingredients.append({
                "label": label,
                "confidence": ing["confidence"],
                "bbox": ing["bbox"],
                "mask_area_ratio": ing["mask_area_ratio"],
                "weight_g": weight_g,
                "calories": calories,
                "nutrition": nutrition
            })

    # Sort by weight descending
    final_ingredients.sort(key=lambda x: x["weight_g"], reverse=True)
    return final_ingredients


@celery.task(queue="default")
def aggregate_results(job_id: str, *args, **kwargs):
    """Finalize a job after all sub-tasks complete, and distribute ingredient weights."""
    job = get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}

    ingredients_list = job.ingredients or []
    class_name = job.class_name
    
    total_weight = None
    if job.portion:
        total_weight = job.portion.get("estimated_weight_grams")
        
    if not total_weight or total_weight <= 0:
        total_weight = TYPICAL_PORTION_GRAMS.get(class_name, 350)

    distributed_ingredients = distribute_ingredient_weights(class_name, ingredients_list, total_weight)

    update_job(
        job_id,
        status="completed",
        completed_at=datetime.utcnow(),
        ingredients=distributed_ingredients,
        progress=_merge_progress(job, {"aggregation": "completed"}),
    )

    return {"job_id": job_id, "status": "completed"}


def _merge_progress(job, updates: dict) -> dict:
    progress = job.progress if job.progress else {}
    progress.update(updates)
    return progress