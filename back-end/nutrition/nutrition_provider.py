from nutrition.usda_client import USDAClient
from nutrition.fatsecret_client import FatSecretClient
from nutrition.food_mapping import USDA_SEARCH_TERMS

MOCK_FOOD_NUTRITION = {
    "pho": {"calories": 130, "protein": 6.0, "total_fat": 3.2, "carbohydrates": 17.0},
    "banh-mi": {"calories": 250, "protein": 8.0, "total_fat": 9.0, "carbohydrates": 35.0},
    "com-tam": {"calories": 200, "protein": 8.0, "total_fat": 6.0, "carbohydrates": 28.0},
    "bun-cha": {"calories": 150, "protein": 6.0, "total_fat": 4.0, "carbohydrates": 22.0},
    "bun-bo-hue": {"calories": 120, "protein": 6.0, "total_fat": 4.0, "carbohydrates": 15.0},
    "goi-cuon": {"calories": 120, "protein": 6.0, "total_fat": 2.5, "carbohydrates": 18.0},
    "banh-xeo": {"calories": 180, "protein": 5.0, "total_fat": 7.0, "carbohydrates": 25.0},
    "bo-kho": {"calories": 110, "protein": 8.0, "total_fat": 4.5, "carbohydrates": 10.0},
    "default": {"calories": 140, "protein": 6.0, "total_fat": 5.0, "carbohydrates": 18.0}
}

MOCK_INGREDIENTS = {
    "beef": {"calories": 250, "protein": 26.0, "total_fat": 15.0, "carbohydrates": 0.0},
    "pork": {"calories": 290, "protein": 20.0, "total_fat": 22.0, "carbohydrates": 0.0},
    "chicken": {"calories": 165, "protein": 31.0, "total_fat": 3.6, "carbohydrates": 0.0},
    "shrimp": {"calories": 85, "protein": 20.0, "total_fat": 0.5, "carbohydrates": 0.0},
    "fish": {"calories": 120, "protein": 18.0, "total_fat": 4.0, "carbohydrates": 1.0},
    "crab": {"calories": 90, "protein": 19.0, "total_fat": 1.0, "carbohydrates": 0.0},
    "squid": {"calories": 92, "protein": 16.0, "total_fat": 1.4, "carbohydrates": 3.0},
    "tofu": {"calories": 140, "protein": 14.0, "total_fat": 8.0, "carbohydrates": 2.5},
    "noodle": {"calories": 110, "protein": 2.0, "total_fat": 0.1, "carbohydrates": 25.0},
    "vermicelli": {"calories": 110, "protein": 2.0, "total_fat": 0.1, "carbohydrates": 25.0},
    "bread": {"calories": 265, "protein": 9.0, "total_fat": 3.2, "carbohydrates": 49.0},
    "bun": {"calories": 265, "protein": 9.0, "total_fat": 3.2, "carbohydrates": 49.0},
    "baguette": {"calories": 265, "protein": 9.0, "total_fat": 3.2, "carbohydrates": 49.0},
    "egg": {"calories": 155, "protein": 13.0, "total_fat": 11.0, "carbohydrates": 1.1},
    "vegetable": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "spinach": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "basil": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "herb": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "scallion": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "cilantro": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "mint": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "lettuce": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "sprout": {"calories": 22, "protein": 2.0, "total_fat": 0.2, "carbohydrates": 3.0},
    "carrot": {"calories": 30, "protein": 1.0, "total_fat": 0.1, "carbohydrates": 7.0},
    "cucumber": {"calories": 15, "protein": 0.6, "total_fat": 0.1, "carbohydrates": 3.6},
    "tomato": {"calories": 18, "protein": 0.9, "total_fat": 0.2, "carbohydrates": 3.9},
    "sauce": {"calories": 50, "protein": 1.0, "total_fat": 0.5, "carbohydrates": 10.0},
    "broth": {"calories": 15, "protein": 1.0, "total_fat": 0.5, "carbohydrates": 1.0},
    "soup": {"calories": 15, "protein": 1.0, "total_fat": 0.5, "carbohydrates": 1.0},
    "peanut": {"calories": 567, "protein": 25.0, "total_fat": 49.0, "carbohydrates": 16.0},
    "onion": {"calories": 40, "protein": 1.1, "total_fat": 0.1, "carbohydrates": 9.0},
    "garlic": {"calories": 149, "protein": 6.4, "total_fat": 0.5, "carbohydrates": 33.0},
    "ginger": {"calories": 80, "protein": 1.8, "total_fat": 0.8, "carbohydrates": 18.0},
    "lime": {"calories": 30, "protein": 0.7, "total_fat": 0.2, "carbohydrates": 10.0},
    "lemon": {"calories": 30, "protein": 0.7, "total_fat": 0.2, "carbohydrates": 10.0},
    "coconut": {"calories": 230, "protein": 2.3, "total_fat": 24.0, "carbohydrates": 6.0},
    "mushroom": {"calories": 35, "protein": 3.0, "total_fat": 0.3, "carbohydrates": 5.0},
    "potato": {"calories": 89, "protein": 1.1, "total_fat": 0.3, "carbohydrates": 23.0},
    "pumpkin": {"calories": 26, "protein": 1.0, "total_fat": 0.1, "carbohydrates": 6.5},
    "banana": {"calories": 89, "protein": 1.1, "total_fat": 0.3, "carbohydrates": 23.0},
    "eel": {"calories": 184, "protein": 18.4, "total_fat": 11.6, "carbohydrates": 0.0},
    "rice": {"calories": 130, "protein": 2.7, "total_fat": 0.3, "carbohydrates": 28.0},
    "sticky rice": {"calories": 150, "protein": 3.0, "total_fat": 0.5, "carbohydrates": 33.0},
    "default": {"calories": 50, "protein": 2.0, "total_fat": 1.0, "carbohydrates": 8.0}
}


def get_mock_nutrition(class_name: str) -> dict:
    class_name = class_name.lower()
    nut = MOCK_FOOD_NUTRITION.get(class_name)
    if not nut:
        for key, val in MOCK_FOOD_NUTRITION.items():
            if key != "default" and key in class_name:
                nut = val
                break
    if not nut:
        nut = MOCK_FOOD_NUTRITION["default"]
        
    return {
        "calories": {"value": float(nut["calories"]), "unit": "kcal"},
        "protein": {"value": float(nut["protein"]), "unit": "g"},
        "total_fat": {"value": float(nut["total_fat"]), "unit": "g"},
        "carbohydrates": {"value": float(nut["carbohydrates"]), "unit": "g"},
    }


def get_mock_ingredient_nutrition(label: str) -> dict:
    label = label.lower()
    nut = None
    for key, val in MOCK_INGREDIENTS.items():
        if key != "default" and key in label:
            nut = val
            break
    if not nut:
        nut = MOCK_INGREDIENTS["default"]
        
    return {
        "calories": {"value": float(nut["calories"]), "unit": "kcal"},
        "protein": {"value": float(nut["protein"]), "unit": "g"},
        "total_fat": {"value": float(nut["total_fat"]), "unit": "g"},
        "carbohydrates": {"value": float(nut["carbohydrates"]), "unit": "g"},
    }


def lookup_nutrition(class_name: str, usda_key: str = "", fatsecret_id: str = "", fatsecret_secret: str = ""):
    """
    Look up nutrition data for a Vietnamese food class.

    Cascades through providers:
      1. USDA FoodData Central (best for Western foods, free key needed)
      2. FatSecret (better Asian food coverage, client ID + secret needed)

    Returns a dict with:
      - "provider": which API returned the data ("usda" or "fatsecret")
      - "description": food description from the API
      - "food_id": ID from the provider
      - "nutrients": dict of {name: {"value": float, "unit": str}}
      - "source": short label like "USDA FDC" or "FatSecret"
      - "errors": list of error messages from failed providers
    """
    from core.cache import get_class_mapping_cached, set_class_mapping_cached
    cached = get_class_mapping_cached(class_name)
    if cached:
        return cached

    search_term = USDA_SEARCH_TERMS.get(class_name, class_name.replace("-", " "))
    errors = []

    # --- Try USDA first ---
    if usda_key:
        client = USDAClient(usda_key)
        try:
            # Try mapped search term
            mapped = USDA_SEARCH_TERMS.get(class_name, "")
            queries = []
            if mapped:
                queries.append(mapped)
                if "Vietnamese" in mapped:
                    queries.append(mapped.replace("Vietnamese", "").strip())
            queries.append(class_name.replace("-", " ").replace("_", " "))

            for query in queries:
                food, nutrients = client.find_food_with_nutrients(query)
                if nutrients:
                    res = {
                        "provider": "usda",
                        "description": food.get("description", ""),
                        "food_id": str(food.get("fdcId", "")),
                        "nutrients": nutrients,
                        "source": "USDA FDC",
                        "errors": [],
                    }
                    set_class_mapping_cached(class_name, res)
                    return res
            errors.append("USDA: no match found with nutrient data")
        except Exception as e:
            errors.append(f"USDA: {e}")

    # --- Try FatSecret ---
    if fatsecret_id and fatsecret_secret:
        client = FatSecretClient(fatsecret_id, fatsecret_secret)

        # Try mapped search term first
        food = client.search_food(search_term)
        if food:
            food_id = food.get("food_id", "")
            _, nutrients = client.get_food_details(food_id)
            if nutrients:
                res = {
                    "provider": "fatsecret",
                    "description": food.get("food_name", ""),
                    "food_id": food_id,
                    "nutrients": nutrients,
                    "source": "FatSecret",
                    "errors": errors,
                }
                set_class_mapping_cached(class_name, res)
                return res

        # Try raw class name
        raw_name = class_name.replace("-", " ").replace("_", " ")
        if raw_name != search_term:
            food = client.search_food(raw_name)
            if food:
                food_id = food.get("food_id", "")
                _, nutrients = client.get_food_details(food_id)
                if nutrients:
                    res = {
                        "provider": "fatsecret",
                        "description": food.get("food_name", ""),
                        "food_id": food_id,
                        "nutrients": nutrients,
                        "source": "FatSecret",
                        "errors": errors,
                    }
                    set_class_mapping_cached(class_name, res)
                    return res

    # --- Fallback to Local Mock ---
    mock_nut = get_mock_nutrition(class_name)
    if mock_nut:
        res = {
            "provider": "mock",
            "description": f"Mock Fallback for {class_name.replace('-', ' ').title()}",
            "food_id": f"mock-{class_name}",
            "nutrients": mock_nut,
            "source": "Local Fallback",
            "errors": errors,
        }
        set_class_mapping_cached(class_name, res)
        return res

    return {"provider": None, "description": "", "food_id": "", "nutrients": {}, "source": "", "errors": errors}


def lookup_ingredient_nutrition(ingredient_label: str, usda_key: str = "", fatsecret_id: str = "", fatsecret_secret: str = ""):
    """Look up nutrition for a single ingredient (used for per-ingredient display)."""
    from core.cache import get_ingredient_nutrition_cached, set_ingredient_nutrition_cached
    cached = get_ingredient_nutrition_cached(ingredient_label)
    if cached:
        return cached

    if usda_key:
        client = USDAClient(usda_key)
        try:
            food, nutrients = client.find_food_with_nutrients(ingredient_label)
            if nutrients:
                res = {
                    "nutrients": nutrients,
                    "source": "USDA FDC",
                    "description": food.get("description", ""),
                }
                set_ingredient_nutrition_cached(ingredient_label, res)
                return res
        except Exception:
            pass

    if fatsecret_id and fatsecret_secret:
        client = FatSecretClient(fatsecret_id, fatsecret_secret)
        food = client.search_food(ingredient_label)
        if food:
            food_id = food.get("food_id", "")
            _, nutrients = client.get_food_details(food_id)
            if nutrients:
                res = {
                    "nutrients": nutrients,
                    "source": "FatSecret",
                }
                set_ingredient_nutrition_cached(ingredient_label, res)
                return res

    # --- Fallback to Local Mock ---
    mock_nut = get_mock_ingredient_nutrition(ingredient_label)
    if mock_nut:
        res = {
            "nutrients": mock_nut,
            "source": "Local Fallback",
            "description": f"Mock Fallback for {ingredient_label.title()}",
        }
        set_ingredient_nutrition_cached(ingredient_label, res)
        return res

    return None