import streamlit as st

from nutrition.usda_client import USDAClient
from nutrition.fatsecret_client import FatSecretClient
from nutrition.food_mapping import USDA_SEARCH_TERMS
from config import get_usda_api_key, get_fatsecret_credentials


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_nutrition(class_name: str, usda_key: str = "", fatsecret_id: str = "", fatsecret_secret: str = ""):
    """
    Look up nutrition data for a Vietnamese food class.

    Cascades through providers:
      1. USDA FoodData Central (best for Western foods, free, no auth needed beyond API key)
      2. FatSecret (better Asian food coverage, requires client ID + secret)

    Returns a dict with:
      - "provider": which API returned the data ("usda" or "fatsecret")
      - "description": food description from the API
      - "food_id": ID from the provider
      - "nutrients": dict of {name: {"value": float, "unit": str}}
      - "source": short label like "USDA FDC" or "FatSecret"
    """
    search_term = USDA_SEARCH_TERMS.get(class_name, class_name.replace("-", " "))

    # --- Try USDA first ---
    if usda_key:
        client = USDAClient(usda_key)
        food_data = client.search_with_fallback(class_name, USDA_SEARCH_TERMS)
        if food_data:
            nutrients = client.get_nutrients(food_data)
            if nutrients:
                return {
                    "provider": "usda",
                    "description": food_data.get("description", ""),
                    "food_id": str(food_data.get("fdcId", "")),
                    "nutrients": nutrients,
                    "source": "USDA FDC",
                }

    # --- Try FatSecret ---
    if fatsecret_id and fatsecret_secret:
        client = FatSecretClient(fatsecret_id, fatsecret_secret)

        # Try mapped term
        food = client.search_food(search_term)
        if food:
            food_id = food.get("food_id", "")
            food_details, nutrients = client.get_food_details(food_id)
            if nutrients:
                return {
                    "provider": "fatsecret",
                    "description": food.get("food_name", ""),
                    "food_id": food_id,
                    "nutrients": nutrients,
                    "source": "FatSecret",
                }

        # Try raw class name
        raw_name = class_name.replace("-", " ").replace("_", " ")
        food = client.search_food(raw_name)
        if food:
            food_id = food.get("food_id", "")
            food_details, nutrients = client.get_food_details(food_id)
            if nutrients:
                return {
                    "provider": "fatsecret",
                    "description": food.get("food_name", ""),
                    "food_id": food_id,
                    "nutrients": nutrients,
                    "source": "FatSecret",
                }

    return None


def lookup_ingredient_nutrition(ingredient_label: str, usda_key: str = "", fatsecret_id: str = "", fatsecret_secret: str = ""):
    """Look up nutrition for a single ingredient (used for per-ingredient display)."""
    if usda_key:
        client = USDAClient(usda_key)
        food = client.search_food(ingredient_label)
        if food:
            nutrients = client.get_nutrients(food)
            if nutrients:
                return {
                    "nutrients": nutrients,
                    "source": "USDA FDC",
                }

    if fatsecret_id and fatsecret_secret:
        client = FatSecretClient(fatsecret_id, fatsecret_secret)
        food = client.search_food(ingredient_label)
        if food:
            food_id = food.get("food_id", "")
            _, nutrients = client.get_food_details(food_id)
            if nutrients:
                return {
                    "nutrients": nutrients,
                    "source": "FatSecret",
                }

    return None