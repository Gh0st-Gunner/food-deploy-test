import streamlit as st
import requests

from config import USDA_BASE_URL


class USDAClient:
    NUTRIENT_IDS = {
        "calories": 208,
        "protein": 203,
        "total_fat": 204,
        "carbohydrates": 205,
        "fiber": 291,
        "sugars": 269,
        "sodium": 307,
        "calcium": 301,
        "iron": 303,
        "vitamin_c": 401,
        "vitamin_a": 320,
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_food(self, query: str, page_size: int = 5):
        params = {
            "query": query,
            "pageSize": page_size,
            "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)"],
            "api_key": self.api_key,
        }
        try:
            resp = requests.get(
                f"{USDA_BASE_URL}/foods/search", params=params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("foods"):
                return data["foods"][0]
        except requests.RequestException:
            pass
        return None

    def get_nutrients(self, food: dict) -> dict:
        result = {}
        for nut in food.get("foodNutrients", []):
            nut_id = nut.get("nutrientId")
            for name, target_id in self.NUTRIENT_IDS.items():
                if nut_id == target_id:
                    result[name] = {
                        "value": nut.get("value", 0),
                        "unit": nut.get("unitName", ""),
                    }
        return result

    def search_with_fallback(self, class_name: str, usda_search_terms: dict):
        from nutrition.food_mapping import USDA_SEARCH_TERMS

        # Try mapped term first
        mapped = usda_search_terms.get(class_name, "")
        if mapped:
            result = self.search_food(mapped)
            if result:
                return result

        # Try English name without "Vietnamese"
        if mapped and "Vietnamese" in mapped:
            fallback = mapped.replace("Vietnamese", "").strip()
            result = self.search_food(fallback)
            if result:
                return result

        # Try raw class name
        raw = class_name.replace("-", " ").replace("_", " ")
        result = self.search_food(raw)
        if result:
            return result

        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search_food(api_key: str, query: str, page_size: int = 5):
    client = USDAClient(api_key)
    return client.search_food(query, page_size)