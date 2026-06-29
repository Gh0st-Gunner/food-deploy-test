import requests

from config import USDA_BASE_URL


class USDAClient:
    # USDA nutrientNumber values for matching
    NUTRIENT_IDS = {
        "calories": "208",
        "protein": "203",
        "total_fat": "204",
        "carbohydrates": "205",
        "fiber": "291",
        "sugars": "269",
        "sodium": "307",
        "calcium": "301",
        "iron": "303",
        "vitamin_c": "401",
        "vitamin_a": "320",
    }

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search_food(self, query: str, page_size: int = 5):
        foods = self.search_foods(query, page_size)
        if foods:
            return foods[0]
        return None

    def search_foods(self, query: str, page_size: int = 10):
        """Search USDA and return all results for fallback matching."""
        from core.cache import get_usda_cached, set_usda_cached
        cached = get_usda_cached(query)
        if cached is not None:
            return cached

        try:
            resp = requests.get(
                f"{USDA_BASE_URL}/foods/search",
                params=[
                    ("query", query),
                    ("pageSize", page_size),
                    ("dataType", "Foundation"),
                    ("dataType", "SR Legacy"),
                    ("dataType", "Survey (FNDDS)"),
                    ("api_key", self.api_key),
                ],
                timeout=10,
            )
            if resp.status_code != 200 and resp.status_code not in [403, 429]:
                # Fallback: retry without dataType filter
                resp = requests.get(
                    f"{USDA_BASE_URL}/foods/search",
                    params={"query": query, "pageSize": page_size, "api_key": self.api_key},
                    timeout=10,
                )
            resp.raise_for_status()
            data = resp.json()
            foods = data.get("foods", [])
            set_usda_cached(query, foods)
            return foods
        except requests.RequestException:
            return []

    def find_food_with_nutrients(self, query: str):
        """Search and return the first food that has nutrient data."""
        foods = self.search_foods(query)
        for food in foods:
            nutrients = self.get_nutrients(food)
            if nutrients:
                return food, nutrients
        return None, {}

    def get_nutrients(self, food: dict) -> dict:
        result = {}
        for nut in food.get("foodNutrients", []):
            nut_num = str(nut.get("nutrientNumber", ""))
            for name, target_num in self.NUTRIENT_IDS.items():
                if nut_num == target_num:
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


def get_rate_limit(api_key: str):
    """Fetch USDA API rate limit info from response headers."""
    try:
        resp = requests.get(
            f"{USDA_BASE_URL}/foods/search",
            params={"query": "a", "pageSize": 1, "api_key": api_key},
            timeout=10,
        )
        limit = resp.headers.get("X-RateLimit-Limit", "N/A")
        remaining = resp.headers.get("X-RateLimit-Remaining", "N/A")
        return int(limit) if limit != "N/A" else None, int(remaining) if remaining != "N/A" else None
    except Exception:
        return None, None


def cached_search_food(api_key: str, query: str, page_size: int = 5):
    client = USDAClient(api_key)
    return client.search_food(query, page_size)