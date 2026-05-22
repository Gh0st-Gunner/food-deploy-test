import hashlib
import time
import requests
import streamlit as st

from config import FATSECRET_CLIENT_ID, FATSECRET_CLIENT_SECRET


class FatSecretClient:
    """FatSecret API client for nutrition data, particularly good for Asian foods."""

    BASE_URL = "https://platform.fatsecret.com/rest/server.api"

    NUTRIENT_MAP = {
        "calories": "Calories",
        "protein": "Protein",
        "total_fat": "Total fat",
        "carbohydrates": "Carbohydrates",
        "fiber": "Fiber",
        "sugars": "Sugars",
        "sodium": "Sodium",
        "calcium": "Calcium",
        "iron": "Iron",
        "vitamin_c": "Vitamin C",
        "vitamin_a": "Vitamin A",
    }

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id or FATSECRET_CLIENT_ID
        self.client_secret = client_secret or FATSECRET_CLIENT_SECRET
        self._access_token = None
        self._token_expires = 0

    def _get_access_token(self):
        """Get OAuth2 access token using client credentials."""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        url = "https://oauth.fatsecret.com/connect/token"
        data = {
            "grant_type": "client_credentials",
            "scope": "basic",
        }
        try:
            resp = requests.post(
                url,
                auth=(self.client_id, self.client_secret),
                data=data,
                timeout=10,
            )
            resp.raise_for_status()
            token_data = resp.json()
            self._access_token = token_data["access_token"]
            self._token_expires = time.time() + token_data.get("expires_in", 86400) - 60
            return self._access_token
        except requests.RequestException:
            return None

    def search_food(self, query: str, max_results: int = 5):
        """Search FatSecret for a food by name. Returns first match or None."""
        token = self._get_access_token()
        if not token:
            return None

        params = {
            "method": "foods.search",
            "search_expression": query,
            "max_results": max_results,
            "format": "json",
        }
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            foods = data.get("foods", {}).get("food", [])
            if isinstance(foods, dict):
                foods = [foods]
            if foods:
                return foods[0]
        except requests.RequestException:
            pass
        return None

    def get_food_details(self, food_id: str):
        """Get detailed nutrition for a food by its ID. Returns per-100g data."""
        token = self._get_access_token()
        if not token:
            return None, {}

        params = {
            "method": "food.get",
            "food_id": food_id,
            "format": "json",
        }
        headers = {"Authorization": f"Bearer {token}"}

        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            food = data.get("food", {})
            servings = food.get("servings", {}).get("serving", [])

            if isinstance(servings, dict):
                servings = [servings]

            # Find per-100g serving, or fallback to first serving
            serving = None
            for s in servings:
                if s.get("serving_description", "").startswith("100 g"):
                    serving = s
                    break

            if not serving and servings:
                serving = servings[0]

            if not serving:
                return food, {}

            # Extract nutrients, normalizing to per-100g
            nutrients = self._extract_nutrients(serving)
            return food, nutrients

        except requests.RequestException:
            return None, {}

    def _extract_nutrients(self, serving: dict) -> dict:
        """Extract and normalize nutrients from a FatSecret serving to per-100g."""
        # FatSecret gives values per serving; check if it's per 100g
        serving_desc = serving.get("serving_description", "")
        metric = float(serving.get("metric_serving_amount", 100) or 100)
        unit = serving.get("metric_serving_unit", "g") or "g"

        # Normalization factor: if serving is not per 100g, scale it
        if unit == "g" and metric > 0:
            factor = 100.0 / metric
        else:
            factor = 1.0

        result = {}
        fatsecret_fields = {
            "calories": "calories",
            "protein": "protein",
            "total_fat": "total_fat",
            "carbohydrates": "carbohydrate",
            "fiber": "fiber",
            "sugars": "sugar",
            "sodium": "sodium",
            "calcium": "calcium",
            "iron": "iron",
            "vitamin_c": "vitamin_c",
            "vitamin_a": "vitamin_a",
        }

        for our_name, fs_key in fatsecret_fields.items():
            value_str = serving.get(fs_key)
            if value_str:
                try:
                    value = float(value_str)
                    result[our_name] = {
                        "value": round(value * factor, 2),
                        "unit": "g" if our_name != "calories" else "kcal",
                    }
                except (ValueError, TypeError):
                    pass

        return result


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fatsecret_search(client_id: str, client_secret: str, query: str, max_results: int = 5):
    client = FatSecretClient(client_id, client_secret)
    return client.search_food(query, max_results)