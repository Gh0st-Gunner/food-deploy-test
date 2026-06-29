import time
import requests

from config import FATSECRET_CLIENT_ID, FATSECRET_CLIENT_SECRET


class FatSecretClient:
    """FatSecret API client for nutrition data, particularly good for Asian foods."""

    BASE_URL = "https://platform.fatsecret.com/rest/server.api"
    TOKEN_URL = "https://oauth.fatsecret.com/connect/token"

    NUTRIENT_MAP = {
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

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id or FATSECRET_CLIENT_ID
        self.client_secret = client_secret or FATSECRET_CLIENT_SECRET
        self._access_token = None
        self._token_expires = 0
        self._last_error = None

    def _get_access_token(self):
        """Get OAuth2 access token using client credentials."""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token

        self._last_error = None

        try:
            resp = requests.post(
                self.TOKEN_URL,
                auth=(self.client_id, self.client_secret),
                data={
                    "grant_type": "client_credentials",
                    "scope": "basic",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )

            if resp.status_code != 200:
                self._last_error = f"Token request failed (HTTP {resp.status_code}): {resp.text[:200]}"
                return None

            token_data = resp.json()
            if "access_token" not in token_data:
                self._last_error = f"No access_token in response: {list(token_data.keys())}"
                return None

            self._access_token = token_data["access_token"]
            self._token_expires = time.time() + token_data.get("expires_in", 86400) - 60
            return self._access_token

        except requests.exceptions.ConnectionError:
            self._last_error = "Connection error — check internet connection"
            return None
        except requests.exceptions.Timeout:
            self._last_error = "Request timed out"
            return None
        except Exception as e:
            self._last_error = f"Unexpected error: {e}"
            return None

    @property
    def last_error(self):
        return self._last_error

    def search_food(self, query: str, max_results: int = 5):
        """Search FatSecret for a food by name. Returns first match or None."""
        from core.cache import get_fatsecret_cached, set_fatsecret_cached
        cached = get_fatsecret_cached(query)
        if cached is not None:
            return cached

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
            if resp.status_code != 200:
                self._last_error = f"Search failed (HTTP {resp.status_code})"
                return None

            data = resp.json()

            # Check for API error
            if "error" in data:
                self._last_error = f"API error: {data['error'].get('message', data['error'])}"
                return None

            foods = data.get("foods", {}).get("food", [])
            if isinstance(foods, dict):
                foods = [foods]
            if foods:
                set_fatsecret_cached(query, foods[0])
                return foods[0]

            self._last_error = f"No results for '{query}'"
            return None

        except requests.RequestException as e:
            self._last_error = f"Request error: {e}"
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
            if resp.status_code != 200:
                return None, {}

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

            nutrients = self._extract_nutrients(serving)
            return food, nutrients

        except requests.RequestException:
            return None, {}

    def _extract_nutrients(self, serving: dict) -> dict:
        """Extract and normalize nutrients from a FatSecret serving to per-100g."""
        metric = float(serving.get("metric_serving_amount", 100) or 100)
        unit = serving.get("metric_serving_unit", "g") or "g"

        # Normalization factor: if serving is not per 100g, scale it
        if unit == "g" and metric > 0:
            factor = 100.0 / metric
        else:
            factor = 1.0

        result = {}
        for our_name, fs_key in self.NUTRIENT_MAP.items():
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


def cached_fatsecret_search(client_id: str, client_secret: str, query: str, max_results: int = 5):
    client = FatSecretClient(client_id, client_secret)
    return client.search_food(query, max_results)