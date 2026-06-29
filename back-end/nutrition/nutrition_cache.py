"""Streamlit-compatible caching wrappers for the nutrition provider.

These wrap the pure-Python nutrition functions with @st.cache_data
so the Streamlit app still gets caching benefits. The Celery workers
use Redis caching via core/cache.py instead.
"""
import streamlit as st

from nutrition.nutrition_provider import lookup_nutrition as _lookup_nutrition
from nutrition.nutrition_provider import lookup_ingredient_nutrition as _lookup_ingredient_nutrition


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_nutrition_cached(class_name, usda_key="", fatsecret_id="", fatsecret_secret=""):
    return _lookup_nutrition(class_name, usda_key=usda_key, fatsecret_id=fatsecret_id, fatsecret_secret=fatsecret_secret)


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_ingredient_nutrition_cached(ingredient_label, usda_key="", fatsecret_id="", fatsecret_secret=""):
    return _lookup_ingredient_nutrition(ingredient_label, usda_key=usda_key, fatsecret_id=fatsecret_id, fatsecret_secret=fatsecret_secret)