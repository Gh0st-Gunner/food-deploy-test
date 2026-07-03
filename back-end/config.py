import streamlit as st
import os

MODELS_DIR = "models"
ASSETS_DIR = "assets"

MODELS_DIR_FALLBACK = os.path.join(os.path.dirname(__file__), "models")

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2.1-hiera-small"
DEPTH_MODEL = "intel/zoedepth-nyu"

DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25

PLATE_DIAMETER_CM = 25.0

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

FATSECRET_CLIENT_ID = ""
FATSECRET_CLIENT_SECRET = ""


def get_usda_api_key():
    try:
        key = st.secrets.get("USDA_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    key = os.environ.get("USDA_API_KEY", "")
    if key:
        return key
    if "usda_api_key" in st.session_state:
        return st.session_state.usda_api_key
    return ""


def get_fatsecret_credentials():
    try:
        cid = st.secrets.get("FATSECRET_CLIENT_ID", "")
        cs = st.secrets.get("FATSECRET_CLIENT_SECRET", "")
        if cid and cs:
            return cid, cs
    except Exception:
        pass
    cid = os.environ.get("FATSECRET_CLIENT_ID", "")
    cs = os.environ.get("FATSECRET_CLIENT_SECRET", "")
    if cid and cs:
        return cid, cs
    if "fatsecret_client_id" in st.session_state and "fatsecret_client_secret" in st.session_state:
        return st.session_state.fatsecret_client_id, st.session_state.fatsecret_client_secret
    return "", ""