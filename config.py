import streamlit as st
import os

MODELS_DIR = "models"
ASSETS_DIR = "assets"

MODELS_DIR_FALLBACK = os.path.join(os.path.dirname(__file__), "models")

GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL = "facebook/sam2.1-hiera-small"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

DEFAULT_BOX_THRESHOLD = 0.3
DEFAULT_TEXT_THRESHOLD = 0.25

PLATE_DIAMETER_CM = 25.0

USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


def get_usda_api_key():
    try:
        return st.secrets.get("USDA_API_KEY", "")
    except Exception:
        pass
    return os.environ.get("USDA_API_KEY", "")