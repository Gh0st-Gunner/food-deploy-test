import streamlit as st
from config import GROUNDING_DINO_MODEL


@st.cache_resource
def load_grounding_dino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL, device_map="auto"
    )
    return model, processor