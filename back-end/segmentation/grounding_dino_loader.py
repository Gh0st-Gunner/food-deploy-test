from config import GROUNDING_DINO_MODEL

_cached_dino = None


def load_grounding_dino():
    global _cached_dino
    if _cached_dino is not None:
        return _cached_dino

    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL, device_map="auto"
    )
    _cached_dino = (model, processor)
    return model, processor


try:
    import streamlit as st

    @st.cache_resource
    def load_grounding_dino_cached():
        return load_grounding_dino()
except ImportError:
    pass