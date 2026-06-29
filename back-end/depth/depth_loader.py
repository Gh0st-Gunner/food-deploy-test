from config import DEPTH_MODEL

_cached_depth = None


def load_depth_model():
    global _cached_depth
    if _cached_depth is not None:
        return _cached_depth

    from transformers import pipeline as hf_pipeline

    _cached_depth = hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
    )
    return _cached_depth


try:
    import streamlit as st

    @st.cache_resource
    def load_depth_model_cached():
        return load_depth_model()
except ImportError:
    pass