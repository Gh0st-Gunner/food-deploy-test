import streamlit as st
from config import DEPTH_MODEL


@st.cache_resource
def load_depth_model():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
    )