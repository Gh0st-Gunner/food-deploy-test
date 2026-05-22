import streamlit as st
from config import SAM2_MODEL


@st.cache_resource
def load_sam2():
    from transformers import Sam2Processor, Sam2Model

    processor = Sam2Processor.from_pretrained(SAM2_MODEL)
    model = Sam2Model.from_pretrained(SAM2_MODEL, device_map="auto")
    return model, processor