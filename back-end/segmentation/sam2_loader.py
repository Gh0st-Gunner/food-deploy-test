from config import SAM2_MODEL

_cached_sam2 = None


def load_sam2():
    global _cached_sam2
    if _cached_sam2 is not None:
        return _cached_sam2

    from transformers import Sam2Processor, Sam2Model

    processor = Sam2Processor.from_pretrained(SAM2_MODEL)
    model = Sam2Model.from_pretrained(SAM2_MODEL, device_map="auto")
    _cached_sam2 = (model, processor)
    return model, processor


try:
    import streamlit as st

    @st.cache_resource
    def load_sam2_cached():
        return load_sam2()
except ImportError:
    pass