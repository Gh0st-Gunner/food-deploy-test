"""
Vietnamese Food Classifier - Streamlit App
Features: Multi-model classification, ingredient detection, nutrition lookup, portion estimation
"""

import streamlit as st

from classification.tab_classification import tab_classification
from classification.tab_comparison import tab_comparison
from info.tab_info import tab_info

st.set_page_config(
    page_title="Vietnamese Food Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.main { padding-top: 0rem; }
.stTabs [data-baseweb="tab-list"] { gap: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    st.title("Vietnamese Food Classifier")
    st.markdown(
        "_AI-powered Vietnamese food recognition, ingredient detection, nutrition & portion analysis_"
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        [
            "Classification & Analysis",
            "Model Comparison",
            "Information",
        ]
    )

    with tab1:
        tab_classification()

    with tab2:
        tab_comparison()

    with tab3:
        tab_info()

    st.divider()
    st.markdown(
        """
<div style='text-align: center; color: #888; padding: 20px;'>
<small>Vietnamese Food Classifier | Built with PyTorch, Grounding DINO, SAM 2 & Depth Anything | © 2026</small>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()