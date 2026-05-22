import os

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

from config import ASSETS_DIR


@st.cache_resource
def generate_pipeline_image():
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")

    color_input = "#FF6B6B"
    color_process = "#4ECDC4"
    color_model = "#95E1D3"
    color_output = "#FFA07A"

    def draw_box(ax, x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.1",
            edgecolor="black",
            facecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight="bold", wrap=True)

    def draw_arrow(ax, x1, y1, x2, y2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="->", mutation_scale=30, linewidth=2.5, color="black"
        )
        ax.add_patch(arrow)

    y_pos = 17

    draw_box(ax, 5, y_pos, 3, 1, "USER INPUT\n(Upload/URL/Paste)", color_input, 11)
    y_pos -= 1.5
    draw_arrow(ax, 5, 17.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 3.5, 1, "IMAGE PREPROCESSING\n(Resize 224x224, Normalize)", color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 3.5, 1, "MODEL SELECTION\n(1 or More Models)", color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 4, 1.2, "PARALLEL INFERENCE\n(Process Simultaneously)", color_process, 10)
    y_pos -= 1.8

    models_y = y_pos + 1
    model_positions = [(1.5, models_y), (5, models_y), (8.5, models_y)]
    model_names = [
        "ResNet50\n(89% acc)",
        "EfficientNet\n(92% acc)",
        "MobileNetV3\n(87% acc)",
    ]

    draw_arrow(ax, 5, y_pos + 1.8, 1.5, models_y + 0.5)
    draw_arrow(ax, 5, y_pos + 1.8, 5, models_y + 0.5)
    draw_arrow(ax, 5, y_pos + 1.8, 8.5, models_y + 0.5)

    for i, (x, y) in enumerate(model_positions):
        draw_box(ax, x, y, 2.2, 1, model_names[i], color_model, 9)

    draw_arrow(ax, 1.5, models_y - 0.5, 1.5, models_y - 1.2)
    draw_arrow(ax, 5, models_y - 0.5, 5, models_y - 1.2)
    draw_arrow(ax, 8.5, models_y - 0.5, 8.5, models_y - 1.2)

    y_pos = models_y - 2

    draw_box(ax, 5, y_pos, 4, 1, "SOFTMAX + PROBABILITY\n(Confidence Scores)", color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 3.5, 1, "RESULTS AGGREGATION\n(Top-3 per Model)", color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 3.5, 1, "CONSENSUS ANALYSIS\n(Model Agreement %)", color_process, 10)
    y_pos -= 1.5
    draw_arrow(ax, 5, y_pos + 1.5, 5, y_pos + 0.5)

    draw_box(ax, 5, y_pos, 3.5, 1.2, "DISPLAY RESULTS\n(Predictions & Scores)", color_output, 11)

    plt.tight_layout()
    return fig


def display_pipeline_diagram():
    assets_png = os.path.join(ASSETS_DIR, "pipeline.png")
    assets_drawio = os.path.join(ASSETS_DIR, "pipeline.drawio")

    if os.path.exists(assets_png):
        st.image(assets_png, use_container_width=True)
        return

    if os.path.exists(assets_drawio):
        try:
            with open(assets_drawio, "rb") as f:
                content = f.read()
                text = content.decode("utf-8", errors="ignore")
                if "<svg" in text.lower():
                    svg_start = text.lower().find("<svg")
                    svg_text = text[svg_start:]
                    components.html(svg_text, height=800, scrolling=True)
                    return
                st.info(
                    "Found assets/pipeline.drawio, but it can't be rendered directly. "
                    "Export to PNG/SVG for display."
                )
                st.download_button(
                    label="Download pipeline.drawio",
                    data=content,
                    file_name="pipeline.drawio",
                    mime="application/xml",
                )
                return
        except Exception:
            st.warning("Failed to load pipeline.drawio. Using generated diagram for now.")

    st.info("Pipeline image not found in assets/. Using generated diagram for now.")
    fig = generate_pipeline_image()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def tab_info():
    st.header("Application Information")

    tab_about, tab_tech, tab_pipeline = st.tabs(["About", "Technologies", "Pipeline"])

    with tab_about:
        st.subheader("Vietnamese Food Classifier")
        st.markdown(
            """
### What is this app?

This is an AI-powered application designed to classify **103 different Vietnamese food categories**.

Users can upload images or provide image URLs to get instant predictions from multiple deep learning models.

### Features

- **Multi-Model Support**: Use one or multiple models simultaneously
- **Flexible Input**: Upload images or provide URLs
- **Consensus Analysis**: Get agreement scores when using multiple models
- **Model Comparison**: Compare performance metrics across models
- **Nutrition & Ingredients**: Detect ingredients and look up USDA nutrition data
- **Portion Estimator**: Estimate food portion size using depth estimation
- **Real-time Predictions**: Fast classification powered by PyTorch

### Terms of Service

1. **Usage**: This app is provided for educational and personal use
2. **Accuracy**: Model predictions are probabilistic; always verify results
3. **Data**: Images are processed locally; no data is stored
4. **Limitations**: Best performance on clear, well-lit food images
5. **Attribution**: Dataset and models created for Vietnamese food classification research

### Disclaimer

This application is for informational purposes. While efforts have been made to ensure accuracy,
the developers make no guarantees about prediction correctness. Use at your own discretion.

Made with love by @Gunner
"""
        )

        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Classes", "103")
        with col2:
            st.metric("Max Models", "6")
        with col3:
            st.metric("Input Methods", "3")

    with tab_tech:
        st.subheader("Technology Stack")

        tech_data = {
            "Category": [
                "ML Framework",
                "ML Framework",
                "Computer Vision",
                "Web Framework",
                "Data Processing",
                "Model Architectures",
                "Ingredient Detection",
                "Depth Estimation",
            ],
            "Technology": [
                "PyTorch",
                "TorchVision",
                "PIL/Pillow",
                "Streamlit",
                "Pandas/NumPy",
                "ResNet, EfficientNet, MobileNetV3",
                "Grounding DINO + SAM 2",
                "Depth Anything V2",
            ],
            "Purpose": [
                "Deep learning models",
                "Pre-trained models",
                "Image processing",
                "Web interface",
                "Data analysis",
                "Classification backbones",
                "Ingredient detection and segmentation",
                "Portion size estimation",
            ],
        }

        df_tech = pd.DataFrame(tech_data)
        st.dataframe(df_tech, use_container_width=True, hide_index=True)

        st.markdown(
            """
### Model Architectures

- **ResNet50/101**: Deep residual networks for robust feature extraction
- **EfficientNet B0/B3**: Scalable models with optimal efficiency
- **MobileNetV3**: Lightweight models for fast inference

### Ingredient Detection

- **Grounding DINO**: Open-vocabulary object detection - finds ingredients by text prompt
- **SAM 2.1**: Segment Anything Model - creates precise pixel-level masks for each ingredient

### Depth Estimation

- **Depth Anything V2**: Monocular depth estimation for portion size guesstimation

### Dataset

- 30VNFoods @QuanDang ~25.2k images
- Vietnamese-foods-extended @Tran Van Nhan ~3387 images
- 100 Vietnamese Food @Karos ~20k images
- Combined by @Le Anh Duy
"""
        )

    with tab_pipeline:
        st.subheader("Application Pipeline")
        st.markdown(
            "_Visual representation of how the application processes your image through multiple AI models_"
        )
        st.divider()

        display_pipeline_diagram()

        st.divider()
        st.subheader("Pipeline Stages Explained")

        stages = [
            ("USER INPUT", "Users can provide input through upload, URL, or paste."),
            ("IMAGE PREPROCESSING", "Resize to 224x224, normalize with ImageNet statistics."),
            ("MODEL SELECTION", "Select one or more AI models for classification."),
            ("PARALLEL INFERENCE", "All selected models process the image simultaneously."),
            ("SOFTMAX + PROBABILITY", "Raw scores converted to probability distribution."),
            ("RESULTS AGGREGATION", "Top predictions collected from each model."),
            ("CONSENSUS ANALYSIS", "Agreement percentage calculated across models."),
            ("DISPLAY RESULTS", "Final predictions with confidence scores shown."),
        ]
        for title, desc in stages:
            with st.expander(f"**{title}**", expanded=False):
                st.markdown(desc)