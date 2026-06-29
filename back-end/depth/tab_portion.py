import numpy as np
import streamlit as st
from PIL import Image

from depth.depth_loader import load_depth_model
from depth.portion_estimator import estimate_portion
from depth.food_densities import get_density
from nutrition.food_mapping import USDA_SEARCH_TERMS, TYPICAL_PORTION_GRAMS
from nutrition.usda_client import USDAClient
from nutrition.nutrition_display import format_class_name
from config import get_usda_api_key


def tab_portion():
    st.header("Portion Estimator")
    st.markdown("Estimate food portion size using depth estimation")

    has_classification = "last_classification" in st.session_state
    class_name = None
    image = None

    if has_classification:
        last = st.session_state.last_classification
        class_name = last.get("class_name")
        image = last.get("image")
        st.info(
            f"Using classification: **{format_class_name(class_name)}** "
            f"({last.get('confidence', 0) * 100:.1f}% confidence)"
        )

    # Image input
    st.subheader("Input Image")
    use_existing = has_classification and st.checkbox(
        "Use classified image", value=True if has_classification else False, key="portion_use_existing"
    )

    if not use_existing:
        uploaded_file = st.file_uploader(
            "Upload a food image", type=["jpg", "jpeg", "png"], key="portion_upload"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    if not image or not class_name:
        st.warning("Please classify an image first, or upload one here.")
        return

    # Reference scale
    st.subheader("Scale Reference")
    scale_method = st.radio(
        "How to determine real-world scale:",
        ["Assume standard plate (25cm diameter)", "Provide reference height"],
        index=0,
    )

    reference_height_cm = None
    if scale_method == "Provide reference height":
        reference_height_cm = st.number_input(
            "Reference object height (cm):",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )

    # Run depth estimation
    if st.button("Estimate Portion Size", type="primary"):
        with st.spinner("Loading depth estimation model (may take a moment on first run)..."):
            depth_pipeline = load_depth_model()

        with st.spinner("Estimating depth and portion..."):
            result = estimate_portion(
                image=image,
                class_name=class_name,
                depth_pipeline=depth_pipeline,
                reference_height_cm=reference_height_cm,
            )

        # Store in session state for cross-tab access
        st.session_state.portion_estimate = result

        # Display results
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Depth Map")
            depth_map = result["depth_map"]

            # Colorize depth map for display
            depth_colored = (depth_map * 255).astype(np.uint8)
            depth_image = Image.fromarray(
                np.stack([depth_colored] * 3, axis=-1)
            )
            st.image(depth_image, caption="Depth Map (brighter = closer)", use_container_width=True)

            # 3D surface plot
            st.subheader("3D Surface")
            try:
                import plotly.graph_objects as go

                # Downsample for performance
                step = max(1, depth_map.shape[0] // 100)
                z = depth_map[::step, ::step]
                fig = go.Figure(data=[go.Surface(z=z)])
                fig.update_layout(
                    scene=dict(zaxis_title="Depth"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="Food Surface Estimate",
                )
                fig.update_traces(
                    colorscale="Viridis",
                    showscale=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for 3D visualization: pip install plotly")

        with col_right:
            st.subheader("Estimated Portion")

            display_name = format_class_name(class_name)
            st.write(f"**Dish:** {display_name}")
            st.write(f"**Scaling method:** {result['scaling_method']}")
            st.write(f"**Food density used:** {result['density_used']:.2f} kg/L")

            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Weight", f"{result['estimated_weight_grams']:.0f} g")
            with col2:
                st.metric("Estimated Volume", f"{result['estimated_volume_ml']:.0f} mL")
            with col3:
                st.metric("Typical Portion", f"{result['typical_portion_grams']} g")

            st.divider()

            # Nutrition adjustment
            api_key = get_usda_api_key()
            if not api_key:
                api_key = st.text_input(
                    "USDA API Key",
                    type="password",
                    help="Get a free key at https://fdc.nal.usda.gov/api-key.html",
                    key="portion_usda_key",
                )

            if api_key:
                client = USDAClient(api_key)
                with st.spinner("Fetching nutrition data..."):
                    food_data = client.search_with_fallback(class_name, USDA_SEARCH_TERMS)

                if food_data:
                    nutrients = client.get_nutrients(food_data)
                    multiplier = result["nutrient_multiplier"]

                    st.write(f"**USDA Match:** {food_data.get('description', 'Unknown')}")
                    st.write(
                        f"**Adjustment:** {multiplier:.2f}x (base nutrition per 100g "
                        f"x {result['estimated_weight_grams']:.0f}g portion)"
                    )

                    st.subheader("Adjusted Nutrition")
                    for name, info in nutrients.items():
                        value = info.get("value", 0)
                        unit = info.get("unit", "")
                        adjusted = value * multiplier
                        display = name.replace("_", " ").title()
                        st.write(f"**{display}**: {adjusted:.1f} {unit}")
                else:
                    st.info("No USDA data found for this dish.")
            else:
                st.info("Enter a USDA API key to see adjusted nutrition values.")

        st.caption(
            "This is an approximate estimation based on monocular depth estimation. "
            "Results may vary significantly. Not a substitute for weighing food."
        )