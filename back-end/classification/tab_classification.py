import time
import streamlit as st
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import Counter

from classification.model_loader import (
    get_available_models,
    load_model,
    load_onnx_model,
)
from classification.predict import predict, predict_onnx
from config import get_usda_api_key, get_fatsecret_credentials, DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD
from nutrition.food_mapping import (
    USDA_SEARCH_TERMS,
    INGREDIENT_PROMPTS,
    TYPICAL_PORTION_GRAMS,
)
from nutrition.usda_client import get_rate_limit
from nutrition.nutrition_provider import lookup_nutrition, lookup_ingredient_nutrition
from nutrition.nutrition_cache import lookup_nutrition_cached, lookup_ingredient_nutrition_cached
from nutrition.nutrition_display import display_nutrition_table, format_class_name
from segmentation.grounding_dino_loader import load_grounding_dino
from segmentation.sam2_loader import load_sam2
from segmentation.ingredient_detector import detect_ingredients
from segmentation.visualize import overlay_ingredients, format_ingredients_legend, PALETTE
from depth.depth_loader import load_depth_model
from depth.portion_estimator import estimate_portion
from depth.food_densities import get_density


def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image, None
    except Exception as e:
        return None, f"Error loading image: {e}"


def _fmt(class_name):
    return class_name.replace("_", " ").replace("-", " ").title()


def tab_classification():
    st.header("Food Classifier & Analysis")
    st.markdown("Classify Vietnamese food, detect ingredients, view nutrition, and estimate portions")

    available_models = get_available_models()

    if not available_models:
        st.error("No models found! Place model checkpoints in 'models/' folder")
        st.info(
            "1. Create a 'models' folder\n"
            "2. Copy your .pth checkpoint files there\n"
            "3. Name them: resnet50.pth, efficientnet_b0.pth, etc."
        )
        return

    # --- Sidebar: Model selection + API key ---
    with st.sidebar:
        st.subheader("Model Selection")
        select_all = st.checkbox("Select All Models", value=False)

        selected_models = {}
        if select_all:
            selected_models = available_models
        else:
            for model_name in available_models:
                if st.checkbox(model_name.upper(), key=f"model_{model_name}"):
                    selected_models[model_name] = available_models[model_name]

        st.divider()

        st.subheader("Nutrition APIs")
        api_key = get_usda_api_key()
        if not api_key:
            api_key = st.text_input(
                "USDA API Key",
                type="password",
                help="Free key at fdc.nal.usda.gov/api-key.html",
                key="usda_key_sidebar",
            )
            if api_key:
                st.session_state.usda_api_key = api_key
        else:
            st.success("USDA API key configured")
            if st.button("API Quota", key="usda_quota_btn"):
                limit, remaining = get_rate_limit(api_key)
                if limit is not None:
                    used = limit - remaining
                    st.info(f"**{remaining:,}** / {limit:,} calls remaining\n\n({used:,} used this hour)")
                else:
                    st.warning("Could not fetch quota info")

        fatsecret_id, fatsecret_secret = get_fatsecret_credentials()
        if not fatsecret_id or not fatsecret_secret:
            with st.expander("FatSecret (optional)", expanded=False):
                st.caption("Better Asian food coverage. Free at platform.fatsecret.com")
                fs_id = st.text_input("Client ID", type="password", key="fs_id_sidebar")
                fs_secret = st.text_input("Client Secret", type="password", key="fs_secret_sidebar")
                if fs_id and fs_secret:
                    st.session_state.fatsecret_client_id = fs_id
                    st.session_state.fatsecret_client_secret = fs_secret
                    fatsecret_id = fs_id
                    fatsecret_secret = fs_secret
        else:
            st.success("FatSecret configured")

        st.divider()

        st.subheader("Detection Settings")
        box_threshold = st.slider(
            "Ingredient detection sensitivity",
            min_value=0.1,
            max_value=0.9,
            value=DEFAULT_BOX_THRESHOLD,
            step=0.05,
            help="Lower = more detections, higher = more selective",
        )

        scale_method = st.radio(
            "Scale reference:",
            ["Standard plate (25cm)", "Custom reference height"],
            index=0,
        )
        reference_height_cm = None
        if scale_method == "Custom reference height":
            reference_height_cm = st.number_input(
                "Reference height (cm):",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
            )

    if not selected_models:
        st.warning("Select at least one model in the sidebar")
        return

    # --- Image input ---
    st.subheader("Input Image")
    input_method = st.radio(
        "Choose input method:",
        ["Upload Image", "Image URL"],
    )

    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_method == "Image URL":
        image_url = st.text_input("Enter image URL:")
        if image_url:
            with st.spinner("Loading image..."):
                image, error = load_image_from_url(image_url)
                if error:
                    st.error(error)

    if not image:
        return

    # ============================================================
    # RUN EVERYTHING AUTOMATICALLY
    # ============================================================
    timings = {}

    # --- 1. Classification ---
    st.divider()
    st.subheader("Classification")

    t0 = time.time()
    predictions_data = {}

    for model_name, model_path in selected_models.items():
        if model_path.endswith(".onnx"):
            session, input_name, class_names = load_onnx_model(model_path)
            if session is not None and class_names:
                preds = predict_onnx(image, session, input_name, class_names, top_k=3)
                predictions_data[model_name] = {
                    "predictions": preds,
                    "accuracy": "N/A",
                    "detected_name": "ONNX",
                }
        else:
            model, class_names, device, detected_name, accuracy = load_model(model_path)
            if model is not None:
                preds = predict(image, model, class_names, device, top_k=3)
                predictions_data[model_name] = {
                    "predictions": preds,
                    "accuracy": accuracy,
                    "detected_name": detected_name,
                }

    timings["Classification"] = time.time() - t0

    if not predictions_data:
        st.error("All models failed to load.")
        return

    all_top_predictions = [d["predictions"][0]["class"] for d in predictions_data.values()]
    vote_counts = Counter(all_top_predictions)
    consensus = vote_counts.most_common(1)[0]
    agreement = (consensus[1] / len(predictions_data)) * 100 if len(predictions_data) > 1 else 100.0

    class_name = consensus[0]
    display_name = _fmt(class_name)
    top_confidence = max(d["predictions"][0]["probability"] for d in predictions_data.values())

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.image(image, caption="Input Image", use_container_width=True)
    with col_right:
        st.metric("Model Agreement", f"{agreement:.0f}%")
        st.success(f"**{display_name}** ({top_confidence:.1%} confidence)")
        if len(predictions_data) > 1:
            st.caption(f"{int(consensus[1])}/{len(predictions_data)} models agree")

        best_model_data = max(predictions_data.values(), key=lambda d: d["predictions"][0]["probability"])
        st.write("**Top 3:**")
        for pred in best_model_data["predictions"]:
            pct = pred["probability"] * 100
            st.write(f"{pred['rank']}. {_fmt(pred['class'])}: {pct:.1f}%")

        with st.expander("Per-model breakdown"):
            for model_name, data in predictions_data.items():
                top = data["predictions"][0]
                acc_text = (
                    f"{data['accuracy']:.2f}%"
                    if isinstance(data["accuracy"], (int, float))
                    else data["accuracy"]
                )
                st.write(
                    f"**{model_name.upper()}** "
                    f"(arch: {data['detected_name']}, acc: {acc_text}) "
                    f"→ {_fmt(top['class'])} at {top['probability']:.1%}"
                )

    st.session_state.last_classification = {
        "class_name": class_name,
        "confidence": top_confidence,
        "image": image,
        "predictions": predictions_data,
    }

    # --- 2. Nutrition ---
    st.divider()
    st.subheader("Nutrition")

    nutrients = {}
    nutrition_source = ""
    nutrition_errors = []

    t0 = time.time()
    nutrition_result = lookup_nutrition_cached(class_name, usda_key=api_key, fatsecret_id=fatsecret_id, fatsecret_secret=fatsecret_secret)
    if nutrition_result and nutrition_result.get("nutrients"):
        nutrients = nutrition_result["nutrients"]
        nutrition_source = nutrition_result["source"]
        nutrition_errors = nutrition_result.get("errors", [])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"**{nutrition_source} Match:** {nutrition_result['description']}")
            st.caption(f"ID: {nutrition_result['food_id']}")
            typical = TYPICAL_PORTION_GRAMS.get(class_name, 300)
            st.info(f"Typical portion: ~{typical}g")
        with col2:
            display_nutrition_table(nutrients)
    elif api_key or (fatsecret_id and fatsecret_secret):
        # Show what went wrong
        msg = "No nutrition data found for this dish."
        if nutrition_result and nutrition_result.get("errors"):
            msg += " Debug info:"
            for err in nutrition_result["errors"]:
                msg += f"\n- {err}"
        st.warning(msg)
        st.info(
            "Try searching manually at [USDA](https://fdc.nal.usda.gov/) or [FatSecret](https://www.fatsecret.com/)."
        )
    else:
        st.info(
            "Add a USDA API key or FatSecret credentials in the sidebar to view nutrition data.\n"
            "- USDA: free key at [fdc.nal.usda.gov/api-key.html](https://fdc.nal.usda.gov/api-key.html)\n"
            "- FatSecret: free at [platform.fatsecret.com](https://platform.fatsecret.com/)"
        )
    timings["Nutrition"] = time.time() - t0

    # --- 3. Ingredient Detection ---
    ingredients_list = INGREDIENT_PROMPTS.get(class_name, [])
    ingredient_results = []

    if ingredients_list:
        st.divider()
        st.subheader("Ingredient Detection")

        with st.expander("Expected ingredients for this dish", expanded=False):
            for ing in ingredients_list:
                st.write(f"- {ing.rstrip('.')}")

        t0 = time.time()
        with st.spinner("Detecting ingredients..."):
            grounding_model, grounding_processor = load_grounding_dino()
            sam_model, sam_processor = load_sam2()
            ingredient_results = detect_ingredients(
                image=image,
                class_name=class_name,
                grounding_model=grounding_model,
                grounding_processor=grounding_processor,
                sam_model=sam_model,
                sam_processor=sam_processor,
                box_threshold=box_threshold,
            )
        timings["Ingredients"] = time.time() - t0

        if ingredient_results:
            st.session_state.ingredient_results = ingredient_results

            overlay = overlay_ingredients(image, ingredient_results)
            col_img, col_legend = st.columns([2, 1])
            with col_img:
                st.image(overlay, caption="Detected Ingredients", use_container_width=True)
            with col_legend:
                legend = format_ingredients_legend(ingredient_results)
                st.write("**Detected:**")
                for item in legend:
                    color_hex = "#{:02x}{:02x}{:02x}".format(*item["color"])
                    st.markdown(
                        f'<span style="color:{color_hex};font-size:1.2em;">&#9632;</span> '
                        f"**{item['label']}** ({item['confidence']:.0%})",
                        unsafe_allow_html=True,
                    )

            # Per-ingredient nutrition
            if (api_key or (fatsecret_id and fatsecret_secret)) and ingredient_results:
                st.write("**Per-ingredient nutrition:**")
                for ing in ingredient_results:
                    label = ing.get("label", "").rstrip(".")
                    confidence = ing.get("confidence", 0)
                    with st.expander(f"{label} ({confidence:.0%})"):
                        ing_result = lookup_ingredient_nutrition_cached(
                            label, usda_key=api_key,
                            fatsecret_id=fatsecret_id, fatsecret_secret=fatsecret_secret,
                        )
                        if ing_result and ing_result.get("nutrients"):
                            st.caption(f"Source: {ing_result['source']}")
                            for name, info in ing_result["nutrients"].items():
                                value = info.get("value", 0)
                                unit = info.get("unit", "")
                                if value > 0:
                                    st.write(
                                        f"**{name.replace('_', ' ').title()}**: {value:.1f} {unit}"
                                    )
                        else:
                            st.info("No nutrition data found for this ingredient.")
        else:
            st.info("No ingredients detected for this dish.")
    else:
        st.divider()
        st.subheader("Ingredient Detection")
        st.info("No ingredient prompts defined for this dish.")

    # --- 4. Portion Estimation ---
    st.divider()
    st.subheader("Portion Estimation")

    t0 = time.time()
    with st.spinner("Estimating portion size..."):
        depth_pipeline = load_depth_model()
        result = estimate_portion(
            image=image,
            class_name=class_name,
            depth_pipeline=depth_pipeline,
            ingredient_masks=ingredient_results if ingredient_results else None,
            reference_height_cm=reference_height_cm,
        )
    timings["Portion"] = time.time() - t0

    st.session_state.portion_estimate = result

    col_depth, col_info = st.columns([1, 1])

    with col_depth:
        depth_map = result.get("depth_map")
        if depth_map is not None:
            depth_colored = (depth_map * 255).astype(np.uint8)
            depth_image = Image.fromarray(np.stack([depth_colored] * 3, axis=-1))
            st.image(depth_image, caption="Depth Map (brighter = closer)", use_container_width=True)

            try:
                import plotly.graph_objects as go

                step = max(1, depth_map.shape[0] // 100)
                z = depth_map[::step, ::step]
                fig = go.Figure(data=[go.Surface(z=z)])
                fig.update_layout(
                    scene=dict(zaxis_title="Depth"),
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"Surface: {display_name}",
                )
                fig.update_traces(colorscale="Viridis", showscale=True)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass
        else:
            st.info("Depth visualization unavailable")

    with col_info:
        method_label = result["scaling_method"].replace("_", " ").title()
        st.metric("Estimated Weight", f"{result['estimated_weight_grams']:.0f} g")
        st.metric("Estimated Volume", f"{result['estimated_volume_ml']:.0f} mL")
        st.metric("Typical Portion", f"{result['typical_portion_grams']} g")
        st.metric("Method", method_label)
        st.metric("Food Density", f"{result['density_used']:.2f} kg/L")
        if result.get("area_ratio"):
            st.metric("Dish Area Ratio", f"{result['area_ratio']:.1%}")

        if nutrients:
            multiplier = result["nutrient_multiplier"]
            st.divider()
            st.write(f"**Adjusted nutrition for ~{result['estimated_weight_grams']:.0f}g portion:**")
            st.caption(f"(per-100g values x {multiplier:.2f})")
            for name, info in nutrients.items():
                value = info.get("value", 0) * multiplier
                unit = info.get("unit", "")
                if value > 0:
                    st.write(f"**{name.replace('_', ' ').title()}**: {value:.1f} {unit}")

    st.caption(
        "Portion estimate uses dish area ratio (via segmentation) scaled against typical portions. "
        "Depth map is for visualization only. Not a substitute for weighing food."
    )

    # ============================================================
    # TIMING SUMMARY
    # ============================================================
    total = sum(timings.values())
    st.divider()
    with st.expander("Performance", expanded=False):
        for step, duration in timings.items():
            st.write(f"**{step}**: {duration:.2f}s")
        st.write(f"**Total**: {total:.2f}s")