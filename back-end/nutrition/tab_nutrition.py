import streamlit as st
from PIL import Image

from config import get_usda_api_key, DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD
from classification.model_loader import get_available_models, load_model, load_onnx_model
from classification.predict import predict, predict_onnx
from nutrition.food_mapping import (
    USDA_SEARCH_TERMS,
    INGREDIENT_PROMPTS,
)
from nutrition.usda_client import USDAClient
from nutrition.nutrition_display import display_nutrition_table, format_class_name
from segmentation.grounding_dino_loader import load_grounding_dino
from segmentation.sam2_loader import load_sam2
from segmentation.ingredient_detector import detect_ingredients
from segmentation.visualize import overlay_ingredients, format_ingredients_legend


def tab_nutrition():
    st.header("Nutrition & Ingredients")
    st.markdown("Classify a food image, detect ingredients, and view nutritional information")

    # Get classification from session state or allow direct upload
    has_classification = "last_classification" in st.session_state
    class_name = None
    image = None

    if has_classification:
        last = st.session_state.last_classification
        class_name = last.get("class_name")
        image = last.get("image")
        st.info(
            f"Using classification from Tab 1: **{format_class_name(class_name)}** "
            f"({last.get('confidence', 0) * 100:.1f}% confidence)"
        )

    # Image input (override or new upload)
    st.subheader("Input Image")
    use_existing = has_classification and st.checkbox(
        "Use classified image", value=True if has_classification else False
    )

    if not use_existing:
        uploaded_file = st.file_uploader(
            "Upload a food image", type=["jpg", "jpeg", "png"], key="nutrition_upload"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

        # Quick classify if new image
        if image and not class_name:
            available_models = get_available_models()
            if available_models:
                model_name, model_path = next(iter(available_models.items()))
                with st.spinner("Classifying..."):
                    if model_path.endswith(".onnx"):
                        session, input_name, cn = load_onnx_model(model_path)
                        if session and cn:
                            preds = predict_onnx(image, session, input_name, cn, top_k=1)
                            class_name = preds[0]["class"]
                    else:
                        model, cn, device, _, _ = load_model(model_path)
                        if model:
                            preds = predict(image, model, cn, device, top_k=1)
                            class_name = preds[0]["class"]

    if not image or not class_name:
        st.warning("Please classify an image in the Classification tab or upload one here.")
        return

    # Display classified food name
    display_name = format_class_name(class_name)
    st.subheader(f"Classified: {display_name}")

    # --- Ingredient Detection Section ---
    ingredients_list = INGREDIENT_PROMPTS.get(class_name, [])

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.image(image, caption="Input Image", use_container_width=True)

        # Show known ingredients
        if ingredients_list:
            st.write("**Known ingredients:**")
            for ing in ingredients_list:
                st.write(f"- {ing.rstrip('.')}")

        # Ingredient detection with Grounding DINO + SAM 2
        if ingredients_list:
            st.divider()
            st.subheader("Detect Ingredients")

            box_threshold = st.slider(
                "Detection sensitivity",
                min_value=0.1,
                max_value=0.9,
                value=DEFAULT_BOX_THRESHOLD,
                step=0.05,
                help="Lower = more detections, Higher = more selective",
            )

            if st.button("Run Ingredient Detection", type="primary", key="run_detection"):
                with st.spinner("Loading Grounding DINO model (first run downloads ~660MB)..."):
                    grounding_model, grounding_processor = load_grounding_dino()

                with st.spinner("Loading SAM 2 model (first run downloads ~184MB)..."):
                    sam_model, sam_processor = load_sam2()

                with st.spinner("Detecting and segmenting ingredients..."):
                    ingredient_results = detect_ingredients(
                        image=image,
                        class_name=class_name,
                        grounding_model=grounding_model,
                        grounding_processor=grounding_processor,
                        sam_model=sam_model,
                        sam_processor=sam_processor,
                        box_threshold=box_threshold,
                    )

                if ingredient_results:
                    st.session_state.ingredient_results = ingredient_results

                    # Show overlay image
                    overlay_image = overlay_ingredients(image, ingredient_results)
                    st.image(overlay_image, caption="Detected Ingredients", use_container_width=True)

                    # Show legend
                    legend = format_ingredients_legend(ingredient_results)
                    st.write("**Detected Ingredients:**")
                    for item in legend:
                        color_hex = "#{:02x}{:02x}{:02x}".format(*item["color"])
                        st.markdown(
                            f'<span style="color:{color_hex};">█</span> '
                            f"**{item['label']}** ({item['confidence']:.0%})",
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("No ingredients detected. Try lowering the detection sensitivity.")

    with col_right:
        # USDA Nutrition lookup
        api_key = get_usda_api_key()

        if not api_key:
            api_key = st.text_input(
                "USDA API Key",
                type="password",
                help="Get a free key at https://fdc.nal.usda.gov/api-key.html",
                key="usda_key_input",
            )

        if api_key:
            client = USDAClient(api_key)

            with st.spinner("Fetching nutrition data..."):
                food_data = client.search_with_fallback(class_name, USDA_SEARCH_TERMS)

            if food_data:
                nutrients = client.get_nutrients(food_data)
                st.write(f"**USDA Match:** {food_data.get('description', 'Unknown')}")
                st.caption(f"FDC ID: {food_data.get('fdcId', 'N/A')}")
                display_nutrition_table(nutrients)

                # Per-ingredient nutrition
                if "ingredient_results" in st.session_state:
                    ingredient_results = st.session_state.ingredient_results
                    st.divider()
                    st.subheader("Per-Ingredient Nutrition")

                    for ing in ingredient_results:
                        label = ing.get("label", "").rstrip(".")
                        with st.expander(f"{label} ({ing.get('confidence', 0):.0%})"):
                            # Try searching USDA for the individual ingredient
                            ing_data = client.search_food(label)
                            if ing_data:
                                ing_nutrients = client.get_nutrients(ing_data)
                                for name, info in ing_nutrients.items():
                                    value = info.get("value", 0)
                                    unit = info.get("unit", "")
                                    if value > 0:
                                        display = name.replace("_", " ").title()
                                        st.write(f"**{display}**: {value:.1f} {unit}")
                            else:
                                st.info("No USDA data found for this ingredient.")
            else:
                st.warning(
                    "No USDA data found for this dish. "
                    "Try searching manually at [FoodData Central](https://fdc.nal.usda.gov/)."
                )
        else:
            st.info(
                "Enter your USDA API key to view nutrition data. "
                "Get a free key at [FoodData Central](https://fdc.nal.usda.gov/api-key.html)."
            )