import streamlit as st

from nutrition.food_mapping import USDA_SEARCH_TERMS, INGREDIENT_PROMPTS


def display_nutrition_table(nutrients: dict, title: str = "Nutrition (per 100g)"):
    if not nutrients:
        st.info("No nutrition data available for this dish.")
        return

    st.subheader(title)
    for name, info in nutrients.items():
        value = info.get("value", 0)
        unit = info.get("unit", "")
        if value > 0:
            display_name = name.replace("_", " ").title()
            st.write(f"**{display_name}**: {value:.1f} {unit}")


def display_per_ingredient_nutrition(ingredient_nutrition: list):
    if not ingredient_nutrition:
        return

    st.subheader("Per-Ingredient Nutrition")
    for item in ingredient_nutrition:
        label = item.get("label", "Unknown")
        nutrients = item.get("nutrients", {})
        with st.expander(f"{label}"):
            if nutrients:
                for name, info in nutrients.items():
                    value = info.get("value", 0)
                    unit = info.get("unit", "")
                    if value > 0:
                        st.write(
                            f"**{name.replace('_', ' ').title()}**: {value:.1f} {unit}"
                        )
            else:
                st.info("No USDA data found for this ingredient.")


def format_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").title()