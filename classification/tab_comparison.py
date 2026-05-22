import streamlit as st
import torch
import pandas as pd

from classification.model_loader import (
    get_available_models,
    load_model,
    load_onnx_model,
)


def tab_comparison():
    st.header("Model Comparison")
    st.markdown("Compare performance metrics across all available models")

    available_models = get_available_models()

    if not available_models:
        st.error("No models found!")
        return

    st.info(f"Found {len(available_models)} model(s)")

    comparison_data = []

    for model_name, model_path in available_models.items():
        with st.spinner(f"Loading {model_name}..."):
            if model_path.endswith(".onnx"):
                session, input_name, class_names = load_onnx_model(model_path)
                if session is not None:
                    comparison_data.append({
                        "Model": model_name.upper(),
                        "Architecture": "ONNX",
                        "Accuracy": 0,
                        "F1-Score": "N/A",
                        "Precision": "N/A",
                        "Recall": "N/A",
                        "Classes": len(class_names),
                        "Checkpoint": model_path,
                    })
            else:
                model, class_names, device, detected_name, accuracy = load_model(model_path)
                if model is not None:
                    try:
                        checkpoint = torch.load(model_path, map_location="cpu")
                        f1_score = checkpoint.get("f1_score", "N/A")
                        precision = checkpoint.get("precision", "N/A")
                        recall = checkpoint.get("recall", "N/A")
                    except Exception:
                        f1_score = precision = recall = "N/A"

                    comparison_data.append({
                        "Model": model_name.upper(),
                        "Architecture": detected_name,
                        "Accuracy": (
                            accuracy
                            if isinstance(accuracy, (int, float))
                            else float(accuracy)
                            if accuracy != "N/A"
                            else 0
                        ),
                        "F1-Score": (
                            f1_score if isinstance(f1_score, str) else f1_score * 100
                        ),
                        "Precision": (
                            precision if isinstance(precision, str) else precision * 100
                        ),
                        "Recall": recall if isinstance(recall, str) else recall * 100,
                        "Classes": len(class_names),
                        "Checkpoint": model_path,
                    })

    if comparison_data:
        df = pd.DataFrame(comparison_data)

        st.subheader("Model Metrics")
        metrics_cols = [
            "Model", "Architecture", "Accuracy", "F1-Score", "Precision", "Recall", "Classes"
        ]
        display_df = df[metrics_cols].copy()

        for col in ["Accuracy", "F1-Score", "Precision", "Recall"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                )

        st.dataframe(display_df, use_container_width=True)

        if len(df) > 0 and "Accuracy" in df.columns:
            best_idx = df["Accuracy"].idxmax()
            best_model = df.loc[best_idx]
            st.success(
                f"Best Model: **{best_model['Model']}** with "
                f"{best_model['Accuracy']:.2f}% accuracy"
            )

        col1, _ = st.columns(2)

        with col1:
            if len(df) > 1:
                st.subheader("Accuracy Comparison")
                chart_data = df.set_index("Model")["Accuracy"].sort_values(ascending=True)
                st.bar_chart(chart_data)

        st.divider()
        st.subheader("Model Details")

        for idx, row in df.iterrows():
            with st.expander(f"{row['Model']} Details"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Architecture", row["Architecture"])
                    accuracy_val = (
                        f"{row['Accuracy']:.2f}%"
                        if isinstance(row["Accuracy"], (int, float))
                        else row["Accuracy"]
                    )
                    st.metric("Accuracy", accuracy_val)
                with col2:
                    f1_val = (
                        f"{row['F1-Score']:.2f}%"
                        if isinstance(row["F1-Score"], (int, float))
                        else row["F1-Score"]
                    )
                    st.metric("F1-Score", f1_val)
                    precision_val = (
                        f"{row['Precision']:.2f}%"
                        if isinstance(row["Precision"], (int, float))
                        else row["Precision"]
                    )
                    st.metric("Precision", precision_val)
                with col3:
                    recall_val = (
                        f"{row['Recall']:.2f}%"
                        if isinstance(row["Recall"], (int, float))
                        else row["Recall"]
                    )
                    st.metric("Recall", recall_val)
                    st.metric("Classes", row["Classes"])
    else:
        st.error("Failed to load model information")