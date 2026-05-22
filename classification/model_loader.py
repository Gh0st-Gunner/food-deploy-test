import os
import glob
import json
import warnings

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort

from config import MODELS_DIR, MODELS_DIR_FALLBACK


def get_available_models():
    models_dir = MODELS_DIR if os.path.isdir(MODELS_DIR) else MODELS_DIR_FALLBACK
    os.makedirs(models_dir, exist_ok=True)
    model_files = glob.glob(os.path.join(models_dir, "*"))
    available_models = {}
    for model_path in model_files:
        name = os.path.basename(model_path)
        if name.endswith(".pth") or name.endswith(".onnx"):
            model_name = name.split(".")[0]
            available_models[model_name] = model_path
    return available_models


def load_class_names_metadata(model_path, checkpoint=None):
    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        if isinstance(class_names, list) and class_names:
            return class_names
    metadata_paths = [
        os.path.splitext(model_path)[0] + ".json",
        os.path.join(os.path.dirname(model_path), "class_names.json"),
    ]
    for metadata_path in metadata_paths:
        if not os.path.exists(metadata_path):
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            class_names = metadata.get("class_names")
        elif isinstance(metadata, list):
            class_names = metadata
        else:
            class_names = None
        if isinstance(class_names, list) and class_names:
            return class_names
    return []


@st.cache_resource
def load_onnx_model(model_path):
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        class_names = load_class_names_metadata(model_path)
        return session, input_name, class_names
    except Exception:
        return None, None, []


@st.cache_resource
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        return None, None, None, None, None

    class_names = load_class_names_metadata(checkpoint_path, checkpoint)
    if not class_names:
        return None, None, None, None, None

    num_classes = len(class_names)
    state_dict_keys = list(checkpoint["model_state_dict"].keys())

    def detect_model_from_state_dict(keys):
        if any("features." in key for key in keys):
            if "classifier.1.weight" in keys:
                try:
                    final_conv_shape = checkpoint["model_state_dict"].get("features.8.1.weight")
                    if final_conv_shape is not None and final_conv_shape.shape[0] >= 1500:
                        return "efficientnet_b3"
                except Exception:
                    pass
                return "efficientnet_b0"
            elif "classifier.3.weight" in keys:
                return "mobilenet_v3_large"
        elif any("layer1." in key for key in keys):
            return "resnet101" if any("layer4.2." in key for key in keys) else "resnet50"
        return "resnet50"

    model_name = checkpoint.get("model_name") or detect_model_from_state_dict(state_dict_keys)

    model = None
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        return None, None, None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except Exception:
            return None, None, None, None, None

    accuracy = checkpoint.get("val_acc", checkpoint.get("best_acc", "N/A"))
    model = model.to(device)
    model.eval()
    return model, class_names, device, model_name, accuracy