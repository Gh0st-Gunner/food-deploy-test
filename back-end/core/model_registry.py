import os
import glob
import json
import warnings
import threading
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from core.settings import get_settings

logger = logging.getLogger(__name__)


def _detect_model_from_state_dict(keys, state_dict):
    """Detect model architecture from checkpoint state dict keys."""
    if any("features." in key for key in keys):
        if "classifier.1.weight" in keys:
            try:
                final_conv_shape = state_dict.get("features.8.1.weight")
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


def _load_class_names(model_path, checkpoint=None):
    """Load class names from checkpoint metadata or sidecar JSON file."""
    if isinstance(checkpoint, dict):
        class_names = checkpoint.get("class_names")
        if isinstance(class_names, list) and class_names:
            return class_names

    for metadata_path in [
        os.path.splitext(model_path)[0] + ".json",
        os.path.join(os.path.dirname(model_path), "class_names.json"),
    ]:
        if not os.path.exists(metadata_path):
            continue
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            class_names = metadata.get("class_names") if isinstance(metadata, dict) else metadata
            if isinstance(class_names, list) and class_names:
                return class_names
        except (OSError, json.JSONDecodeError):
            continue
    return []


class ModelRegistry:
    """Singleton that loads and caches ML models for the worker process lifetime.

    Thread-safe lazy initialization. Models are loaded on first access
    and kept in memory for the process lifetime.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._models = {}
                    instance._loaded = set()
                    instance._settings = get_settings()
                    cls._instance = instance
        return cls._instance

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Classification models ---

    def list_classification_models(self) -> dict:
        """Return {model_name: model_path} for all available .pth and .onnx files."""
        models_dir = self._settings.models_dir
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        available = {}
        for path in glob.glob(os.path.join(models_dir, "*")):
            name = os.path.basename(path)
            if name.endswith(".pth") or name.endswith(".onnx"):
                model_name = name.split(".")[0]
                available[model_name] = path
        return available

    def get_classifier(self, model_name: str):
        """Load and cache a PyTorch classification model.

        Returns: (model, class_names, device, model_name, accuracy)
        """
        cache_key = f"cls_{model_name}"
        if cache_key in self._models:
            return self._models[cache_key]

        available = self.list_classification_models()
        model_path = available.get(model_name)
        if not model_path:
            raise FileNotFoundError(f"Model not found: {model_name}")

        device = self.device
        checkpoint = torch.load(model_path, map_location=device)

        class_names = _load_class_names(model_path, checkpoint)
        if not class_names:
            raise ValueError(f"No class names found for model: {model_name}")

        num_classes = len(class_names)
        state_dict = checkpoint["model_state_dict"]
        state_dict_keys = list(state_dict.keys())

        arch = checkpoint.get("model_name") or _detect_model_from_state_dict(state_dict_keys, state_dict)

        model = None
        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "resnet101":
            model = models.resnet101(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif arch == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif arch == "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=None)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.load_state_dict(state_dict, strict=False)

        accuracy = checkpoint.get("val_acc", checkpoint.get("best_acc", "N/A"))
        model = model.to(device)
        model.eval()

        result = (model, class_names, device, arch, accuracy)
        self._models[cache_key] = result
        self._loaded.add(cache_key)
        logger.info("Loaded classification model: %s (arch: %s)", model_name, arch)
        return result

    def get_onnx_model(self, model_name: str):
        """Load and cache an ONNX classification model.

        Returns: (session, input_name, class_names)
        """
        import onnxruntime as ort

        cache_key = f"onnx_{model_name}"
        if cache_key in self._models:
            return self._models[cache_key]

        available = self.list_classification_models()
        model_path = available.get(model_name)
        if not model_path:
            raise FileNotFoundError(f"ONNX model not found: {model_name}")

        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        class_names = _load_class_names(model_path)

        result = (session, input_name, class_names)
        self._models[cache_key] = result
        self._loaded.add(cache_key)
        logger.info("Loaded ONNX model: %s", model_name)
        return result

    # --- Segmentation models ---

    def get_grounding_dino(self):
        """Load and cache Grounding DINO model + processor."""
        cache_key = "grounding_dino"
        if cache_key in self._models:
            return self._models[cache_key]

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        processor = AutoProcessor.from_pretrained(self._settings.grounding_dino_model)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._settings.grounding_dino_model, device_map="auto"
        )

        result = (model, processor)
        self._models[cache_key] = result
        self._loaded.add(cache_key)
        logger.info("Loaded Grounding DINO model")
        return result

    def get_sam2(self):
        """Load and cache SAM 2 model + processor."""
        cache_key = "sam2"
        if cache_key in self._models:
            return self._models[cache_key]

        from transformers import Sam2Processor, Sam2Model

        processor = Sam2Processor.from_pretrained(self._settings.sam2_model)
        model = Sam2Model.from_pretrained(self._settings.sam2_model, device_map="auto")

        result = (model, processor)
        self._models[cache_key] = result
        self._loaded.add(cache_key)
        logger.info("Loaded SAM 2 model")
        return result

    # --- Depth model ---

    def get_depth_model(self):
        """Load and cache Depth Anything V2 pipeline."""
        cache_key = "depth"
        if cache_key in self._models:
            return self._models[cache_key]

        from transformers import pipeline as hf_pipeline

        pipeline = hf_pipeline(
            task="depth-estimation",
            model=self._settings.depth_model,
        )

        self._models[cache_key] = pipeline
        self._loaded.add(cache_key)
        logger.info("Loaded Depth Anything V2 model")
        return pipeline

    # --- Preloading ---

    def preload(self, *model_groups: str):
        """Preload model groups at worker startup.

        Args:
            model_groups: One or more of "classification", "dino", "sam2", "depth"
        """
        for group in model_groups:
            try:
                if group == "classification":
                    for name in self.list_classification_models():
                        path = self.list_classification_models()[name]
                        if path.endswith(".onnx"):
                            self.get_onnx_model(name)
                        else:
                            self.get_classifier(name)
                elif group == "dino":
                    self.get_grounding_dino()
                elif group == "sam2":
                    self.get_sam2()
                elif group == "depth":
                    self.get_depth_model()
            except Exception as e:
                logger.error("Failed to preload %s: %s", group, e)