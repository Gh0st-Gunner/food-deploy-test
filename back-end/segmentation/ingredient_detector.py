import torch
import numpy as np
from PIL import Image

from nutrition.food_mapping import INGREDIENT_PROMPTS
from config import DEFAULT_BOX_THRESHOLD, DEFAULT_TEXT_THRESHOLD


def detect_ingredients(
    image: Image.Image,
    class_name: str,
    grounding_model,
    grounding_processor,
    sam_model,
    sam_processor,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
):
    prompts = INGREDIENT_PROMPTS.get(class_name, [])
    if not prompts:
        return []

    # Grounding DINO: detect bounding boxes from text prompts
    text_input = " ".join(prompts)
    inputs = grounding_processor(
        images=image, text=[text_input], return_tensors="pt"
    ).to(grounding_model.device)

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu().tolist()
    scores = results["scores"].cpu().tolist()
    labels = results["labels"]

    if not boxes:
        return []

    # Convert string labels to list
    if isinstance(labels, str):
        labels = [labels] * len(boxes)
    elif hasattr(labels, "tolist"):
        labels = labels.tolist()

    # SAM 2: segment all detected boxes in batch
    # Format input_boxes as [1, num_boxes, 4] for batch processing
    all_boxes = torch.tensor(boxes, dtype=torch.float32).unsqueeze(0)  # [1, N, 4]

    sam_inputs = sam_processor(
        images=image,
        input_boxes=[[b] for b in boxes],
        return_tensors="pt",
    ).to(sam_model.device)

    orig_w, orig_h = image.size

    detected_ingredients = []

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Process one box at a time to avoid batch issues
        single_box = torch.tensor([[box]], dtype=torch.float32)

        single_inputs = sam_processor(
            images=image,
            input_boxes=[[[float(b) for b in box]]],
            return_tensors="pt",
        ).to(sam_model.device)

        with torch.no_grad():
            single_outputs = sam_model(**single_inputs, multimask_output=False)

        # Manually resize mask to original image size
        pred_mask = single_outputs.pred_masks.cpu().squeeze()  # [H, W] or [1, H, W]

        if pred_mask.ndim == 3:
            pred_mask = pred_mask[0]

        # Resize mask to original image size using PIL
        mask_pil = Image.fromarray((pred_mask.numpy() * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        mask_array = np.array(mask_pil) > 128  # binary mask at original resolution

        detected_ingredients.append({
            "label": str(label),
            "confidence": float(score),
            "bbox": [float(b) for b in box],
            "mask": mask_array,
            "mask_pixel_count": int(mask_array.sum()),
        })

    return detected_ingredients