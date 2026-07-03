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

    orig_w, orig_h = image.size

    # --- VRAM Guard: Downscale image if either dimension exceeds 1024px ---
    MAX_SIZE = 1024
    if orig_w > MAX_SIZE or orig_h > MAX_SIZE:
        if orig_w > orig_h:
            new_w = MAX_SIZE
            new_h = int(orig_h * (MAX_SIZE / orig_w))
        else:
            new_h = MAX_SIZE
            new_w = int(orig_w * (MAX_SIZE / orig_h))
            
        try:
            resample_filter = Image.Resampling.BILINEAR
        except AttributeError:
            resample_filter = Image.BILINEAR
            
        image = image.resize((new_w, new_h), resample_filter)

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
    formatted_boxes = [[[float(b) for b in box] for box in boxes]]

    sam_inputs = sam_processor(
        images=image,
        input_boxes=formatted_boxes,
        return_tensors="pt",
    ).to(sam_model.device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs, multimask_output=False)

    # sam_outputs.pred_masks has shape [1, num_boxes, 1, H_model, W_model]
    pred_masks = sam_outputs.pred_masks.cpu()

    detected_ingredients = []
    scale_x = orig_w / image.size[0]
    scale_y = orig_h / image.size[1]

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        # Extract the mask for the i-th box
        mask_tensor = pred_masks[0, i, 0]  # shape [H_model, W_model]

        # Convert to PIL and resize to original image resolution
        mask_pil = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        mask_array = np.array(mask_pil) > 128  # binary mask at original resolution

        # Scale bounding box back to original resolution coordinates
        orig_box = [
            box[0] * scale_x,
            box[1] * scale_y,
            box[2] * scale_x,
            box[3] * scale_y
        ]

        detected_ingredients.append({
            "label": str(label),
            "confidence": float(score),
            "bbox": orig_box,
            "mask": mask_array,
            "mask_pixel_count": int(mask_array.sum()),
        })

    return detected_ingredients