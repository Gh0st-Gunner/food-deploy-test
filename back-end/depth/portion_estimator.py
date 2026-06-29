import numpy as np
from PIL import Image

from depth.food_densities import get_density
from nutrition.food_mapping import TYPICAL_PORTION_GRAMS
from config import PLATE_DIAMETER_CM


def estimate_portion(
    image: Image.Image,
    class_name: str,
    depth_pipeline=None,
    ingredient_masks=None,
    reference_height_cm=None,
    sam_model=None,
    sam_processor=None,
):
    """Estimate portion size using SAM 2 area ratio (primary) + depth visualization."""
    img_w, img_h = image.size
    total_pixels = img_w * img_h

    # --- Primary method: SAM 2 area ratio ---
    # If we have ingredient masks from detection, use their combined area
    food_pixel_count = 0
    scaling_method = "typical_portion"

    if ingredient_masks:
        for ing in ingredient_masks:
            mask = ing.get("mask")
            if mask is not None:
                food_pixel_count += int(np.sum(mask))
    else:
        # Use SAM 2 to segment the whole dish
        food_pixel_count, scaling_method = _segment_dish(
            image, class_name, sam_model, sam_processor
        )

    area_ratio = min(food_pixel_count / total_pixels, 1.0) if total_pixels > 0 else 0.5

    # Scale typical portion by area ratio
    # A dish that fills the whole image is likely a full portion (ratio ~1.0)
    # A dish that fills 25% of the image might be a half portion, etc.
    typical = TYPICAL_PORTION_GRAMS.get(class_name, 300)

    # Use a power curve: small area = small portion, but with diminishing returns
    # ratio^0.7 gives a more realistic scaling than linear
    # At ratio=1.0 -> weight = typical
    # At ratio=0.5 -> weight = typical * 0.62 (not 0.5)
    # At ratio=0.25 -> weight = typical * 0.37 (not 0.25)
    portion_factor = area_ratio ** 0.7 if area_ratio > 0 else 1.0

    estimated_weight_g = typical * portion_factor
    scaling_method = "area_ratio" if ingredient_masks or food_pixel_count > 0 else scaling_method

    # Sanity bounds
    estimated_weight_g = max(estimated_weight_g, 10.0)
    if estimated_weight_g > typical * 3:
        estimated_weight_g = typical * 2.0

    # --- Depth map for visualization only ---
    depth_map = None
    if depth_pipeline is not None:
        depth_result = depth_pipeline(image)
        depth_raw = np.array(depth_result["depth"]).astype(np.float32)
        d_min = depth_raw.min()
        d_max = depth_raw.max()
        if d_max - d_min > 0:
            depth_map = (depth_raw - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_raw)

    density = get_density(class_name)
    estimated_volume_ml = estimated_weight_g / density if density > 0 else estimated_weight_g
    nutrient_multiplier = estimated_weight_g / 100.0

    return {
        "depth_map": depth_map,
        "estimated_weight_grams": round(estimated_weight_g, 1),
        "estimated_volume_ml": round(estimated_volume_ml, 1),
        "density_used": density,
        "scaling_method": scaling_method,
        "nutrient_multiplier": round(nutrient_multiplier, 2),
        "typical_portion_grams": typical,
        "area_ratio": round(area_ratio, 3),
        "ingredient_volumes": None,
    }


def _segment_dish(image, class_name, sam_model, sam_processor):
    """Use SAM 2 to segment the whole dish when ingredient masks aren't available."""
    import torch

    if sam_model is None or sam_processor is None:
        return 0, "typical_portion"

    img_w, img_h = image.size

    # Use automatic mask generation — segment center region as dish
    # Feed the dish name as a point prompt at image center
    center_x, center_y = img_w // 2, img_h // 2

    try:
        inputs = sam_processor(
            images=image,
            input_points=[[[center_x, center_y]]],
            input_labels=[[1]],
            return_tensors="pt",
        ).to(sam_model.device)

        with torch.no_grad():
            outputs = sam_model(**inputs, multimask_output=False)

        pred_mask = outputs.pred_masks.cpu().squeeze()
        if pred_mask.ndim == 3:
            pred_mask = pred_mask[0]

        mask_pil = Image.fromarray((pred_mask.numpy() * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((img_w, img_h), Image.NEAREST)
        mask_array = np.array(mask_pil) > 128

        pixel_count = int(mask_array.sum())
        return pixel_count, "sam_segmentation"

    except Exception:
        return 0, "typical_portion"