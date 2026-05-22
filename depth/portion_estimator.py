import numpy as np
from PIL import Image

from depth.food_densities import get_density
from nutrition.food_mapping import TYPICAL_PORTION_GRAMS
from config import PLATE_DIAMETER_CM


def estimate_portion(
    image: Image.Image,
    class_name: str,
    depth_pipeline,
    ingredient_masks=None,
    reference_height_cm=None,
):
    # Generate depth map
    depth_result = depth_pipeline(image)
    depth_map = np.array(depth_result["depth"]).astype(np.float32)

    # Normalize to 0-1
    d_min = depth_map.min()
    d_max = depth_map.max()
    if d_max - d_min > 0:
        depth_map = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_map = np.zeros_like(depth_map)

    # Determine real-world scale
    img_h, img_w = depth_map.shape

    if reference_height_cm and reference_height_cm > 0:
        scale_cm_per_pixel = reference_height_cm / img_h
        scaling_method = "reference"
    else:
        # Assume standard plate diameter across image width
        scale_cm_per_pixel = PLATE_DIAMETER_CM / img_w
        scaling_method = "plate_assumption"

    # Compute volume
    # Invert depth: closer objects (food) have lower depth values
    # Use (1 - depth) as approximate height of food
    pixel_area_cm2 = scale_cm_per_pixel ** 2
    food_height = (1.0 - depth_map) * scale_cm_per_pixel

    total_volume_cm3 = float(np.sum(food_height * pixel_area_cm2))
    total_volume_ml = total_volume_cm3  # 1 cm^3 = 1 mL

    # If we have ingredient masks, also compute per-ingredient volume
    ingredient_volumes = None
    if ingredient_masks:
        ingredient_volumes = []
        for ing in ingredient_masks:
            mask = ing.get("mask")
            if mask is not None and mask.shape == (img_h, img_w):
                ing_volume = float(np.sum(food_height[mask] * pixel_area_cm2))
                ingredient_volumes.append(ing_volume)
            else:
                ingredient_volumes.append(0.0)

    # Apply density to get weight
    density = get_density(class_name)
    estimated_weight_g = total_volume_ml * density

    # Sanity clamp: if wildly off typical portion, cap at 2x
    typical = TYPICAL_PORTION_GRAMS.get(class_name, 300)
    if estimated_weight_g > typical * 5:
        estimated_weight_g = typical * 2.0

    # Clamp minimum
    estimated_weight_g = max(estimated_weight_g, 10.0)

    nutrient_multiplier = estimated_weight_g / 100.0

    return {
        "depth_map": depth_map,
        "estimated_weight_grams": round(estimated_weight_g, 1),
        "estimated_volume_ml": round(total_volume_ml, 1),
        "density_used": density,
        "scaling_method": scaling_method,
        "nutrient_multiplier": round(nutrient_multiplier, 2),
        "typical_portion_grams": typical,
        "ingredient_volumes": ingredient_volumes,
    }