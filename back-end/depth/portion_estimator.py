import numpy as np
from PIL import Image
import cv2

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
    
    # --- VRAM Guard: Downscale image if either dimension exceeds 1024px ---
    MAX_SIZE = 1024
    if img_w > MAX_SIZE or img_h > MAX_SIZE:
        if img_w > img_h:
            new_w = MAX_SIZE
            new_h = int(img_h * (MAX_SIZE / img_w))
        else:
            new_h = MAX_SIZE
            new_w = int(img_w * (MAX_SIZE / img_h))
            
        try:
            resample_filter = Image.Resampling.BILINEAR
        except AttributeError:
            resample_filter = Image.BILINEAR
            
        image = image.resize((new_w, new_h), resample_filter)
        img_w, img_h = image.size

    total_pixels = img_w * img_h

    # --- 1. Segment the dish / food mask ---
    dish_mask = None
    scaling_method = "typical_portion"

    if ingredient_masks:
        # If we have ingredient masks, we combine them to form the food/dish mask
        combined_mask = np.zeros((img_h, img_w), dtype=bool)
        for ing in ingredient_masks:
            mask = ing.get("mask")
            if mask is not None:
                # Ensure mask is boolean and correct size
                if mask.shape != (img_h, img_w):
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((img_w, img_h), Image.NEAREST)
                    mask = np.array(mask_pil) > 128
                combined_mask = combined_mask | mask
        if np.sum(combined_mask) > 0:
            dish_mask = combined_mask
            scaling_method = "area_ratio"
    else:
        # Use SAM 2 to segment the whole dish
        dish_mask, scaling_method = _segment_dish(
            image, class_name, sam_model, sam_processor
        )

    # --- 2. Fit Ellipse to Plate & Calculate Camera Tilt / Physical Scale ---
    fit_success = False
    major_axis = 0.0
    minor_axis = 0.0
    tilt_ratio = 1.0
    measured_diameter_px = 0.0
    physical_scale_cm_per_px = None
    food_pixel_count = 0

    if dish_mask is not None and np.sum(dish_mask) > 0:
        food_pixel_count = int(np.sum(dish_mask))
        try:
            # Convert mask to CV_8UC1
            mask_uint8 = (dish_mask.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    (xc, yc), (d1, d2), angle = ellipse
                    major_axis = max(d1, d2)
                    minor_axis = min(d1, d2)
                    
                    if minor_axis > 0:
                        tilt_ratio = major_axis / minor_axis
                        # Cap tilt_ratio at 2.5 (about 66 degrees camera tilt) to prevent extreme scaling
                        tilt_ratio = min(max(tilt_ratio, 1.0), 2.5)
                        
                        image_diagonal = np.sqrt(img_w**2 + img_h**2)
                        if major_axis > 1.2 * image_diagonal or major_axis < 0.1 * max(img_w, img_h):
                            fit_success = False
                            print(f"Plate ellipse fitting discarded as anomaly: major_axis={major_axis:.1f}px (limit: {1.2 * image_diagonal:.1f}px)")
                        else:
                            fit_success = True
                            measured_diameter_px = major_axis
        except Exception as e:
            print(f"Plate ellipse fitting failed: {e}")

    plate_diameter = reference_height_cm or PLATE_DIAMETER_CM or 25.0

    if fit_success and measured_diameter_px > 0:
        physical_scale_cm_per_px = plate_diameter / measured_diameter_px
        actual_area_cm2 = food_pixel_count * (physical_scale_cm_per_px ** 2) * tilt_ratio
    else:
        # Fallback if ellipse fit failed: use typical area ratio scale
        area_ratio = min(food_pixel_count / total_pixels, 1.0) if total_pixels > 0 else 0.5
        standard_plate_area = np.pi * ((plate_diameter / 2) ** 2)
        actual_area_cm2 = area_ratio * standard_plate_area

    # --- 3. Estimate Physical Height from Depth Map ---
    depth_map = None
    avg_height_cm = 1.5  # default fallback average food height in cm
    
    if depth_pipeline is not None:
        try:
            depth_result = depth_pipeline(image)
            depth_raw = np.array(depth_result["depth"]).astype(np.float32)
            d_min = depth_raw.min()
            d_max = depth_raw.max()
            if d_max - d_min > 0:
                depth_map = (depth_raw - d_min) / (d_max - d_min)
            else:
                depth_map = np.zeros_like(depth_raw)
                
            # Estimate height if we have a mask and depth map
            if dish_mask is not None and food_pixel_count > 0 and depth_map is not None:
                # Resize depth map to match image/mask dimensions if necessary
                if depth_map.shape != (img_h, img_w):
                    depth_map = cv2.resize(depth_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                
                food_depths = depth_map[dish_mask]
                
                # Find base plate depth from boundary of dish mask
                kernel = np.ones((5, 5), np.uint8)
                eroded_mask = cv2.erode(dish_mask.astype(np.uint8), kernel, iterations=1)
                boundary_mask = (dish_mask.astype(np.uint8) - eroded_mask) > 0
                
                if np.sum(boundary_mask) > 0:
                    base_depth = np.median(depth_map[boundary_mask])
                else:
                    base_depth = np.percentile(food_depths, 10)
                
                # Relative heights above the plate (larger values mean closer/higher)
                heights_rel = np.maximum(food_depths - base_depth, 0.0)
                
                # Map relative depth range to physical cm
                if physical_scale_cm_per_px is not None:
                    # Height scale is proportional to physical plate size
                    depth_to_cm_factor = plate_diameter * 0.3
                else:
                    depth_to_cm_factor = 8.0  # standard 8cm height range fallback
                
                food_heights_cm = heights_rel * depth_to_cm_factor
                avg_height_cm = float(np.mean(food_heights_cm))
                # Bound avg_height_cm to realistic food heights: 0.5cm to 8.0cm
                avg_height_cm = min(max(avg_height_cm, 0.5), 8.0)
                
        except Exception as e:
            print(f"Depth height estimation failed: {e}")

    # --- 4. Blended Weight Calculation & Safety Bounds ---
    typical = TYPICAL_PORTION_GRAMS.get(class_name, 300)
    density = get_density(class_name)

    # Volume-based estimation: Area * Height (1 cm^3 = 1 ml)
    estimated_volume_ml = actual_area_cm2 * avg_height_cm
    weight_by_volume = estimated_volume_ml * (density if density > 0 else 1.0)

    # Heuristic area-ratio power-curve (original method)
    area_ratio = min(food_pixel_count / total_pixels, 1.0) if total_pixels > 0 else 0.5
    portion_factor = area_ratio ** 0.7 if area_ratio > 0 else 1.0
    weight_by_area_ratio = typical * portion_factor

    # Blend or choose method
    if fit_success and depth_map is not None:
        estimated_weight_g = weight_by_volume
        scaling_method = "3d_volume_estimation"
    else:
        estimated_weight_g = weight_by_area_ratio
        if scaling_method == "typical_portion" and food_pixel_count > 0:
            scaling_method = "area_ratio"

    # Safety bounds relative to typical portion to avoid extreme anomalies
    min_weight = max(typical * 0.15, 15.0)
    max_weight = typical * 2.5
    estimated_weight_g = min(max(estimated_weight_g, min_weight), max_weight)

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
        "tilt_ratio": round(tilt_ratio, 2),
        "average_height_cm": round(avg_height_cm, 2),
        "plate_diameter_cm": plate_diameter,
        "measured_diameter_px": round(measured_diameter_px, 1) if fit_success else None,
        "ingredient_volumes": None,
    }


def _segment_dish(image, class_name, sam_model, sam_processor):
    """Use SAM 2 to segment the whole dish when ingredient masks aren't available."""
    import torch

    if sam_model is None or sam_processor is None:
        return None, "typical_portion"

    img_w, img_h = image.size
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

        return mask_array, "sam_segmentation"

    except Exception:
        return None, "typical_portion"