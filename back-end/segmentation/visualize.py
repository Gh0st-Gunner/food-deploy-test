import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (255, 87, 87),   # red
    (87, 199, 133),   # green
    (87, 133, 255),   # blue
    (255, 199, 87),   # yellow
    (199, 87, 255),   # purple
    (87, 255, 223),   # cyan
    (255, 159, 87),   # orange
    (159, 87, 255),   # violet
    (87, 199, 255),   # light blue
    (255, 87, 159),   # pink
    (133, 255, 87),   # lime
    (255, 255, 87),   # light yellow
]


def overlay_ingredients(
    image: Image.Image,
    ingredients: list,
    alpha: float = 0.35,
):
    overlay = image.copy().convert("RGBA")
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for i, ing in enumerate(ingredients):
        color = PALETTE[i % len(PALETTE)]
        mask = ing.get("mask")

        if mask is None:
            continue

        # Ensure mask matches image size
        if mask.shape != (image.height, image.width):
            continue

        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask] = (*color, int(alpha * 255))
        mask_pil = Image.fromarray(mask_rgba, "RGBA")
        mask_layer = Image.alpha_composite(mask_layer, mask_pil)

    result = Image.alpha_composite(overlay, mask_layer)

    # Draw bounding boxes and labels
    draw = ImageDraw.Draw(result)
    for i, ing in enumerate(ingredients):
        color = PALETTE[i % len(PALETTE)]
        bbox = ing.get("bbox")
        label = ing.get("label", "").rstrip(".")
        confidence = ing.get("confidence", 0)

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label} ({confidence:.0%})"
        draw.text((x1, max(0, y1 - 15)), text, fill=color)

    return result.convert("RGB")


def format_ingredients_legend(ingredients: list) -> list:
    legend = []
    for i, ing in enumerate(ingredients):
        color = PALETTE[i % len(PALETTE)]
        label = ing.get("label", "").rstrip(".")
        confidence = ing.get("confidence", 0)
        pixel_count = ing.get("mask_pixel_count", 0)
        legend.append({
            "color": color,
            "label": label,
            "confidence": confidence,
            "pixel_count": pixel_count,
            "index": i,
        })
    return legend