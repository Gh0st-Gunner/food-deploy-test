from PIL import Image
import numpy as np


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    return Image.fromarray(array)