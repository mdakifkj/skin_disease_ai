from typing import Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
import config

def ensure_rgb(image: Image.Image) -> Image.Image:
    """Converts RGBA, Grayscale (L), or Palette (P) images to standard 3-channel RGB."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image

def resize_image(image: Image.Image, target_size: Tuple[int, int] = config.IMAGE_SIZE) -> Image.Image:
    """Resizes an image using high-quality resampling (LANCZOS/BICUBIC)."""
    return image.resize(target_size, Image.Resampling.LANCZOS)

def preprocess_pil_image(
    image: Image.Image,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Takes a PIL Image and preprocesses it for deep learning inference:
    1. Converts to RGB.
    2. Resizes to model input target size (e.g., 224x224).
    3. Converts to float32 NumPy array.
    4. Applies normalization (scale [0, 1] or raw [0, 255] float32 suitable for EfficientNet).
    5. Adds batch dimension: returns shape (1, height, width, 3).
    """
    rgb_img = ensure_rgb(image)
    resized_img = resize_image(rgb_img, target_size)
    img_array = np.array(resized_img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
        
    # Expand dims for batch: (1, 224, 224, 3)
    batch_tensor = np.expand_dims(img_array, axis=0)
    return batch_tensor

def preprocess_image_from_path(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    normalize: bool = True
) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads an image from filesystem, preprocesses it for model inference,
    and returns both the preprocessed tensor and the original PIL image.
    """
    img = Image.open(image_path)
    img.load()
    tensor = preprocess_pil_image(img, target_size=target_size, normalize=normalize)
    return tensor, img
