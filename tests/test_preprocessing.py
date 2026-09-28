import io
from PIL import Image
import numpy as np
from werkzeug.datastructures import FileStorage

from utils.validation import validate_image_upload, is_allowed_file
from utils.preprocessing import ensure_rgb, resize_image, preprocess_pil_image
from utils.dataset_loader import scan_dataset_directory
from utils.helpers import generate_safe_unique_filename
import config

def create_dummy_image(size=(100, 100), color="red", format="JPEG"):
    """Helper to create an in-memory image FileStorage object."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return FileStorage(stream=buf, filename=f"test_sample.{format.lower()}")

def test_allowed_extensions():
    assert is_allowed_file("photo.jpg") is True
    assert is_allowed_file("skin.PNG") is True
    assert is_allowed_file("lesion.webp") is True
    assert is_allowed_file("document.pdf") is False
    assert is_allowed_file("script.py") is False

def test_validate_image_upload_valid():
    file = create_dummy_image((300, 300))
    is_valid, msg, pil_img = validate_image_upload(file)
    assert is_valid is True
    assert msg == ""
    assert pil_img is not None
    assert pil_img.size == (300, 300)

def test_validate_image_upload_too_small():
    file = create_dummy_image((20, 20))
    is_valid, msg, pil_img = validate_image_upload(file)
    assert is_valid is False
    assert "resolution too low" in msg.lower()

def test_preprocess_pil_image():
    img = Image.new("RGBA", (400, 400), color=(255, 0, 0, 255))
    tensor = preprocess_pil_image(img, target_size=(224, 224), normalize=True)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0
    assert tensor.min() >= 0.0

def test_dataset_scanner():
    report = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    assert report["exists"] is True
    assert "train" in report["splits"]
    assert "validation" in report["splits"]
    assert "test" in report["splits"]

def test_safe_unique_filename():
    fname = generate_safe_unique_filename("sample skin image.JPG")
    assert fname.endswith(".jpg")
    assert " " not in fname
    assert len(fname) > 10
