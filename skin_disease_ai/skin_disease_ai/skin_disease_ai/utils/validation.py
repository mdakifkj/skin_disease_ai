import os
from io import BytesIO
from typing import Tuple, Optional
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
import config

def is_allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS

def validate_image_upload(file: FileStorage) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates an uploaded image file:
    1. Checks if file object is present.
    2. Checks filename and allowed extensions.
    3. Verifies file size within maximum upload limits.
    4. Opens and decodes image using PIL to verify it is not corrupted.
    5. Verifies image dimensions are within valid boundaries.
    
    Returns:
        (is_valid, error_message, PIL.Image instance or None)
    """
    if file is None or file.filename == "":
        return False, "No file was selected for upload. Please choose an image file.", None

    filename = file.filename
    if not is_allowed_file(filename):
        allowed_str = ", ".join(config.ALLOWED_EXTENSIONS)
        return False, f"Unsupported file format. Allowed formats: {allowed_str.upper()}.", None

    # Read bytes to check file content and size safely
    file_bytes = file.read()
    file.seek(0)  # Reset pointer for subsequent reads/saves

    if len(file_bytes) == 0:
        return False, "The uploaded file is empty.", None

    if len(file_bytes) > config.MAX_CONTENT_LENGTH:
        max_mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        return False, f"File size exceeds maximum allowed limit of {max_mb} MB.", None

    # Image corruption check
    try:
        img = Image.open(BytesIO(file_bytes))
        img.verify()  # Verify image integrity
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return False, f"Corrupt or invalid image file: The file could not be decoded. ({str(e)})", None

    # Re-open image after verify() since verify() closes image stream
    try:
        img = Image.open(BytesIO(file_bytes))
        img.load()  # Force load pixel data
    except Exception as e:
        return False, f"Failed to read image pixel data: {str(e)}", None

    # Dimension validation
    width, height = img.size
    if width < config.MIN_IMAGE_DIMENSION or height < config.MIN_IMAGE_DIMENSION:
        return False, (
            f"Image resolution too low ({width}x{height}px). "
            f"Minimum required dimension is {config.MIN_IMAGE_DIMENSION}x{config.MIN_IMAGE_DIMENSION}px."
        ), None

    if width > config.MAX_IMAGE_DIMENSION or height > config.MAX_IMAGE_DIMENSION:
        return False, (
            f"Image resolution too high ({width}x{height}px). "
            f"Maximum allowed dimension is {config.MAX_IMAGE_DIMENSION}x{config.MAX_IMAGE_DIMENSION}px."
        ), None

    return True, "", img
