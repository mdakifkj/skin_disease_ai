import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

import config
from utils.preprocessing import preprocess_pil_image
from utils.helpers import get_severity_info

SEVERITY_UNAVAILABLE_MESSAGE = "Severity prediction is currently unavailable because a trained severity model has not been provided."

class SeverityModelNotAvailableError(Exception):
    """Raised when severity inference is requested but no trained severity model exists."""
    pass

class SeverityPredictor:
    """Handles deep learning inference for skin condition severity classification (Mild, Moderate, Severe)."""

    def __init__(self, model_path: Path = config.SEVERITY_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.classes = config.SEVERITY_CLASSES
        self.display_names = config.SEVERITY_DISPLAY_NAMES
        self.load_model()

    def load_model(self) -> bool:
        """Loads the Keras severity model if weights file exists on disk."""
        if self.model_path.exists() and self.model_path.is_file():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(self.model_path))
                return True
            except Exception as e:
                print(f"[SeverityPredictor] Error loading model from {self.model_path}: {e}")
                self.model = None
                return False
        else:
            self.model = None
            return False

    def is_available(self) -> bool:
        """Returns True if a real trained severity model is loaded."""
        return self.model is not None

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Runs severity classification inference.
        
        Args:
            image_input: Either a PIL Image or a preprocessed NumPy batch tensor (1, 224, 224, 3).
            
        Returns:
            Dict containing:
            - severity: string key ("mild", "moderate", "severe")
            - severity_display_name: human-readable name ("Mild Severity", etc.)
            - severity_confidence: float (0.0 to 1.0)
            - severity_confidence_percent: float (0.0 to 100.0)
            - severity_probabilities: dict mapping every severity class to probability & display name
            - info: educational severity notes
            - is_trained_model: True
            
        Raises:
            SeverityModelNotAvailableError if model file is not present.
        """
        if not self.is_available():
            raise SeverityModelNotAvailableError(SEVERITY_UNAVAILABLE_MESSAGE)

        if isinstance(image_input, Image.Image):
            tensor = preprocess_pil_image(image_input, target_size=config.IMAGE_SIZE, normalize=True)
        elif isinstance(image_input, np.ndarray):
            tensor = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        raw_predictions = self.model.predict(tensor, verbose=0)
        probabilities = raw_predictions[0]

        top_index = int(np.argmax(probabilities))
        predicted_severity = self.classes[top_index]
        confidence = float(probabilities[top_index])

        severity_probabilities = {}
        for idx, cls_key in enumerate(self.classes):
            prob = float(probabilities[idx])
            s_name = self.display_names.get(cls_key, cls_key.title())
            severity_probabilities[cls_key] = {
                "severity_key": cls_key,
                "display_name": s_name,
                "probability": prob,
                "confidence_percent": round(prob * 100, 2),
            }

        info = get_severity_info(predicted_severity)

        return {
            "severity": predicted_severity,
            "severity_display_name": self.display_names.get(predicted_severity, predicted_severity.title()),
            "severity_confidence": float(confidence),
            "severity_confidence_percent": round(confidence * 100, 2),
            "severity_probabilities": severity_probabilities,
            "info": info,
            "is_trained_model": True,
        }
