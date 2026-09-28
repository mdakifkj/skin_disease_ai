import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

import config
from utils.preprocessing import preprocess_pil_image
from utils.helpers import get_disease_info

class ModelNotAvailableError(Exception):
    """Raised when a prediction is requested but the trained model file is not found."""
    pass

class DiseasePredictor:
    """Handles deep learning inference for skin disease classification using EfficientNetB0."""

    def __init__(self, model_path: Path = config.DISEASE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.classes = config.DISEASE_CLASSES
        self.display_names = config.DISEASE_DISPLAY_NAMES
        self.load_model()

    def load_model(self) -> bool:
        """Loads the Keras model if the weights file exists on disk."""
        if self.model_path.exists() and self.model_path.is_file():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(self.model_path))
                return True
            except Exception as e:
                print(f"[DiseasePredictor] Error loading model from {self.model_path}: {e}")
                self.model = None
                return False
        else:
            self.model = None
            return False

    def is_available(self) -> bool:
        """Returns True if a real trained model is loaded and ready."""
        return self.model is not None

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """
        Runs disease classification model inference.
        
        Args:
            image_input: Either a PIL Image or a preprocessed NumPy batch tensor of shape (1, 224, 224, 3).
            
        Returns:
            Dict with keys:
            - disease: primary predicted disease class key
            - disease_display_name: human-readable disease name
            - disease_confidence: float (0.0 to 1.0)
            - disease_confidence_percent: float (0.0 to 100.0)
            - disease_probabilities: dict mapping every class key to its probability and display name
            - top_predictions: sorted list of top class predictions
            - info: educational context dictionary
            - is_trained_model: True
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                f"Trained disease classification model not found at '{self.model_path}'. "
                "Disease classification is currently unavailable because a trained disease model has not been provided."
            )

        # Preprocess if image_input is a PIL Image
        if isinstance(image_input, Image.Image):
            tensor = preprocess_pil_image(image_input, target_size=config.IMAGE_SIZE, normalize=True)
        elif isinstance(image_input, np.ndarray):
            tensor = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        raw_predictions = self.model.predict(tensor, verbose=0)
        probabilities = raw_predictions[0]

        top_index = int(np.argmax(probabilities))
        predicted_class = self.classes[top_index]
        confidence = float(probabilities[top_index])

        # Full probability dictionary across all classes
        disease_probabilities = {}
        top_predictions = []

        for idx, cls_key in enumerate(self.classes):
            prob = float(probabilities[idx])
            d_name = self.display_names.get(cls_key, cls_key.replace("_", " ").title())
            disease_probabilities[cls_key] = {
                "class_key": cls_key,
                "display_name": d_name,
                "probability": prob,
                "confidence_percent": round(prob * 100, 2),
            }
            top_predictions.append({
                "class_key": cls_key,
                "display_name": d_name,
                "probability": prob,
                "confidence_percent": round(prob * 100, 2),
            })

        top_predictions.sort(key=lambda x: x["probability"], reverse=True)
        info = get_disease_info(predicted_class)

        return {
            "disease": predicted_class,
            "disease_display_name": self.display_names.get(predicted_class, predicted_class),
            "disease_confidence": float(confidence),
            "disease_confidence_percent": round(confidence * 100, 2),
            "disease_probabilities": disease_probabilities,
            "top_predictions": top_predictions,
            "info": info,
            "is_trained_model": True,
        }
