import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import config
from utils.helpers import get_disease_info

class ModelNotAvailableError(Exception):
    """Raised when a prediction is requested but the trained model file is not found."""
    pass

class DiseasePredictor:
    """Handles deep learning inference for skin disease classification."""

    def __init__(self, model_path: Path = config.DISEASE_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.classes = config.DISEASE_CLASSES
        self.display_names = config.DISEASE_DISPLAY_NAMES
        self._load_model_if_available()

    def _load_model_if_available(self):
        """Loads the Keras model if the weights file exists on disk."""
        if self.model_path.exists() and self.model_path.is_file():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(self.model_path))
            except Exception as e:
                print(f"[DiseasePredictor] Error loading model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

    def is_available(self) -> bool:
        """Returns True if a real trained model is loaded and ready."""
        return self.model is not None

    def predict(self, preprocessed_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Runs model inference on preprocessed image batch tensor.
        
        Args:
            preprocessed_tensor: Shape (1, 224, 224, 3), normalized float32.
            
        Returns:
            Dict containing predicted_class, display_name, confidence, top_predictions list, and info.
        """
        if not self.is_available():
            raise ModelNotAvailableError(
                f"Trained disease classification model not found at '{self.model_path}'. "
                "Please train the model using 'python training/train_disease.py' or place a trained "
                ".keras model file in the 'models/disease_model/' directory."
            )

        # Real model inference
        raw_predictions = self.model.predict(preprocessed_tensor, verbose=0)
        probabilities = raw_predictions[0]

        top_index = int(np.argmax(probabilities))
        predicted_class = self.classes[top_index]
        confidence = float(probabilities[top_index])

        # Top predictions ranking
        top_k_indices = np.argsort(probabilities)[::-1]
        top_predictions = []
        for idx in top_k_indices:
            cls_key = self.classes[idx]
            top_predictions.append({
                "class_key": cls_key,
                "display_name": self.display_names.get(cls_key, cls_key),
                "confidence_percent": round(float(probabilities[idx]) * 100, 2),
                "probability": float(probabilities[idx]),
            })

        info = get_disease_info(predicted_class)

        return {
            "predicted_class": predicted_class,
            "display_name": self.display_names.get(predicted_class, predicted_class),
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "top_predictions": top_predictions,
            "information": info,
            "is_trained_model": True,
        }
