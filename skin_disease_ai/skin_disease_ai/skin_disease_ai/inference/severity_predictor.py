import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import config
from utils.helpers import get_severity_info

class SeverityModelNotAvailableError(Exception):
    """Raised when a prediction is requested but the trained severity model file is not found."""
    pass

class SeverityPredictor:
    """Handles deep learning inference for skin condition severity classification."""

    def __init__(self, model_path: Path = config.SEVERITY_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.classes = config.SEVERITY_CLASSES
        self.display_names = config.SEVERITY_DISPLAY_NAMES
        self._load_model_if_available()

    def _load_model_if_available(self):
        """Loads the Keras severity model if the weights file exists on disk."""
        if self.model_path.exists() and self.model_path.is_file():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(self.model_path))
            except Exception as e:
                print(f"[SeverityPredictor] Error loading model from {self.model_path}: {e}")
                self.model = None
        else:
            self.model = None

    def is_available(self) -> bool:
        """Returns True if a real trained model is loaded and ready."""
        return self.model is not None

    def predict(self, preprocessed_tensor: np.ndarray) -> Dict[str, Any]:
        """
        Runs severity model inference on preprocessed image batch tensor.
        
        Args:
            preprocessed_tensor: Shape (1, 224, 224, 3), normalized float32.
            
        Returns:
            Dict containing predicted_severity, display_name, confidence, breakdown, and info.
        """
        if not self.is_available():
            raise SeverityModelNotAvailableError(
                f"Trained severity classification model not found at '{self.model_path}'. "
                "Severity prediction requires reliable severity-annotated images. "
                "Please train the model using 'python training/train_severity.py' once severity data is available."
            )

        raw_predictions = self.model.predict(preprocessed_tensor, verbose=0)
        probabilities = raw_predictions[0]

        top_index = int(np.argmax(probabilities))
        predicted_severity = self.classes[top_index]
        confidence = float(probabilities[top_index])

        breakdown = []
        for idx, cls_key in enumerate(self.classes):
            breakdown.append({
                "severity_key": cls_key,
                "display_name": self.display_names.get(cls_key, cls_key),
                "confidence_percent": round(float(probabilities[idx]) * 100, 2),
                "probability": float(probabilities[idx]),
            })

        info = get_severity_info(predicted_severity)

        return {
            "predicted_severity": predicted_severity,
            "display_name": self.display_names.get(predicted_severity, predicted_severity),
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "breakdown": breakdown,
            "information": info,
            "is_trained_model": True,
        }
