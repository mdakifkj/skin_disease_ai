from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import config

def find_last_conv_layer(model) -> Optional[str]:
    """Finds the name of the last convolutional layer in a Keras model (e.g. 'top_conv' in EfficientNet)."""
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4 and ("conv" in layer.name.lower() or "top_activation" in layer.name.lower()):
            return layer.name
    return None

def compute_gradcam_heatmap(
    model,
    img_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap
    for explainability.
    
    Args:
        model: Trained Keras Model
        img_tensor: Preprocessed image tensor of shape (1, 224, 224, 3)
        target_class_idx: Target class index to explain (defaults to top predicted class)
        last_conv_layer_name: Name of target convolutional layer
        
    Returns:
        2D numpy array heatmap normalized to [0, 1]
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

    # Create a gradient model that maps input image to conv layer activations & output predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]

    # Compute gradients of the loss with respect to the output feature map of the target conv layer
    grads = tape.gradient(loss, conv_outputs)

    # Guided pooling: average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolution feature maps by corresponding gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain features that have a positive influence on the class
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(
    original_image_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> Path:
    """
    Superimposes the Grad-CAM heatmap over the original input image and saves it.
    
    Args:
        original_image_path: Path to original RGB image
        heatmap: 2D float array in [0, 1]
        output_path: Path to write overlaid image
        alpha: Heatmap blend weight (0.0 to 1.0)
        colormap: OpenCV colormap
        
    Returns:
        Path to the saved Grad-CAM image
    """
    img = cv2.imread(str(original_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load original image at {original_image_path}")

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend original image and heatmap
    superimposed = cv2.addWeighted(img, 1.0 - alpha, colored_heatmap, alpha, 0)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), superimposed)

    return output_path
