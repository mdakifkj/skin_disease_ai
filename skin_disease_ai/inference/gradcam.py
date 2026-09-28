from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

import config

def find_last_conv_layer(model) -> Optional[str]:
    """
    Finds the name of the last convolutional layer in a Keras model
    (e.g., 'top_conv' in EfficientNetB0 or any convolutional layer).
    """
    for layer in reversed(model.layers):
        if hasattr(layer, "output") and len(layer.output.shape) == 4:
            if "conv" in layer.name.lower() or "top_activation" in layer.name.lower():
                return layer.name
    # Fallback to sub-model layers if model contains a Functional backbone
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if hasattr(sub_layer, "output") and len(sub_layer.output.shape) == 4:
                    if "conv" in sub_layer.name.lower() or "top_activation" in sub_layer.name.lower():
                        return sub_layer.name
    return None

def compute_gradcam_heatmap(
    model,
    img_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes a Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap.
    
    Args:
        model: Trained Keras Model
        img_tensor: Preprocessed float32 image tensor of shape (1, 224, 224, 3)
        target_class_idx: Target class index to explain (defaults to highest scoring class)
        last_conv_layer_name: Name of target convolutional layer
        
    Returns:
        2D numpy array heatmap normalized to [0, 1]
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)
        if last_conv_layer_name is None:
            raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

    # Try building Grad-CAM graph directly or via sub-backbone
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception:
        # If last_conv_layer is nested in a sub-layer/backbone
        base_layer = None
        for l in model.layers:
            if hasattr(l, "get_layer"):
                try:
                    _ = l.get_layer(last_conv_layer_name)
                    base_layer = l
                    break
                except Exception:
                    pass
        if base_layer is not None:
            sub_grad_model = tf.keras.models.Model(
                inputs=base_layer.inputs,
                outputs=[base_layer.get_layer(last_conv_layer_name).output, base_layer.output]
            )
            grad_model = sub_grad_model
        else:
            raise ValueError(f"Target convolutional layer '{last_conv_layer_name}' not found.")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # Return uniform heatmap if gradient is none
        return np.zeros((conv_outputs.shape[1], conv_outputs.shape[2]), dtype=np.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_and_save_gradcam(
    model,
    preprocessed_tensor: np.ndarray,
    original_image_path: Union[str, Path],
    output_filename: str,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.4
) -> Optional[str]:
    """
    Computes Grad-CAM and creates a blended heatmap overlay on the original image,
    saving the result safely in static/uploads/.
    
    Returns:
        output_filename if successful, None if Grad-CAM could not be generated.
    """
    try:
        heatmap = compute_gradcam_heatmap(model, preprocessed_tensor, target_class_idx=target_class_idx)
        
        orig_path = Path(original_image_path)
        img_bgr = cv2.imread(str(orig_path))
        if img_bgr is None:
            return None

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Superimpose
        superimposed = cv2.addWeighted(img_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)

        output_path = config.UPLOAD_FOLDER / output_filename
        cv2.imwrite(str(output_path), superimposed)

        return output_filename
    except Exception as e:
        print(f"[Grad-CAM Warning] Grad-CAM generation failed: {e}")
        return None
