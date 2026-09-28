"""
Evaluation Script for Skin Disease Model on Test Split.
Computes real classification metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import config
from utils.dataset_loader import scan_dataset_directory

def evaluate_disease_model():
    print("=" * 70)
    print("SKIN DISEASE MODEL EVALUATION")
    print("=" * 70)

    if not config.DISEASE_MODEL_PATH.exists():
        print(f"Error: Disease model not found at {config.DISEASE_MODEL_PATH}. Train the model first.")
        return

    status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    test_count = status["splits"].get("test", {}).get("total_split_images", 0)
    if test_count == 0:
        print(f"Error: Test split in {config.TEST_DIR} is empty. Add test images to evaluate.")
        return

    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    print(f"Loading trained model from {config.DISEASE_MODEL_PATH}...")
    model = tf.keras.models.load_model(str(config.DISEASE_MODEL_PATH))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=config.DISEASE_CLASSES, digits=4))

    print("=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(f"\nOverall Test Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_disease_model()
