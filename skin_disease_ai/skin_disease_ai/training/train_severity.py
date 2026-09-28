"""
AI-Based Skin Condition Severity Classification: Training Pipeline.
Trains a separate 3-class model (Mild, Moderate, Severe).

IMPORTANT:
- Severity prediction requires images annotated specifically for clinical severity.
- Never convert disease categories into severity labels.

Usage:
    python training/train_severity.py [--epochs 25] [--batch-size 32] [--learning-rate 0.0001]
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils.dataset_loader import (
    set_random_seed,
    scan_dataset_directory,
    compute_dataset_class_weights,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Severity Classification Model")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE[0], help="Image input resolution")
    parser.add_argument("--train-dir", type=str, default=str(config.SEVERITY_TRAIN_DIR), help="Severity train path")
    parser.add_argument("--val-dir", type=str, default=str(config.SEVERITY_VAL_DIR), help="Severity validation path")
    parser.add_argument("--model-path", type=str, default=str(config.SEVERITY_MODEL_PATH), help="Severity model output path")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    return parser.parse_args()

def build_severity_model(num_classes: int = len(config.SEVERITY_CLASSES), input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Constructs a transfer learning model for 3-class severity grading (Mild, Moderate, Severe).
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="severity_input")
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.15, fill_mode="nearest", name="aug_rot")(x)
    x = layers.RandomZoom(0.1, fill_mode="nearest", name="aug_zoom")(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="severity_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinSeverity_EfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_severity_plots(history_dict: dict, output_dir: Path):
    """Generates training & validation curves for severity classifier."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, len(history_dict.get("loss", [])) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history_dict.get("accuracy", []), label="Train Accuracy", marker="o")
        if "val_accuracy" in history_dict:
            ax1.plot(epochs_range, history_dict.get("val_accuracy", []), label="Val Accuracy", marker="s")
        ax1.set_title("Severity Model: Accuracy vs Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, history_dict.get("loss", []), label="Train Loss", marker="o", color="crimson")
        if "val_loss" in history_dict:
            ax2.plot(epochs_range, history_dict.get("val_loss", []), label="Val Loss", marker="s", color="darkorange")
        ax2.set_title("Severity Model: Loss vs Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "severity_training_curves.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[Plot] Severity training curves saved to: {plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate severity plot: {e}")

def main():
    args = parse_args()
    set_random_seed(args.seed)

    print("=" * 75)
    print("AI-BASED SKIN SEVERITY CLASSIFICATION: TRAINING PIPELINE")
    print("=" * 75)
    print(f"Target Severity Classes: {', '.join(config.SEVERITY_CLASSES)}")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}, Seed={args.seed}")

    # 1. Check Severity Dataset
    status = scan_dataset_directory(config.SEVERITY_DATASET_DIR, config.SEVERITY_CLASSES)
    print("\n" + "-" * 50)
    print("SEVERITY DATASET STATUS:")
    print("-" * 50)
    print(f"Directory: {status['base_directory']}")
    print(f"Total Images: {status['total_images']}")
    for split_name, split_info in status["splits"].items():
        print(f"\n  Split [{split_name.upper()}]: {split_info.get('total_split_images', 0)} total images")
        for cls_name, count in split_info.get("class_counts", {}).items():
            print(f"    - {cls_name:10s}: {count:5d} images")

    if not status["is_ready_for_training"]:
        print("\n" + "=" * 75)
        print("[SEVERITY DATASET REQUIREMENT & SCIENTIFIC INTEGRITY NOTICE]")
        print("Severity prediction requires medically annotated severity labels (mild, moderate, severe).")
        print("Disease identity (e.g. Acne vs Psoriasis) MUST NOT be conflated with severity.")
        print("To train the severity classifier:")
        print("1. Obtain a dataset with dermatologist-verified severity ratings.")
        print(f"2. Place severity-labelled images into:")
        for cls in config.SEVERITY_CLASSES:
            print(f"   - severity_dataset/train/{cls}/")
            print(f"   - severity_dataset/validation/{cls}/")
            print(f"   - severity_dataset/test/{cls}/")
        print("\nPipeline is verified and ready. Halting severity training until dataset is populated.")
        print("=" * 75)
        return

    # 2. Compute Class Weights
    class_weights = compute_dataset_class_weights(Path(args.train_dir), config.SEVERITY_CLASSES)

    # 3. Load TensorFlow Datasets
    import tensorflow as tf
    image_dim = (args.image_size, args.image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        image_size=image_dim,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
        seed=args.seed
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        image_size=image_dim,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)

    # 4. Build Model
    model = build_severity_model(
        num_classes=len(config.SEVERITY_CLASSES),
        input_shape=(args.image_size, args.image_size, 3),
        learning_rate=args.learning_rate
    )
    model.summary()

    # 5. Callbacks
    model_output_path = Path(args.model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.SEVERITY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_log_path = config.SEVERITY_RESULTS_DIR / "severity_training_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(model_output_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(csv_log_path))
    ]

    print(f"\nStarting Severity Model Training ({args.epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    model.save(str(model_output_path))
    print(f"\n[Success] Trained severity model saved to: {model_output_path}")

    save_severity_plots(history.history, config.SEVERITY_RESULTS_DIR)

    history_json_path = config.SEVERITY_RESULTS_DIR / "severity_training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"[Log] Severity history JSON saved to: {history_json_path}")

if __name__ == "__main__":
    main()
