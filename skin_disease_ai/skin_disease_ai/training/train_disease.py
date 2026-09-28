"""
AI-Based Skin Disease Classification: Transfer Learning Training Pipeline.
Uses EfficientNetB0 backbone with data augmentation, class weighting, and callbacks.

Usage:
    python training/train_disease.py [--epochs 25] [--batch-size 32] [--learning-rate 0.0001]
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
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model (EfficientNetB0)")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Initial Adam learning rate")
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE[0], help="Image input resolution (e.g. 224)")
    parser.add_argument("--train-dir", type=str, default=str(config.TRAIN_DIR), help="Training dataset path")
    parser.add_argument("--val-dir", type=str, default=str(config.VAL_DIR), help="Validation dataset path")
    parser.add_argument("--model-path", type=str, default=str(config.DISEASE_MODEL_PATH), help="Model output path (.keras)")
    parser.add_argument("--fine-tune", action="store_true", help="Unfreeze top layers of EfficientNet for fine-tuning")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")
    return parser.parse_args()

def build_disease_model(num_classes: int, input_shape=(224, 224, 3), learning_rate=1e-4):
    """
    Constructs an EfficientNetB0 transfer learning architecture.
    Includes data augmentation layers, GlobalAveragePooling, Dropout, and Dense Softmax head.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # 1. Base pretrained model (ImageNet weights)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze backbone for initial feature extraction

    # 2. Input and Augmentation Pipeline
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.2, fill_mode="nearest", name="aug_rotation")(x)
    x = layers.RandomZoom(0.15, fill_mode="nearest", name="aug_zoom")(x)
    x = layers.RandomContrast(0.15, name="aug_contrast")(x)

    # 3. Backbone Feature Extraction
    x = base_model(x, training=False)

    # 4. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_feature_layer")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_classification")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_EfficientNetB0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base_model

def save_training_plots(history_dict: dict, output_dir: Path):
    """Generates and saves training and validation Loss & Accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, len(history_dict.get("loss", [])) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy Plot
        ax1.plot(epochs_range, history_dict.get("accuracy", []), label="Train Accuracy", marker="o")
        if "val_accuracy" in history_dict:
            ax1.plot(epochs_range, history_dict.get("val_accuracy", []), label="Val Accuracy", marker="s")
        ax1.set_title("Disease Model: Accuracy vs Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Loss Plot
        ax2.plot(epochs_range, history_dict.get("loss", []), label="Train Loss", marker="o", color="crimson")
        if "val_loss" in history_dict:
            ax2.plot(epochs_range, history_dict.get("val_loss", []), label="Val Loss", marker="s", color="darkorange")
        ax2.set_title("Disease Model: Loss vs Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"[Plot] Training loss & accuracy curves saved to: {plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate training plot: {e}")

def main():
    args = parse_args()
    set_random_seed(args.seed)

    print("=" * 75)
    print("AI-BASED SKIN DISEASE DETECTION: EFFICIENTNET-B0 TRAINING PIPELINE")
    print("=" * 75)
    print(f"Target Classes ({len(config.DISEASE_CLASSES)}): {', '.join(config.DISEASE_CLASSES)}")
    print(f"Input Shape: ({args.image_size}, {args.image_size}, 3)")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.learning_rate}, Seed={args.seed}")

    # 1. Inspect Dataset Directory
    status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    print("\n" + "-" * 50)
    print("DATASET STATUS REPORT:")
    print("-" * 50)
    print(f"Root Directory: {status['base_directory']}")
    print(f"Total Dataset Images: {status['total_images']}")
    for split_name, split_info in status["splits"].items():
        print(f"\n  Split [{split_name.upper()}]: {split_info.get('total_split_images', 0)} total images")
        for cls_name, count in split_info.get("class_counts", {}).items():
            print(f"    - {cls_name:20s}: {count:5d} images")

    if not status["is_ready_for_training"]:
        print("\n" + "=" * 75)
        print("[DATASET REQUIREMENT NOTICE]")
        print("Real image dataset is not ready in 'dataset/train/' and 'dataset/validation/'.")
        print("To train the model on genuine skin disease data:")
        print("1. Download an academic skin disease dataset (e.g. DermNet, ISIC, HAM10000).")
        print("2. Place images in their respective class subfolders:")
        for cls in config.DISEASE_CLASSES:
            print(f"   - dataset/train/{cls}/")
            print(f"   - dataset/validation/{cls}/")
            print(f"   - dataset/test/{cls}/")
        print("\nPipeline is verified and ready. Halting training safely until dataset is populated.")
        print("=" * 75)
        return

    # 2. Compute Class Weights for Imbalance
    class_weights = compute_dataset_class_weights(Path(args.train_dir), config.DISEASE_CLASSES)
    if class_weights:
        print("\nCalculated Class Weights (for class imbalance compensation):")
        for idx, w in class_weights.items():
            print(f"  Class {idx} ({config.DISEASE_CLASSES[idx]}): {w:.3f}")

    # 3. Initialize TensorFlow Data Loaders
    import tensorflow as tf
    image_dim = (args.image_size, args.image_size)

    print("\nLoading TensorFlow Datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        image_size=image_dim,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
        seed=args.seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        image_size=image_dim,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    )

    # Prefetch optimization for I/O performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 4. Build Model Architecture
    model, base_model = build_disease_model(
        num_classes=len(config.DISEASE_CLASSES),
        input_shape=(args.image_size, args.image_size, 3),
        learning_rate=args.learning_rate
    )
    print("\nModel Architecture Summary:")
    model.summary()

    # 5. Prepare Callbacks & Output Directories
    model_output_path = Path(args.model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.DISEASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_log_path = config.DISEASE_RESULTS_DIR / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_output_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path))
    ]

    # 6. Execute Stage 1: Feature Extraction
    print(f"\nStarting Transfer Learning Feature Extraction ({args.epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # 7. Optional Stage 2: Fine-Tuning top layers of EfficientNet
    if args.fine_tune:
        print("\nStarting Fine-Tuning Stage (unfreezing top 20 layers of EfficientNetB0)...")
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + config.FINE_TUNE_EPOCHS,
            initial_epoch=len(history.history["loss"]),
            callbacks=callbacks,
            class_weight=class_weights
        )

        for k in history.history:
            if k in fine_tune_history.history:
                history.history[k].extend(fine_tune_history.history[k])

    # 8. Save Final Model and Training Curves
    model.save(str(model_output_path))
    print(f"\n[Success] Trained model saved to: {model_output_path}")

    save_training_plots(history.history, config.DISEASE_RESULTS_DIR)

    # Save training history JSON
    history_json_path = config.DISEASE_RESULTS_DIR / "training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"[Log] Training metrics JSON saved to: {history_json_path}")

if __name__ == "__main__":
    main()
