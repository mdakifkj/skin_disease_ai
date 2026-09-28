"""
Training Script for Skin Condition Severity Classification.

Note: Severity prediction requires dedicated severity-annotated images (Mild, Moderate, Severe).
Do not substitute disease labels for severity labels.

Usage:
    python training/train_severity.py [--epochs 25] [--batch-size 32]
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from utils.dataset_loader import scan_dataset_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Train Severity Classification Model")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    return parser.parse_args()

def build_severity_model(num_classes: int = len(config.SEVERITY_CLASSES), input_shape=config.INPUT_SHAPE):
    """
    Constructs a transfer learning model for 3-class severity grading.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="severity_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinSeverity_EfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    args = parse_args()
    print("=" * 70)
    print("AI-BASED SKIN SEVERITY PREDICTION: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Check severity dataset availability
    status = scan_dataset_directory(config.SEVERITY_DATASET_DIR, config.SEVERITY_CLASSES)
    print(f"Checking severity dataset directory at: {config.SEVERITY_DATASET_DIR}")
    print(f"Total severity-annotated images found: {status['total_images']}")

    if not status["is_ready_for_training"]:
        print("\n[IMPORTANT LIMITATION / NOTICE]")
        print("Severity prediction requires medically annotated severity labels (mild, moderate, severe).")
        print("Do NOT confuse disease labels with severity ratings.")
        print(f"Please place severity-labelled images into '{config.SEVERITY_DATASET_DIR}' before training.")
        return

    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.SEVERITY_TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.SEVERITY_VAL_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    )

    model = build_severity_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.SEVERITY_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print(f"\nStarting severity model training for {args.epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    config.SEVERITY_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(config.SEVERITY_MODEL_PATH))
    print(f"\nSeverity model saved to: {config.SEVERITY_MODEL_PATH}")

if __name__ == "__main__":
    main()
