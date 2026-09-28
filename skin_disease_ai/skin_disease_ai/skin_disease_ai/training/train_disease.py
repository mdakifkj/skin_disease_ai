"""
Training Script for Skin Disease Classification using Transfer Learning (EfficientNetB0).

Usage:
    python training/train_disease.py [--epochs 25] [--batch-size 32] [--learning-rate 0.0001]
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
    parser = argparse.ArgumentParser(description="Train Skin Disease Classification Model")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    return parser.parse_args()

def build_disease_model(num_classes: int = len(config.DISEASE_CLASSES), input_shape=config.INPUT_SHAPE):
    """
    Constructs an EfficientNetB0 transfer learning model architecture.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    # Base pretrained model
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze initial feature extractor

    inputs = layers.Input(shape=input_shape)
    
    # EfficientNet has built-in rescaling, but data augmentation can be attached
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SkinDisease_EfficientNetB0")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    args = parse_args()
    print("=" * 70)
    print("AI-BASED SKIN DISEASE DETECTION: MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Check dataset availability
    status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    print(f"Checking dataset directory at: {config.DATASET_DIR}")
    print(f"Total images found: {status['total_images']}")
    
    if not status["is_ready_for_training"]:
        print("\n[WARNING] Dataset is not ready for training!")
        print("Please place publicly licensed skin disease images inside the appropriate class folders:")
        for cls in config.DISEASE_CLASSES:
            print(f"  - dataset/train/{cls}/")
            print(f"  - dataset/validation/{cls}/")
            print(f"  - dataset/test/{cls}/")
        print("\nCannot train model without real dataset images. Exiting cleanly.")
        return

    print("\nDataset verified. Proceeding with TensorFlow dataset initialization...")
    import tensorflow as tf

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    )

    model = build_disease_model(num_classes=len(config.DISEASE_CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.DISEASE_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nStarting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save final model
    config.DISEASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(config.DISEASE_MODEL_PATH))
    print(f"\nModel training complete! Model saved to: {config.DISEASE_MODEL_PATH}")

if __name__ == "__main__":
    main()
