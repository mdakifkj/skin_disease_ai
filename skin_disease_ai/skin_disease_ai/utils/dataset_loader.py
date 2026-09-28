import os
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import config

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def set_random_seed(seed: int = config.RANDOM_SEED):
    """Sets global random seeds for reproducibility across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def scan_dataset_directory(base_dir: Path, expected_classes: List[str]) -> Dict[str, Any]:
    """
    Scans a dataset root directory (with train, validation, test subfolders)
    and counts samples per class.
    
    Returns a comprehensive structured report dictionary.
    """
    splits = ["train", "validation", "test"]
    report = {
        "base_directory": str(base_dir),
        "exists": base_dir.exists(),
        "splits": {},
        "total_images": 0,
        "is_ready_for_training": False,
        "missing_classes": [],
        "warnings": [],
    }

    if not base_dir.exists():
        report["warnings"].append(f"Directory '{base_dir}' does not exist.")
        return report

    total_count = 0

    for split in splits:
        split_dir = base_dir / split
        split_data = {
            "exists": split_dir.exists(),
            "class_counts": {},
            "total_split_images": 0,
        }

        if not split_dir.exists():
            split_data["warnings"] = [f"Split directory '{split}' not found."]
        else:
            for cls_name in expected_classes:
                cls_dir = split_dir / cls_name
                if cls_dir.exists() and cls_dir.is_dir():
                    images = [
                        f for f in cls_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
                    ]
                    count = len(images)
                else:
                    count = 0
                    if cls_name not in report["missing_classes"]:
                        report["missing_classes"].append(cls_name)

                split_data["class_counts"][cls_name] = count
                split_data["total_split_images"] += count
                total_count += count

        report["splits"][split] = split_data

    report["total_images"] = total_count
    train_count = report["splits"].get("train", {}).get("total_split_images", 0)
    val_count = report["splits"].get("validation", {}).get("total_split_images", 0)
    
    # Ready if both train and validation splits have samples
    report["is_ready_for_training"] = (train_count > 0 and val_count > 0)

    return report

def compute_dataset_class_weights(train_dir: Path, classes: List[str]) -> Optional[Dict[int, float]]:
    """
    Calculates balanced class weights to address potential class imbalances in training data:
    weight = total_samples / (n_classes * class_samples)
    """
    counts = []
    for cls in classes:
        cls_dir = train_dir / cls
        if cls_dir.exists():
            images = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
            counts.append(len(images))
        else:
            counts.append(0)

    total_samples = sum(counts)
    num_classes = len(classes)

    if total_samples == 0 or min(counts) == 0:
        return None

    class_weights = {}
    for idx, count in enumerate(counts):
        class_weights[idx] = float(total_samples / (num_classes * count))

    return class_weights

def get_overall_dataset_status() -> Dict[str, Any]:
    """Returns dataset status summary for both disease and severity datasets."""
    disease_status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    severity_status = scan_dataset_directory(config.SEVERITY_DATASET_DIR, config.SEVERITY_CLASSES)
    
    return {
        "disease_dataset": disease_status,
        "severity_dataset": severity_status,
    }
