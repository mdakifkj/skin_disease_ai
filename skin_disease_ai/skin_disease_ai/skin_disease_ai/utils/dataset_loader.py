import os
from pathlib import Path
from typing import Dict, List, Any
import config

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

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

    all_splits_have_data = True
    total_count = 0

    for split in splits:
        split_dir = base_dir / split
        split_data = {
            "exists": split_dir.exists(),
            "class_counts": {},
            "total_split_images": 0,
        }

        if not split_dir.exists():
            all_splits_have_data = False
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

            if split_data["total_split_images"] == 0:
                all_splits_have_data = False

        report["splits"][split] = split_data

    report["total_images"] = total_count
    # Ready if at least train split has images across expected classes
    train_has_images = report["splits"].get("train", {}).get("total_split_images", 0) > 0
    val_has_images = report["splits"].get("validation", {}).get("total_split_images", 0) > 0
    report["is_ready_for_training"] = train_has_images and val_has_images

    return report

def get_overall_dataset_status() -> Dict[str, Any]:
    """Returns dataset status summary for both disease and severity datasets."""
    disease_status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    severity_status = scan_dataset_directory(config.SEVERITY_DATASET_DIR, config.SEVERITY_CLASSES)
    
    return {
        "disease_dataset": disease_status,
        "severity_dataset": severity_status,
    }
