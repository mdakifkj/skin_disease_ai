"""
AI-Based Skin Severity Model: Test Split Evaluation Script.
Computes real classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
and generates evaluation plots in results/severity/.

Usage:
    python training/evaluate_severity.py [--model-path models/severity_model/severity_classifier.keras]
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import config
from utils.dataset_loader import scan_dataset_directory

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Severity Classification Model")
    parser.add_argument("--model-path", type=str, default=str(config.SEVERITY_MODEL_PATH), help="Path to trained severity model")
    parser.add_argument("--test-dir", type=str, default=str(config.SEVERITY_TEST_DIR), help="Path to severity test dataset")
    parser.add_argument("--output-dir", type=str, default=str(config.SEVERITY_RESULTS_DIR), help="Output directory")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    return parser.parse_args()

def generate_severity_evaluation_plots(cm: np.ndarray, report_dict: dict, classes: list, output_dir: Path):
    """Generates confusion matrix heatmap and metrics bar chart for severity model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(7, 6))
        display_labels = [config.SEVERITY_DISPLAY_NAMES.get(c, c) for c in classes]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar=True
        )
        plt.title("Severity Model: Confusion Matrix (Test Split)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Predicted Severity", fontsize=11, fontweight="semibold")
        plt.ylabel("Ground Truth Severity", fontsize=11, fontweight="semibold")
        plt.tight_layout()
        cm_path = output_dir / "severity_confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[Plot] Severity confusion matrix saved to: {cm_path}")

        # 2. Per-Class Bar Chart
        class_keys = [c for c in classes if c in report_dict]
        if class_keys:
            precisions = [report_dict[c]["precision"] * 100 for c in class_keys]
            recalls = [report_dict[c]["recall"] * 100 for c in class_keys]
            f1_scores = [report_dict[c]["f1-score"] * 100 for c in class_keys]

            x = np.arange(len(class_keys))
            width = 0.25

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width, precisions, width, label="Precision (%)", color="#3b82f6")
            ax.bar(x, recalls, width, label="Recall (%)", color="#10b981")
            ax.bar(x + width, f1_scores, width, label="F1-Score (%)", color="#f59e0b")

            ax.set_ylabel("Score (%)", fontsize=11, fontweight="semibold")
            ax.set_title("Severity Model: Class Performance", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([config.SEVERITY_DISPLAY_NAMES.get(c, c) for c in class_keys])
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            plt.tight_layout()
            metrics_plot_path = output_dir / "severity_class_metrics.png"
            plt.savefig(metrics_plot_path, dpi=200)
            plt.close()
            print(f"[Plot] Severity metrics chart saved to: {metrics_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate severity evaluation plots: {e}")

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    print("=" * 75)
    print("SKIN SEVERITY MODEL: COMPREHENSIVE TEST EVALUATION")
    print("=" * 75)

    if not model_path.exists():
        print(f"\n[ERROR] Trained severity model not found at: {model_path}")
        print("Train the model first using: python training/train_severity.py (when severity data is available).")
        print("Halting evaluation cleanly.")
        return

    status = scan_dataset_directory(config.SEVERITY_DATASET_DIR, config.SEVERITY_CLASSES)
    test_count = status["splits"].get("test", {}).get("total_split_images", 0)

    if test_count == 0:
        print(f"\n[ERROR] No severity test images found in: {test_dir}")
        print("Please place test split images into 'severity_dataset/test/<mild|moderate|severe>/'.")
        print("Halting evaluation cleanly.")
        return

    import tensorflow as tf
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    print(f"Loading trained severity model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
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

    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=config.SEVERITY_CLASSES, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=config.SEVERITY_CLASSES, output_dict=True, zero_division=0)

    print("\n" + "=" * 60)
    print("SEVERITY CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)
    print(f"Overall Accuracy: {acc * 100:.2f}% | F1 (Macro): {f1_macro * 100:.2f}%")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "severity_classification_report.txt", "w") as f:
        f.write(report_text)

    metrics_payload = {
        "model": "EfficientNetB0_Severity_Classifier",
        "total_test_samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": config.SEVERITY_CLASSES,
    }

    with open(output_dir / "severity_evaluation_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    generate_severity_evaluation_plots(cm, report_dict, config.SEVERITY_CLASSES, output_dir)
    print(f"\n[Saved] Severity evaluation artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()
