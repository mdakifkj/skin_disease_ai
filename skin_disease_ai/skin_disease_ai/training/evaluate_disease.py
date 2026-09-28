"""
AI-Based Skin Disease Model: Test Split Evaluation Script.
Computes real classification metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix)
and generates evaluation plots in results/disease/.

Usage:
    python training/evaluate_disease.py [--model-path models/disease_model/efficientnet_b0_skin_disease.keras] [--test-dir dataset/test]
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
    parser = argparse.ArgumentParser(description="Evaluate Skin Disease Classification Model")
    parser.add_argument("--model-path", type=str, default=str(config.DISEASE_MODEL_PATH), help="Path to trained .keras model")
    parser.add_argument("--test-dir", type=str, default=str(config.TEST_DIR), help="Path to test dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(config.DISEASE_RESULTS_DIR), help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")
    return parser.parse_args()

def generate_evaluation_plots(cm: np.ndarray, report_dict: dict, classes: list, output_dir: Path):
    """Generates and saves confusion matrix heatmap and per-class performance bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(9, 7))
        display_labels = [config.DISEASE_DISPLAY_NAMES.get(c, c) for c in classes]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=display_labels,
            yticklabels=display_labels,
            cbar=True
        )
        plt.title("Disease Model: Confusion Matrix (Test Split)", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Predicted Class", fontsize=12, fontweight="semibold")
        plt.ylabel("Ground Truth Class", fontsize=12, fontweight="semibold")
        plt.xticks(rotation=40, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[Plot] Confusion matrix saved to: {cm_path}")

        # 2. Per-Class Precision, Recall, F1 Bar Chart
        class_keys = [c for c in classes if c in report_dict]
        if class_keys:
            precisions = [report_dict[c]["precision"] * 100 for c in class_keys]
            recalls = [report_dict[c]["recall"] * 100 for c in class_keys]
            f1_scores = [report_dict[c]["f1-score"] * 100 for c in class_keys]

            x = np.arange(len(class_keys))
            width = 0.25

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width, precisions, width, label="Precision (%)", color="#3b82f6")
            ax.bar(x, recalls, width, label="Recall (%)", color="#10b981")
            ax.bar(x + width, f1_scores, width, label="F1-Score (%)", color="#f59e0b")

            ax.set_ylabel("Score (%)", fontsize=11, fontweight="semibold")
            ax.set_title("Disease Model: Per-Class Evaluation Metrics", fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([config.DISEASE_DISPLAY_NAMES.get(c, c) for c in class_keys], rotation=35, ha="right")
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            plt.tight_layout()
            metrics_plot_path = output_dir / "per_class_metrics.png"
            plt.savefig(metrics_plot_path, dpi=200)
            plt.close()
            print(f"[Plot] Per-class metrics chart saved to: {metrics_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not generate evaluation plots: {e}")

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    print("=" * 75)
    print("SKIN DISEASE MODEL: COMPREHENSIVE TEST EVALUATION")
    print("=" * 75)

    # 1. Verify Model Existence
    if not model_path.exists():
        print(f"\n[ERROR] Trained model file not found at: {model_path}")
        print("Please train the model first using: python training/train_disease.py")
        print("Halting evaluation cleanly.")
        return

    # 2. Verify Test Dataset
    status = scan_dataset_directory(config.DATASET_DIR, config.DISEASE_CLASSES)
    test_count = status["splits"].get("test", {}).get("total_split_images", 0)

    if test_count == 0:
        print(f"\n[ERROR] No test images found in: {test_dir}")
        print("Please place test split images into 'dataset/test/<class_name>/' to run evaluation.")
        print("Halting evaluation cleanly.")
        return

    print(f"Loading trained model from {model_path}...")
    import tensorflow as tf
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    model = tf.keras.models.load_model(str(model_path))

    # 3. Load Test Data
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=config.IMAGE_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    )

    y_true = []
    y_pred_probs = []

    print(f"Evaluating {test_count} test images across {len(config.DISEASE_CLASSES)} classes...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.argmax(np.array(y_pred_probs), axis=1)

    # 4. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, target_names=config.DISEASE_CLASSES, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=config.DISEASE_CLASSES, output_dict=True, zero_division=0)

    # 5. Display Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report_text)

    print("=" * 60)
    print("OVERALL SUMMARY METRICS")
    print("=" * 60)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro * 100:.2f}%")
    print(f"Recall (Macro):     {rec_macro * 100:.2f}%")
    print(f"F1-Score (Macro):   {f1_macro * 100:.2f}%")
    print(f"Precision (Weight): {prec_weighted * 100:.2f}%")
    print(f"Recall (Weight):    {rec_weighted * 100:.2f}%")
    print(f"F1-Score (Weight):  {f1_weighted * 100:.2f}%")

    # 6. Save Artifacts to results/disease/
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report_text)

    metrics_payload = {
        "model": "EfficientNetB0_Disease_Classifier",
        "total_test_samples": int(len(y_true)),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": config.DISEASE_CLASSES,
    }

    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    generate_evaluation_plots(cm, report_dict, config.DISEASE_CLASSES, output_dir)
    print(f"\n[Saved] All evaluation artifacts saved to: {output_dir}")

if __name__ == "__main__":
    main()
