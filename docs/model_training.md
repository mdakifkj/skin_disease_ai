# AI Model Training & Evaluation Guide

## 1. Transfer Learning with EfficientNetB0

Transfer learning enables the neural network to utilize rich low-level and mid-level feature representations learned from 1.4 million natural images in ImageNet, significantly improving sample efficiency and convergence speed.

---

## 2. Disease Classifier Training

```bash
python training/train_disease.py --epochs 25 --batch-size 32 --learning-rate 0.0001
```

### Supported Arguments:
- `--epochs`: Integer (default: `25`)
- `--batch-size`: Integer (default: `32`)
- `--learning-rate`: Float (default: `0.0001`)
- `--image-size`: Integer (default: `224`)
- `--fine-tune`: Boolean flag to unfreeze upper EfficientNet layers
- `--seed`: Random seed (default: `42`)

### Training Callbacks:
1. **EarlyStopping**: Halts training if validation loss fails to improve for 6 consecutive epochs.
2. **ReduceLROnPlateau**: Halves the learning rate if validation loss stagnates for 3 epochs.
3. **ModelCheckpoint**: Automatically serializes the `.keras` model with the highest validation accuracy to `models/disease_model/`.
4. **CSVLogger**: Records per-epoch loss and accuracy values to `results/disease/training_history.csv`.

---

## 3. Severity Model Training

```bash
python training/train_severity.py --epochs 25 --batch-size 32 --learning-rate 0.0001
```

Outputs:
- Serialized Model: `models/severity_model/severity_classifier.keras`
- Training Curves: `results/severity/severity_training_curves.png`

---

## 4. Model Evaluation

```bash
# Evaluate Disease Classifier
python training/evaluate_disease.py

# Evaluate Severity Classifier
python training/evaluate_severity.py
```

Outputs:
- `results/disease/classification_report.txt`
- `results/disease/evaluation_metrics.json`
- `results/disease/confusion_matrix.png`
- `results/disease/per_class_metrics.png`
