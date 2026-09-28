# System Architecture & Technical Specifications

## 1. High-Level Architecture Overview

DermaVision AI is designed with a decoupled, modular architecture adhering to clean software engineering principles and ethical medical AI standards.

```
+-------------------------------------------------------------------------+
|                              Client Browser                             |
|    HTML5 | Bootstrap 5.3 | Inter Font | Chart.js 4.4 | JavaScript ES6   |
+-------------------------------------------------------------------------+
                                    |  (HTTP / Multipart)
                                    v
+-------------------------------------------------------------------------+
|                        Flask Web Application Router                     |
|           app.py (/ | /detect | /predict | /result | /history)          |
+-------------------------------------------------------------------------+
       |                                |                     |
       v                                v                     v
+---------------+             +--------------------+   +-------------------+
|  Validation   |             | Dual-Branch Neural |   |  SQLite Database  |
|  & Security   |             | Network Inference  |   | (SQLAlchemy ORM)  |
+---------------+             +--------------------+   +-------------------+
| - Extension   |             | 1. Disease Model:  |   | - PredictionRecord|
| - Dimensions  |             |    EfficientNetB0  |   | - Timestamps (UTC)|
| - Corruption  |             | 2. Severity Branch:|   | - Class Confidence|
| - File limits |             |    3-Class Dense   |   | - No Personal Data|
+---------------+             | 3. Grad-CAM        |   +-------------------+
                              |    Explainability  |
                              +--------------------+
```

---

## 2. Component Specifications

### 2.1 Preprocessing Pipeline
- **Resampling**: Uses high-fidelity Lanczos bicubic interpolation to downsample or upsample raw clinical images into the standard $224 \times 224$ pixel input format.
- **Color Space**: Ensures invariant 3-channel RGB representation (converting grayscale $L$, RGBA, and palette-indexed modes).
- **Normalization**: Scales 8-bit integer pixel data into float32 tensors with range $[0.0, 1.0]$.
- **Batch Shaping**: Expands dimensions to $(1, 224, 224, 3)$ matching Keras tensor input conventions.

### 2.2 Dual-Branch AI Models
1. **Disease Classification Branch**:
   - **Backbone**: EfficientNetB0 initialized with ImageNet pretrained feature extraction filters.
   - **Regularization**: Batch Normalization, Spatial Dropout (0.35), Dense Dropout (0.25).
   - **Head**: 256-unit Dense ReLU layer followed by a 7-class Softmax activation.
2. **Severity Estimation Branch**:
   - **Backbone**: Transfer learning model fine-tuned specifically on clinician-annotated severity scales (*Mild*, *Moderate*, *Severe*).
   - **Safety Feature**: If no validated severity weights exist, the pipeline emits a clean `"Unavailable"` status without fabricating diagnostic estimates.

### 2.3 Visual Explainability (Grad-CAM)
- Extracts feature maps from the penultimate convolutional activation layer (`top_conv`).
- Calculates the first-order gradient of the winning class score $y^c$ with respect to feature activation maps $A^k$:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
- Generates a localization map:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- Colorizes the normalized heatmap using the OpenCV `COLORMAP_JET` matrix and superimposes it over the original image at $\alpha = 0.4$ opacity.

---

## 3. Database Schema

The system uses an SQLite database managed via SQLAlchemy (`database/models.py`):

```sql
CREATE TABLE prediction_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(512) NOT NULL,
    predicted_disease VARCHAR(100),
    disease_display_name VARCHAR(150),
    disease_confidence FLOAT,
    top_predictions_json TEXT,
    predicted_severity VARCHAR(50),
    severity_display_name VARCHAR(100),
    severity_confidence FLOAT,
    gradcam_filename VARCHAR(255),
    is_model_trained BOOLEAN DEFAULT 0,
    notes TEXT,
    created_at DATETIME NOT NULL
);
```
