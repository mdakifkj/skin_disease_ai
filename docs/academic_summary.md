# Academic Project Summary & Presentation Brief

## 1. Project Title
**AI-Based Skin Disease Detection and Severity Prediction using Deep Learning**

---

## 2. Problem Statement
Dermatological disorders impact over 1.9 billion people globally, ranking among the top causes of non-fatal disease burden. In conventional clinical practice, early identification is constrained by a shortage of certified dermatologists, high geographic disparity in healthcare access, and subjective visual assessments. Furthermore, conventional machine learning models often function as opaque "black boxes" that fail to offer visual justification for predictions, conflate disease category with severity level, or fabricate diagnostic metrics.

---

## 3. Objectives
1. Build a robust transfer-learning computer vision framework based on **EfficientNetB0** for multi-class skin lesion classification across 7 target conditions.
2. Establish a dedicated secondary classification branch for **severity estimation** (*Mild*, *Moderate*, *Severe*) with strict data integrity isolation.
3. Integrate **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate visual heatmap overlays highlighting lesion boundaries and morphological features.
4. Implement a full-stack web application (Flask + Bootstrap 5 + Chart.js) with local SQLite audit persistence, drag-and-drop ingestion, and REST API support.
5. Enforce ethical AI guidelines, ensuring non-medical disclaimers are prominently displayed and no synthetic metrics are fabricated.

---

## 4. Existing System vs. Proposed System

### Existing Systems:
- Limited to binary classification (e.g. Melanoma vs Non-Melanoma).
- Monolithic architectures that conflate disease identity with severity rating.
- Lack visual interpretability or gradient-based attention heatmaps.
- Often run only as isolated Jupyter notebooks without full-stack web interfaces or audit trails.

### Proposed System (DermaVision AI):
- Multi-class classification across 7 clinically relevant dermatological conditions.
- Decoupled dual-branch inference for disease classification and severity scoring.
- Visual explainability via Grad-CAM heatmap overlays on convolutional feature maps.
- Complete full-stack web application with responsive healthcare UI, Dark/Light modes, interactive Chart.js visualizations, and SQLite database logging.
- Comprehensive security checks (file format, dimension bounds, size limits, corruption detection, shielded stack traces).

---

## 5. System Modules
1. **Validation & Security Ingestion Module**: Verifies MIME types, dimensions, file size ($\le 16\text{ MB}$), and image integrity.
2. **Preprocessing Pipeline Module**: Lanczos resampling ($224 \times 224$), RGB standardization, and float32 normalization.
3. **Disease Inference Engine**: EfficientNetB0 backbone + custom Dense Softmax head.
4. **Severity Inference Engine**: Dedicated 3-class severity classifier with missing-weight handling.
5. **Grad-CAM Explainability Module**: Computes gradients on final conv layers (`top_conv`) and renders heatmap overlays.
6. **Database Audit Module**: SQLAlchemy SQLite ORM managing prediction histories.
7. **Interactive Frontend Dashboard**: Bootstrap 5 + Chart.js responsive UI with live search/filter.
8. **RESTful API Service**: Endpoint (`POST /predict`) for automated client integration.

---

## 6. Hardware & Software Requirements

### Hardware Requirements:
- **Processor**: Intel Core i5/i7 (8th Gen or newer) / AMD Ryzen 5/7 or Apple Silicon
- **RAM**: Minimum 8 GB (16 GB recommended for model training)
- **Storage**: Minimum 5 GB available disk space for datasets and models
- **GPU (Optional)**: NVIDIA GPU with CUDA support for accelerated model training

### Software Requirements:
- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Programming Language**: Python 3.12+ / Python 3.13
- **Frameworks & Libraries**: Flask 3.x, Flask-SQLAlchemy, TensorFlow 2.15+, OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pillow
- **Frontend Assets**: HTML5, CSS3, Bootstrap 5.3.3, Bootstrap Icons, Chart.js 4.4.1

---

## 7. Dataset Taxonomy
- **Disease Dataset (7 Classes)**: Acne Vulgaris, Eczema, Fungal Infection (Tinea), Healthy/Normal Skin, Impetigo, Psoriasis, Rosacea.
- **Severity Dataset (3 Classes)**: Mild, Moderate, Severe (isolated branch).

---

## 8. Algorithms & Deep Learning Techniques
- **Transfer Learning**: EfficientNetB0 (pretrained on ImageNet)
- **Regularization**: Spatial Dropout (0.35), Dense Dropout (0.25), Batch Normalization
- **Optimization**: Adam Optimizer with learning rate scheduler (`ReduceLROnPlateau`)
- **Loss Function**: Sparse Categorical Crossentropy with inverse-frequency class weighting
- **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM)

---

## 9. Expected Output
- Multi-class predicted probability distribution.
- Primary condition and confidence percentage.
- Condition severity estimate (*Mild*, *Moderate*, *Severe*) or clean unavailable notice if unannotated.
- Superimposed Grad-CAM visual heatmap overlay.
- Interactive Chart.js probability breakdown chart.
- Searchable and filterable historical audit record in SQLite.

---

## 10. Advantages
- **High Computational Efficiency**: EfficientNet compound scaling optimizes parameter efficiency.
- **Explainable Decisions**: Eliminates black-box opacity through Grad-CAM localization.
- **Data Integrity**: Never confuses disease taxonomy with severity grades.
- **Production-Ready Web Architecture**: Secure, responsive, and mobile-ready.

---

## 11. Limitations & Future Scope

### Limitations:
- Clinical photographic variability (uneven lighting, motion blur, angle).
- Dataset representation disparities across diverse Fitzpatrick phototypes.
- Educational and research demonstrator only; strictly not a certified diagnostic device.

### Future Scope:
- Integration of multi-modal metadata (patient age, lesion duration, anatomical site).
- Deployment of mobile-optimized quantization (TensorFlow Lite / ONNX).
- Expansion of disease taxonomy to rare dermatological disorders and melanoma screening.
