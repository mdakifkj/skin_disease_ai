# REST API & Integration Specifications

## 1. Prediction API (`POST /predict`)

Accepts multipart form-data containing a clinical image and returns structured inference predictions.

### Request:
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `image`: File binary (JPG, PNG, WEBP, $\le 16\text{ MB}$)

### Successful Response (`200 OK`):
```json
{
  "status": "success",
  "disease": "acne",
  "disease_confidence": 0.9421,
  "severity": "unavailable",
  "severity_confidence": null,
  "disease_probabilities": {
    "acne": {
      "class_key": "acne",
      "display_name": "Acne Vulgaris",
      "probability": 0.9421,
      "confidence_percent": 94.21
    },
    "eczema": {
      "class_key": "eczema",
      "display_name": "Eczema (Atopic Dermatitis)",
      "probability": 0.0215,
      "confidence_percent": 2.15
    }
  },
  "severity_probabilities": {
    "message": "Severity prediction is currently unavailable because a trained severity model has not been provided."
  },
  "original_image": "uploads/20260928_104500_a1b2c3d4.jpg",
  "gradcam_image": "uploads/gradcam_20260928_104500_a1b2c3d4.jpg",
  "disclaimer": "This system is for educational and research purposes only and is not a substitute for professional medical diagnosis. Users should consult a qualified healthcare professional for medical advice."
}
```

---

## 2. Health & Status Endpoints

### 2.1 Health Check (`GET /api/health`)
```json
{
  "status": "online",
  "disease_model_available": true,
  "severity_model_available": false,
  "version": "1.0.0-final",
  "disclaimer": "This system is for educational and research purposes only and is not a substitute for professional medical diagnosis. Users should consult a qualified healthcare professional for medical advice."
}
```

### 2.2 Dataset Status (`GET /api/dataset-status`)
Returns JSON report detailing image counts per class and split across the disease and severity datasets.
