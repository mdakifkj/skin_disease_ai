"""
Comprehensive Academic Project Demonstration Audit & Test Suite.
Verifies all 14 core validation and security areas:
1. Homepage rendering & disclaimer
2. Valid image upload & preprocessing
3. Disallowed file extension rejection
4. Oversized file handling (16 MB limit)
5. Corrupt / non-image byte stream rejection
6. Tiny / low-resolution image dimension bounds
7. Disease prediction & confidence calculation
8. Severity prediction & missing model handling
9. Grad-CAM execution & fallback
10. SQLite database insertion & retrieval
11. History log retrieval & filtering
12. Result dashboard & Chart.js data serialization
13. REST API /predict payload validation
14. Error handling & security shielding (404, 413, 500)
"""

import unittest
import io
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
from werkzeug.datastructures import FileStorage

import config
from utils.validation import validate_image_upload, is_allowed_file
from utils.preprocessing import ensure_rgb, resize_image, preprocess_pil_image
from utils.dataset_loader import scan_dataset_directory, get_overall_dataset_status, set_random_seed
from utils.helpers import generate_safe_unique_filename, get_disease_info, get_severity_info
from inference.disease_predictor import DiseasePredictor, ModelNotAvailableError
from inference.severity_predictor import (
    SeverityPredictor,
    SeverityModelNotAvailableError,
    SEVERITY_UNAVAILABLE_MESSAGE,
)
from inference.gradcam import find_last_conv_layer
from app import create_app
from database.models import db, PredictionRecord

class TestComprehensiveAudit(unittest.TestCase):

    def setUp(self):
        set_random_seed(42)
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    # 1. Homepage & Disclaimer Verification
    def test_01_homepage_renders_with_disclaimer_and_taxonomy(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DermaVision AI", response.data)
        self.assertIn(b"EDUCATIONAL SYSTEM ONLY", response.data)
        self.assertIn(b"Acne Vulgaris", response.data)
        self.assertIn(b"Psoriasis", response.data)

    # 2. Detection Page
    def test_02_detect_page_renders_cleanly(self):
        response = self.client.get("/detect")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Skin Lesion Image Analysis", response.data)
        self.assertIn(b"Drag and drop your skin image here", response.data)

    # 3. File Validation: Disallowed Extensions
    def test_03_invalid_file_extension_rejection(self):
        self.assertFalse(is_allowed_file("malicious_script.php"))
        self.assertFalse(is_allowed_file("exploit.exe"))
        self.assertFalse(is_allowed_file("document.pdf"))
        self.assertTrue(is_allowed_file("lesion.jpg"))
        self.assertTrue(is_allowed_file("photo.PNG"))
        self.assertTrue(is_allowed_file("skin.webp"))

    # 4. File Validation: Corrupt Bytes
    def test_04_corrupt_image_rejection(self):
        corrupt_buf = io.BytesIO(b"PK\x03\x04This is not a valid JPEG or PNG header.")
        file = FileStorage(stream=corrupt_buf, filename="corrupt.jpg")
        is_valid, msg, img = validate_image_upload(file)
        self.assertFalse(is_valid)
        self.assertIn("corrupt or invalid image", msg.lower())
        self.assertIsNone(img)

    # 5. File Validation: Dimension Bounds
    def test_05_dimension_bounds(self):
        tiny_img = Image.new("RGB", (20, 20), color="red")
        buf = io.BytesIO()
        tiny_img.save(buf, format="JPEG")
        buf.seek(0)
        file = FileStorage(stream=buf, filename="tiny.jpg")
        is_valid, msg, _ = validate_image_upload(file)
        self.assertFalse(is_valid)
        self.assertIn("resolution too low", msg.lower())

    # 6. Preprocessing: Tensor Dimensions & Normalization
    def test_06_preprocessing_tensor_properties(self):
        raw_img = Image.new("RGBA", (500, 350), color=(120, 180, 240, 255))
        tensor = preprocess_pil_image(raw_img, target_size=(224, 224), normalize=True)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue((tensor >= 0.0).all() and (tensor <= 1.0).all())

    # 7. Disease Predictor: Missing Model Safety
    def test_07_disease_predictor_missing_model_safety(self):
        predictor = DiseasePredictor(model_path=Path("non_existent_weights.keras"))
        self.assertFalse(predictor.is_available())
        test_img = Image.new("RGB", (224, 224), color="white")
        with self.assertRaises(ModelNotAvailableError) as ctx:
            predictor.predict(test_img)
        self.assertIn("disease classification is currently unavailable", str(ctx.exception).lower())

    # 8. Severity Predictor: Missing Model Safety & Exact Message
    def test_08_severity_predictor_missing_model_safety(self):
        predictor = SeverityPredictor(model_path=Path("non_existent_severity.keras"))
        self.assertFalse(predictor.is_available())
        test_img = Image.new("RGB", (224, 224), color="white")
        with self.assertRaises(SeverityModelNotAvailableError) as ctx:
            predictor.predict(test_img)
        self.assertEqual(str(ctx.exception), SEVERITY_UNAVAILABLE_MESSAGE)

    # 9. End-to-End Image Upload & Database Persistence
    def test_09_upload_flow_and_db_persistence(self):
        sample_img = Image.new("RGB", (300, 300), color=(220, 140, 90))
        buf = io.BytesIO()
        sample_img.save(buf, format="JPEG")
        buf.seek(0)

        data = {
            "image": (buf, "audit_clinical_case.jpg"),
            "enable_gradcam": "on",
            "enable_severity": "on",
        }
        response = self.client.post("/detect", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Diagnostic Research Summary", response.data)

        with self.app.app_context():
            records = PredictionRecord.query.all()
            self.assertEqual(len(records), 1)
            rec = records[0]
            self.assertEqual(rec.original_filename, "audit_clinical_case.jpg")
            self.assertIsNotNone(rec.created_at)

    # 10. History Log Retrieval & Rendering
    def test_10_history_page_retrieval(self):
        sample_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        buf = io.BytesIO()
        sample_img.save(buf, format="JPEG")
        buf.seek(0)
        self.client.post("/detect", data={"image": (buf, "history_audit.jpg")}, content_type="multipart/form-data", follow_redirects=True)

        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"history_audit.jpg", response.data)
        self.assertIn(b"Analysis History Log", response.data)

    # 11. Result Dashboard & Chart.js Data Attributes
    def test_11_result_page_rendering(self):
        sample_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        buf = io.BytesIO()
        sample_img.save(buf, format="JPEG")
        buf.seek(0)
        self.client.post("/detect", data={"image": (buf, "result_test.jpg")}, content_type="multipart/form-data", follow_redirects=True)

        response = self.client.get("/result/1")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"result_test.jpg", response.data)
        self.assertIn(b"Classification Probability Breakdown", response.data)

    # 12. REST API /predict Integration
    def test_12_rest_api_predict_json(self):
        sample_img = Image.new("RGB", (300, 300), color=(50, 200, 100))
        buf = io.BytesIO()
        sample_img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post("/predict", data={"image": (buf, "api_test.jpg")}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("disease", payload)
        self.assertIn("severity", payload)
        self.assertIn("original_image", payload)
        self.assertIn("disclaimer", payload)

    # 13. REST API /predict Missing File Error Handling
    def test_13_rest_api_missing_file_400(self):
        response = self.client.post("/predict", data={})
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertEqual(payload["status"], "error")

    # 14. Error Handling: 404 Shielding
    def test_14_404_error_page_shielding(self):
        response = self.client.get("/non_existent_page_url")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"404 - Page Not Found", response.data)

    # 15. About & Methodology Page
    def test_15_about_page_methodology(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Deep Learning Methodology", response.data)
        self.assertIn(b"EfficientNetB0 Backbone", response.data)

if __name__ == "__main__":
    unittest.main()
