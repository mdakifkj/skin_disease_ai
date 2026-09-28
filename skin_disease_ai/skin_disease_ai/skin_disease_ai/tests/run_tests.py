import unittest
import io
import json
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np
from werkzeug.datastructures import FileStorage

import config
from utils.validation import validate_image_upload, is_allowed_file
from utils.preprocessing import ensure_rgb, resize_image, preprocess_pil_image
from utils.dataset_loader import scan_dataset_directory, get_overall_dataset_status
from utils.helpers import generate_safe_unique_filename, get_disease_info, get_severity_info
from app import create_app
from database.models import db, PredictionRecord

class TestValidationAndPreprocessing(unittest.TestCase):

    def test_allowed_extensions(self):
        self.assertTrue(is_allowed_file("skin_lesion.jpg"))
        self.assertTrue(is_allowed_file("sample.PNG"))
        self.assertTrue(is_allowed_file("test.webp"))
        self.assertFalse(is_allowed_file("report.pdf"))
        self.assertFalse(is_allowed_file("malicious.exe"))

    def test_image_validation_valid(self):
        img = Image.new("RGB", (250, 250), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        file = FileStorage(stream=buf, filename="sample.jpg")

        is_valid, msg, pil_img = validate_image_upload(file)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
        self.assertIsNotNone(pil_img)
        self.assertEqual(pil_img.size, (250, 250))

    def test_image_validation_invalid_dimension(self):
        img = Image.new("RGB", (20, 20), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        file = FileStorage(stream=buf, filename="tiny.jpg")

        is_valid, msg, pil_img = validate_image_upload(file)
        self.assertFalse(is_valid)
        self.assertIn("resolution too low", msg.lower())

    def test_preprocess_tensor_shape_and_norm(self):
        img = Image.new("RGBA", (300, 300), color=(100, 150, 200, 255))
        tensor = preprocess_pil_image(img, target_size=(224, 224), normalize=True)
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertLessEqual(tensor.max(), 1.0)
        self.assertGreaterEqual(tensor.min(), 0.0)

    def test_dataset_directory_structure(self):
        status = get_overall_dataset_status()
        self.assertTrue(status["disease_dataset"]["exists"])
        self.assertTrue(status["severity_dataset"]["exists"])
        self.assertIn("train", status["disease_dataset"]["splits"])
        self.assertIn("validation", status["disease_dataset"]["splits"])
        self.assertIn("test", status["disease_dataset"]["splits"])

    def test_helpers_metadata(self):
        acne_info = get_disease_info("acne")
        self.assertEqual(acne_info["title"], "Acne Vulgaris")
        mild_info = get_severity_info("mild")
        self.assertEqual(mild_info["label"], "Mild")

class TestFlaskEndpoints(unittest.TestCase):

    def setUp(self):
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

    def test_home_page_renders_with_disclaimer(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DermaVision AI", response.data)
        self.assertIn(b"EDUCATIONAL DEMONSTRATION ONLY", response.data)

    def test_detect_page_renders(self):
        response = self.client.get("/detect")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Skin Lesion Image Analysis", response.data)

    def test_about_page_renders(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"System Architecture", response.data)

    def test_history_page_empty(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No Analysis History Found", response.data)

    def test_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")

    def test_image_upload_and_pipeline_execution(self):
        img = Image.new("RGB", (250, 250), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        data = {
            "image": (buf, "clinical_sample.jpg"),
            "enable_severity": "on",
            "enable_gradcam": "on",
        }
        response = self.client.post("/detect", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction Results", response.data)

        # Check DB entry exists
        with self.app.app_context():
            records = PredictionRecord.query.all()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].original_filename, "clinical_sample.jpg")

if __name__ == "__main__":
    unittest.main()
