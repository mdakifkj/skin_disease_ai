import io
from PIL import Image
import pytest
from app import create_app
from database.models import db, PredictionRecord

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["WTF_CSRF_ENABLED"] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"DermaVision AI" in response.data
    assert b"EDUCATIONAL DEMONSTRATION ONLY" in response.data

def test_detect_page_get(client):
    response = client.get("/detect")
    assert response.status_code == 200
    assert b"Skin Lesion Image Analysis" in response.data

def test_about_page(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"System Architecture" in response.data

def test_history_page(client):
    response = client.get("/history")
    assert response.status_code == 200

def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "online"

def test_api_dataset_status(client):
    response = client.get("/api/dataset-status")
    assert response.status_code == 200
    data = response.get_json()
    assert "disease_dataset" in data
    assert "severity_dataset" in data

def test_upload_and_prediction_pipeline(client):
    # Create sample image
    img = Image.new("RGB", (300, 300), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    data = {
        "image": (img_byte_arr, "test_skin_sample.jpg"),
        "enable_severity": "on",
        "enable_gradcam": "on",
    }

    response = client.post("/detect", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Diagnostic Research Summary" in response.data
