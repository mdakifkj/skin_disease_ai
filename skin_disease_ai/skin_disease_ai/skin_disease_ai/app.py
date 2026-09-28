import os
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

import config
from database.database import init_db, save_prediction_record, get_all_records, get_record_by_id
from utils.validation import validate_image_upload
from utils.preprocessing import preprocess_pil_image
from utils.dataset_loader import get_overall_dataset_status
from utils.helpers import generate_safe_unique_filename, get_disease_info, get_severity_info
from inference.disease_predictor import DiseasePredictor, ModelNotAvailableError
from inference.severity_predictor import SeverityPredictor, SeverityModelNotAvailableError

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = config.SQLALCHEMY_TRACK_MODIFICATIONS
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    # Ensure upload directory exists
    config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Initialize SQLite Database
    init_db(app)

    # Initialize Predictors
    disease_predictor = DiseasePredictor()
    severity_predictor = SeverityPredictor()

    @app.route("/")
    def index():
        status = get_overall_dataset_status()
        return render_template(
            "index.html",
            disclaimer=config.MEDICAL_DISCLAIMER,
            disease_classes=config.DISEASE_DISPLAY_NAMES,
            severity_classes=config.SEVERITY_DISPLAY_NAMES,
            dataset_status=status,
            disease_model_loaded=disease_predictor.is_available(),
            severity_model_loaded=severity_predictor.is_available(),
        )

    @app.route("/detect", methods=["GET", "POST"])
    def detect():
        if request.method == "POST":
            if "image" not in request.files:
                flash("No file part in the request.", "danger")
                return redirect(request.url)

            file = request.files["image"]
            is_valid, error_msg, pil_img = validate_image_upload(file)

            if not is_valid:
                flash(error_msg, "danger")
                return redirect(request.url)

            # Generate safe filename and save original upload
            original_filename = secure_filename(file.filename)
            unique_filename = generate_safe_unique_filename(original_filename)
            saved_filepath = config.UPLOAD_FOLDER / unique_filename
            pil_img.save(saved_filepath)

            # Preprocess image into float32 normalized tensor (1, 224, 224, 3)
            img_tensor = preprocess_pil_image(pil_img, target_size=config.IMAGE_SIZE)

            predicted_disease = None
            disease_display_name = None
            disease_confidence = None
            top_preds_json = None
            is_model_trained = False

            # Run disease model inference if model file is available
            try:
                disease_result = disease_predictor.predict(img_tensor)
                predicted_disease = disease_result["predicted_class"]
                disease_display_name = disease_result["display_name"]
                disease_confidence = disease_result["confidence"]
                top_preds_json = json.dumps(disease_result["top_predictions"])
                is_model_trained = True
            except ModelNotAvailableError:
                # Expected when model weights are not yet trained
                predicted_disease = "pending_model"
                disease_display_name = "Model Pending Training"
                disease_confidence = None
                top_preds_json = json.dumps([])
                is_model_trained = False

            # Severity model inference
            predicted_severity = None
            severity_display_name = None
            severity_confidence = None

            enable_severity = request.form.get("enable_severity") == "on"
            if enable_severity:
                try:
                    severity_result = severity_predictor.predict(img_tensor)
                    predicted_severity = severity_result["predicted_severity"]
                    severity_display_name = severity_result["display_name"]
                    severity_confidence = severity_result["confidence"]
                except SeverityModelNotAvailableError:
                    predicted_severity = "pending_annotation"
                    severity_display_name = "Severity Pending Training"
                    severity_confidence = None

            # Persist record in SQLite
            record = save_prediction_record(
                filename=unique_filename,
                original_filename=original_filename,
                filepath=str(saved_filepath),
                predicted_disease=predicted_disease,
                disease_display_name=disease_display_name,
                disease_confidence=disease_confidence,
                top_predictions_json=top_preds_json,
                predicted_severity=predicted_severity,
                severity_display_name=severity_display_name,
                severity_confidence=severity_confidence,
                gradcam_filename=None,
                is_model_trained=is_model_trained,
                notes="Stage 1 verified analysis run.",
            )

            if not is_model_trained:
                flash(
                    "Image uploaded and preprocessed successfully! Note: Disease classification model weights have not been trained yet. Train the model using 'python training/train_disease.py' once dataset images are placed.",
                    "info"
                )

            return redirect(url_for("view_result", record_id=record.id))

        return render_template(
            "detect.html",
            disclaimer=config.MEDICAL_DISCLAIMER,
        )

    @app.route("/result/<int:record_id>")
    def view_result(record_id):
        record = get_record_by_id(record_id)
        if not record:
            flash("Prediction record not found.", "warning")
            return redirect(url_for("history"))

        top_predictions = []
        if record.top_predictions_json:
            try:
                top_predictions = json.loads(record.top_predictions_json)
            except Exception:
                top_predictions = []

        disease_info = get_disease_info(record.predicted_disease)
        severity_info = get_severity_info(record.predicted_severity)

        return render_template(
            "result.html",
            record=record,
            top_predictions=top_predictions,
            disease_info=disease_info,
            severity_info=severity_info,
            disclaimer=config.MEDICAL_DISCLAIMER,
        )

    @app.route("/history")
    def history():
        records = get_all_records(limit=50)
        return render_template(
            "history.html",
            records=records,
            disclaimer=config.MEDICAL_DISCLAIMER,
        )

    @app.route("/about")
    def about():
        return render_template(
            "about.html",
            disclaimer=config.MEDICAL_DISCLAIMER,
            disease_classes=config.DISEASE_DISPLAY_NAMES,
            severity_classes=config.SEVERITY_DISPLAY_NAMES,
        )

    @app.route("/api/health")
    def api_health():
        return jsonify({
            "status": "online",
            "disease_model_available": disease_predictor.is_available(),
            "severity_model_available": severity_predictor.is_available(),
            "version": "1.0.0-stage1",
        })

    @app.route("/api/dataset-status")
    def api_dataset_status():
        status = get_overall_dataset_status()
        return jsonify(status)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=config.DEBUG)
