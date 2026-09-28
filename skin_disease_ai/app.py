import os
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

import config
from database.database import init_db, save_prediction_record, get_all_records, get_record_by_id
from utils.validation import validate_image_upload, is_allowed_file
from utils.preprocessing import preprocess_pil_image
from utils.dataset_loader import get_overall_dataset_status
from utils.helpers import generate_safe_unique_filename, get_disease_info, get_severity_info
from inference.disease_predictor import DiseasePredictor, ModelNotAvailableError
from inference.severity_predictor import (
    SeverityPredictor,
    SeverityModelNotAvailableError,
    SEVERITY_UNAVAILABLE_MESSAGE,
)
from inference.gradcam import generate_and_save_gradcam

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

    # Initialize Predictors (will load models if weight files exist)
    disease_predictor = DiseasePredictor()
    severity_predictor = SeverityPredictor()

    @app.errorhandler(413)
    def file_too_large(e):
        max_mb = config.MAX_CONTENT_LENGTH // (1024 * 1024)
        error_msg = f"The uploaded file is too large. Maximum allowed size is {max_mb} MB."
        if request.path == "/predict" or request.is_json:
            return jsonify({"error": error_msg, "status": "failed"}), 413
        flash(error_msg, "danger")
        return redirect(url_for("detect"))

    @app.errorhandler(500)
    def internal_error(e):
        error_msg = "An unexpected server error occurred during processing."
        if request.path == "/predict" or request.is_json:
            return jsonify({"error": error_msg, "status": "failed"}), 500
        flash(error_msg, "danger")
        return redirect(url_for("index"))

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

            # Preprocess image
            img_tensor = preprocess_pil_image(pil_img, target_size=config.IMAGE_SIZE, normalize=True)

            predicted_disease = None
            disease_display_name = None
            disease_confidence = None
            top_preds_json = None
            is_model_trained = False
            gradcam_filename = None

            # 1. Disease Inference
            try:
                disease_result = disease_predictor.predict(img_tensor)
                predicted_disease = disease_result["disease"]
                disease_display_name = disease_result["disease_display_name"]
                disease_confidence = disease_result["disease_confidence"]
                top_preds_json = json.dumps(disease_result["top_predictions"])
                is_model_trained = True

                # Generate Grad-CAM if enabled and disease model is loaded
                enable_gradcam = request.form.get("enable_gradcam") == "on"
                if enable_gradcam and disease_predictor.is_available():
                    grad_name = f"gradcam_{unique_filename}"
                    gradcam_filename = generate_and_save_gradcam(
                        model=disease_predictor.model,
                        preprocessed_tensor=img_tensor,
                        original_image_path=saved_filepath,
                        output_filename=grad_name
                    )

            except ModelNotAvailableError:
                predicted_disease = "pending_model"
                disease_display_name = "Disease Model Pending Training"
                disease_confidence = None
                top_preds_json = json.dumps([])
                is_model_trained = False
            except Exception as e:
                print(f"[Error] Disease prediction failure: {e}")
                flash("An error occurred during disease prediction.", "warning")
                predicted_disease = "error"
                disease_display_name = "Prediction Error"

            # 2. Severity Inference
            predicted_severity = None
            severity_display_name = None
            severity_confidence = None

            enable_severity = request.form.get("enable_severity") == "on"
            if enable_severity:
                try:
                    severity_result = severity_predictor.predict(img_tensor)
                    predicted_severity = severity_result["severity"]
                    severity_display_name = severity_result["severity_display_name"]
                    severity_confidence = severity_result["severity_confidence"]
                except SeverityModelNotAvailableError:
                    predicted_severity = "unavailable"
                    severity_display_name = SEVERITY_UNAVAILABLE_MESSAGE
                    severity_confidence = None
                except Exception as e:
                    print(f"[Error] Severity prediction failure: {e}")
                    predicted_severity = "error"
                    severity_display_name = "Severity Estimation Error"

            # 3. Persist record in SQLite
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
                gradcam_filename=gradcam_filename,
                is_model_trained=is_model_trained,
                notes="Standard user inference run.",
            )

            if not is_model_trained:
                flash(
                    "Image uploaded and preprocessed successfully. Notice: Trained disease model weights were not found. Run 'python training/train_disease.py' after placing dataset images.",
                    "info"
                )

            return redirect(url_for("view_result", record_id=record.id))

        return render_template(
            "detect.html",
            disclaimer=config.MEDICAL_DISCLAIMER,
        )

    @app.route("/predict", methods=["POST"])
    def predict_api():
        """
        Dedicated JSON API endpoint for image prediction.
        Accepts multipart/form-data with 'image' file field.
        Returns standardized JSON prediction payload.
        """
        if "image" not in request.files:
            return jsonify({
                "status": "error",
                "error": "No file attached under key 'image'.",
                "disclaimer": config.MEDICAL_DISCLAIMER,
            }), 400

        file = request.files["image"]
        is_valid, error_msg, pil_img = validate_image_upload(file)

        if not is_valid:
            return jsonify({
                "status": "error",
                "error": error_msg,
                "disclaimer": config.MEDICAL_DISCLAIMER,
            }), 400

        # Save uploaded file
        original_filename = secure_filename(file.filename)
        unique_filename = generate_safe_unique_filename(original_filename)
        saved_filepath = config.UPLOAD_FOLDER / unique_filename
        pil_img.save(saved_filepath)

        img_tensor = preprocess_pil_image(pil_img, target_size=config.IMAGE_SIZE, normalize=True)

        disease_name = None
        disease_confidence = None
        disease_probabilities = {}
        gradcam_image_path = None

        # Disease model inference
        try:
            disease_res = disease_predictor.predict(img_tensor)
            disease_name = disease_res["disease"]
            disease_confidence = disease_res["disease_confidence"]
            disease_probabilities = disease_res["disease_probabilities"]

            if disease_predictor.is_available():
                grad_name = f"gradcam_{unique_filename}"
                grad_res = generate_and_save_gradcam(
                    model=disease_predictor.model,
                    preprocessed_tensor=img_tensor,
                    original_image_path=saved_filepath,
                    output_filename=grad_name
                )
                if grad_res:
                    gradcam_image_path = f"uploads/{grad_res}"
        except ModelNotAvailableError:
            disease_name = "unavailable"
            disease_confidence = None
            disease_probabilities = {"message": "Disease model weights not found on server."}

        # Severity model inference
        severity_name = None
        severity_confidence = None
        severity_probabilities = {}

        try:
            severity_res = severity_predictor.predict(img_tensor)
            severity_name = severity_res["severity"]
            severity_confidence = severity_res["severity_confidence"]
            severity_probabilities = severity_res["severity_probabilities"]
        except SeverityModelNotAvailableError:
            severity_name = "unavailable"
            severity_confidence = None
            severity_probabilities = {"message": SEVERITY_UNAVAILABLE_MESSAGE}

        response_payload = {
            "status": "success",
            "disease": disease_name,
            "disease_confidence": disease_confidence,
            "severity": severity_name,
            "severity_confidence": severity_confidence,
            "disease_probabilities": disease_probabilities,
            "severity_probabilities": severity_probabilities,
            "original_image": f"uploads/{unique_filename}",
            "gradcam_image": gradcam_image_path,
            "disclaimer": config.MEDICAL_DISCLAIMER,
        }
        return jsonify(response_payload), 200

    @app.route("/result")
    def result_redirect():
        """Redirects generic /result to history or latest item."""
        records = get_all_records(limit=1)
        if records:
            return redirect(url_for("view_result", record_id=records[0].id))
        return redirect(url_for("detect"))

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
            severity_unavailable_message=SEVERITY_UNAVAILABLE_MESSAGE,
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
            "version": "1.0.0-stage3",
            "disclaimer": config.MEDICAL_DISCLAIMER,
        })

    @app.route("/api/dataset-status")
    def api_dataset_status():
        status = get_overall_dataset_status()
        return jsonify(status)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=config.DEBUG)
