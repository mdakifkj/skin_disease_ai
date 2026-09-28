from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionRecord(db.Model):
    """Stores past image uploads, disease classifications, severity predictions, and Grad-CAM metadata."""
    __tablename__ = "prediction_records"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    
    # Disease prediction
    predicted_disease = db.Column(db.String(100), nullable=True)
    disease_display_name = db.Column(db.String(150), nullable=True)
    disease_confidence = db.Column(db.Float, nullable=True)
    top_predictions_json = db.Column(db.Text, nullable=True)  # JSON string of top-3 or top-k
    
    # Severity prediction
    predicted_severity = db.Column(db.String(50), nullable=True)
    severity_display_name = db.Column(db.String(100), nullable=True)
    severity_confidence = db.Column(db.Float, nullable=True)
    
    # Explainability / Grad-CAM
    gradcam_filename = db.Column(db.String(255), nullable=True)
    
    # Metadata
    is_model_trained = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "predicted_disease": self.predicted_disease,
            "disease_display_name": self.disease_display_name,
            "disease_confidence": round(self.disease_confidence * 100, 2) if self.disease_confidence is not None else None,
            "predicted_severity": self.predicted_severity,
            "severity_display_name": self.severity_display_name,
            "severity_confidence": round(self.severity_confidence * 100, 2) if self.severity_confidence is not None else None,
            "gradcam_filename": self.gradcam_filename,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
