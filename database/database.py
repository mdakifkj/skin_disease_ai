from database.models import db, PredictionRecord

def init_db(app):
    """Initializes the database with the Flask application context."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

def save_prediction_record(
    filename: str,
    original_filename: str,
    filepath: str,
    predicted_disease: str = None,
    disease_display_name: str = None,
    disease_confidence: float = None,
    top_predictions_json: str = None,
    predicted_severity: str = None,
    severity_display_name: str = None,
    severity_confidence: float = None,
    gradcam_filename: str = None,
    is_model_trained: bool = False,
    notes: str = None,
) -> PredictionRecord:
    """Helper to persist a prediction event into SQLite."""
    record = PredictionRecord(
        filename=filename,
        original_filename=original_filename,
        filepath=filepath,
        predicted_disease=predicted_disease,
        disease_display_name=disease_display_name,
        disease_confidence=disease_confidence,
        top_predictions_json=top_predictions_json,
        predicted_severity=predicted_severity,
        severity_display_name=severity_display_name,
        severity_confidence=severity_confidence,
        gradcam_filename=gradcam_filename,
        is_model_trained=is_model_trained,
        notes=notes,
    )
    db.session.add(record)
    db.session.commit()
    return record

def get_all_records(limit: int = 50):
    """Retrieve history records sorted by newest first."""
    return PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).limit(limit).all()

def get_record_by_id(record_id: int):
    """Retrieve a single prediction record by primary key."""
    return db.session.get(PredictionRecord, record_id)
