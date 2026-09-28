import os
import uuid
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
import config

DISEASE_INFORMATION = {
    "acne": {
        "title": "Acne Vulgaris",
        "description": "A common inflammatory skin condition that occurs when hair follicles become clogged with oil and dead skin cells, leading to comedones, papules, pustules, or cysts.",
        "common_characteristics": "Blackheads, whiteheads, pimples, oily skin, possible scarring.",
        "research_notes": "Frequently assessed using lesion count and inflammatory severity indices (such as Leeds or GAGS scoring in clinical dermatology).",
    },
    "eczema": {
        "title": "Eczema (Atopic Dermatitis)",
        "description": "A chronic inflammatory condition marked by dry, pruritic (itchy), and erythematous patches that can weep or become lichenified.",
        "common_characteristics": "Dry red skin, intense itching, cracked or leathery patches, skin barrier dysfunction.",
        "research_notes": "Clinical research often evaluates severity via EASI (Eczema Area and Severity Index) or SCORAD.",
    },
    "fungal_infection": {
        "title": "Fungal Infection (Tinea / Dermatophytosis)",
        "description": "Superficial fungal infection of keratinized tissue (skin, hair, nails) commonly caused by dermatophytes (e.g., Tinea corporis, Tinea pedis, Tinea versicolor).",
        "common_characteristics": "Annular (ring-shaped) erythematous scaly plaques with central clearing, itching, scaling edges.",
        "research_notes": "Computer vision models typically focus on identifying characteristic active expanding borders.",
    },
    "healthy": {
        "title": "Healthy / Normal Skin",
        "description": "Normal skin tissue with intact barrier function, absence of active erythematous, scaly, pustular, or pigmented lesions.",
        "common_characteristics": "Even pigmentation, no inflammatory papules or open lesions, normal skin turgor.",
        "research_notes": "Crucial negative control class to prevent false-positive disease classifications in computer vision models.",
    },
    "impetigo": {
        "title": "Impetigo",
        "description": "A contagious superficial bacterial skin infection, predominantly caused by Staphylococcus aureus or Streptococcus pyogenes.",
        "common_characteristics": "Honey-colored crusted erosions, erythematous base, fluid-filled vesicles.",
        "research_notes": "Common in pediatric dermatology datasets; distinct crusting pattern provides key morphological features.",
    },
    "psoriasis": {
        "title": "Psoriasis",
        "description": "A chronic autoimmune dermatosis characterized by rapid skin cell proliferation, forming thick, silvery-scaled erythematous plaques.",
        "common_characteristics": "Well-demarcated pink or red plaques covered with silvery-white scales, commonly on extensor surfaces (elbows, knees, scalp).",
        "research_notes": "Clinical severity is standardly scored using PASI (Psoriasis Area and Severity Index).",
    },
    "rosacea": {
        "title": "Rosacea",
        "description": "A chronic vascular and inflammatory skin condition primarily affecting central facial regions.",
        "common_characteristics": "Facial erythema, telangiectasia (visible blood vessels), papulopustular breakouts, facial flushing.",
        "research_notes": "Distinguished in ML benchmarks from acne by absence of comedones and predilection for central face.",
    },
}

SEVERITY_INFORMATION = {
    "mild": {
        "label": "Mild",
        "description": "Localized, low lesion count or mild erythema; minimal skin barrier breakdown.",
        "color_class": "text-success",
        "badge_class": "bg-success",
    },
    "moderate": {
        "label": "Moderate",
        "description": "Widespread involvement, elevated inflammatory papules/plaques, or noticeable surface disruption.",
        "color_class": "text-warning",
        "badge_class": "bg-warning text-dark",
    },
    "severe": {
        "label": "Severe",
        "description": "Extensive surface area involvement, deep inflammatory lesions, confluent plaques, or severe crusting/exudation.",
        "color_class": "text-danger",
        "badge_class": "bg-danger",
    },
}

def generate_safe_unique_filename(original_filename: str) -> str:
    """Generates a secure, timestamped unique filename to prevent collisions and path injection."""
    clean_name = secure_filename(original_filename)
    stem, ext = os.path.splitext(clean_name)
    if not ext:
        ext = ".jpg"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{unique_suffix}{ext.lower()}"

def get_disease_info(disease_key: str) -> dict:
    """Returns educational metadata dictionary for a disease key."""
    return DISEASE_INFORMATION.get(disease_key, {
        "title": disease_key.replace("_", " ").title() if disease_key else "Unknown",
        "description": "No educational description available for this class.",
        "common_characteristics": "N/A",
        "research_notes": "N/A",
    })

def get_severity_info(severity_key: str) -> dict:
    """Returns educational metadata for a severity level."""
    return SEVERITY_INFORMATION.get(severity_key, {
        "label": severity_key.title() if severity_key else "Unknown",
        "description": "Severity scale metadata pending training annotation.",
        "color_class": "text-secondary",
        "badge_class": "bg-secondary",
    })
