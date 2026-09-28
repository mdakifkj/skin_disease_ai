# Dataset Specifications & Preparation Guidelines

## 1. Disease Classification Dataset

The system supports 7 distinct target dermatological classes:

| Class Key | Display Name | Clinical Overview | Recommended Sources |
| :--- | :--- | :--- | :--- |
| `acne` | Acne Vulgaris | Inflammatory lesions of hair follicles & sebaceous glands | ISIC, DermNet NZ, SD-198 |
| `eczema` | Eczema (Atopic Dermatitis) | Pruritic, erythematous dry scaly plaques | DermNet NZ, Fitzpatrick 17k |
| `fungal_infection` | Fungal Infection (Tinea) | Superficial annular scaly lesions with active borders | DermNet NZ |
| `healthy` | Healthy / Normal Skin | Intact skin tissue without inflammatory lesions | PadChest / Normal Dermatology Controls |
| `impetigo` | Impetigo | Superficial bacterial skin infection with honey-colored crusts | DermNet NZ |
| `psoriasis` | Psoriasis | Chronic erythematous plaques with thick silvery scales | DermNet NZ, HAM10000 |
| `rosacea` | Rosacea | Facial vascular flushing, erythema, and telangiectasia | DermNet NZ, SD-198 |

---

## 2. Directory Structure Conventions

Datasets must be organized into standard directory splits:

```text
dataset/
├── train/
│   ├── acne/
│   ├── eczema/
│   ├── fungal_infection/
│   ├── healthy/
│   ├── impetigo/
│   ├── psoriasis/
│   └── rosacea/
├── validation/
│   ├── acne/
│   ├── eczema/
│   ├── fungal_infection/
│   ├── healthy/
│   ├── impetigo/
│   ├── psoriasis/
│   └── rosacea/
└── test/
    ├── acne/
    ├── eczema/
    ├── fungal_infection/
    ├── healthy/
    ├── impetigo/
    ├── psoriasis/
    └── rosacea/
```

---

## 3. Severity Classification Dataset

```text
severity_dataset/
├── train/
│   ├── mild/
│   ├── moderate/
│   └── severe/
├── validation/
│   ├── mild/
│   ├── moderate/
│   └── severe/
└── test/
    ├── mild/
    ├── moderate/
    └── severe/
```

> ### 🛑 Critical Medical Integrity Note
> Disease category datasets (e.g. DermNet) do not contain reliable severity scores. To maintain scientific integrity, disease labels **must never be converted into severity scores**. Severity models must only be trained when dermatologist-annotated severity ratings (such as EASI or PASI scores) are available.
