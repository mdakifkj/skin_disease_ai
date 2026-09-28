# Quality Assurance & Testing Suite

## 1. Automated Test Execution

Run the comprehensive unit, security, and integration test suite:

```bash
python tests/test_audit_suite.py
```

---

## 2. Test Coverage Matrix

| Test Suite Category | Tested Scenarios | Expected Behavior |
| :--- | :--- | :--- |
| **Input Validation** | Valid image, corrupt byte stream, zero bytes, tiny dimensions ($<50\text{px}$), oversized dimensions | Emits clear descriptive error message without crashing |
| **Security & Uploads** | Disallowed extensions (`.exe`, `.pdf`, `.py`), file size $>16\text{ MB}$, path traversal injection | Rejected safely with HTTP 400/413 status code |
| **Model Ingestion** | Missing `.keras` weight files | Raises `ModelNotAvailableError` / emits graceful notice without fake metrics |
| **Database ORM** | SQLite insertion, retrieval, JSON serialization | Records preserved in `prediction_records` without storing PII |
| **REST API** | `POST /predict`, `GET /api/health`, `GET /api/dataset-status` | Returns valid JSON with disclaimer |
| **Frontend Web Routes** | `/`, `/detect`, `/result/<id>`, `/history`, `/about` | HTTP 200 OK, renders semantic Bootstrap 5 templates |
| **Error Handling** | 404 Not Found, 413 File Too Large, 500 Server Error | Shields stack traces and renders user-friendly error templates |
