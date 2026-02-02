from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import joblib

app = FastAPI(title="Student Attendance Event Prediction API", version="0.1.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")

class PredictRequest(BaseModel):
    student_id: str = Field(..., examples=["STU_001234"])
    event_id: str = Field(..., examples=["EVT_2026_0001"])
    features: Dict[str, Any] = Field(default_factory=dict)

class PredictResponse(BaseModel):
    student_id: str
    event_id: str
    attendance_probability: float
    model_version: str = "mvp-heuristic"

_model = None

def load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/attendance", response_model=PredictResponse)
def predict_attendance(req: PredictRequest):
    model = load_model()

    # MVP fallback: if no trained model is present, use a simple heuristic.
    if model is None:
        # Example heuristic knobs (safe defaults)
        registered = bool(req.features.get("student_registered_for_event", False))
        food = bool(req.features.get("food_provided", False))
        certificate = bool(req.features.get("certificate_offered", False))
        exam = bool(req.features.get("is_exam_period", False))

        p = 0.35
        p += 0.25 if registered else 0.0
        p += 0.10 if food else 0.0
        p += 0.10 if certificate else 0.0
        p -= 0.20 if exam else 0.0
        p = max(0.01, min(0.99, p))

        return PredictResponse(
            student_id=req.student_id,
            event_id=req.event_id,
            attendance_probability=float(p),
            model_version="mvp-heuristic",
        )

    # If model exists, it should be a sklearn Pipeline with predict_proba.
    X = [req.features]
    proba = float(model.predict_proba(X)[0][1])
    return PredictResponse(
        student_id=req.student_id,
        event_id=req.event_id,
        attendance_probability=proba,
        model_version="sklearn-pipeline-v1",
    )

