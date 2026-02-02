import pandas as pd

STUDENTS_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "students_master.csv")

_students_df = None
def load_students():
    global _students_df
    if _students_df is None and os.path.exists(STUDENTS_CSV):
        _students_df = pd.read_csv(STUDENTS_CSV)
        _students_df["student_id"] = _students_df["student_id"].astype(str)
    return _students_df

def sigmoid(z: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-z))

@app.post("/predict/attendance", response_model=PredictResponse)
def predict_attendance(req: PredictRequest):
    model = load_model()

    # If trained model exists, use it
    if model is not None:
        X = [req.features]
        proba = float(model.predict_proba(X)[0][1])
        return PredictResponse(
            student_id=req.student_id,
            event_id=req.event_id,
            attendance_probability=proba,
            model_version="sklearn-pipeline-v1",
        )

    # Rule-based MVP using your data
    students = load_students()
    att = None
    if students is not None:
        row = students[students["student_id"] == str(req.student_id)]
        if len(row) > 0:
            att = float(row.iloc[0].get("course_attendance_rate", 0.0))

    # Inputs you can pass from frontend
    has_conflict = bool(req.features.get("has_timetable_conflict", False))
    is_exam = bool(req.features.get("is_exam_period", False))
    registered = bool(req.features.get("student_registered_for_event", False))

    # Convert attendance % (0..100) into a 0..1 signal
    att_sig = 0.0 if att is None else max(0.0, min(1.0, att / 100.0))

    # Simple scoring -> probability
    z = -0.3
    z += 1.2 * att_sig
    z += 0.8 if registered else 0.0
    z -= 1.0 if has_conflict else 0.0
    z -= 0.6 if is_exam else 0.0

    p = max(0.01, min(0.99, sigmoid(z)))

    return PredictResponse(
        student_id=req.student_id,
        event_id=req.event_id,
        attendance_probability=float(p),
        model_version="mvp-attendance+timetable",
    )
