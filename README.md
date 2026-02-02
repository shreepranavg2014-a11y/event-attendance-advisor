# Student Attendance Event Prediction (Christ University)

Predicts the probability that a student will attend a specific event, to support planning, outreach, and resource allocation.

## What’s in this repo (MVP)
- FastAPI service (Vercel-ready): `POST /predict/attendance`, `GET /health`
- Baseline training pipeline (scikit-learn): `src/train.py`
- Placeholder dataset: `data/training.csv`

## Quickstart (local)
1. Install dependencies:
   - `pip install -r requirements.txt`

2. Train a baseline model:
   - `python src/train.py`

3. Run API locally (optional):
   - `uvicorn api.index:app --reload`

4. Test:
   - `GET http://127.0.0.1:8000/health`
   - `POST http://127.0.0.1:8000/predict/attendance`

Example request:
```json
{
  "student_id": "STU_001",
  "event_id": "EVT_001",
  "features": {
    "student_registered_for_event": 1,
    "food_provided": 1,
    "certificate_offered": 0,
    "is_exam_period": 0,
    "event_category": "Technical",
    "student_dept": "AI_DS"
  }
}
