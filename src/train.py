import os
import json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

DATA_PATH = os.path.join("data", "training.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
META_PATH = os.path.join(MODEL_DIR, "metadata.json")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    y = df["attendance"].astype(int)
    X = df.drop(columns=["attendance"])

    # Identify column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=200, class_weight="balanced")

    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    dump(pipe, MODEL_PATH)

    meta = {
        "model_type": "LogisticRegression",
        "features": list(X.columns),
        "target": "attendance",
        "notes": "Baseline model. Replace with XGBoost/LightGBM later."
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")

if __name__ == "__main__":
    main()

