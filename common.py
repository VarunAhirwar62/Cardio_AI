"""Shared config + helpers used by both train.py and app.py."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"

TASKS = {
    "heart": {
        "title": "Heart Disease Detection",
        "file": DATA_DIR / "heart.csv",          # UCI Heart Disease (Cleveland, 14 columns)
        "target": "target",
        "numeric": ["age", "trestbps", "chol", "thalach", "oldpeak"],
        "categorical": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "drop": [],
        "beta": 1.0,   # F1 -> balanced precision/recall
    },
    "stroke": {
        "title": "Stroke Risk Prediction",
        "file": DATA_DIR / "stroke.csv",         # Kaggle Stroke Prediction Dataset (5110 rows)
        "target": "stroke",
        "numeric": ["age", "avg_glucose_level", "bmi"],
        "categorical": ["gender", "hypertension", "heart_disease", "ever_married",
                        "work_type", "Residence_type", "smoking_status"],
        "drop": ["id"],
        "beta": 2.0,   # F2 -> favour recall (missing a stroke is costlier than a false alarm)
    },
}


def _cat(v):
    """Normalise categories so 1, 1.0 and '1' all become '1' (train == app)."""
    if pd.isna(v):
        return np.nan
    try:
        return str(int(float(v)))
    except (ValueError, TypeError):
        return str(v).strip()


def clean_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy().replace("?", np.nan)
    for c in cfg["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cfg["categorical"]:
        df[c] = df[c].map(_cat).astype(object)
    return df[cfg["numeric"] + cfg["categorical"]]


def load_dataset(task: str):
    cfg = TASKS[task]
    if not cfg["file"].exists():
        raise FileNotFoundError(f"Dataset missing: {cfg['file']}  (see README step 2)")
    df = pd.read_csv(cfg["file"])
    df = df.drop(columns=[c for c in cfg["drop"] if c in df.columns]).drop_duplicates()
    if task == "stroke":
        df = df[df["gender"] != "Other"]
    y = (pd.to_numeric(df[cfg["target"]], errors="coerce") > 0).astype(int)  # UCI target 1-4 -> 1
    return clean_features(df, cfg), y
