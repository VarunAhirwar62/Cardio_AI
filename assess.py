"""Logic for the 'Assess patient' workflow: field schema, row building, explanations, history DB."""
import json
import sqlite3
from datetime import datetime
import pandas as pd
from common import ROOT, TASKS, load_dataset

DB_PATH = ROOT / "patient_history.db"
YN = {"No": 0, "Yes": 1}
SEX = {"Female": 0, "Male": 1}
CP = {"Asymptomatic": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Typical angina": 3}
ECG = {"Normal": 1, "ST-T abnormality": 2, "LV hypertrophy": 0}
SLOPE = {"Upsloping": 2, "Flat": 1, "Downsloping": 0}
THAL = {"Normal": 2, "Fixed defect": 1, "Reversible defect": 3}


def N(key, label, lo, hi, step, kind, group, help=None):
    c = int if kind == "int" else float
    return dict(key=key, label=label, kind=kind, min=c(lo), max=c(hi), step=c(step), group=group, help=help)


def S(key, label, options, group, help=None):
    return dict(key=key, label=label, kind="select", options=list(options), group=group, help=help)


FIELDS = [
    N("age", "Age (years)", 1, 120, 1, "int", "basic"),
    S("sex", "Sex", ["Female", "Male"], "basic"),
    N("bp", "Resting blood pressure, systolic (mm Hg)", 70, 250, 1, "int", "basic"),
    N("glucose", "Blood glucose (mg/dl)", 40, 500, 1, "float", "basic",
      "Heart model treats > 120 as 'high fasting sugar'; stroke model uses it as average glucose."),
    N("chol", "Total cholesterol (mg/dl)", 100, 600, 1, "float", "heart"),
    S("cp", "Chest pain type", CP, "heart"),
    S("restecg", "Resting ECG", ECG, "heart"),
    N("thalach", "Max heart rate achieved", 60, 220, 1, "int", "heart"),
    S("exang", "Exercise-induced angina", YN, "heart"),
    N("oldpeak", "ST depression (oldpeak)", 0, 7, 0.1, "float", "heart"),
    S("slope", "ST slope", SLOPE, "heart"),
    S("ca", "Major vessels coloured (0-4)", [0, 1, 2, 3, 4], "heart"),
    S("thal", "Thalassemia", THAL, "heart"),
    N("bmi", "BMI (kg/m²)", 10, 70, 0.1, "float", "stroke"),
    S("hypertension", "Hypertension", YN, "stroke"),
    S("heart_disease", "Known heart disease", YN, "stroke"),
    S("ever_married", "Ever married", ["Yes", "No"], "stroke"),
    S("work_type", "Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"], "stroke"),
    S("residence", "Residence", ["Urban", "Rural"], "stroke"),
    S("smoking", "Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"], "stroke"),
]
BY_KEY = {f["key"]: f for f in FIELDS}
LABEL = {f["key"]: f["label"] for f in FIELDS}
REQUIRED = {
    "heart": ["age", "sex", "bp", "glucose", "chol", "cp", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
    "stroke": ["sex", "age", "hypertension", "heart_disease", "ever_married", "work_type", "residence", "glucose", "bmi", "smoking"],
}
COLNAMES = {"trestbps": "Resting BP", "chol": "Cholesterol", "thalach": "Max heart rate", "oldpeak": "ST depression",
            "cp": "Chest pain", "restecg": "Resting ECG", "exang": "Exercise angina", "ca": "Vessels coloured",
            "thal": "Thalassemia", "fbs": "High fasting sugar", "slope": "ST slope", "sex": "Sex", "age": "Age",
            "gender": "Sex", "avg_glucose_level": "Glucose", "bmi": "BMI", "hypertension": "Hypertension",
            "heart_disease": "Heart disease", "ever_married": "Ever married", "work_type": "Work type",
            "Residence_type": "Residence", "smoking_status": "Smoking"}

DEMOS = {
    "high": dict(age=62, sex="Male", bp=140, glucose=105.0, chol=268.0, cp="Asymptomatic", restecg="Normal", thalach=120,
                 exang="Yes", oldpeak=3.0, slope="Flat", ca=2, thal="Reversible defect", bmi=29.0, hypertension="Yes",
                 heart_disease="Yes", ever_married="Yes", work_type="Private", residence="Urban", smoking="formerly smoked"),
    "healthy": dict(age=35, sex="Female", bp=118, glucose=85.0, chol=190.0, cp="Non-anginal pain", restecg="Normal",
                    thalach=175, exang="No", oldpeak=0.0, slope="Upsloping", ca=0, thal="Normal", bmi=22.0,
                    hypertension="No", heart_disease="No", ever_married="No", work_type="Private", residence="Urban",
                    smoking="never smoked"),
}


def coerce(key, v):
    """Make a value match the widget type (int / float / option). Returns None if it cannot."""
    f = BY_KEY[key]
    if f["kind"] == "select":
        return v if v in f["options"] else None
    try:
        return int(round(float(v))) if f["kind"] == "int" else float(v)
    except (TypeError, ValueError):
        return None


def load_demo(name):  # used as a button callback
    import streamlit as st
    for k, v in DEMOS[name].items():
        st.session_state[k] = v


def missing_fields(task, v):
    return [LABEL[k] for k in REQUIRED[task] if v.get(k) is None]


def to_row(task, v):
    if task == "heart":
        return dict(age=v["age"], sex=SEX[v["sex"]], cp=CP[v["cp"]], trestbps=v["bp"], chol=v["chol"],
                    fbs=int(v["glucose"] > 120), restecg=ECG[v["restecg"]], thalach=v["thalach"], exang=YN[v["exang"]],
                    oldpeak=v["oldpeak"], slope=SLOPE[v["slope"]], ca=v["ca"], thal=THAL[v["thal"]])
    return dict(gender=v["sex"], age=v["age"], hypertension=YN[v["hypertension"]], heart_disease=YN[v["heart_disease"]],
                ever_married=v["ever_married"], work_type=v["work_type"], Residence_type=v["residence"],
                avg_glucose_level=v["glucose"], bmi=v["bmi"], smoking_status=v["smoking"])


def band_of(p):
    return "LOW" if p < 0.30 else "MODERATE" if p < 0.60 else "HIGH"


def reference_row(task):
    """Typical patient (median / most common value) used for the 'what pushed the risk' explanation."""
    X, _ = load_dataset(task)
    cfg = TASKS[task]
    ref = {c: float(X[c].median()) for c in cfg["numeric"]}
    ref.update({c: X[c].mode().iloc[0] for c in cfg["categorical"]})
    return ref


def explain(model, X, ref, top=6):
    """Model-agnostic local explanation: swap each feature for the typical value and see how the risk moves."""
    base = float(model.predict_proba(X)[0, 1])
    out = {}
    for c in X.columns:
        X2 = X.copy()
        X2[c] = ref[c]
        out[c] = base - float(model.predict_proba(X2)[0, 1])
    return sorted(out.items(), key=lambda kv: -abs(kv[1]))[:top]


# ---------------------------------------------------------------- history (SQLite)
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS assessments(id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT,
                   ts TEXT, task TEXT, probability REAL, band TEXT, inputs TEXT)""")
    return con


def save_assessment(patient_id, task, p, band, inputs):
    with _db() as con:
        con.execute("INSERT INTO assessments(patient_id,ts,task,probability,band,inputs) VALUES(?,?,?,?,?,?)",
                    (patient_id or "UNKNOWN", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task, p, band,
                     json.dumps(inputs, default=str)))


def fetch_history(patient_id=None):
    with _db() as con:
        q, args = "SELECT patient_id,ts,task,probability,band,inputs FROM assessments", ()
        if patient_id:
            q, args = q + " WHERE patient_id=?", (patient_id,)
        return pd.read_sql_query(q + " ORDER BY ts", con, params=args)
