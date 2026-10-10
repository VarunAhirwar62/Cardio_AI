import json
import joblib
import pandas as pd
import streamlit as st
from common import MODEL_DIR, TASKS, clean_features

st.set_page_config(page_title="CardioAI – Heart Disease & Stroke", page_icon="🫀", layout="wide")

SEX = {"Female": 0, "Male": 1}
CP = {"Asymptomatic": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Typical angina": 3}
ECG = {"Normal": 1, "ST-T abnormality": 2, "LV hypertrophy": 0}
SLOPE = {"Upsloping": 2, "Flat": 1, "Downsloping": 0}
THAL = {"Normal": 2, "Fixed defect": 1, "Reversible defect": 3}
YN = {"No": 0, "Yes": 1}


@st.cache_resource(show_spinner="Training models for the first time (1-2 min)...")
def load(task):
    p = MODEL_DIR / f"{task}.joblib"
    if not p.exists():
        import train
        train.run(task)
    return joblib.load(p)


def metrics(task):
    p = MODEL_DIR / f"{task}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def show_result(task, row):
    art = load(task)
    X = clean_features(pd.DataFrame([row]), TASKS[task])
    p = float(art["model"].predict_proba(X)[0, 1])
    band = "LOW" if p < 0.30 else "MODERATE" if p < 0.60 else "HIGH"
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk probability", f"{p:.1%}")
    c2.metric("Risk level", band)
    c3.metric("Model verdict", "Positive" if p >= art["threshold"] else "Negative",
              help=f"Decision threshold tuned during training: {art['threshold']:.2f}")
    st.progress(min(p, 1.0))
    (st.error if band == "HIGH" else st.warning if band == "MODERATE" else st.success)(
        {"HIGH": "High risk – please consult a cardiologist/doctor soon.",
         "MODERATE": "Moderate risk – lifestyle changes and a medical check-up are advisable.",
         "LOW": "Low risk – keep up healthy habits and regular check-ups."}[band])
    m = metrics(task)
    if m:
        st.caption("Top factors the model relies on (global importance):")
        st.bar_chart(pd.Series(m["importance"]).head(6))


def heart_page():
    st.header("🫀 Heart Disease Detection")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 1, 120, 54)
    sex = c1.selectbox("Sex", SEX)
    cp = c1.selectbox("Chest pain type", CP)
    trestbps = c2.number_input("Resting BP (mm Hg)", 70, 250, 130)
    chol = c2.number_input("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = c2.selectbox("Fasting sugar > 120 mg/dl", YN)
    restecg = c3.selectbox("Resting ECG", ECG)
    thalach = c3.number_input("Max heart rate achieved", 60, 220, 150)
    exang = c3.selectbox("Exercise-induced angina", YN)
    oldpeak = c1.number_input("ST depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope = c2.selectbox("ST slope", SLOPE)
    ca = c3.selectbox("Major vessels coloured (0-4)", [0, 1, 2, 3, 4])
    thal = c1.selectbox("Thalassemia", THAL)
    if st.button("Detect", type="primary"):
        show_result("heart", dict(age=age, sex=SEX[sex], cp=CP[cp], trestbps=trestbps, chol=chol,
                                  fbs=YN[fbs], restecg=ECG[restecg], thalach=thalach, exang=YN[exang],
                                  oldpeak=oldpeak, slope=SLOPE[slope], ca=ca, thal=THAL[thal]))


def stroke_page():
    st.header("🧠 Stroke Risk Prediction")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    age = c1.number_input("Age", 1, 120, 55)
    married = c1.selectbox("Ever married", ["Yes", "No"])
    work = c2.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    res = c2.selectbox("Residence", ["Urban", "Rural"])
    smoke = c2.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    hyp = c3.selectbox("Hypertension", YN)
    hd = c3.selectbox("Heart disease", YN)
    glu = c3.number_input("Avg glucose level (mg/dl)", 40.0, 400.0, 105.0)
    bmi = c1.number_input("BMI", 10.0, 70.0, 27.0)
    if st.button("Predict", type="primary"):
        show_result("stroke", dict(gender=gender, age=age, hypertension=YN[hyp], heart_disease=YN[hd],
                                   ever_married=married, work_type=work, Residence_type=res,
                                   avg_glucose_level=glu, bmi=bmi, smoking_status=smoke))


def performance_page():
    st.header("📊 Model Performance")
    for t, cfg in TASKS.items():
        m = metrics(t)
        st.subheader(cfg["title"])
        if not m:
            st.info("Not trained yet.")
            continue
        a, b, c, d = st.columns(4)
        a.metric("Best model", m["best_model"])
        b.metric("ROC-AUC", f"{m['test']['roc_auc']:.3f}")
        c.metric("Recall", f"{m['test']['recall']:.3f}")
        d.metric("Precision", f"{m['test']['precision']:.3f}")
        st.write(f"Hold-out test set (20%) · dataset rows: {m['n_rows']} · positive rate: {m['positive_rate']:.1%}")
        st.dataframe(pd.DataFrame({"Cross-val ROC-AUC": m["cv_roc_auc"]}).T)
        st.dataframe(pd.DataFrame([[m["test"]["tn"], m["test"]["fp"]], [m["test"]["fn"], m["test"]["tp"]]],
                                  index=["Actual: No", "Actual: Yes"], columns=["Pred: No", "Pred: Yes"]))


def about_page():
    st.header("About")
    st.markdown("""
**CardioAI** is a two-model cardiovascular screening system.

| Module | Dataset | Purpose |
|---|---|---|
| Heart Disease Detection | UCI Heart Disease (Cleveland), 14 clinical attributes | Detects presence of heart disease |
| Stroke Risk Prediction | Kaggle Stroke Prediction Dataset (5,110 records) | Predicts stroke risk |

**Pipeline:** cleaning → imputation → scaling/one-hot → 4 models (LogReg, Random Forest, Gradient Boosting, XGBoost) compared by 5-fold CV ROC-AUC →
decision threshold tuned on out-of-fold predictions → evaluated on untouched hold-out set.

⚠️ *Educational project. Not a medical device – never use it as a substitute for a doctor.*
""")


page = st.sidebar.radio("Navigate", ["Heart Disease Detection", "Stroke Prediction", "Model Performance", "About"])
if page != "Model Performance" and page != "About":
    key = "heart" if page.startswith("Heart") else "stroke"
    if load(key) is None:
        st.error("Model not found. Run `python train.py` first (see README).")
        st.stop()
{"Heart Disease Detection": heart_page, "Stroke Prediction": stroke_page,
 "Model Performance": performance_page, "About": about_page}[page]()
