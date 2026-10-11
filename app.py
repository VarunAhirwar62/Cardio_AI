import hashlib
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import assess as A
from common import MODEL_DIR, TASKS, clean_features
from report_parser import extract_text, parse_report

st.set_page_config(page_title="CardioAI – Heart Disease & Stroke", page_icon="🫀", layout="wide")


@st.cache_resource(show_spinner="Training models for the first time (1-2 min)...")
def load(task):
    p = MODEL_DIR / f"{task}.joblib"
    if not p.exists():
        import train
        train.run(task)
    return joblib.load(p)


@st.cache_data
def reference(task):
    return A.reference_row(task)


def metrics(task):
    load(task)
    return json.loads((MODEL_DIR / f"{task}_metrics.json").read_text())


MSG = {"HIGH": "High risk – please consult a cardiologist/doctor soon.",
       "MODERATE": "Moderate risk – a medical check-up and lifestyle changes are advisable.",
       "LOW": "Low risk – keep up healthy habits and regular check-ups."}


def render_result(r):
    st.subheader(TASKS[r["task"]]["title"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated risk", f"{r['p']:.1%}")
    c2.metric("Risk level", r["band"])
    c3.metric("Model verdict", "Positive" if r["verdict"] else "Negative", help=f"Decision threshold: {r['thr']:.2f}")
    st.progress(min(r["p"], 1.0))
    (st.error if r["band"] == "HIGH" else st.warning if r["band"] == "MODERATE" else st.success)(MSG[r["band"]])
    st.caption("What moved this estimate up (+) or down (−) compared with a typical patient "
               "(approximate, model-based; not a medical explanation):")
    st.bar_chart(pd.Series({A.COLNAMES.get(k, k): v for k, v in r["why"]}))


# ---------------------------------------------------------------- Assess patient
def report_block():
    f = st.file_uploader("Upload a medical report (digital PDF works best; images/scans need OCR)",
                         type=["pdf", "png", "jpg", "jpeg"])
    st.caption("🔒 Do not upload real patient reports to a public deployment. Use synthetic/sample reports for demos.")
    if not f:
        return
    data = f.getvalue()
    h = hashlib.md5(data).hexdigest()
    if st.session_state.get("report_hash") != h:
        try:
            text, notes = extract_text(data, f.name)
            found, warns = parse_report(text)
        except Exception as e:
            st.error(f"Could not read this report: {e}")
            return
        for k, (v, _) in found.items():
            cv = A.coerce(k, v)
            if cv is not None:
                st.session_state[k] = cv
        st.session_state.update(report_hash=h, report_found=found, report_warns=notes + warns, report_text=text)
    found = st.session_state.get("report_found", {})
    if found:
        st.success(f"Found {len(found)} value(s) in the report and filled them in below. "
                   "**Please verify each one** – extraction can be wrong.")
        st.dataframe(pd.DataFrame([(A.LABEL[k], v, s) for k, (v, s) in found.items()],
                                  columns=["Field", "Extracted value", "Matched text in report"]),
                     hide_index=True, width="stretch")
    else:
        st.warning("No known values were found in this report. Enter the values manually below.")
    for w in st.session_state.get("report_warns", []):
        st.warning(w)
    with st.expander("Raw text read from the report"):
        st.text(st.session_state.get("report_text", "")[:4000])
    st.info("Reports rarely contain ECG, ST slope, vessels or thalassemia results. Fields still empty must be "
            "filled by you – missing values are never assumed to be normal.")


def widget(f, col):
    k = f["key"]
    if f["kind"] == "select":
        col.selectbox(f["label"], f["options"], index=None, key=k, placeholder="Select…", help=f["help"])
    else:
        col.number_input(f["label"], min_value=f["min"], max_value=f["max"], step=f["step"], value=None, key=k,
                         placeholder="Enter value", help=f["help"])


def assess_page():
    st.header("🩺 Assess patient")
    st.sidebar.markdown("**Demo patients**")
    st.sidebar.button("Load high-risk demo", on_click=A.load_demo, args=("high",))
    st.sidebar.button("Load healthy demo", on_click=A.load_demo, args=("healthy",))
    mode = st.radio("How do you want to enter the data?", ["✍️ Manual entry", "📄 Upload medical report"], horizontal=True)
    if mode.startswith("📄"):
        report_block()
    st.text_input("Patient ID (optional – used for history)", key="patient_id", placeholder="e.g. P-001")
    run = st.multiselect("Run assessments", ["Heart disease", "Stroke"], default=["Heart disease", "Stroke"], key="run_tasks")
    for group, title in (("basic", "Basic details & vitals"), ("heart", "Heart-specific (cholesterol, ECG, stress test)"),
                         ("stroke", "Stroke-specific (history & lifestyle)")):
        st.markdown(f"##### {title}")
        cols = st.columns(3)
        for i, f in enumerate([x for x in A.FIELDS if x["group"] == group]):
            widget(f, cols[i % 3])

    if st.button("Run assessment", type="primary"):
        vals = {f["key"]: st.session_state.get(f["key"]) for f in A.FIELDS}
        results = []
        for name, task in (("Heart disease", "heart"), ("Stroke", "stroke")):
            if name not in run:
                continue
            miss = A.missing_fields(task, vals)
            if miss:
                st.error(f"Cannot run {name} assessment – missing: " + ", ".join(miss))
                continue
            art = load(task)
            X = clean_features(pd.DataFrame([A.to_row(task, vals)]), TASKS[task])
            p = float(art["model"].predict_proba(X)[0, 1])
            band = A.band_of(p)
            results.append(dict(task=task, p=p, band=band, verdict=p >= art["threshold"], thr=art["threshold"],
                                why=A.explain(art["model"], X, reference(task))))
            A.save_assessment(st.session_state.get("patient_id"), task, p, band, vals)
        st.session_state["last"] = results
    for r in st.session_state.get("last", []):
        st.divider()
        render_result(r)
    if st.session_state.get("last"):
        st.caption("⚠️ Educational estimate, not a diagnosis. In an emergency call 112 / 108.")


# ---------------------------------------------------------------- Batch
def batch_page():
    st.header("📑 Batch prediction (CSV / Excel)")
    task = "heart" if st.radio("Model", ["Heart disease", "Stroke"], horizontal=True) == "Heart disease" else "stroke"
    cfg = TASKS[task]
    cols = cfg["numeric"] + cfg["categorical"]
    demo = [A.to_row(task, A.DEMOS[k]) for k in ("high", "healthy")]
    tmpl = pd.DataFrame(demo, columns=cols)
    tmpl.insert(0, "patient_id", ["P-001", "P-002"])
    st.download_button("Download template CSV", tmpl.to_csv(index=False), f"{task}_template.csv", "text/csv")
    st.caption("Columns must use the dataset's numeric codes (see the template). Rows with any missing value are "
               "skipped and flagged – never silently filled.")
    f = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if not f:
        return
    df = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    df.columns = [str(c).strip() for c in df.columns]
    absent = [c for c in cols if c not in df.columns]
    if absent:
        st.error("Missing columns: " + ", ".join(absent))
        return
    ok = df[cols].replace("?", np.nan).notna().all(axis=1)
    art = load(task)
    out = df.copy()
    out["risk_probability"], out["risk_level"], out["verdict"] = np.nan, "", ""
    out["status"] = np.where(ok, "OK", "Skipped – missing values")
    if ok.any():
        p = art["model"].predict_proba(clean_features(df[ok], cfg))[:, 1]
        out.loc[ok, "risk_probability"] = p.round(3)
        out.loc[ok, "risk_level"] = [A.band_of(x) for x in p]
        out.loc[ok, "verdict"] = np.where(p >= art["threshold"], "Positive", "Negative")
    st.success(f"{int(ok.sum())} of {len(df)} rows predicted.")
    st.dataframe(out, width="stretch")
    st.download_button("Download results", out.to_csv(index=False), f"{task}_predictions.csv", "text/csv")


# ---------------------------------------------------------------- History
def history_page():
    st.header("🗂️ Patient history")
    st.caption("Saved in a local SQLite file. On free cloud hosting this storage is temporary and not private – "
               "use demo data only.")
    allh = A.fetch_history()
    if allh.empty:
        st.info("No assessments saved yet. Run one from 'Assess patient'.")
        return
    pid = st.selectbox("Patient", ["All"] + sorted(allh["patient_id"].unique()))
    h = allh if pid == "All" else allh[allh["patient_id"] == pid]
    c1, c2, c3 = st.columns(3)
    c1.metric("Assessments", len(h))
    c2.metric("Patients", h["patient_id"].nunique())
    c3.metric("High-risk results", int((h["band"] == "HIGH").sum()))
    if pid != "All":
        st.line_chart(h.pivot_table(index="ts", columns="task", values="probability"))
    st.dataframe(h.drop(columns=["inputs"]), width="stretch", hide_index=True)
    st.download_button("Download history", h.to_csv(index=False), "history.csv", "text/csv")


# ---------------------------------------------------------------- Performance / About
def performance_page():
    st.header("📊 Model performance")
    for t, cfg in TASKS.items():
        m = metrics(t)
        st.subheader(cfg["title"])
        a, b, c, d = st.columns(4)
        a.metric("Best model", m["best_model"])
        b.metric("ROC-AUC", f"{m['test']['roc_auc']:.3f}")
        c.metric("Recall", f"{m['test']['recall']:.3f}")
        d.metric("Precision", f"{m['test']['precision']:.3f}")
        st.write(f"Hold-out test (20%) · rows: {m['n_rows']} · positive rate: {m['positive_rate']:.1%}")
        st.dataframe(pd.DataFrame({"Cross-val ROC-AUC": m["cv_roc_auc"]}).T)
        t_ = m["test"]
        st.dataframe(pd.DataFrame([[t_["tn"], t_["fp"]], [t_["fn"], t_["tp"]]], index=["Actual: No", "Actual: Yes"],
                                  columns=["Pred: No", "Pred: Yes"]))


def about_page():
    st.header("About")
    st.markdown("""
**CardioAI** estimates heart-disease and stroke risk from manually entered values, an uploaded medical report, or a CSV/Excel batch.

**Pipeline:** input / report extraction → validation (missing values are flagged, never assumed normal) → ML model
(best of Logistic Regression, Random Forest, Gradient Boosting, XGBoost by 5-fold CV) → risk estimate + factors that moved it → saved history.

| Module | Dataset |
|---|---|
| Heart disease | UCI Heart Disease (Cleveland), 302 unique patients |
| Stroke | Kaggle Stroke Prediction Dataset, 5,109 patients (4.9% positive) |

**Limitations:** small public datasets, not clinically validated, no external validation, report extraction can misread values,
the heart model needs tests (ECG, vessels, thalassemia) that ordinary reports lack.
**Future work:** login & roles, doctor dashboard, validated hospital data, SHAP explanations, insurer workflow.

⚠️ *Educational project, not a medical device. Never a substitute for a doctor.*
""")


PAGES = {"Assess patient": assess_page, "Batch (CSV/Excel)": batch_page, "Patient history": history_page,
         "Model performance": performance_page, "About": about_page}
PAGES[st.sidebar.radio("Navigate", list(PAGES))]()
