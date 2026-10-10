"""Train + compare models for both tasks.   Usage: python train.py"""
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import MODEL_DIR, TASKS, load_dataset

SEED = 42


def build(cfg, clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cfg["numeric"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cfg["categorical"]),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def candidates(spw=1.0):
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
    }


def best_threshold(y, proba, beta):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, proba >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def run(task):
    cfg = TASKS[task]
    X, y = load_dataset(task)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"\n=== {cfg['title']} | rows={len(X)} | positive rate={y.mean():.1%} ===")

    spw = float((ytr == 0).sum() / (ytr == 1).sum())  # class-imbalance weight for XGBoost
    cv_auc = {}
    for name, clf in candidates(spw).items():
        cv_auc[name] = float(cross_val_score(build(cfg, clf), Xtr, ytr, cv=cv, scoring="roc_auc").mean())
        print(f"  CV ROC-AUC  {name:<20} {cv_auc[name]:.3f}")
    best = max(cv_auc, key=cv_auc.get)

    # threshold chosen on out-of-fold predictions (no test leakage)
    oof = cross_val_predict(build(cfg, candidates(spw)[best]), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(ytr, oof, cfg["beta"])

    model = build(cfg, candidates(spw)[best]).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {
        "best_model": best, "threshold": thr, "cv_roc_auc": cv_auc,
        "test": {"roc_auc": roc_auc_score(yte, proba), "accuracy": accuracy_score(yte, pred),
                 "precision": precision_score(yte, pred, zero_division=0),
                 "recall": recall_score(yte, pred), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_rows": int(len(X)), "positive_rate": float(y.mean()),
    }
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
    metrics["importance"] = dict(sorted(zip(X.columns, map(float, imp.importances_mean)),
                                        key=lambda kv: -kv[1]))
    print(f"  BEST: {best} | threshold={thr:.2f} | test AUC={metrics['test']['roc_auc']:.3f} "
          f"recall={metrics['test']['recall']:.3f} precision={metrics['test']['precision']:.3f}")

    model.fit(X, y)  # refit on all data for deployment
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": thr}, MODEL_DIR / f"{task}.joblib")
    (MODEL_DIR / f"{task}_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    for t in TASKS:
        run(t)
    print("\nDone. Now run:  streamlit run app.py")
