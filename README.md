# CardioAI – Heart Disease Detection + Stroke Prediction (Streamlit)

## 1. Setup
    pip install -r requirements.txt

## 2. Datasets (already included in `data/`)
| File | Source | Rows |
|---|---|---|
| `data/heart.csv` | UCI Heart Disease (Cleveland), Kaggle-processed copy | 1025 (302 unique after duplicate removal) |
| `data/stroke.csv` | Kaggle "Stroke Prediction Dataset" (fedesoriano) | 5110 |

Originals: https://archive.ics.uci.edu/dataset/45/heart+disease (CC BY 4.0) and the Kaggle stroke dataset.
`heart.csv` uses the Kaggle 0-3 encoding for `cp`/`slope`, thal 1=normal, 2=fixed, 3=reversible.
If you replace the file with another encoding, edit the dictionaries at the top of `app.py`.

## 3. Train
    python train.py

## 4. Run
    streamlit run app.py

## Extending
- Add LightGBM/CatBoost in `candidates()` in train.py (XGBoost is already included)
- Add SHAP explanations per patient
- Add the Cleveland+Hungary+Switzerland+VA combined data (920 rows) for more robustness
