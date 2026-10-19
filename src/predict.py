"""Phase 8: score employees with the saved model (batch or single)."""
import joblib
import pandas as pd
from src.config import MODEL_PATH, CLASSES
from src.preprocessing import add_features, get_X_y


def load_model():
    return joblib.load(MODEL_PATH)


def score(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """df must contain RAW columns (as in employee_features.csv). Returns df + prediction columns."""
    model = model or load_model()
    d = add_features(df)
    X = d.drop(columns=[c for c in ["perf_band_next", "employee_id", "gender", "age"] if c in d])
    proba = model.predict_proba(X)
    out = d.copy()
    out["predicted_band"] = model.predict(X)
    for c in CLASSES:
        out[f"prob_{c}"] = proba[:, list(model.classes_).index(c)]
    return out
