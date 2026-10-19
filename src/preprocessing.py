"""Phase 5: feature engineering + preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.config import TARGET, ID_COL, SENSITIVE


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new columns that capture domain knowledge."""
    df = df.copy()
    df["delivery_x_quality"] = df["on_time_delivery_rate"] * df["code_review_score"]
    df["bugs_per_project"] = df["bug_count"] / df["projects_count"].clip(lower=1)
    df["learning_index"] = df["training_hours"] / 10 + df["certifications_count"] + df["hackathons_participated"]
    df["absence_total"] = df["sick_days"] + df["unplanned_absences"]
    df["feedback_avg"] = df[["peer_feedback_score", "manager_score"]].mean(axis=1)
    return df


def get_X_y(df: pd.DataFrame):
    """Drop id, target and sensitive columns (fairness policy) -> model inputs."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL] + [c for c in SENSITIVE if c in df])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes(include="object").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", RobustScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
