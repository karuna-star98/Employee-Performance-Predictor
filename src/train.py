"""Phase 6: train baseline + tuned models, pick the winner by macro-F1 (cross-validated)."""
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from src.config import MODEL_PATH, OUT_DIR, SEED, TARGET
from src.preprocessing import build_preprocessor, get_X_y


def split(df):
    """Stratified 80/20 split. employee_id is unique per row, so no employee appears in both sets."""
    return train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)


def train_models(train_df):
    X, y = get_X_y(train_df)
    pre = build_preprocessor(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}

    # --- 1. Baseline: Logistic Regression ---
    log = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    results["LogisticRegression"] = cross_val_score(log, X, y, cv=cv, scoring="f1_macro").mean()
    fitted["LogisticRegression"] = log.fit(X, y)

    # --- 2. Tuned Random Forest ---
    rf = Pipeline([("pre", pre), ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1))])
    grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [8, 12, None], "clf__min_samples_leaf": [3, 5]}
    gs = GridSearchCV(rf, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    results["RandomForest(tuned)"] = gs.best_score_
    fitted["RandomForest(tuned)"] = gs.best_estimator_

    # --- 3. Gradient Boosting (XGBoost-style, built into sklearn so no extra install) ---
    gb = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        class_weight="balanced", learning_rate=0.05, max_iter=200, random_state=SEED))])
    results["GradientBoosting"] = cross_val_score(gb, X, y, cv=cv, scoring="f1_macro").mean()
    fitted["GradientBoosting"] = gb.fit(X, y)

    best_name = max(results, key=results.get)
    comparison = pd.Series(results).sort_values(ascending=False).round(4).rename("cv_f1_macro")
    comparison.to_csv(OUT_DIR / "model_comparison.csv")
    print("\nModel comparison (5-fold CV macro-F1):\n", comparison.to_string())
    print(f"\nBest model: {best_name}")
    if best_name.startswith("RandomForest"):
        print("Best params:", gs.best_params_)
    joblib.dump(fitted[best_name], MODEL_PATH)
    json.dump({"best_model": best_name, "cv_scores": results}, open(OUT_DIR / "training_summary.json", "w"), indent=2)
    return fitted[best_name], best_name, comparison
