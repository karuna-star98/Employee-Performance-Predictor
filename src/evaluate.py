"""Phase 7: evaluate on the untouched hold-out test set."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from src.config import CLASSES, IMG_DIR, OUT_DIR, SEED
from src.preprocessing import get_X_y


def evaluate(model, test_df):
    X, y = get_X_y(test_df)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    # one-vs-rest ROC-AUC, averaged over the 3 classes
    auc = roc_auc_score(pd.get_dummies(y)[CLASSES], proba[:, order], average="macro")

    metrics = {"accuracy": accuracy_score(y, pred),
               "macro_f1": f1_score(y, pred, average="macro"),
               "macro_roc_auc": auc,
               "per_class_f1": dict(zip(CLASSES, f1_score(y, pred, average=None, labels=CLASSES)))}
    print("\n=== HOLD-OUT TEST RESULTS ===")
    print(classification_report(y, pred, labels=CLASSES, digits=3))
    print(f"Accuracy {metrics['accuracy']:.3f} | Macro-F1 {metrics['macro_f1']:.3f} | ROC-AUC {auc:.3f}")
    json.dump(metrics, open(OUT_DIR / "test_metrics.json", "w"), indent=2)

    cm = confusion_matrix(y, pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (test set)")
    fig.tight_layout(); fig.savefig(IMG_DIR / "07_confusion_matrix.png", dpi=150); plt.close(fig)
    return metrics, pred


def global_importance(model, test_df):
    """Permutation importance on the whole pipeline: how much does macro-F1 drop if a column is shuffled?"""
    X, y = get_X_y(test_df)
    r = permutation_importance(model, X, y, n_repeats=8, scoring="f1_macro", random_state=SEED, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", header=["importance"])
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#4e79a7")
    ax.set_title("Top 15 Performance Drivers (permutation importance)")
    ax.set_xlabel("Drop in macro-F1 when feature is shuffled")
    fig.tight_layout(); fig.savefig(IMG_DIR / "08_feature_importance.png", dpi=150); plt.close(fig)
    print("\nTop 10 drivers:\n", imp.head(10).round(4).to_string())
    return imp
