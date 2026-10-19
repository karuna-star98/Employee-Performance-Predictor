"""Phase 8: fairness audit. Sensitive columns were NOT used by the model; here we check outcomes anyway."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.config import IMG_DIR, OUT_DIR, TARGET


def group_report(test_df, y_pred, group_col, min_size=30):
    rows = []
    t = test_df.reset_index(drop=True)
    pred = pd.Series(y_pred)
    for g in t[group_col].dropna().unique():
        idx = (t[group_col] == g).values
        if idx.sum() < min_size:          # tiny groups give noisy statistics
            continue
        rows.append({"group_col": group_col, "group": g, "n": int(idx.sum()),
                     "actual_high_rate": round((t.loc[idx, TARGET] == "High").mean(), 3),
                     "high_selection_rate": round((pred[idx] == "High").mean(), 3),
                     "low_flag_rate": round((pred[idx] == "Low").mean(), 3),
                     "macro_f1": round(f1_score(t.loc[idx, TARGET], pred[idx], average="macro"), 3)})
    return pd.DataFrame(rows)


def run_fairness_audit(test_df, y_pred):
    test_df = test_df.copy()
    test_df["age_group"] = pd.cut(test_df["age"], [0, 29, 39, 49, 100], labels=["<30", "30-39", "40-49", "50+"]).astype(str)
    reports = [group_report(test_df, y_pred, c) for c in ["gender", "age_group", "department", "job_level"]]
    rep = pd.concat(reports, ignore_index=True)
    # Disparate-impact ratio (4/5ths rule): min group rate / max group rate for 'High' selection
    di = {}
    for c, g in rep.groupby("group_col"):
        mx = g.high_selection_rate.max()
        di[c] = round(g.high_selection_rate.min() / mx, 3) if mx > 0 else np.nan
    rep.to_csv(OUT_DIR / "fairness_report.csv", index=False)
    print("\n=== FAIRNESS AUDIT (High-band selection rate by group) ===")
    print(rep.to_string(index=False))
    di = {k: float(v) for k, v in di.items()}
    print("\nDisparate-impact ratios (>= 0.80 passes the 4/5ths rule):", di)
    print("NOTE: compare 'actual_high_rate' with 'high_selection_rate'. If the model only mirrors real "
          "base-rate differences (and groups are small), that is not necessarily model bias.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, c in zip(axes, ["gender", "department"]):
        d = rep[rep.group_col == c]
        ax.bar(d.group, d.high_selection_rate, color="#76b7b2")
        ax.set_title(f"Predicted 'High' rate by {c}"); ax.set_ylim(0, max(0.4, d.high_selection_rate.max() + .05))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(IMG_DIR / "09_fairness.png", dpi=150); plt.close(fig)
    return rep, di
