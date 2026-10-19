"""Phase 9: one-image HR insights dashboard (great README screenshot)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from src.config import CLASSES, IMG_DIR

COL = {"Low": "#e15759", "Medium": "#f2b134", "High": "#59a14f"}


def build_dashboard(scored: pd.DataFrame, importance: pd.Series, metrics: dict):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    fig.suptitle("HR Insights Dashboard — Employee Performance Predictor", fontsize=18, fontweight="bold")

    # KPI strip
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    n = len(scored); n_low = (scored.predicted_band == "Low").sum(); n_high = (scored.predicted_band == "High").sum()
    kpis = [("Employees scored", f"{n:,}"), ("Predicted Low (at risk)", f"{n_low} ({n_low/n:.0%})"),
            ("Predicted High (promotable)", f"{n_high} ({n_high/n:.0%})"),
            ("Model macro-F1 (test)", f"{metrics['macro_f1']:.2f}"), ("Accuracy (test)", f"{metrics['accuracy']:.2f}")]
    for i, (k, v) in enumerate(kpis):
        ax.text(0, 0.9 - i * 0.2, k, fontsize=11, color="gray")
        ax.text(0, 0.82 - i * 0.2, v, fontsize=17, fontweight="bold")

    # predicted band by department
    ax = fig.add_subplot(gs[0, 1:])
    ct = pd.crosstab(scored.department, scored.predicted_band).reindex(columns=CLASSES, fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=ax, color=[COL[c] for c in CLASSES])
    ax.set_title("Predicted performance bands by department"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=20)

    # top drivers
    ax = fig.add_subplot(gs[1, 0]); top = importance.head(8)[::-1]
    ax.barh(top.index, top.values, color="#4e79a7"); ax.set_title("Top performance drivers")

    # risk by job level
    ax = fig.add_subplot(gs[1, 1])
    r = scored.groupby("job_level").apply(lambda d: (d.predicted_band == "Low").mean(), include_groups=False)
    ax.bar(r.index, r.values, color="#e15759"); ax.set_title("% predicted Low by job level")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # top at-risk list
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); ax.set_title("Top 8 at-risk employees (P(Low))", loc="left")
    risk = scored.sort_values("prob_Low", ascending=False).head(8)
    for i, (_, rr) in enumerate(risk.iterrows()):
        ax.text(0, 0.92 - i * 0.115, f"{rr.employee_id}  {rr.department:<13} P(Low)={rr.prob_Low:.1%}", fontsize=10, family="monospace")
    fig.savefig(IMG_DIR / "10_hr_dashboard.png", dpi=150, bbox_inches="tight"); plt.close(fig)
