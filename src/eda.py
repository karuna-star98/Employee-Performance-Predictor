"""Phase 4 + 9: EDA charts. Every chart is saved to images/ (your GitHub proof)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from src.config import IMG_DIR, TARGET, CLASSES, ID_COL

sns.set_theme(style="whitegrid")
PALETTE = {"Low": "#e15759", "Medium": "#f2b134", "High": "#59a14f"}


def run_eda(df: pd.DataFrame):
    # 1. class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].value_counts().reindex(CLASSES)
    ax.bar(counts.index, counts.values, color=[PALETTE[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 15, f"{v} ({v/len(df):.0%})", ha="center")
    ax.set_title("Performance Band Distribution"); ax.set_ylabel("Employees")
    fig.tight_layout(); fig.savefig(IMG_DIR / "01_class_balance.png", dpi=150); plt.close(fig)

    # 2. correlation heatmap
    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Correlation Heatmap (numeric features)")
    fig.tight_layout(); fig.savefig(IMG_DIR / "02_correlation_heatmap.png", dpi=150); plt.close(fig)

    # 3. key drivers by band (boxplots)
    cols = ["on_time_delivery_rate", "code_review_score", "manager_score",
            "peer_feedback_score", "avg_task_delay_days", "training_hours"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), cols):
        sns.boxplot(data=df, x=TARGET, y=c, order=CLASSES, palette=PALETTE, ax=ax, hue=TARGET, legend=False)
        ax.set_title(c); ax.set_xlabel("")
    fig.suptitle("Key Drivers vs Performance Band", fontsize=14)
    fig.tight_layout(); fig.savefig(IMG_DIR / "03_drivers_by_band.png", dpi=150); plt.close(fig)

    # 4. department performance mix
    ct = pd.crosstab(df["department"], df[TARGET], normalize="index")[CLASSES]
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 5), color=[PALETTE[c] for c in CLASSES])
    ax.set_title("Performance Mix by Department"); ax.set_ylabel("Share"); plt.xticks(rotation=30)
    plt.tight_layout(); plt.savefig(IMG_DIR / "04_department_mix.png", dpi=150); plt.close()

    # 5. training hours: non-linear?
    tmp = df.dropna(subset=["training_hours"]).copy()
    tmp["train_bucket"] = pd.cut(tmp["training_hours"], [-1, 15, 30, 45, 60, 101],
                                 labels=["0-15", "16-30", "31-45", "46-60", "60+"])
    rate = tmp.groupby("train_bucket", observed=True)[TARGET].apply(lambda s: (s == "High").mean())
    fig, ax = plt.subplots(figsize=(6, 4)); rate.plot(kind="bar", color="#4e79a7", ax=ax)
    ax.set_title("High-Performer Rate by Training Hours"); ax.set_ylabel("Share High")
    plt.xticks(rotation=0); fig.tight_layout(); fig.savefig(IMG_DIR / "05_training_vs_high.png", dpi=150); plt.close(fig)

    # 6. mutual information ranking
    X = df.drop(columns=[TARGET, ID_COL]).select_dtypes("number").fillna(0)
    mi = pd.Series(mutual_info_classif(X, df[TARGET], random_state=13), index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8)); mi.tail(15).plot(kind="barh", color="#76b7b2", ax=ax)
    ax.set_title("Top 15 Features by Mutual Information")
    fig.tight_layout(); fig.savefig(IMG_DIR / "06_mutual_information.png", dpi=150); plt.close(fig)
    return mi.sort_values(ascending=False)
