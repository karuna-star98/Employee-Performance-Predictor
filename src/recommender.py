"""Phase 8-9: turn predictions into per-employee drivers + coaching actions.

Driver logic (simple + explainable):
  for each feature, compute how far the employee is from the company median (z-score),
  flip the sign if 'lower is better', multiply by the model's global importance,
  and surface the most negative scores as 'key concerns' and most positive as 'strengths'.
"""
import numpy as np
import pandas as pd
from src.config import ID_COL, REPORT_DIR

# +1 = higher is better, -1 = lower is better
DIRECTION = {
    "on_time_delivery_rate": 1, "avg_task_delay_days": -1, "code_review_score": 1, "bug_count": -1,
    "qa_defect_density": -1, "story_points_completed": 1, "billable_hours_ratio": 1,
    "training_hours": 1, "certifications_count": 1, "hackathons_participated": 1,
    "unplanned_absences": -1, "sick_days": -1, "peer_feedback_score": 1, "manager_score": 1,
    "kudos_count": 1, "delivery_x_quality": 1, "bugs_per_project": -1, "learning_index": 1,
    "absence_total": -1, "feedback_avg": 1, "avg_login_hours": 1,
}

ACTIONS = {
    "on_time_delivery_rate": "Sprint-planning & estimation workshop; weekly check-in on deadlines",
    "avg_task_delay_days": "Break tasks into smaller tickets; pair with a senior for time-boxing",
    "delivery_x_quality": "Define a 'done' checklist combining deadline + review standards",
    "code_review_score": "Pair programming + clean-code / code-review course",
    "bug_count": "Add unit-test coverage goals; attend a quality engineering workshop",
    "bugs_per_project": "Test-driven development mini-course; peer code walkthroughs",
    "qa_defect_density": "Shift-left testing practice with the QA team",
    "story_points_completed": "Review workload blockers with manager; reprioritise backlog",
    "training_hours": "Assign a targeted certification path (e.g. AWS Cloud Practitioner, Azure DP-900)",
    "certifications_count": "Fund one role-relevant certification this quarter",
    "learning_index": "Create a 90-day personal learning plan with L&D",
    "hackathons_participated": "Invite to the next internal hackathon / innovation sprint",
    "unplanned_absences": "Wellbeing check-in; discuss flexibility or workload with HR",
    "absence_total": "Confidential wellbeing check-in with HR partner",
    "sick_days": "Confidential wellbeing check-in with HR partner",
    "peer_feedback_score": "360-degree feedback session; communication & collaboration coaching",
    "manager_score": "1:1 expectation-setting with manager; agree SMART goals",
    "feedback_avg": "Structured feedback loop: bi-weekly 1:1s for the next 2 months",
    "kudos_count": "Recognition programme: encourage peers to log wins",
    "billable_hours_ratio": "Review project allocation and utilisation with delivery manager",
    "avg_login_hours": "Discuss working-pattern and workload balance",
}


def explain_employees(df_features, importance: pd.Series, reference_df: pd.DataFrame, top_k=3):
    """Return DataFrame: employee_id, concerns (list), strengths (list)."""
    cols = [c for c in DIRECTION if c in df_features and c in importance.index]
    med = reference_df[cols].median()
    std = reference_df[cols].std().replace(0, 1)
    w = importance.reindex(cols).clip(lower=0)
    w = w / w.sum()
    z = (df_features[cols] - med) / std
    z = z.mul(pd.Series(DIRECTION)[cols], axis=1).fillna(0).clip(-3, 3)
    contrib = z.mul(w, axis=1)
    out = []
    for i, idx in enumerate(df_features.index):
        row = contrib.loc[idx].sort_values()
        out.append({ID_COL: df_features.loc[idx, ID_COL],
                    "concerns": [c for c in row.index[:top_k] if row[c] < 0],
                    "strengths": [c for c in row.index[::-1][:top_k] if row[c] > 0]})
    return pd.DataFrame(out)


def build_one_pager(row, review_in_days=60):
    band = row["predicted_band"]
    p = {"Low": row["prob_Low"], "Medium": row["prob_Medium"], "High": row["prob_High"]}
    emoji = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}[band]
    md = [f"# Employee Performance One-Pager — {row[ID_COL]}", "",
          f"**Department:** {row['department']}  |  **Level:** {row['job_level']}  |  "
          f"**Experience:** {row['experience_years']} yrs", "",
          f"## {emoji} Predicted next-cycle band: **{band}**",
          f"Confidence → Low {p['Low']:.0%} · Medium {p['Medium']:.0%} · High {p['High']:.0%}", "",
          "## Key drivers holding performance back"]
    md += [f"- `{c}` = {row[c]}" for c in row["concerns"]] or ["- None significant"]
    md += ["", "## Strengths"]
    md += [f"- `{c}` = {row[c]}" for c in row["strengths"]] or ["- None significant"]
    md += ["", "## Recommended actions"]
    acts = [ACTIONS[c] for c in row["concerns"] if c in ACTIONS][:2]
    if band == "High":
        acts = ["Stretch assignment / mentoring role", "Review promotion readiness at next calibration"]
    md += [f"{i+1}. {a}" for i, a in enumerate(acts)] or ["1. Continue current plan"]
    md += ["", f"**Review in:** {review_in_days} days",
           "", "> Decision-support only. A human manager must review before any action. "
               "Model trained on synthetic data."]
    return "\n".join(md)


def write_reports(scored_df, max_reports=25):
    """Write one-pagers for the highest-risk employees (highest P(Low))."""
    top = scored_df.sort_values("prob_Low", ascending=False).head(max_reports)
    for _, r in top.iterrows():
        (REPORT_DIR / f"{r[ID_COL]}.md").write_text(build_one_pager(r), encoding="utf-8")
    return len(top)
