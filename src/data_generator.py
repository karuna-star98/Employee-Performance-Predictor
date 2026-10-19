"""Synthetic HR dataset generator (the 'virtual company').

Idea: every employee has a hidden 'true performance' score built from realistic
drivers (delivery, quality, feedback, learning, attendance ...) plus noise.
We then convert that hidden score into Low / Medium / High bands.
Gender and age have NO causal effect, so the fairness audit should look clean.
"""
import numpy as np
import pandas as pd
from src.config import DATA_PATH, N_EMPLOYEES, SEED, TARGET


def _z(x):
    return (x - x.mean()) / x.std()


def generate_employees(n=N_EMPLOYEES, seed=SEED):
    rng = np.random.default_rng(seed)

    # ---------- demographics / role context ----------
    age = rng.integers(22, 58, n)
    experience = np.clip(age - 22 - rng.integers(0, 4, n) + rng.normal(0, 1, n), 0, 35).round(1)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.45, 0.03])
    education = rng.choice(["Bachelors", "Masters", "PhD"], n, p=[0.62, 0.34, 0.04])
    department = rng.choice(["Engineering", "Data Science", "QA", "DevOps", "HR", "Sales"],
                            n, p=[0.32, 0.14, 0.14, 0.12, 0.08, 0.20])
    job_level = np.where(experience < 3, "L1",
                np.where(experience < 7, "L2",
                np.where(experience < 12, "L3", "L4")))
    manager_tenure = np.clip(rng.normal(4, 2.5, n), 0.3, 15).round(1)

    # ---------- hidden latent ability / motivation ----------
    ability = rng.normal(0, 1, n)
    motivation = rng.normal(0, 1, n)

    # ---------- work signals ----------
    projects = np.clip(rng.poisson(3 + 0.3 * ability, n), 1, 9)
    delay = np.clip(rng.normal(4 - 1.6 * ability - 0.8 * motivation, 3, n), -3, 30).round(1)
    on_time = np.clip(0.78 + 0.07 * ability + 0.04 * motivation + rng.normal(0, 0.07, n), 0.2, 1).round(3)

    # ---------- quality ----------
    bug = np.clip(rng.poisson(np.exp(1.8 - 0.35 * ability), n), 0, 60)
    review = np.clip(3.4 + 0.45 * ability + rng.normal(0, 0.4, n), 1, 5).round(2)
    defect_density = np.clip(rng.normal(2.2 - 0.5 * ability, 0.7, n), 0, 8).round(2)

    # ---------- productivity ----------
    story_points = np.clip(rng.normal(55 + 9 * ability + 4 * motivation, 12, n), 5, 120).round(0)
    billable = np.clip(0.72 + 0.05 * ability + rng.normal(0, 0.08, n), 0.3, 1).round(3)

    # ---------- engagement / learning ----------
    training = np.clip(rng.normal(30 + 7 * motivation, 12, n), 0, 100).round(0)
    certs = np.clip(rng.poisson(np.exp(0.2 + 0.35 * motivation + 0.1 * ability), n), 0, 8)
    hackathons = np.clip(rng.poisson(np.exp(-0.5 + 0.4 * motivation), n), 0, 6)

    # ---------- attendance ----------
    sick = np.clip(rng.poisson(5, n), 0, 25)
    absences = np.clip(rng.poisson(np.exp(0.6 - 0.35 * motivation), n), 0, 15)
    login = np.clip(rng.normal(8.2 + 0.3 * motivation, 0.8, n), 5, 12).round(2)

    # ---------- feedback ----------
    peer = np.clip(3.5 + 0.35 * ability + 0.3 * motivation + rng.normal(0, 0.45, n), 1, 5).round(2)
    mgr_prev = np.clip(3.4 + 0.5 * ability + 0.2 * motivation + rng.normal(0, 0.5, n), 1, 5).round(2)
    kudos = np.clip(rng.poisson(np.exp(1.0 + 0.4 * ability + 0.2 * motivation), n), 0, 30)

    # ---------- HR signals ----------
    promos = np.clip(rng.poisson(np.exp(-0.9 + 0.35 * ability + 0.1 * experience / 10), n), 0, 3)
    salary_pct = np.clip(50 + 8 * ability + 4 * experience / 5 + rng.normal(0, 14, n), 1, 99).round(0)

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(1, n + 1)],
        "age": age, "gender": gender, "education": education,
        "experience_years": experience, "department": department,
        "job_level": job_level, "manager_tenure": manager_tenure,
        "projects_count": projects, "avg_task_delay_days": delay,
        "on_time_delivery_rate": on_time, "bug_count": bug,
        "code_review_score": review, "qa_defect_density": defect_density,
        "story_points_completed": story_points, "billable_hours_ratio": billable,
        "training_hours": training, "certifications_count": certs,
        "hackathons_participated": hackathons, "sick_days": sick,
        "unplanned_absences": absences, "avg_login_hours": login,
        "peer_feedback_score": peer, "manager_score": mgr_prev,
        "kudos_count": kudos, "promotions_in_2y": promos,
        "salary_percentile_band": salary_pct,
    })

    # ---------- performance score (the "virtual appraisal") ----------
    score = (
        0.22 * _z(df.on_time_delivery_rate) - 0.12 * _z(df.avg_task_delay_days)
        + 0.14 * _z(df.code_review_score) - 0.08 * _z(df.bug_count)
        + 0.14 * _z(df.manager_score) + 0.10 * _z(df.peer_feedback_score)
        + 0.10 * _z(df.story_points_completed) + 0.05 * _z(df.kudos_count)
        + 0.06 * _z(df.training_hours) + 0.04 * _z(df.certifications_count)
        - 0.06 * _z(df.unplanned_absences) + 0.03 * _z(df.billable_hours_ratio)
        + np.where(df.department.isin(["Data Science", "Engineering"]), 0.05, 0.0)
        + rng.normal(0, 0.28, n)                      # real life is noisy
    )
    q_low, q_high = np.quantile(score, [0.25, 0.80])  # 25% Low, 55% Medium, 20% High
    df[TARGET] = np.where(score <= q_low, "Low", np.where(score >= q_high, "High", "Medium"))

    # ---------- inject realistic dirt: missing values ----------
    for col, frac in {"peer_feedback_score": 0.04, "manager_score": 0.03,
                      "training_hours": 0.02, "code_review_score": 0.03,
                      "avg_login_hours": 0.02}.items():
        df.loc[rng.random(n) < frac, col] = np.nan
    return df


def main():
    df = generate_employees()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} employees x {df.shape[1]} columns -> {DATA_PATH}")
    print(df[TARGET].value_counts(normalize=True).round(3))
    return df


if __name__ == "__main__":
    main()
