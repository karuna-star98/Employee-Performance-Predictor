"""Central configuration: paths, column groups, random seed.
Change things here once instead of hunting through every file."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "images"
REPORT_DIR = OUT_DIR / "employee_reports"

DATA_PATH = DATA_DIR / "employee_features.csv"
MODEL_PATH = MODEL_DIR / "employee_perf_model.pkl"

SEED = 13
N_EMPLOYEES = 3000
TARGET = "perf_band_next"          # High / Medium / Low
ID_COL = "employee_id"
CLASSES = ["Low", "Medium", "High"]

# Sensitive attributes: NEVER fed to the model. Kept only for the fairness audit.
SENSITIVE = ["gender", "age"]

# Valid ranges used by the data-quality checks
RANGE_RULES = {
    "experience_years": (0, 45),
    "on_time_delivery_rate": (0, 1),
    "manager_score": (0, 5),
    "peer_feedback_score": (0, 5),
    "code_review_score": (0, 5),
    "avg_task_delay_days": (-10, 60),
    "billable_hours_ratio": (0, 1),
}

for d in (DATA_DIR, MODEL_DIR, OUT_DIR, IMG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)
