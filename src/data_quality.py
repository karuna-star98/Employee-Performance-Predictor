"""Phase 3: data validation + cleaning."""
import pandas as pd
from src.config import RANGE_RULES, ID_COL


def run_quality_checks(df: pd.DataFrame) -> dict:
    report = {"rows": len(df), "columns": df.shape[1],
              "duplicate_ids": int(df[ID_COL].duplicated().sum()),
              "duplicate_rows": int(df.duplicated().sum()),
              "missing_by_column": df.isna().sum()[df.isna().sum() > 0].to_dict(),
              "out_of_range": {}}
    for col, (lo, hi) in RANGE_RULES.items():
        if col in df:
            report["out_of_range"][col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate employees and null-out impossible values (imputed later in the pipeline).
    Imputation is NOT done here on purpose: doing it inside the sklearn Pipeline avoids leakage."""
    df = df.drop_duplicates(subset=ID_COL).copy()
    for col, (lo, hi) in RANGE_RULES.items():
        if col in df:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = float("nan")
    return df
