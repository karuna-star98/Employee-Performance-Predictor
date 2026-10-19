"""Run with:  pytest -q"""
import pandas as pd
from src import data_generator, data_quality, predict
from src.preprocessing import add_features, get_X_y
from src.config import TARGET, SENSITIVE


def test_generator_shape_and_classes():
    df = data_generator.generate_employees(n=400, seed=1)
    assert len(df) == 400 and df.employee_id.is_unique
    assert set(df[TARGET]) == {"Low", "Medium", "High"}


def test_quality_checks_detect_bad_values():
    df = data_generator.generate_employees(n=100, seed=1)
    df.loc[0, "on_time_delivery_rate"] = 5
    assert data_quality.run_quality_checks(df)["out_of_range"]["on_time_delivery_rate"] == 1
    assert pd.isna(data_quality.clean(df).loc[0, "on_time_delivery_rate"])


def test_sensitive_columns_not_in_model_inputs():
    df = add_features(data_generator.generate_employees(n=100, seed=1))
    X, _ = get_X_y(df)
    assert not set(SENSITIVE) & set(X.columns) and "employee_id" not in X.columns


def test_feature_engineering_adds_columns():
    df = add_features(data_generator.generate_employees(n=50, seed=1))
    assert {"delivery_x_quality", "learning_index", "absence_total"} <= set(df.columns)
