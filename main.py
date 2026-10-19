"""One command runs the whole project:  python main.py

Pipeline: generate data -> quality checks -> EDA -> train -> evaluate -> explain
          -> fairness audit -> score everyone -> HR one-pagers -> dashboard
"""
import json
import pandas as pd
from src.preprocessing import add_features
from src import data_generator, data_quality, eda, train, evaluate, fairness, recommender, predict, dashboard
from src.config import DATA_PATH, OUT_DIR, TARGET


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    banner("PHASE 2: Create synthetic company data")
    df = data_generator.main()
    print(df.head(5).T.head(12))

    banner("PHASE 3: Data quality checks + cleaning")
    report = data_quality.run_quality_checks(df)
    print(json.dumps(report, indent=2, default=str))
    json.dump(report, open(OUT_DIR / "data_quality_report.json", "w"), indent=2, default=str)
    df = data_quality.clean(df)

    banner("PHASE 4: EDA")
    mi = eda.run_eda(df)
    print("Top mutual-information features:\n", mi.head(8).round(3))

    banner("PHASE 5-6: Feature engineering + model training")
    df = add_features(df)
    train_df, test_df = train.split(df)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    model, name, _ = train.train_models(train_df)

    banner("PHASE 7: Evaluation on hold-out test set")
    metrics, test_pred = evaluate.evaluate(model, test_df)
    importance = evaluate.global_importance(model, test_df)

    banner("PHASE 8a: Fairness audit")
    fairness.run_fairness_audit(test_df, test_pred)

    banner("PHASE 8b: Score all employees + recommendations")
    scored = predict.score(df, model)
    exp = recommender.explain_employees(scored, importance, reference_df=df)
    scored = scored.merge(exp, on="employee_id")
    scored[["employee_id", "department", "job_level", "predicted_band", "prob_Low", "prob_Medium",
            "prob_High", "concerns", "strengths"]].round(3).to_csv(OUT_DIR / "predictions.csv", index=False)
    n = recommender.write_reports(scored)
    print(f"Wrote predictions.csv and {n} one-pagers in outputs/employee_reports/")

    banner("PHASE 9: Dashboard")
    dashboard.build_dashboard(scored, importance, metrics)
    print("Saved images/10_hr_dashboard.png")

    banner("DONE")
    print(f"Model: {name} | Test accuracy {metrics['accuracy']:.3f} | Macro-F1 {metrics['macro_f1']:.3f}")
    print("Open the images/ and outputs/ folders — that is your project proof.")


if __name__ == "__main__":
    main()
