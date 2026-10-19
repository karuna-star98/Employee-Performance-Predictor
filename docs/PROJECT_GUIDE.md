# Project Guide — Employee Performance Predictor

Everything beyond the code: concepts, stack choices, simulation, GitHub steps, proof plan, interview prep, troubleshooting.

---
## A. Explanation

**Simple:** Look at how an employee has worked recently (deadlines, quality, feedback, learning, attendance) and predict whether their next review will be Low, Medium or High, early enough to help.

**Technical:** Multiclass supervised classification on tabular data. Target `perf_band_next` in {Low, Medium, High}; features from the last 6–12 months, all available *before* the appraisal (no label leakage). Preprocessing in an sklearn `Pipeline`, models compared by cross-validated macro-F1, evaluated on a stratified hold-out set, explained with permutation importance, audited for group fairness.

**How companies use it:** identify high performers (stretch roles, promotion review) · flag likely low performers early (coaching, workload fixes) · plan training budgets · support retention conversations.

**Workflow:** data → preprocessing → analysis → model → prediction → insights → HR decisions (with a human in the loop).

## B. Tech stack options
| Option | Tools | Models | Difficulty | Good for |
|---|---|---|---|---|
| A (Easy) | Python, Pandas, Matplotlib | Logistic Regression, Decision Tree | Low | First ML project |
| **B (Intermediate) ← chosen** | + scikit-learn Pipelines, Seaborn, Streamlit | Logistic, Random Forest, Gradient Boosting, CV tuning | Medium | **Best for placements** |
| C (Advanced) | + XGBoost/LightGBM, SHAP, FastAPI, MLflow, Docker | Tuned boosting, calibration | High | Experienced / ML-engineer roles |

Option B is chosen: complete lifecycle, recognisable tools, runs on any laptop. Upgrade to C later (see Future work).

## C. Architecture & data flow
See the diagram in `README.md`. `main.py` calls each `src/` module in order; each module saves its artefacts (CSV/JSON/PNG/PKL) so every step leaves visible proof.

## Phases (what / why / output / common mistake)
| Phase | What | Why | Output | Common mistake |
|---|---|---|---|---|
| 1 Setup | venv + requirements | Reproducibility | working env | Installing globally |
| 2 Data | `data_generator.py` | No real HR data | `data/employee_features.csv` | Features that directly encode the label |
| 3 Cleaning | range checks, duplicates, NaNs | Garbage in, garbage out | `data_quality_report.json` | Imputing before the split (leakage) |
| 4 EDA | 6 charts + mutual information | Guides modelling | `images/01–06` | Skipping class-balance check |
| 5 Features | 5 engineered columns + Pipeline | Domain knowledge | in-code | Using gender/age as inputs |
| 6 Model | 3 models, 5-fold CV, macro-F1 | Fair comparison | `model_comparison.csv`, `.pkl` | Tuning on the test set |
| 7 Evaluation | test report, confusion matrix, AUC | Real generalisation | `test_metrics.json` | Reporting only accuracy |
| 8 Insights | importance, fairness, one-pagers | Turn predictions into actions | `outputs/` | Presenting correlation as causation |
| 9 Visualisation | dashboard PNG, Streamlit | Communicate | `images/10` | Cluttered charts |
| 10 GitHub | push repo + screenshots | Proof | public repo | Committing `venv/` |

## Installation (Windows / Mac)
```bash
python --version                      # needs 3.9+
python -m venv venv
venv\Scripts\activate                 # Windows
source venv/bin/activate              # Mac / Linux
pip install -r requirements.txt
```

## Code map
| File | Role |
|---|---|
| `src/config.py` | paths, seed, sensitive columns, validation ranges |
| `src/data_generator.py` | virtual company: hidden ability/motivation → features → performance score → bands |
| `src/data_quality.py` | validation + cleaning |
| `src/preprocessing.py` | feature engineering + ColumnTransformer |
| `src/eda.py` | charts |
| `src/train.py` | model comparison and saving |
| `src/evaluate.py` | metrics, confusion matrix, permutation importance |
| `src/fairness.py` | group audit |
| `src/recommender.py` | per-employee drivers → actions → one-pager |
| `src/predict.py` | scoring raw data |
| `src/dashboard.py` | HR dashboard image |
| `app.py` | Streamlit app (lookup, what-if simulator, team overview) |

## E. Virtual simulation (step by step)
1. **Hire the virtual company:** `data_generator.py` creates 3,000 employees across 6 departments, 4 job levels.
2. **Give each person hidden traits:** `ability` and `motivation` (random, never shown to the model).
3. **Generate observable signals:** delivery rate, delays, bugs, review score, training, absences, feedback, kudos... each influenced by the hidden traits plus noise.
4. **Run the "appraisal":** a weighted sum of standardised signals + random noise = true score; bottom 25% → Low, top 20% → High, rest Medium.
5. **Add real-world mess:** ~2–4% missing values in five columns.
6. **Train the AI** on 80%, hide 20% as the "next cycle".
7. **Predict** the hidden 20%; compare with the true bands.
8. **HR uses it:** dashboard → pick at-risk employees → read one-pager → manager has a coaching conversation → re-check in 60 days.

Proof to save: dataset preview, class balance, driver boxplots, correlation heatmap, confusion matrix, importance chart, fairness chart, dashboard, a sample one-pager, terminal run log (`outputs/run_log.txt`).

## F. GitHub upload
```bash
cd Employee-Performance-Predictor
git init
git add .
git commit -m "Initial commit: project structure and requirements"
git branch -M main
git remote add origin https://github.com/<your-username>/employee-performance-predictor.git
git push -u origin main
```
Create the empty repo on github.com first (no README/license, since you already have them).

- **Repo name:** `employee-performance-predictor`
- **Description:** End-to-end people-analytics ML project: predicts employee performance bands, explains drivers, audits fairness and generates HR coaching actions (synthetic data).
- **Topics:** `machine-learning` `data-science` `people-analytics` `hr-analytics` `scikit-learn` `python` `streamlit` `classification` `fairness` `portfolio-project`

## Proof-building plan (5 days)
| Day | Commit | Message | Screenshot |
|---|---|---|---|
| 1 Setup | folders, requirements, config, .gitignore | `chore: project structure and environment setup` | folder tree |
| 2 Dataset | generator + quality checks + CSV | `feat: synthetic HR data generator and quality checks` | dataset head, quality report |
| 3 EDA | eda.py + charts | `feat: EDA with 6 visualisations` | heatmap, boxplots |
| 4 Model | preprocessing, train, evaluate | `feat: pipeline, model comparison, evaluation` | terminal model comparison, confusion matrix |
| 5 Results | fairness, recommender, dashboard, README | `feat: fairness audit, recommendations, dashboard, docs` | dashboard, one-pager, app |

Commit one thing per day with a real message; it looks (and is) more credible than one giant upload.

## G. Interview preparation
**1. Explain the project.** I built a classifier that predicts an employee's next-cycle performance band from work, quality, feedback, learning and attendance signals, explains the drivers, audits fairness, and turns results into coaching actions. Data is synthetic from a simulator I designed.

**2. Why synthetic data?** Real HR data is confidential. I documented the generation logic, so every result is honest about being simulated, and the pipeline is built to accept a real CSV with the same schema.

**3. Why macro-F1 rather than accuracy?** Classes are imbalanced (25/55/20). Macro-F1 weights each class equally, so a model can't look good by only predicting Medium.

**4. Why did Logistic Regression beat Random Forest?** The simulator's score is mostly linear, so a linear model fits it. It's a reminder to compare models rather than assume complex = better.

**5. How did you prevent leakage?** Features only use pre-appraisal information; imputation/scaling happen inside the Pipeline (fit on train folds only); stratified split; IDs and target removed from inputs.

**6. How did you handle fairness?** Excluded gender and age from the model, audited selection rate, F1 and disparate-impact ratio per group, compared with true base rates, and noted small-sample limits instead of claiming "bias-free".

**7. How do you explain a single prediction?** Per-employee drivers = distance from the median × direction × global permutation importance, mapped to concrete actions. SHAP would be the next upgrade.

**8. What would break in production?** Data drift, label delay, manager bias baked into ratings, and using predictions punitively. I'd add time-based validation, drift monitoring (PSI), and human review.

**9. Which errors matter most?** Predicting Low employees as Medium (missed support) and High as Low (unfair flag). I look at per-class recall/precision and the confusion matrix.

**10. How would you improve it?** Real data (e.g. IBM HR Analytics), time-split validation, SHAP, calibration, attrition model, FastAPI + MLflow.

**To HR (non-technical):** "It's an early-warning and talent-spotting tool. It shows who might need support and why, so managers can help sooner. It never decides alone."

**To a technical interviewer:** "Stratified 80/20 split, sklearn Pipeline with ColumnTransformer, 5-fold CV on macro-F1, three model families, permutation importance, group fairness metrics, and a rules layer for recommendations."

## Future improvements
Real HR dataset · SHAP · deep learning (only worthwhile with large real data) · attrition prediction · richer dashboard (Streamlit/Power BI) · real-time scoring API · MLflow + drift monitoring.

## Troubleshooting
| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root (`python main.py`), not from inside `src/` |
| `ModuleNotFoundError: sklearn` | Activate the venv, then `pip install -r requirements.txt` |
| `'python' is not recognized` (Windows) | Reinstall Python with "Add to PATH" ticked, or use `py` |
| PowerShell blocks venv activation | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `FileNotFoundError: employee_perf_model.pkl` | Run `python main.py` before `streamlit run app.py` |
| Model pickle fails to load | Use the same scikit-learn version that trained it; re-run `main.py` |
| `ValueError: Input contains NaN` | Keep the imputer inside the Pipeline; don't drop it |
| Streamlit port busy | `streamlit run app.py --server.port 8502` |
| Git: `remote origin already exists` | `git remote set-url origin <url>` |
| Git push rejected | `git pull origin main --rebase`, then push |
| Large files rejected | Add them to `.gitignore`; the CSV here is small |
