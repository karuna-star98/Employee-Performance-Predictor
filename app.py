"""Streamlit HR app.  Run:  streamlit run app.py   (after `python main.py`)"""
import pandas as pd
import streamlit as st
from src import predict, recommender
from src.config import DATA_PATH, OUT_DIR

st.set_page_config(page_title="Employee Performance Predictor", page_icon="📊", layout="wide")
st.title("📊 Employee Performance Predictor")
st.caption("Decision-support tool — trained on synthetic data. A human must review every recommendation.")


@st.cache_resource
def _model():
    return predict.load_model()


@st.cache_data
def _data():
    return pd.read_csv(DATA_PATH), pd.read_csv(OUT_DIR / "feature_importance.csv", index_col=0)["importance"]


model = _model()
raw, imp = _data()
tab1, tab2, tab3 = st.tabs(["🔍 Employee lookup", "🧪 What-if simulator", "🏢 Team overview"])

with tab1:
    emp = st.selectbox("Employee ID", raw["employee_id"].head(500))
    row = raw[raw.employee_id == emp]
    s = predict.score(row, model)
    s = s.merge(recommender.explain_employees(s, imp, predict.score(raw, model)), on="employee_id")
    r = s.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted band", r.predicted_band)
    c2.metric("P(Low)", f"{r.prob_Low:.0%}"); c3.metric("P(Medium)", f"{r.prob_Medium:.0%}"); c4.metric("P(High)", f"{r.prob_High:.0%}")
    st.markdown(recommender.build_one_pager(r))

with tab2:
    st.write("Change an employee's metrics and see how the prediction moves.")
    base = raw.iloc[[0]].copy()
    a, b, c = st.columns(3)
    base["on_time_delivery_rate"] = a.slider("On-time delivery rate", 0.2, 1.0, 0.75)
    base["code_review_score"] = a.slider("Code review score", 1.0, 5.0, 3.4)
    base["manager_score"] = b.slider("Manager score", 1.0, 5.0, 3.4)
    base["peer_feedback_score"] = b.slider("Peer feedback", 1.0, 5.0, 3.5)
    base["avg_task_delay_days"] = c.slider("Avg task delay (days)", -3.0, 30.0, 4.0)
    base["training_hours"] = c.slider("Training hours", 0, 100, 30)
    res = predict.score(base, model).iloc[0]
    st.subheader(f"Predicted band: {res.predicted_band}")
    st.bar_chart(pd.Series({"Low": res.prob_Low, "Medium": res.prob_Medium, "High": res.prob_High}))

with tab3:
    scored = predict.score(raw, model)
    dept = st.selectbox("Department", ["All"] + sorted(raw.department.unique()))
    if dept != "All":
        scored = scored[scored.department == dept]
    st.bar_chart(scored.predicted_band.value_counts())
    st.dataframe(scored.sort_values("prob_Low", ascending=False)
                 [["employee_id", "department", "job_level", "predicted_band", "prob_Low"]].head(25))
