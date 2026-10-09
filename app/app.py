import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==================================================
# PATHS AND MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"
HISTORY_PATH = BASE_DIR / "reports" / "prediction_history.csv"

HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "Date_Time",
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Score",
    "Participation",
    "Predicted_Score",
    "Performance_Category",
]

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================

def load_history():
    if HISTORY_PATH.exists():
        try:
            history = pd.read_csv(HISTORY_PATH)
            if all(column in history.columns for column in COLUMNS):
                return history[COLUMNS]
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    return pd.DataFrame(columns=COLUMNS)


def save_prediction(record):
    history = load_history()
    updated = pd.concat(
        [history, pd.DataFrame([record])],
        ignore_index=True,
    )
    updated.to_csv(HISTORY_PATH, index=False)


def get_category(score):
    if score >= 70:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 50:
        return "Average"
    return "Poor"


# ==================================================
# HEADER
# ==================================================

st.title("🎓 Student Performance Prediction System")
st.write(
    "Predict student examination scores and review "
    "previous predictions."
)

history = load_history()

# ==================================================
# DASHBOARD SUMMARY
# ==================================================

st.header("📊 Dashboard")

total = len(history)

average_score = (
    pd.to_numeric(history["Predicted_Score"], errors="coerce").mean()
    if total
    else 0
)

excellent_count = (
    history["Performance_Category"].eq("Excellent").sum()
    if total
    else 0
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)
col2.metric("Average Predicted Score", f"{average_score:.2f}")
col3.metric("Excellent Predictions", int(excellent_count))

st.divider()

# ==================================================
# INPUT FORM
# ==================================================

st.header("📝 Predict a Student's Performance")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5,
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
        )

        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
        )

    with col2:
        previous_score = st.number_input(
            "Previous Score",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

        participation = st.number_input(
            "Participation (1–10)",
            min_value=1.0,
            max_value=10.0,
            value=8.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "🔮 Predict and Save",
        use_container_width=True,
    )

# ==================================================
# PREDICTION AND SAVING
# ==================================================

if submitted:
    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Assignment_Score": [assignment_score],
        "Previous_Score": [previous_score],
        "Participation": [participation],
    })

    score = float(model.predict(student)[0])
    score = max(0.0, min(100.0, score))
    category = get_category(score)

    record = {
        "Date_Time": datetime.now().astimezone().strftime(
            "%Y-%m-%d %H:%M:%S %Z"
        ),
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Assignment_Score": assignment_score,
        "Previous_Score": previous_score,
        "Participation": participation,
        "Predicted_Score": round(score, 2),
        "Performance_Category": category,
    }

    save_prediction(record)

    st.session_state["last_prediction"] = record
    st.rerun()

# Show the most recent result after rerunning
if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]

    st.subheader("Latest Prediction")
    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Final Score",
        f"{result['Predicted_Score']:.2f} / 100",
    )
    col2.metric(
        "Performance Category",
        result["Performance_Category"],
    )

    st.caption(
        "This is a model estimate, not a guaranteed examination result."
    )

st.divider()

# ==================================================
# CATEGORY SUMMARY
# ==================================================

st.header("📈 Performance Category Summary")

history = load_history()

if not history.empty:
    category_counts = (
        history["Performance_Category"]
        .value_counts()
        .reindex(["Excellent", "Good", "Average", "Poor"], fill_value=0)
    )

    st.bar_chart(category_counts)

    st.divider()

    # ==================================================
    # HISTORY TABLE AND DOWNLOAD
    # ==================================================

    st.header("📚 Prediction History")

    st.dataframe(
        history.sort_values("Date_Time", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info(
        "No predictions have been saved yet. "
        "Complete the form above to create your first record."
    )

# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption("Student Performance Prediction System | Machine Learning Project")