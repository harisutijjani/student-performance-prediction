import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODEL_PATH = BASE_DIR / "models" / "student_performance_model.pkl"

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Machine Learning Model Evaluation")
st.write(
    "Evaluate the trained model using a held-out test dataset."
)

# Load dataset and saved model
data = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

features = [
    "Study_Hours",
    "Attendance",
    "Assignment_Score",
    "Previous_Score",
    "Participation",
]

X = data[features]
y = data["Final_Score"]

# Use the same split as the original training script
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
c3.metric("R² Score", f"{r2:.3f}")

st.caption(
    "Lower MAE and MSE are better. An R² closer to 1 generally "
    "indicates a better fit on the test data."
)

st.divider()

st.subheader("Actual vs Predicted Scores")

comparison = pd.DataFrame({
    "Actual Score": y_test.to_numpy(),
    "Predicted Score": predictions,
})

fig, ax = plt.subplots()
ax.scatter(
    comparison["Actual Score"],
    comparison["Predicted Score"],
)
ax.plot([0, 100], [0, 100], linestyle="--")
ax.set_xlabel("Actual Score")
ax.set_ylabel("Predicted Score")
ax.set_title("Actual vs Predicted Final Scores")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
st.pyplot(fig)
plt.close(fig)

st.write(
    "Points closer to the dashed diagonal represent predictions "
    "closer to the actual scores."
)

st.divider()

st.subheader("Feature Correlations with Final Score")

correlations = data[features + ["Final_Score"]].corr(numeric_only=True)
target_correlations = (
    correlations["Final_Score"]
    .drop("Final_Score")
    .sort_values()
)

st.bar_chart(target_correlations)

st.caption(
    "Correlation shows how features relate to final scores in this "
    "dataset. It does not prove that a feature causes better performance."
)

st.divider()

st.subheader("Dataset and Test Split")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", len(data))
c2.metric("Training Records", len(X_train))
c3.metric("Testing Records", len(X_test))

st.subheader("Test Predictions")
st.dataframe(
    comparison.round(2),
    use_container_width=True,
    hide_index=True,
)