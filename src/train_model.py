import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("data/student_data.csv")

# Features
X = data[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Previous_Score",
        "Participation"
    ]
]

# Target
y = data["Final_Score"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Complete!")
print("------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nActual vs Predicted Scores:")
print("---------------------------")

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(2)
})

print(comparison.head(10))

# Plot actual vs predicted scores
plt.scatter(y_test, predictions)

plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Final Scores")

plt.show()

# Save the trained model
joblib.dump(model, "models/student_performance_model.pkl")

print("\nModel saved successfully!")
print("Location: models/student_performance_model.pkl")