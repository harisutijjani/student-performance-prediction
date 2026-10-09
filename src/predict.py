import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/student_performance_model.pkl")


def get_number(prompt, minimum, maximum):
    """Get a valid number within a specified range."""

    while True:
        try:
            value = float(input(prompt))

            if value < minimum or value > maximum:
                print(
                    f"Invalid input! Please enter a value "
                    f"between {minimum} and {maximum}."
                )
            else:
                return value

        except ValueError:
            print("Invalid input! Please enter a number.")


print("========================================")
print("   STUDENT PERFORMANCE PREDICTION")
print("========================================")

study_hours = get_number(
    "Enter Study Hours: ", 0, 24
)

attendance = get_number(
    "Enter Attendance (%): ", 0, 100
)

assignment_score = get_number(
    "Enter Assignment Score: ", 0, 100
)

previous_score = get_number(
    "Enter Previous Score: ", 0, 100
)

participation = get_number(
    "Enter Participation (1-10): ", 1, 10
)


# Create student data
new_student = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Assignment_Score": [assignment_score],
    "Previous_Score": [previous_score],
    "Participation": [participation]
})


# Make prediction
predicted_score = model.predict(new_student)[0]

# Keep score between 0 and 100
predicted_score = max(0, min(100, predicted_score))


# Determine performance category
if predicted_score >= 70:
    performance = "Excellent"
elif predicted_score >= 60:
    performance = "Good"
elif predicted_score >= 50:
    performance = "Average"
else:
    performance = "Poor"


# Display result
print("\n========================================")
print("             PREDICTION RESULT")
print("========================================")
print(f"Predicted Final Score: {predicted_score:.2f}")
print(f"Performance Category: {performance}")
print("========================================")