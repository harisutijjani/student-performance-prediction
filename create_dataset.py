import pandas as pd
import numpy as np

# Make the results reproducible
np.random.seed(42)

# Number of students
number_of_students = 200

# Generate student data
study_hours = np.random.uniform(1, 10, number_of_students)
attendance = np.random.uniform(50, 100, number_of_students)
assignment_score = np.random.uniform(40, 100, number_of_students)
previous_score = np.random.uniform(40, 100, number_of_students)
participation = np.random.uniform(1, 10, number_of_students)

# Calculate final score with some realistic variation
final_score = (
    study_hours * 3
    + attendance * 0.2
    + assignment_score * 0.2
    + previous_score * 0.25
    + participation * 1.5
    + np.random.normal(0, 3, number_of_students)
)

# Keep final scores between 0 and 100
final_score = np.clip(final_score, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    "Study_Hours": study_hours.round(2),
    "Attendance": attendance.round(2),
    "Assignment_Score": assignment_score.round(2),
    "Previous_Score": previous_score.round(2),
    "Participation": participation.round(2),
    "Final_Score": final_score.round(2)
})

# Save dataset
data.to_csv("data/student_data.csv", index=False)

print("Dataset created successfully!")
print(f"Number of students: {len(data)}")
print("\nFirst 5 students:")
print(data.head())