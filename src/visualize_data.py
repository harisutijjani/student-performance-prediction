import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("data/student_data.csv")

# Create a scatter plot
plt.scatter(data["Study_Hours"], data["Final_Score"])

# Add labels and title
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")

# Display the graph
plt.show()