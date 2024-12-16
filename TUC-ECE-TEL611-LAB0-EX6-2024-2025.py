# Import necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
# Load the dataset
california = fetch_california_housing(as_frame=True)
california_data = california.frame
# Display the first 5 rows
print("First 5 rows of the California Housing dataset:")
print(california_data.head())
# Inspect the dataset
print("\nDataset Information:")
print(california_data.info())
# Check for missing values
print("\nMissing Values in Each Column:")
print(california_data.isnull().sum())
# Statistical summary of the dataset
print("\nStatistical Summary:")
print(california_data.describe())

import matplotlib.pyplot as plt
# Scatter plot for MedInc vs MedHouseVal
plt.figure(figsize=(8, 6))
plt.scatter(california_data['MedInc'], california_data['MedHouseVal'], c='blue', edgecolor='k', alpha=0.7)
plt.title("Relationship Between Median Income and Median House Value")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.show()
# Correlation matrix
correlation_matrix = california_data.corr()
# Display correlation of features with MedHouseVal
print("\nCorrelation of Features with Median House Value (MedHouseVal):")
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Prepare the data
X = california_data[['MedInc']]  # Feature: median income
y = california_data['MedHouseVal']  # Target: median house value
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error (MSE) of the Model:", mse)

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title("Linear Regression: MedInc vs MedHouseVal")
plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.legend()
plt.show()
