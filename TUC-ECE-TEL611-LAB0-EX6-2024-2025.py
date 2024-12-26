"""
File Name: california_housing_linear_regression.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a linear regression analysis on the California Housing dataset.
    The goal is to predict the median house value based on the median income using a simple linear regression model.
    It also demonstrates basic data exploration techniques, including dataset inspection, handling missing values, and visualizing the relationship between features.

Key Components:
    1. **Data Loading**: The California Housing dataset is loaded using `fetch_california_housing` from scikit-learn, which provides a dataframe containing housing features and target labels.
    2. **Data Inspection**: The dataset is inspected to identify missing values and get a statistical summary.
    3. **Data Visualization**: A scatter plot is generated to visualize the relationship between median income (MedInc) and median house value (MedHouseVal). A regression line is also plotted to visualize the model's predictions.
    4. **Correlation Analysis**: The correlation matrix is computed to understand the relationships between features and the target variable.
    5. **Model Training**: A simple linear regression model is trained using the median income feature to predict the median house value. The model's performance is evaluated using mean squared error (MSE).
    6. **Model Evaluation**: The model's accuracy is assessed based on the MSE, and the regression line is visualized against the actual data points.

Usage:
    - This script is designed to guide students through the process of exploring a dataset, performing linear regression, and evaluating the model's performance.
    - It provides insights into the relationship between economic factors (like median income) and housing prices.

Dependencies:
    - pandas
    - matplotlib
    - scikit-learn

Instructions:
    - Run the script to load the California Housing dataset, inspect and visualize the data, and train a linear regression model.
    - Update the dataset path if necessary and execute the script in an appropriate Python environment with the required libraries.

Notes:
    - This example uses median income as the single feature for prediction. More sophisticated models can be built by including additional features.
    - The mean squared error is used as a measure of the model's accuracy. Lower MSE indicates a better fit of the model to the data.
    - The correlation analysis helps identify which features have the strongest relationship with the target variable.

Questions for Students:
    1. Why is it important to explore and visualize data before building a model?
    2. What does the correlation matrix tell you about the relationship between the features and the target variable?
    3. Why might the linear regression model not perform well with only one feature?
    4. How would you improve the model by incorporating more features?
    5. What other evaluation metrics could you use to assess the performance of your regression model?
"""

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
