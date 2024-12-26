"""
File Name: least_squares_regression.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the use of the Least Squares method to perform linear regression on a synthetic dataset. 
    The dataset is generated with a linear relationship between the variables 'x' and 'y' (i.e., y = 3x + 7), with added 
    Gaussian noise. The script calculates the coefficients of the regression line using the Least Squares method, 
    predicts the values based on the model, visualizes the data and the fitted line, and computes the Residual Sum of Squares (RSS).

    Key Components:
    - Data Generation: The script generates synthetic data points where 'y' is linearly related to 'x' with added Gaussian noise.
    - Least Squares Solution: The script calculates the coefficients (β₀ and β₁) of the regression line using the 
      Least Squares method. It computes these values based on the mean of 'x' and 'y' and their covariance.
    - Model Prediction: After calculating the regression coefficients, the script predicts the 'y' values for the dataset 
      using the fitted line.
    - Visualization: The script visualizes the original data points and the regression line using a scatter plot and a line plot.
    - Residual Sum of Squares (RSS): The script calculates the RSS to evaluate the goodness of fit of the regression model.

Usage:
    - This script is useful for demonstrating how the Least Squares method works for simple linear regression.
    - It shows how to compute the coefficients of a linear model, how to use it for prediction, and how to evaluate the model 
      using the RSS.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to generate the synthetic dataset, fit a linear regression model, and visualize the results.
    - The script will display the dataset, the fitted regression line, and the RSS of the model.

Notes:
    - This script assumes that the data is linearly related and the relationship between 'x' and 'y' can be modeled 
      using a simple linear equation.
"""

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dataset
np.random.seed(42)  # For reproducibility

# Generate random data points
x = np.linspace(0, 10, 50)  # 50 evenly spaced values between 0 and 10
y = 3 * x + 7 + np.random.normal(0, 5, size=x.shape)  # y = 3x + 7 + noise

# Step 2: Compute Least Squares Solution
# Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Compute coefficients
beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
beta_0 = y_mean - beta_1 * x_mean

print(f"Least Squares Coefficients: β₀ = {beta_0:.2f}, β₁ = {beta_1:.2f}")

# Step 3: Predict Values Using the Regression Line
y_pred = beta_0 + beta_1 * x

# Step 4: Visualize the Data and Fitted Line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='Data Points')  # Original data
plt.plot(x, y_pred, color='red', label='Regression Line')  # Fitted line
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression')
plt.legend()
plt.show()

# Step 5: Compute and Print Residual Sum of Squares (RSS)
rss = np.sum((y - y_pred)**2)
print(f"Residual Sum of Squares (RSS): {rss:.2f}")
