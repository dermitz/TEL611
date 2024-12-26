"""
File Name: limitation_of_lse_with_non_linear_data.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the limitation of Least Squares Estimation (LSE) when applied to non-linear data.
    A synthetic dataset with a cubic relationship between the input variable `x` and target variable `y` is 
    generated, with added noise. The script applies a linear regression model (using LSE) to the non-linear data,
    illustrating how the model fails to capture the true non-linear relationship. The results are visualized by 
    plotting the non-linear data and the linear fit.

    Key Components:
    - Dataset Generation: Synthetic data is generated with a cubic relationship `y = 3 * x^3 - 3 * x + 5` plus random noise.
    - Linear Regression: The script uses Least Squares Estimation (LSE) to fit a linear model to the non-linear data.
    - Model Visualization: A scatter plot is generated showing both the actual non-linear data and the linear fit, highlighting 
      the limitations of the linear model.
    
Usage:
    - This script can be used to demonstrate the limitations of linear regression (LSE) when applied to non-linear data.
    - The script generates synthetic data, fits a linear regression model, and visualizes the results.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to observe how LSE fails to model non-linear data accurately.
    - The results will be plotted, showing the non-linear data and the linear fit.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data (quadratic relationship with noise)
np.random.seed(42)  # For reproducibility
x = np.linspace(-0, 10, 30)  # x values
y = 3*x**3- 3 * x + 5 + np.random.normal(0, 10, len(x))  #  with noise

# Fit a linear model using LSE
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

# Calculate slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b = (sum_y - m * sum_x) / n

# Generate the predicted line
y_pred = m * x + b

# Plot the results
plt.scatter(x, y, color='blue', label='Non-linear data (quadratic)')
plt.plot(x, y_pred, color='red', label='LSE linear fit (fails)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Limitation of LSE with Non-Linear Data')
plt.legend()
plt.grid()
plt.show()

# Display the LSE results
print(f"Linear Fit: y = {m:.2f}x + {b:.2f}")
