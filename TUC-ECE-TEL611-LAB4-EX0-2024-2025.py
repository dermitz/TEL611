"""
File Name: least_squares_error_1d.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the application of Least Squares Error (LSE) to fit a linear model to a set of 1D data points. 
    The script computes the best-fit line using the Least Squares method, which minimizes the sum of squared differences 
    between the observed data points and the predicted values from the linear model. It then visualizes the data points 
    and the best-fit line on a plot.

    Key Components:
    - Dataset: The script uses a set of 1D data points (x, y) to demonstrate the application of the LSE method.
    - Linear Model Fitting: The slope (m) and intercept (b) of the best-fit line are calculated using the Least Squares method.
    - Visualization: The data points and the resulting best-fit line are plotted for visualization.

Usage:
    - This script can be used to demonstrate the application of Least Squares Error to find the best-fit line for 1D data.
    - The results will be displayed as a plot showing the data points and the corresponding best-fit line.

Dependencies:
    - numpy
    - matplotlib

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib) are installed in your Python environment.
    - Run the script to calculate and visualize the best-fit line for the given data points.
    - The results will be printed in the console and displayed in a plot with the data points and the best-fit line.
"""

import numpy as np
import matplotlib.pyplot as plt

# Given data points
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])

# Number of points
n = len(x)

# Calculate sums
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

# Calculate slope (m) and intercept (b)
m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
b = (sum_y - m * sum_x) / n

# Generate the best-fit line
y_pred = m * x + b

# Display results
print(f"Best-fit line: y = {m:.2f}x + {b:.2f}")

# Plot the data points
plt.scatter(x, y, color='blue', label='Data points')
# Plot the best-fit line
plt.plot(x, y_pred, color='red', label='Best-fit line')
# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Error - 1D')
plt.legend()
plt.grid()
plt.show()
