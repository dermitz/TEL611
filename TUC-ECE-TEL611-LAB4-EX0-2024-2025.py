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
