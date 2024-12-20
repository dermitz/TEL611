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
