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
