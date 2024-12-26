import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate the dataset
np.random.seed(42)  # For reproducibility
n_samples = 100
x1 = np.random.uniform(0, 10, n_samples)  # First feature
x2 = np.random.uniform(0, 10, n_samples)  # Second feature
epsilon = np.random.normal(0, 2, n_samples)  # Random noise

# Target variable with known relationship
y = 3 * x1 + 2 * x2 + 5 + epsilon

# Combine features into a DataFrame
data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

# Step 2: Fit a multiple linear regression model using scikit-learn
X = data[['x1', 'x2']].values  # Features matrix
y = data['y'].values  # Target variable

# Initialize and fit the LinearRegression model
model = LinearRegression()
model.fit(X, y)

# Extract coefficients and intercept
intercept = model.intercept_
m1, m2 = model.coef_


# Step 3: Predict values
y_pred = model.predict(X)

# Step 4: Evaluate the model
mse = mean_squared_error(y, y_pred)

# Step 5: Print results
print("Scikit-Learn Coefficients:")
print(f"Intercept (b): {intercept:.2f}")
print(f"Coefficient for x1 (m1): {m1:.2f}")
print(f"Coefficient for x2 (m2): {m2:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot actual data and predictions
ax.scatter(x1, x2, y, color='blue', label='Data (y)')
ax.scatter(x1, x2, y_pred, color='red', label='Predictions (y_pred)')

# Labels and legend
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Multiple Linear Regression Using Scikit-Learn')
ax.legend()
plt.show()