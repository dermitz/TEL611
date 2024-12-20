import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Generate the dataset
np.random.seed(42)  # For reproducibility
n_samples = 100

# Simulating features
size = np.random.uniform(50, 200, n_samples)  # Size in square meters
bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
age = np.random.uniform(1, 50, n_samples)  # Age of the house

# Simulating target variable (Price)
epsilon = np.random.normal(0, 10, n_samples)  # Random noise
price = 50 + 0.8 * size + 10 * bedrooms - 0.5 * age + epsilon  # Price in thousand euros

# Combine data into a DataFrame
data = pd.DataFrame({'Size': size, 'Bedrooms': bedrooms, 'Age': age, 'Price': price})

# Step 2: Fit the model using scikit-learn
X = data[['Size', 'Bedrooms', 'Age']]  # Features matrix
y = data['Price']  # Target variable

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Extract coefficients and intercept
intercept = model.intercept_
coefficients = model.coef_

# Step 3: Predict and evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# Step 4: Print results
print("Linear Regression Model:")
print(f"Intercept (b): {intercept:.2f}")
print(f"Coefficient for Size: {coefficients[0]:.2f}")
print(f"Coefficient for Bedrooms: {coefficients[1]:.2f}")
print(f"Coefficient for Age: {coefficients[2]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Step 5: Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Relationship with Size
axes[0].scatter(data['Size'], data['Price'], color='blue', label='Actual Prices')
axes[0].scatter(data['Size'], y_pred, color='red', label='Predicted Prices')
axes[0].set_title('Price vs. Size')
axes[0].set_xlabel('Size (square meters)')
axes[0].set_ylabel('Price (thousand euros)')
axes[0].legend()

# Relationship with Bedrooms
axes[1].scatter(data['Bedrooms'], data['Price'], color='blue', label='Actual Prices')
axes[1].scatter(data['Bedrooms'], y_pred, color='red', label='Predicted Prices')
axes[1].set_title('Price vs. Bedrooms')
axes[1].set_xlabel('Number of Bedrooms')
axes[1].set_ylabel('Price (thousand euros)')
axes[1].legend()

# Relationship with Age
axes[2].scatter(data['Age'], data['Price'], color='blue', label='Actual Prices')
axes[2].scatter(data['Age'], y_pred, color='red', label='Predicted Prices')
axes[2].set_title('Price vs. Age')
axes[2].set_xlabel('Age (years)')
axes[2].set_ylabel('Price (thousand euros)')
axes[2].legend()

plt.tight_layout()
plt.show()
