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

# Step 3: Predict prices for the dataset and calculate residuals
y_pred = model.predict(X)
residuals = y - y_pred

# Calculate the standard deviation of residuals
residual_std = np.std(residuals)

# Step 4: Predict price range for a given input
def predict_price_range(size_input, bedrooms_input, age_input, model, residual_std):
    # Create feature array for prediction
    features = np.array([[size_input, bedrooms_input, age_input]])
    
    # Predict the price
    predicted_price = model.predict(features)[0]
    
    # Calculate price range (±1 standard deviation of residuals)
    lower_bound = predicted_price - residual_std
    upper_bound = predicted_price + residual_std
    
    return predicted_price, lower_bound, upper_bound

# Example input: customer's desired house features
size_input = float(input("Enter the size of the house (in square meters): "))
bedrooms_input = int(input("Enter the number of bedrooms: "))
age_input = float(input("Enter the age of the house (in years): "))

# Predict price range
predicted_price, lower_bound, upper_bound = predict_price_range(size_input, bedrooms_input, age_input, model, residual_std)

# Display the result
print("\n--- Predicted Price Information ---")
print(f"Predicted Price: {predicted_price:.2f} thousand euros")
print(f"Price Range: {lower_bound:.2f} - {upper_bound:.2f} thousand euros")



