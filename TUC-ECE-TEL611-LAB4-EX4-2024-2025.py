"""
File Name: house_price_prediction.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a simple linear regression model to predict house prices based on features such as
    the size of the house, number of bedrooms, and age of the house. The dataset is synthetically generated 
    using random values for these features and a linear relationship to determine the target variable (Price).

    Key Components:
    - Dataset Generation: Randomly generates a dataset with three input features (Size, Bedrooms, Age) and 
      the corresponding target variable (Price). The price is modeled as a linear function of these features 
      with some added noise.
    - Linear Regression Model: Uses scikit-learn's `LinearRegression` to train the model on the dataset.
    - Residuals Calculation: Computes the residuals (the differences between the actual and predicted prices) 
      and calculates the standard deviation of these residuals to assess model performance.
    - Prediction with Confidence Interval: Allows the user to input house features (size, number of bedrooms, 
      and age) and predicts the house price along with a range (±1 standard deviation) indicating uncertainty 
      in the prediction.
    
Usage:
    - This script can be used to predict house prices for a given set of features.
    - The user can enter the features for a house (size, number of bedrooms, and age) and the model will output 
      the predicted price and its range.
    
Dependencies:
    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    
Instructions:
    - Ensure that the required dependencies (NumPy, pandas, scikit-learn, matplotlib) are installed in your 
      Python environment.
    - Run the script to train the model on a synthetic dataset and make predictions.
    - When prompted, enter the size, number of bedrooms, and age of the house to receive the predicted price 
      and its range.

"""

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



