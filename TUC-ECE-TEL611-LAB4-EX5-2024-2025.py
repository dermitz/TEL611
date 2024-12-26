"""
File Name: xor_mlp_classifier.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a solution for the XOR problem using a Multi-layer Perceptron (MLP) classifier. The XOR 
    problem is a classic binary classification problem where the output is 1 when either of the two inputs is 1, 
    but not both. The script utilizes an MLP neural network with one hidden layer and ReLU activation to model 
    the XOR logic function. 

    Key components:
    - Dataset: A small binary classification dataset representing the XOR problem, with two binary inputs and 
      a binary output.
    - MLPClassifier: A neural network classifier used to approximate the XOR function. It uses one hidden layer 
      with three neurons and ReLU activation.
    - Evaluation: The model is trained on the XOR dataset, and its performance is evaluated using the accuracy score.
    - Visualization: The input data and the decision boundary are visualized using a scatter plot.

Usage:
    - This script is designed to solve the XOR problem using a neural network model.
    - The XOR problem is used as a simple example to showcase the capabilities of multi-layer neural networks in 
      solving non-linear problems.
    - The script visualizes the XOR data, trains the MLP model, and prints the accuracy and predicted results for each input.

Dependencies:
    - numpy
    - matplotlib
    - sklearn

Instructions:
    - Ensure that NumPy, scikit-learn, and Matplotlib are installed in your Python environment.
    - Run the script to train the MLP model on the XOR dataset.
    - The script will output the model's accuracy and display the input-output mapping for each test case.
    - A scatter plot will be generated, showing the XOR data and indicating the class labels with different colors.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
y = np.array([0, 1, 1, 0])  # Outputs (XOR)

# Visualize the data
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR Problem')
plt.legend()
plt.grid()
plt.show()

# Create a Multi-Layer Perceptron (MLP) neural network
mlp = MLPClassifier(hidden_layer_sizes=(3,), activation='relu', max_iter=1000, random_state=42)

# Train the model
mlp.fit(X, y)

# Make predictions
y_pred = mlp.predict(X)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
print("Model accuracy:", accuracy)

# Display predictions
for i in range(len(X)):
    print(f"Input: {X[i]} | Expected Output: {y[i]} | Predicted Output: {y_pred[i]}")
