# Reimport necessary libraries after reset
import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 0, 0, 1])  # Target values

# Initial parameters
w_sigmoid = np.array([0.0, 0.0])  # Initial weights
b_sigmoid = 0.0  # Initial bias
learning_rate = 0.1  # Smaller learning rate for smooth convergence
epochs = 5  # Maximum number of epochs

# Training the Sigmoid Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_sigmoid, X[i]) + b_sigmoid
        
        # Apply Sigmoid activation
        y = sigmoid(z)
        
        # Compute error
        error = t[i] - y
        
        # Compute gradient and update weights and bias
        gradient = error * y * (1 - y)
        w_sigmoid += learning_rate * gradient * X[i]
        b_sigmoid += learning_rate * gradient
        print('w=',w_sigmoid,'b=',b_sigmoid)

# Final weights and bias after training with Sigmoid
w_sigmoid, b_sigmoid
print("Final weights and bias after training with Sigmoid")
print('w=',w_sigmoid,'b=',b_sigmoid)