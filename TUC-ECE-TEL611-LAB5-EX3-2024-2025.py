import numpy as np

# Hyperbolic Tangent (Tanh) function
def tanh(z):
    return np.tanh(z)

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 0, 0, 1])  # Target values

# Initial parameters
w_tanh = np.array([0.0, 0.0])  # Initial weights
b_tanh = 0.0  # Initial bias
learning_rate = 0.1  # Learning rate
epochs = 10  # Maximum number of epochs

# Training the Tanh Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_tanh, X[i]) + b_tanh

        # Apply Tanh activation
        y = tanh(z)

        # Compute error
        error = t[i] - y

        # Compute gradient and update weights and bias
        gradient = error * (1 - y**2)  # Derivative of tanh(z) is 1 - tanh^2(z)
        w_tanh += learning_rate * gradient * X[i]
        b_tanh += learning_rate * gradient
        print('w=',w_tanh,'b=',b_tanh)

# Final weights and bias after training with Tanh
print("Final weights:", w_tanh)
print("Final bias:", b_tanh)
