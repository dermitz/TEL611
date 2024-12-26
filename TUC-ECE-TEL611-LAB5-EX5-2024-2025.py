#Stochastic Neuron Model
import numpy as np

# Sigmoid function for probability calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
t = np.array([0, 1, 1, 0])  # Target values (example)

# Initial parameters
w_stochastic = np.array([0.0, 0.0])  # Initial weights
b_stochastic = 0.0  # Initial bias
learning_rate = 0.1  # Learning rate
epochs = 10  # Maximum number of epochs

# Training the Stochastic Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute weighted sum
        z = np.dot(w_stochastic, X[i]) + b_stochastic

        # Compute activation probability using sigmoid
        p = sigmoid(z)

        # Stochastic firing: output is 1 with probability p, else 0
        y = np.random.binomial(1, p)

        # Compute error
        error = t[i] - y

        # Update weights and bias
        w_stochastic += learning_rate * error * X[i]
        b_stochastic += learning_rate * error
        print('w=',w_stochastic,'b=',b_stochastic)

# Final weights and bias after training with the Stochastic Neuron
print("Final weights and bias after training with the Stochastic Neuron")
print("Final weights:", w_stochastic)
print("Final bias:", b_stochastic)
