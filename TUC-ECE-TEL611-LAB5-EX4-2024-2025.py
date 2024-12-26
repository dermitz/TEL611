import numpy as np

# SoftMax function
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Numerical stability
    return exp_z / np.sum(exp_z)

# Dataset (inputs and one-hot encoded targets for multi-class classification)
X = np.array([
    [1, 2],
    [1, 3],
    [0, 2],
    [2, 1]
])
t = np.array([
    [1, 0, 0],  # Class 1
    [0, 1, 0],  # Class 2
    [0, 0, 1],  # Class 3
    [1, 0, 0]   # Class 1
])

# Parameters
num_classes = 3
num_features = X.shape[1]
w_softmax = np.zeros((num_features, num_classes))  # Weights matrix
b_softmax = np.zeros(num_classes)  # Bias vector
learning_rate = 0.1
epochs = 10

# Training the SoftMax Neuron
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute logits (z)
        z = np.dot(X[i], w_softmax) + b_softmax

        # Apply SoftMax activation
        y = softmax(z)

        # Compute error
        error = t[i] - y

        # Update weights and bias
        w_softmax += learning_rate * np.outer(X[i], error)
        b_softmax += learning_rate * error
        print('w=',w_softmax,'b=',b_softmax)
# Final weights and bias after training with SoftMax
print("Final weights and bias after training with SoftMax")
print("Final weights:", w_softmax)
print("Final bias:", b_softmax)
