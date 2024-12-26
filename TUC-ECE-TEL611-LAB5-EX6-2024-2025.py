import numpy as np
import matplotlib.pyplot as plt

# Define the input features (X) and target outputs (t) for the AND operation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input Features
t = np.array([0, 0, 0, 1])  # Target outputs

# Activation functions with corrected logic
def linear_threshold_neuron(x, threshold=1.5):
    return 1 if np.sum(x) >= threshold else 0

def relu_neuron(x, threshold=1.5):
    return 1 if np.sum(x) >= threshold else 0  # ReLU outputs 1 only if sum exceeds threshold

def sigmoid_neuron(x, threshold=0.75):
    output = 1 / (1 + np.exp(-np.sum(x)))
    return 1 if output >= threshold else 0

def tanh_neuron(x, threshold=0.8):
    output = np.tanh(np.sum(x))
    return 1 if output >= threshold else 0

def stochastic_neuron(x, prob=0.5, threshold=0.8):
    deterministic_output = 1 / (1 + np.exp(-np.sum(x)))
    return 1 if deterministic_output >= threshold else 0

# Train and evaluate the neuron for multiple epochs
def train_and_evaluate(neuron_function, X, t, epochs=10, *args, **kwargs):
    all_predictions = []

    for epoch in range(epochs):
        epoch_predictions = []
        for x in X:
            pred = neuron_function(x, *args, **kwargs)
            epoch_predictions.append(pred)
        all_predictions.append(np.array(epoch_predictions))
    
    return np.array(all_predictions)

# Run the comparison with parametrized epochs
neurons = {
    'Linear Threshold': linear_threshold_neuron,
    'ReLU': relu_neuron,
    'Sigmoid': sigmoid_neuron,
    'Tanh': tanh_neuron,
    'Stochastic': stochastic_neuron
}

# Set the number of epochs
epochs = 10

# Results dictionary to store predictions
results = {}

for name, neuron_func in neurons.items():
    preds = train_and_evaluate(neuron_func, X, t, epochs=epochs)
    results[name] = preds

# Plot the results for each neuron after the final epoch
plt.figure(figsize=(10, 8))

for i, (name, preds) in enumerate(results.items(), 1):
    plt.subplot(2, 3, i)
    plt.scatter(X[:, 0], X[:, 1], c=preds[-1], cmap='bwr', marker='o')
    plt.title(f'{name} Neuron (Epoch {epochs})')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Output the predictions after the final epoch
for name, preds in results.items():
    print(f'{name} predictions after {epochs} epochs: {preds[-1]}')
