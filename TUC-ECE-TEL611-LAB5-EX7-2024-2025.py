import numpy as np
import tensorflow as tf

# Define the input features (X) and target outputs (t) for the AND operation
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input Features
t = np.array([0, 0, 0, 1])  # Target outputs

# Linear Threshold Neuron using NumPy
def linear_threshold_neuron(x, threshold=0.5):
    return (np.sum(x) > threshold).astype(int)

# ReLU Neuron using TensorFlow
def relu_neuron(x):
    return tf.keras.activations.relu(np.sum(x))

# Sigmoid Neuron using TensorFlow (fixed)
def sigmoid_neuron(x):
    return tf.keras.activations.sigmoid(float(np.sum(x))).numpy()

# Tanh Neuron using TensorFlow
def tanh_neuron(x):
    return tf.keras.activations.tanh(float(np.sum(x))).numpy()

# Stochastic Neuron using TensorFlow and NumPy
def stochastic_neuron(x):
    prob = tf.keras.activations.sigmoid(float(np.sum(x))).numpy()  # Logistic probability
    return np.random.binomial(1, prob)

# Train and evaluate the neurons
def train_and_evaluate(neuron_function, X, t, epochs=10, *args, **kwargs):
    """
    Train and evaluate neuron functions for a specified number of epochs.
    """
    predictions = []
    for epoch in range(epochs):
        epoch_predictions = []
        for x in X:
            pred = neuron_function(x, *args, **kwargs)
            epoch_predictions.append(pred)
        predictions.append(epoch_predictions)
    return np.array(predictions)

# Define the neurons to evaluate
neurons = {
    'Linear Threshold': linear_threshold_neuron,
    'ReLU': relu_neuron,
    'Sigmoid': sigmoid_neuron,
    'Tanh': tanh_neuron,
    'Stochastic': stochastic_neuron
}

# Training and results collection
results = {}
epochs = 10

for name, neuron_func in neurons.items():
    preds = train_and_evaluate(neuron_func, X, t, epochs)
    results[name] = preds

# Display results
for name, preds in results.items():
    print(f"{name} predictions after {epochs} epochs:")
    print(preds[-1])  # Final predictions
    print("Over epochs:\n", preds)
    print("-" * 40)
