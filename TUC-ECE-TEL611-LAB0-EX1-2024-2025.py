"""
File Name: sine_cosine_wave_plot.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script generates and visualizes sine and cosine waveforms using `matplotlib` and `numpy`. It contains two plots:
    1. A combined plot of sine and cosine waves on the same graph.
    2. A set of subplots displaying the sine and cosine waves separately.

Key Components:
    1. **Data Generation**: The `numpy` library is used to generate a sequence of time points `t` and compute the sine and cosine values (`y` and `z`).
    2. **Plot 1**: A single plot containing both the sine and cosine waves, with a legend and grid.
    3. **Plot 2**: Two subplots—one for the sine wave and one for the cosine wave—arranged vertically, with labels and legends.
    4. **Plot Customization**: Titles, axis labels, legends, and grid lines are added for clarity.

Usage:
    - Run the script in a Python environment to visualize sine and cosine waveforms.
    - The script will display two sets of plots:
        1. A combined plot of sine and cosine.
        2. Two separate subplots, one for the sine wave and one for the cosine wave.

Dependencies:
    - matplotlib
    - numpy

Instructions:
    - Run the script in a Python environment.
    - The first plot will display both sine and cosine waves on the same graph.
    - The second plot will display the sine and cosine waves in separate subplots.

Notes:
    - This script demonstrates basic plotting functionality with `matplotlib`, including the use of subplots.
    - The `numpy.linspace` function is used to generate evenly spaced points, and the sine and cosine functions are computed using `numpy.sin` and `numpy.cos`.
    - The `tight_layout` method is used to adjust the layout to prevent overlapping of subplots.

"""

import matplotlib.pyplot as plt
import numpy as np

# Generate data
t = np.linspace(0, 10, 10000)  # 10000 points between 0 and 10
y = np.sin(t)                  # Sine wave
z = np.cos(t)                  # Cosine wave

# First figure: Combined sine and cosine plots
plt.figure(0)
plt.plot(t, y, label="sin(x)", color="blue", linestyle="--")  
plt.plot(t, z, label="cos(x)", color="green", linestyle="-")
plt.title("Sine and Cosine Waves")      # Title
plt.xlabel("Time (t)")                  # X-axis label
plt.ylabel("Amplitude")                 # Y-axis label
plt.legend(loc="upper right")           # Legend
plt.grid(True)                          # Grid lines
plt.show()

# Second figure: Subplots for sine and cosine
plt.figure(1)

# Subplot 1: Sine wave
plt.subplot(2, 1, 1)  # 2 rows, 1 column, subplot 1
plt.plot(t, y, label="sin(x)", color="blue", linestyle="--")
plt.title("Sine Wave")               # Title
plt.ylabel("Amplitude")              # Y-axis label
plt.grid(True)                       # Grid lines
plt.legend(loc="upper right")        # Legend

# Subplot 2: Cosine wave
plt.subplot(2, 1, 2)  # 2 rows, 1 column, subplot 2
plt.plot(t, z, label="cos(x)", color="green", linestyle="-")
plt.title("Cosine Wave")             # Title
plt.xlabel("Time (t)")               # X-axis label
plt.ylabel("Amplitude")              # Y-axis label
plt.grid(True)                       # Grid lines
plt.legend(loc="upper right")        # Legend

plt.tight_layout()                   # Adjust layout to prevent overlap
plt.show()
 
