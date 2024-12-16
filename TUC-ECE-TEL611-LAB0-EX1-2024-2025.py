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
 