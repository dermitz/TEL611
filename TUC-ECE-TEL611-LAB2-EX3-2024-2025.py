import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu1, sigma1 = 2, 1  # Class ω1 (Spam)
mu2, sigma2 = 4, 1  # Class ω2 (Not Spam)
P_w1, P_w2 = 0.4, 0.6  # Priors
lambda_21 = 5  # Cost of classifying Spam as Not Spam
lambda_12 = 1  # Cost of classifying Not Spam as Spam

# Step 1: Define the likelihoods
x = np.linspace(-2, 8, 1000)
p_x_w1 = norm.pdf(x, mu1, sigma1)  # p(x | ω1)
p_x_w2 = norm.pdf(x, mu2, sigma2)  # p(x | ω2)

# Step 2: Compute the posterior probabilities
p_x = p_x_w1 * P_w1 + p_x_w2 * P_w2  # Total evidence
P_w1_x = (p_x_w1 * P_w1) / p_x  # Posterior P(ω1 | x)
P_w2_x = (p_x_w2 * P_w2) / p_x  # Posterior P(ω2 | x)

# Step 3: Decision rule - Minimize Average Risk
# Compute the risk for each class
risk_w1 = lambda_21 * P_w1_x  # Risk for classifying as ω1
risk_w2 = lambda_12 * P_w2_x  # Risk for classifying as ω2

# Decision boundary: Compare risks
decision = np.where(risk_w1 < risk_w2, 1, 2)

# Step 4: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, risk_w1, label="Risk for Spam (ω1)", color='red')
plt.plot(x, risk_w2, label="Risk for Not Spam (ω2)", color='blue')
plt.fill_between(x, 0, 1, where=decision == 1, color='red', alpha=0.1, label="Classified as Spam")
plt.fill_between(x, 0, 1, where=decision == 2, color='blue', alpha=0.1, label="Classified as Not Spam")
plt.title("Bayesian Decision Rule with Average Risk")
plt.xlabel("Feature x (e.g., Word Frequency)")
plt.ylabel("Risk")
plt.legend()
plt.grid()
plt.show()

# Step 5: Compute Average Risk
average_risk = np.trapz(np.minimum(risk_w1, risk_w2), x)  # Integrate over x
print(f"Average Risk: {average_risk:.4f}")
