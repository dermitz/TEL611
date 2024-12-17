import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu1, sigma1 = 2, 1  # Class ω1 (Spam)
mu2, sigma2 = 4, 1  # Class ω2 (Not Spam)
P_w1, P_w2 = 0.4, 0.6  # Priors

# Function to calculate decision boundary and plot results
def bayes_decision_with_risk(lambda_21, lambda_12, noise_std=0):
    x = np.linspace(-2, 8, 1000)
    
    # Likelihoods
    p_x_w1 = norm.pdf(x, mu1, sigma1)  # p(x | ω1)
    p_x_w2 = norm.pdf(x, mu2, sigma2)  # p(x | ω2)
    
    # Add Gaussian noise if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=len(x))
        p_x_w1 = norm.pdf(x + noise, mu1, sigma1)
        p_x_w2 = norm.pdf(x + noise, mu2, sigma2)
    
    # Posterior probabilities
    p_x = p_x_w1 * P_w1 + p_x_w2 * P_w2
    P_w1_x = (p_x_w1 * P_w1) / p_x  # P(ω1 | x)
    P_w2_x = (p_x_w2 * P_w2) / p_x  # P(ω2 | x)
    
    # Risk calculation
    risk_w1 = lambda_21 * P_w1_x  # Risk for classifying as ω1
    risk_w2 = lambda_12 * P_w2_x  # Risk for classifying as ω2
    
    # Decision boundary: Compare risks
    decision = np.where(risk_w1 < risk_w2, 1, 2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, risk_w1, label="Risk for Spam (ω1)", color='red')
    plt.plot(x, risk_w2, label="Risk for Not Spam (ω2)", color='blue')
    plt.fill_between(x, 0, 1, where=decision == 1, color='red', alpha=0.1, label="Classified as Spam")
    plt.fill_between(x, 0, 1, where=decision == 2, color='blue', alpha=0.1, label="Classified as Not Spam")
    plt.title(f"Bayesian Decision with λ21={lambda_21}, λ12={lambda_12}, Noise Std={noise_std}")
    plt.xlabel("Feature x (e.g., Word Frequency)")
    plt.ylabel("Risk")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Average Risk
    average_risk = np.trapz(np.minimum(risk_w1, risk_w2), x)
    print(f"Average Risk: {average_risk:.4f}")

# Step 1: Test the impact of costs
print("=== Impact of Costs on Decision Boundary ===")
costs = [(5, 1), (10, 1), (15, 1), (5, 3), (5, 2)]
for lambda_21, lambda_12 in costs:
    bayes_decision_with_risk(lambda_21, lambda_12)

# Step 2: Add Gaussian noise to observed data
print("\n=== Impact of Gaussian Noise on Decision Boundary ===")
noise_levels = [0.1, 0.5, 1.0]
for noise_std in noise_levels:
    bayes_decision_with_risk(lambda_21=5, lambda_12=1, noise_std=noise_std)
