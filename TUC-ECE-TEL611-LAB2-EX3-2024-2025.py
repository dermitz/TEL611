"""
File Name: bayesian_decision_with_average_risk.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements the Bayesian decision rule with average risk minimization in a binary classification context.
    It models a situation where two classes (Spam and Not Spam) are classified based on a feature, such as word frequency,
    and considers the costs of misclassification (λ21 for classifying Spam as Not Spam and λ12 for classifying Not Spam as Spam).

    The script calculates the posterior probabilities of each class given the observed feature and computes the risks 
    associated with each class under different classification decisions. The decision rule aims to minimize the average risk
    by comparing the risks for both classes. The decision boundary is then plotted to visualize how the decision regions 
    change based on the costs and prior probabilities.

    The impact of the misclassification costs and priors is illustrated through plots, and the average risk is computed 
    using integration.

Key Components:
    - Likelihood Calculation: The likelihood of observing the feature given each class (Spam and Not Spam) is computed 
      using normal distributions.
    - Bayesian Update: Posterior probabilities are computed using Bayes' theorem based on the likelihoods and priors.
    - Risk Calculation: The risks associated with each class are calculated based on the misclassification costs.
    - Decision Rule: The decision rule selects the class that minimizes the expected risk, which is compared between 
      the two classes.
    - Visualization: The decision boundary is visualized by plotting the risk functions and the classified regions.
    - Average Risk: The script computes the average risk by integrating the minimum risk across the feature space.

Usage:
    - This script demonstrates the process of Bayesian decision-making with risk minimization in a binary classification problem.
    - It visualizes how the decision boundary and classification regions are influenced by misclassification costs and priors.

Dependencies:
    - numpy
    - matplotlib
    - scipy

Instructions:
    - Ensure that the required dependencies (NumPy, matplotlib, scipy) are installed in your Python environment.
    - Run the script to observe the decision boundary, risks, and average risk under the given conditions.
    - The script will output the average risk value and plot the decision boundary with labeled classification regions.

Notes:
    - The decision-making process is based on minimizing the expected risk, which is influenced by the priors and misclassification costs.
    - This script is useful for understanding how Bayesian decision theory can be applied in a classification context with risks involved.
"""

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
