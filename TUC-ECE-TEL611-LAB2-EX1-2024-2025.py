import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
import seaborn as sns
# Parameters for distributions
mean_X, std_X = 70, 10
mean_Y, std_Y = 65, 15
cov_XY = 50
# Define bivariate normal distribution for joint calculations
mean = [mean_X, mean_Y]
cov_matrix = [[std_X**2, cov_XY], [cov_XY, std_Y**2]]
dist = multivariate_normal(mean=mean, cov=cov_matrix)
# Define range for plotting PDFs
x = np.linspace(30, 110, 1000)
# Step 1: Plot PDFs of X and Y
plt.figure(figsize=(10, 5))
plt.plot(x, norm.pdf(x, mean_X, std_X), label="PDF of X (Group A)")
plt.plot(x, norm.pdf(x, mean_Y, std_Y), label="PDF of Y (Group B)")
plt.xlabel("Test Scores")
plt.ylabel("Probability Density")
plt.title("Probability Density Functions for X and Y")
plt.legend()
plt.grid(True)
plt.show()
# Step 2: Calculate expectations and variances
expected_X, expected_Y = mean_X, mean_Y
variance_X, variance_Y = std_X**2, std_Y**2
cov_XY_val = cov_XY
corr_XY = cov_XY_val / (std_X * std_Y)
print(f"Expectation of X: {expected_X}")
print(f"Variance of X: {variance_X}")
print(f"Expectation of Y: {expected_Y}")
print(f"Variance of Y: {variance_Y}")
print(f"Covariance of X and Y: {cov_XY_val}")
print(f"Correlation of X and Y: {corr_XY}")
# Step 3: Joint Probability P(X > T and Y > T)
T = 75
joint_prob = 1 - dist.cdf([T, T])
print(f"Joint probability P(X > {T} and Y > {T}): {joint_prob}")
# Step 4: Entropy and Mutual Information
entropy_X = 0.5 * np.log(2 * np.pi * np.e * variance_X)
entropy_Y = 0.5 * np.log(2 * np.pi * np.e * variance_Y)
joint_entropy = 0.5 * np.log((2 * np.pi * np.e)**2 * np.linalg.det(cov_matrix))
mutual_information = entropy_X + entropy_Y - joint_entropy
print(f"Entropy of X: {entropy_X}")
print(f"Entropy of Y: {entropy_Y}")
print(f"Joint Entropy H(X, Y): {joint_entropy}")
print(f"Mutual Information I(X; Y): {mutual_information}")
