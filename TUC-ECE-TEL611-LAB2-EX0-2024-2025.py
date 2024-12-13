import numpy as np

# Sample data (frequency counts from the problem scenario)
total_students = 200
students_play_sports_excels_math = 50
students_play_sports_no_math = 40
students_no_sports_excels_math = 30
students_no_sports_no_math = 80

# Marginal Probabilities
P_play_sports = (students_play_sports_excels_math + students_play_sports_no_math) / total_students
P_excels_math = (students_play_sports_excels_math + students_no_sports_excels_math) / total_students
P_no_sports = (students_no_sports_excels_math + students_no_sports_no_math) / total_students
P_no_math = (students_play_sports_no_math + students_no_sports_no_math) / total_students

print("Marginal Probabilities:")
print(f"P(Plays Sports): {P_play_sports:.2f}")
print(f"P(Excels in Math): {P_excels_math:.2f}")

# Joint Probability
P_play_sports_and_excels_math = students_play_sports_excels_math / total_students
print("\nJoint Probability:")
print(f"P(Plays Sports and Excels in Math): {P_play_sports_and_excels_math:.2f}")

# Conditional Probabilities
P_excels_math_given_sports = P_play_sports_and_excels_math / P_play_sports
P_sports_given_excels_math = P_play_sports_and_excels_math / P_excels_math
print("\nConditional Probabilities:")
print(f"P(Excels in Math | Plays Sports): {P_excels_math_given_sports:.2f}")
print(f"P(Plays Sports | Excels in Math): {P_sports_given_excels_math:.2f}")

# Sum Rule: Verify P(Excels in Math) using sum rule
P_excels_math_sum_rule = P_play_sports_and_excels_math + (students_no_sports_excels_math / total_students)
print("\nVerification using Sum Rule:")
print(f"P(Excels in Math) (Sum Rule): {P_excels_math_sum_rule:.2f}")

# Product Rule: Verify joint probability using product rule
P_play_sports_and_excels_math_product_rule = P_play_sports * P_excels_math_given_sports
print("\nVerification using Product Rule:")
print(f"P(Plays Sports and Excels in Math) (Product Rule): {P_play_sports_and_excels_math_product_rule:.2f}")

# Bayes' Theorem: Verify P(Plays Sports | Excels in Math)
P_sports_given_excels_math_bayes = (P_excels_math_given_sports * P_play_sports) / P_excels_math
print("\nVerification using Bayes' Theorem:")
print(f"P(Plays Sports | Excels in Math) (Bayes' Theorem): {P_sports_given_excels_math_bayes:.2f}")

# Entropy Calculation
import math

# Calculate entropy for sports participation
P_sports_values = [P_play_sports, P_no_sports]
entropy_sports = -sum([p * math.log2(p) for p in P_sports_values if p > 0])
print("\nEntropy Calculation:")
print(f"Entropy of Sports Participation: {entropy_sports:.2f} bits")
