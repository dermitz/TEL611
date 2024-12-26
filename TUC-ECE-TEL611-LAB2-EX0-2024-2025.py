"""
File Name: probability_and_entropy_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script performs probability analysis based on a scenario involving students' participation in sports and their performance in mathematics.
    The analysis includes marginal, joint, and conditional probabilities, as well as verification using the sum and product rules, and Bayes' theorem.
    Additionally, it calculates the entropy of the sports participation variable.

Key Components:
    1. Marginal Probabilities: Computes the probability of individual events (e.g., the probability of a student playing sports, excelling in math).
    2. Joint Probability: Calculates the probability of two events occurring together (e.g., the probability of a student both playing sports and excelling in math).
    3. Conditional Probabilities: Calculates the probability of one event given the occurrence of another (e.g., the probability of excelling in math given that a student plays sports).
    4. Sum Rule: Verifies the marginal probability of excelling in math by summing the joint probabilities.
    5. Product Rule: Verifies the joint probability using the product of marginal and conditional probabilities.
    6. Bayes' Theorem: Uses Bayes' theorem to calculate the conditional probability of a student playing sports given that they excel in math.
    7. Entropy Calculation: Computes the entropy of sports participation, which measures the uncertainty in this variable.

Usage:
    - This script provides an example of probability calculations in a real-world scenario and can be applied to understanding relationships between different events.
    - The entropy calculation measures the uncertainty of sports participation in the student population.
    - The script is useful for practicing conditional probabilities, sum/product rules, and Bayes' theorem.

Dependencies:
    - numpy
    - math

Instructions:
    - Run the script to see the computed probabilities, including marginal, joint, and conditional probabilities.
    - Verify these probabilities using the sum and product rules, as well as Bayes' theorem.
    - The script will also calculate the entropy of the sports participation variable.

Notes:
    - The sample data used in this script represents a scenario where 200 students are surveyed, with information on their participation in sports and their performance in math.
    - The entropy calculation assumes a discrete probability distribution for sports participation.

"""

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
