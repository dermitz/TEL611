"""
File Name: titanic_data_analysis.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script performs basic data analysis and visualization on the Titanic dataset. It loads the dataset, handles missing values, performs group-based analysis, and visualizes survival rates by gender. The goal is to provide an example of how to work with real-world datasets using Pandas and Matplotlib.

Key Components:
    1. **Data Loading**: The Titanic dataset is loaded from a remote URL using Pandas.
    2. **Handling Missing Data**: Missing values in the 'Age' column are filled with the median of the column.
    3. **Group-based Analysis**: The survival rate is analyzed based on the passenger class and gender.
    4. **Data Visualization**: The survival rates by gender are visualized using a bar plot with Matplotlib.
    
Usage:
    - Run the script in a Python environment to perform basic data analysis on the Titanic dataset.
    - The script will load the dataset, handle missing values, compute survival rates, and display a bar chart of survival rates by gender.

Dependencies:
    - pandas
    - matplotlib

Instructions:
    - Run the script in a Python environment.
    - The Titanic dataset will be fetched from the provided URL.
    - The script will output the first few rows of the dataset, missing value statistics, survival rates by class, and a bar plot of survival rates by gender.

Notes:
    - This script is a demonstration of basic data preprocessing and visualization techniques using real-world data.
    - The dataset contains information about passengers aboard the Titanic, including whether they survived, their age, gender, passenger class, and more.


"""

import pandas as pd
import matplotlib.pyplot as plt
# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)
# Display the first few rows
print("== First 5 Rows ==")
print(titanic.head())
# Check for missing values
print("\n== Missing Values ==")
print(titanic.isnull().sum())
# Fill missing age with the median
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
# Analyze survival rates by class
survival_rates = titanic.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rates by Class:")
print(survival_rates)
# Visualize survival rates by gender using Matplotlib
gender_survival = titanic.groupby('Sex')['Survived'].mean()
genders = gender_survival.index
survival_values = gender_survival.values
plt.figure(figsize=(8, 6))
plt.bar(genders, survival_values, color=['blue', 'pink'], edgecolor='black')
plt.title('Survival Rates by Gender', fontsize=14)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 1)  # Survival rates range between 0 and 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
