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
