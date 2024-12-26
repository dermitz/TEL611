"""
File Name: sklearn_datasets.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script demonstrates the loading and displaying of various datasets from the `sklearn.datasets` module. 
    It covers how to load datasets and convert them into a Pandas DataFrame for better readability and inspection. 
    Specifically, it loads the 'Wine' dataset, which is used to classify wines based on various chemical properties.

Key Components:
    1. **Dataset Loading**: The script loads several datasets, including 'Wine' using `load_wine()`, which provides data on different wines.
    2. **Pandas DataFrame**: After loading the dataset, it is converted into a Pandas DataFrame for better handling and readability.
    3. **Data Inspection**: The script displays the first 5 rows of the 'Wine' dataset for inspection.

Usage:
    - Run the script to load datasets from `sklearn` and display the first few rows of the 'Wine' dataset.
    - This is useful for familiarizing oneself with the format and features of different machine learning datasets.

Dependencies:
    - pandas
    - scikit-learn (sklearn)

Instructions:
    - Execute the script in a Python environment where scikit-learn and pandas are installed.
    - The script will print the first 5 rows of the 'Wine' dataset, which includes chemical properties like alcohol content, malic acid, etc.

Notes:
    - The dataset provides measurements of 13 different features for wines that are classified into three categories. Each feature is related to the chemical composition of the wines, and the target variable is the class label indicating the type of wine.
    - The other datasets (iris, digits, breast cancer, etc.) are commented out but can be activated by uncommenting the respective lines.


"""

from sklearn.datasets import load_iris,load_digits,load_wine,load_breast_cancer
from sklearn.datasets import fetch_california_housing,fetch_lfw_people,fetch_20newsgroups
import pandas as pd
#iris = load_iris()
#digits = load_digits()
wine = load_wine()
#cancer = load_breast_cancer()
#housing = fetch_california_housing()
#faces = fetch_lfw_people(min_faces_per_person=20)
#newsgroups = fetch_20newsgroups(subset='train')
# Convert to a pandas DataFrame for better readability
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
print(wine_df.head(5))
