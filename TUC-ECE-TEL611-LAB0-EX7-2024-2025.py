"""
File Name: car_evaluation_exercise.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script implements a comprehensive exercise using the Car Evaluation dataset.
    It demonstrates the complete data science workflow, from data loading and preprocessing to data visualization, model training, and evaluation.
    The goal is to predict the class of a car based on its attributes (buying, maintenance, doors, persons, luggage boot, and safety).

Key Components:
    1. **Data Loading**: The Car Evaluation dataset is loaded from a CSV file. The dataset lacks headers, so the column names are manually defined.
    2. **Data Preprocessing**: Categorical variables are encoded using `LabelEncoder` to convert them into numerical format for model training.
    3. **Data Visualization**: Two types of visualizations are generated: a bar chart showing the class distribution and a scatter plot between 'persons' and 'safety' features.
    4. **Model Training**: A Decision Tree Classifier is trained using the preprocessed data. The data is split into training and testing sets, and model evaluation is done using accuracy and classification report.
    5. **Model Evaluation**: The script evaluates the performance of the trained Decision Tree Classifier using standard classification metrics.

Usage:
    - This script is designed for educational purposes, providing a step-by-step guide to loading, preprocessing, visualizing, and modeling data.
    - It demonstrates how to apply machine learning techniques to real-world categorical data and evaluate model performance.

Dependencies:
    - pandas
    - matplotlib
    - scikit-learn

Instructions:
    - Replace the file path `path_to_your_car_data_file.csv` with the actual location of the Car Evaluation dataset.
    - Run the script to load, preprocess, visualize, and model the Car Evaluation data.

Notes:
    - The dataset contains categorical variables that need to be encoded before training the model.
    - The Decision Tree Classifier is used as a simple model for this classification task, though other models could be applied for comparison.
    - Data visualization is used to gain insights into the distribution of data and the relationships between features.

Questions for Students:
    1. Why is it important to encode categorical features before training a machine learning model?
    2. How do the visualizations help in understanding the dataset?
    3. What performance metrics should be used to evaluate classification models, and why?
    4. How would you improve the model’s performance if needed?
"""


# Car Evaluation Dataset Exercise
# Comprehensive Exercise for Educational Purpose

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the Dataset
def load_data(file_path):
    """
    Load the Car Evaluation dataset.

    Args:
        file_path (str): Path to the car dataset file.

    Returns:
        DataFrame: Loaded dataset.
    """
    try:
        # Define column names since the dataset lacks headers
        file_path = 'C:/datasets/car.data'
        column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        data = pd.read_csv(file_path, header=None, names=column_names)
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None

# Step 2: Data Inspection and Preprocessing
def preprocess_data(df):
    """
    Preprocess the Car Evaluation dataset.

    Args:
        df (DataFrame): The dataset.

    Returns:
        DataFrame: Preprocessed dataset with encoded values.
    """
    # Display dataset info
    print("\n== Dataset Info ==")
    print(df.info())

    print("\n== Unique Values in Each Column ==")
    for column in df.columns:
        print(f"{column}: {df[column].unique()}")

    # Encode categorical variables using LabelEncoder
    encoder = LabelEncoder()
    for column in df.columns:
        df[column] = encoder.fit_transform(df[column])

    print("\n== Dataset after Encoding ==")
    print(df.head())
    return df

# Step 3: Data Visualization
def visualize_data(df):
    """
    Visualize the dataset.

    Args:
        df (DataFrame): The dataset.

    Returns:
        None
    """
    # Plot class distribution
    plt.figure(figsize=(6, 4))
    df['class'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # Scatter plot: persons vs safety
    plt.figure(figsize=(6, 4))
    plt.scatter(df['persons'], df['safety'], c=df['class'], cmap='viridis', edgecolor='k')
    plt.title("Persons vs Safety")
    plt.xlabel("Persons")
    plt.ylabel("Safety")
    plt.colorbar(label='Class')
    plt.show()

# Step 4: Train-Test Split and Model Training
def train_model(df):
    """
    Train and evaluate a Decision Tree Classifier.

    Args:
        df (DataFrame): The dataset.

    Returns:
        None
    """
    # Features and target
    X = df.drop(columns=['class'])
    y = df['class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    print("\n== Classification Report ==")
    print(classification_report(y_test, y_pred))

    print("\n== Accuracy Score ==")
    print("Accuracy:", accuracy_score(y_test, y_pred))

# Main Workflow
if __name__ == "__main__":
    # Step 1: Load the dataset
    file_path = 'path_to_your_car_data_file.csv'  # Update with the correct path to your dataset
    car_data = load_data(file_path)

    if car_data is not None:
        # Step 2: Preprocess the data
        processed_data = preprocess_data(car_data)

        # Step 3: Visualize the data
        visualize_data(processed_data)

        # Step 4: Train and evaluate the model
        train_model(processed_data)
