"""
File Name: iris_gaussian_naive_bayes.py
Author: Dr. Dermitzakis Emm. Eleftherios
Date: 2024-12-26
Version: 1.0

Description:
    This script applies the Gaussian Naive Bayes classifier to the Iris dataset for multi-class classification. 
    The dataset contains three classes of iris flowers (Setosa, Versicolor, and Virginica), each with four features 
    (sepal length, sepal width, petal length, and petal width). The script performs the following steps:

    1. Loads the Iris dataset from scikit-learn.
    2. Splits the data into training and testing sets (70% training, 30% testing).
    3. Initializes a Gaussian Naive Bayes classifier.
    4. Trains the classifier on the training data.
    5. Makes predictions on the test set.
    6. Evaluates the model's performance by calculating the accuracy and generating a confusion matrix.
    7. Visualizes the confusion matrix using `ConfusionMatrixDisplay` from scikit-learn.

Key Components:
    - Gaussian Naive Bayes Classifier: A probabilistic classifier that assumes features are normally distributed.
    - Model Training: The classifier is trained on the training data using the `fit` method.
    - Model Evaluation: Accuracy is computed using `accuracy_score`, and performance is visualized through a confusion matrix.
    - Visualization: The confusion matrix is displayed with color coding using `ConfusionMatrixDisplay`.

Usage:
    - This script demonstrates the use of Gaussian Naive Bayes for classifying the Iris dataset, a popular dataset for machine learning tasks.
    - The confusion matrix helps assess the performance of the classifier in terms of true positives, false positives, true negatives, and false negatives.
    - Run the script to evaluate the classifier's accuracy and visualize the confusion matrix for multi-class classification.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - scikit-learn

Instructions:
    - Ensure that the required dependencies (NumPy, pandas, matplotlib, scikit-learn) are installed in your Python environment.
    - Run the script to load the Iris dataset, train the Gaussian Naive Bayes classifier, and display the results.
    - The script will output the accuracy of the classifier and display the confusion matrix plot.

Notes:
    - The Iris dataset is commonly used for classification tasks and is available directly from scikit-learn.
    - Gaussian Naive Bayes assumes that the features are conditionally independent given the class and follow a Gaussian distribution.
    - The confusion matrix plot provides insights into how well the classifier performs for each class in the multi-class problem.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Classes (Iris Setosa, Versicolor, Virginica)
# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Step 3: Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()
# Step 4: Train the classifier
gnb.fit(X_train, y_train)
# Step 5: Make predictions on the test set
y_pred = gnb.predict(X_test)
# Step 6: Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# Print accuracy
print(f'Accuracy of the Gaussian Naive Bayes classifier: {accuracy:.2f}')
# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Iris Dataset')
plt.show()

