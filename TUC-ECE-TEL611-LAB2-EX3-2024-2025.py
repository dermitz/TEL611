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

