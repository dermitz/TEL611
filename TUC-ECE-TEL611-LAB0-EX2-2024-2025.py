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
