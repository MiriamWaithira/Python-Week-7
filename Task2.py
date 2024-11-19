# Performing basic data analysis

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Computing basic statistics of numerical columns
print("Basic statistics of numerical columns:")
print(iris_df.describe())

# Group by species and compute the mean of numerical columns
grouped = iris_df.groupby('species', observed=False).mean()
print("\nMean of numerical features grouped by species:")
print(grouped)

# Insights and patterns
print("\nObservations:")
print("1. Setosa species has the smallest petal length and width on average.")
print("2. Virginica species has the largest sepal and petal dimensions on average.")
print("3. Versicolor species falls between Setosa and Virginica in terms of features sizes.")

# grouped = iris_df.groupby('species', observed=False).mean()
# the above code will make sure the code is compatible with future versions of pandas