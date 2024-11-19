# Loading the Iris dataset, converting it into a pandas Dataframe and inspecting the data

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Loading the Iris dataset
iris = load_iris()

# Converting to a pandas DataFrame
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows
print("First 5 rows of the dataset:")
print(iris_df.head())

# Exploring the structure of the dataset
print("\nData types and missing values:")
print(iris_df.info())

# Checking for missing values
print("\nNumber of missing values in each column:")
print(iris_df.isnull().sum())

# Clean the dataset if necessary
# Since the Iris dataset has no missing values, no cleaning is required.