# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Line chart showing trends (e.g., cumulative petal length for simplicity)
iris_df['cumulative_petal_length'] = iris_df['petal length (cm)'].cumsum()
plt.figure(figsize=(8, 5))
plt.plot(iris_df.index, iris_df['cumulative_petal_length'], label='Cumulative Petal Length', color='blue')
plt.title('Cumulative Petal Length Trend')
plt.xlabel('Index')
plt.ylabel('Cumulative Petal Length (cm)')
plt.legend()
plt.grid()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# Histogram: Distribution of sepal length
plt.figure(figsize=(8, 5))
plt.hist(iris_df['sepal length (cm)'], bins=10, color='teal', alpha=0.7, edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Scatter plot: Sepal length vs. Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df, palette='cool')
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()