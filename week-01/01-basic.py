# Databricks notebook source
# MAGIC %md
# MAGIC ## Load  and Verify the Data

# COMMAND ----------

# Basic Python Imports

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC The Iris dataset is available from the UCI Machine Learning Repository at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data.
# MAGIC 
# MAGIC The features are given in the following order:
# MAGIC 
# MAGIC     sepal length
# MAGIC     sepal width
# MAGIC     petal length
# MAGIC     petal width
# MAGIC     flower class 

# COMMAND ----------

# MAGIC %md
# MAGIC Please use the following as the column names in your dataframe:

# COMMAND ----------

# Define list of column names

columns = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'flower_class'
]

# COMMAND ----------

# MAGIC %md
# MAGIC Read the Iris data into a Pandas DataFrame using the `pd.read_csv()` function. Load the data using the URL to the data set at the UCI Machine Learning Repository.

# COMMAND ----------

# Create Function to load dataset

def load_dataframe():
    """
    Write a function that returns the iris dataset loaded in a dataframe. 
    Make sure that the columns are properly named using the variable
    columns defined in the previous cell and that header rows
    are properly handled.
    """
    # Write your code here.
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris_df = pd.read_csv(URL, header=None)
    iris_df.columns = columns
    
    return iris_df

# COMMAND ----------

# MAGIC %md
# MAGIC Your `load_dataframe` function should pass all of these tests. If all tests pass, you will see the text `"All tests pass."` following the execution of the cell.

# COMMAND ----------

# Verify Dataset load

iris_df = load_dataframe()
assert type(iris_df) == pd.DataFrame
assert iris_df.shape == (150,5)
assert list(iris_df.columns) == columns
assert list(iris_df.iloc[4]) == [5.0,3.6,1.4,0.2, "Iris-setosa"]
print("All tests pass.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Display the shape of your DataFrame to verify that the DataFrame has the correct shape 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Display the head of your DataFrame to verify that the data was properly loaded.

# COMMAND ----------

# Display Dataset

from IPython.display import display

display(iris_df.shape)
display(iris_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Describe the DataFrame

# COMMAND ----------

# MAGIC %md
# MAGIC Write a function that returns the description of a DataFrame.

# COMMAND ----------

# Display the currently defined variables

%whos

# COMMAND ----------

# the description of a dataframe.

description = iris_df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Your `dataframe_description` function should pass these tests. If all tests pass you should see the text `"All tests pass."` and the description of the Iris Dataframe.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use `sns.boxplot()` to Display Box Plots of the Features

# COMMAND ----------

# Display the data types of the `iris_df` Dataframe

iris_df.dtypes

# COMMAND ----------

# Box plots of the features

_, ax = plt.subplots(1,4, figsize=(20,5))
sns.boxplot(iris_df.sepal_length, ax=ax[0])
sns.boxplot(iris_df.sepal_width, ax=ax[1])
sns.boxplot(iris_df.petal_length, ax=ax[2])
sns.boxplot(iris_df.petal_width, ax=ax[3])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use `sns.distplot` to display Distribution Plots of the Features

# COMMAND ----------

# Distribution plots of the features

_, ax = plt.subplots(1,4, figsize=(20,5))
sns.distplot(iris_df.sepal_length, ax=ax[0])
sns.distplot(iris_df.sepal_width, ax=ax[1])
sns.distplot(iris_df.petal_length, ax=ax[2])
sns.distplot(iris_df.petal_width, ax=ax[3])

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### True/False

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Sepal Width is qualitative.
# MAGIC 2. Sepal Length is quantitative.
# MAGIC 3. Petal Width is quantitative.
# MAGIC 4. Petal Length is qualitative.
# MAGIC 5. Flower Class is qualitative.
# MAGIC 6. If predicting Flower Class, this is a classification problem. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical Response

# COMMAND ----------

# MAGIC %md
# MAGIC 1. What is the range of petal width?
# MAGIC 2. What is the mean of sepal length?
# MAGIC 3. What is the median of petal length?
# MAGIC 4. What is the standard deviation of sepal width?
