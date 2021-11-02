# Databricks notebook source
# MAGIC %md
# MAGIC ## PCA on Categorical Features

# COMMAND ----------

# Import Numerical Python Libraries

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# COMMAND ----------

# load the data

IRIS = load_iris()

# COMMAND ----------

# Redefine the column names

names = [(name
  .replace(" ", "")
  .replace("(cm)", "")) for name in IRIS.feature_names]

# COMMAND ----------

# Create Dummy Variables from the Label

df = pd.DataFrame(IRIS.data)
df.columns = names
df['label'] = IRIS.target
df.label = df.label.astype('category')
df = pd.get_dummies(df)
df = df.sample(df.shape[0])

# COMMAND ----------

# Display distribution plots

fig = plt.figure(figsize=(20,6))
for i, col in enumerate(df.columns):
    fig.add_subplot(1,7,1+i)
    sns.distplot(df[col])
    plt.xlim(-3,3)

# COMMAND ----------

# Define two scaling functions

def gelman_scale(series):
    return (series - series.mean())/(2*series.std())

def standard_scale(series):
    return (series - series.mean())/(series.std())

# COMMAND ----------

# Scale Numerical Features

df_gelman = df.copy()
df_standard = df.copy()

df_gelman.sepallength = gelman_scale(df.sepallength)
df_gelman.sepalwidth = gelman_scale(df.sepalwidth)
df_gelman.petallength = gelman_scale(df.petallength)
df_gelman.petalwidth = gelman_scale(df.petalwidth)
df_standard.sepallength = standard_scale(df.sepallength)
df_standard.sepalwidth = standard_scale(df.sepalwidth)
df_standard.petallength = standard_scale(df.petallength)
df_standard.petalwidth = standard_scale(df.petalwidth)

# COMMAND ----------

# Display Distribution Plots

fig = plt.figure(figsize=(20,6))
for i, col in enumerate(df.columns):
    fig.add_subplot(2,7,1+i)
    sns.distplot(df_gelman[col])
    plt.xlim(-3,3)
    fig.add_subplot(2,7,8+i)
    sns.distplot(df_standard[col])
    plt.xlim(-3,3)

# COMMAND ----------

# Variance of label 0

df_standard.label_0.var()

# COMMAND ----------

# Variance of a boolean vector with 0.3-0.7 distribution

np.array(3*[True]+7*[False]).var()

# COMMAND ----------

# Variance of a boolean vector with 0.5-0.5 distribution

np.array(5*[True]+5*[False]).var()

# COMMAND ----------

# Covariance Matrix of Standard Scaled Data

df_standard.cov()

# COMMAND ----------

# Covariance Matrix of Gelman Scaled Data

df_gelman.cov()

# COMMAND ----------

# Eigenvectors of Gelman Scaled Data

np.linalg.eig(4*df_gel_cov)[1].T

# COMMAND ----------

# Principal Component Analysis

from sklearn.decomposition import PCA

pca_gelman = PCA()
pca_standard = PCA()
pca_numerical = PCA()
pca_gelman.fit(df_gelman)
pca_numerical.fit(df_standard)
pca_standard.fit(df_standard)

# COMMAND ----------

# Principal Components of Gelman Scaled Data

pca_gelman.components_

# COMMAND ----------

# PCA results

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def pca_results(dataframe, pca):


    dimensions = dimensions = ['PC {}'.format(i) for i in range(1,len(pca.components_)+1)]

    components = pd.DataFrame(np.round(pca.components_, 4), columns = dataframe.columns)
    components.index = dimensions

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    fig, ax = plt.subplots(figsize = (14,8))

    components.plot(ax = ax, kind = 'bar', colormap=cm.viridis);
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)


    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

pca_results(df_gelman, pca_gelman)
pca_results(df_standard, pca_standard);
