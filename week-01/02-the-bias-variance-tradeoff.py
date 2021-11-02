# Databricks notebook source
# MAGIC %md
# MAGIC ## The Bias-Variance Tradeoff

# COMMAND ----------

# Set up Numerical Python environment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline
plt.rc('figure', figsize=(20, 6))

# COMMAND ----------

# Load Iris dataset

from sklearn.datasets import load_iris

feature_names = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width'
]

IRIS = load_iris()
iris_df = pd.DataFrame(IRIS.data, columns=feature_names)
labels = IRIS.target_names

# COMMAND ----------

# MAGIC %md
# MAGIC In his 1996 paper, "The Lack of A Priori Distinctions between Learning Algorithms," David Wolpert asserts that "There is No Free Lunch in Machine Learning", essentially saying that the performance of on machine learning models are equivalent when averaged across all possible problems. In other words, there is no way to know which model is going to perform at best for a particular problem. In practice this means that we must have strong methods for assessing the performance of our models.

# COMMAND ----------

# MAGIC %md
# MAGIC Thinking about model performance is complex. We cannot simply choose the model that performs best with the data that we have. The reason for this is that the data we have represents a sample of the actual data that we could possibly collect now or in the future. The "best model" is not necessarily one that performs best on the data that we have. The best model is a model that performs extremely well on the data that we have but it's also capable of generalizing to new data. In statistical learning, we have a framework for thinking about this called The Bias-Variance Tradeoff.

# COMMAND ----------

# MAGIC %md
# MAGIC This framework is difficult to understand. Adding to the difficulty is the fact that both "bias" and "variance" are important concepts for working in applied statistics **and** the meaning of these terms is difficult to reconcile with their meaning when thinking about the Bias-Variance Tradeoff.

# COMMAND ----------

# MAGIC %md
# MAGIC Let us try to come to a high level understanding of these terms in this setting before looking at them in application. You might think of **bias** as the extent to which a particular model can learn to represent an underlying physical phenomenon. A low bias means that the model was able to learn the phenomenon well. For example, looking at the Iris data set, if we are building a regression model to predict the petal width then bias would be the degree to which our model was able to learn the relationship between petal length and petal width or sepal width and petal width. 

# COMMAND ----------

# MAGIC %md
# MAGIC **Variance** on the other hand is the extent to which a particular model would change if fit with different data. We previously looked at sampling our data set. We saw the measured means of value change with each sample. From this we can infer that a model predicting the meaning with only a few points of data has a high variance.

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Gory Details

# COMMAND ----------

# MAGIC %md
# MAGIC If we have split our data into a training set and a testing set, then we can think of choosing the best model in terms of optimizing the expected test error, $MSE_{test}$
# MAGIC 
# MAGIC Let's consider sources of possible test error:
# MAGIC 
# MAGIC $$MSE_{test} = \mathbb{E}\left[(y-\widehat{y})^2\right] = \text{Var}\left(\widehat{y}\right) + \left(\text{Bias}\left(\widehat{y}\right)\right)^2 + \text{Var}\left(\epsilon\right)$$
# MAGIC 
# MAGIC This is intended to be a conceptual and not an actual calculation to be performed. Let's think about what each of these terms might represent. The variance is error introduced to the model by the specific choice of training data. Of course this isn't something that we choose, at least not with out using randomness, but the training data that is used will impact the model. By nature, variance is a squared value and that's always positive. Bias is introduced by choosing a specific model. Note that it is squared here and thus also always positive. The last term is the variance caused by noise in the system. We have no way of controlling this, nor of actually knowing what is truly noise and what is model variance or bias. Again this term is always positive.
# MAGIC 
# MAGIC The important thing is that all three of these terms are always positive. The impact of this is that one kind of error cannot be offset by another kind of error. A high variance cannot be offset by a low bias. **In order to choose the best model, we are going to need to simultaneously minimize both bias and variance.** The problem is changing an aspect of our model to decrease one Will typically increase the other -- The Bias-Variance Tradeoff.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Assessment

# COMMAND ----------

# MAGIC %md
# MAGIC Consider these eight models for predicting petal width $x_{pw}$ on the Iris data set, using the other three features, petal length $x_{pl}$, sepal length $x_{sl}$, and sepal width $x_{sw}$

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{align*}
# MAGIC \widehat{f}_0 = \widehat{x}_{pw} &= \beta_0 \\
# MAGIC \widehat{f}_1 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{pl}\\
# MAGIC \widehat{f}_2 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{sl}\\
# MAGIC \widehat{f}_3 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{sw}\\
# MAGIC \widehat{f}_4 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{pl} + \beta_1x_{sl}\\
# MAGIC \widehat{f}_5 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{pl} + \beta_1x_{sw}\\
# MAGIC \widehat{f}_6 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{sl} + \beta_1x_{sw}\\
# MAGIC \widehat{f}_7 = \widehat{x}_{pw} &= \beta_0 + \beta_1x_{pl} + \beta_1x_{sl}+ \beta_1x_{sw}\\
# MAGIC \end{align*}

# COMMAND ----------

# MAGIC %md
# MAGIC We will split our dataset into ten randomly chosen subsets of 50 points each. We will assess the performance of each of these models. We will use the mean of the performance to represent bias and the standard deviation of the performance to represent variance.

# COMMAND ----------

# Create 10 sample datasets from full data set

samples = []

for _ in range(10):
    samples.append(iris_df.sample(50))

# COMMAND ----------

# Create 10 sample datasets from full data set with list comprehension

samples = [iris_df.sample(50) for _ in range(10)]

# COMMAND ----------

# Show length of each sample in the list `samples`

[len(sample_set) for sample_set in samples]

# COMMAND ----------

# Import the design matrices library, `dmatrices` from `patsy`

from patsy import dmatrices

# COMMAND ----------

# Import the Linear Regression Model from Sklearn

from sklearn.linear_model import LinearRegression

# COMMAND ----------

# Simple example of computing a list of squared differences

a = np.array((1,2,3))
b = np.array((4,5,6))

(b - a)**2

# COMMAND ----------

# Define a Mean Squared Error function

def MSE(actual, predicted):
    return sum((actual - predicted)**2)/len(actual)

# COMMAND ----------

# Define a function for running our tests

def test_func(model_description):
    
    test = dict()
    
    test['samples'] = [
        dmatrices(model_description, sample) 
        for sample in samples
    ]
    
    test['models']  = [
        LinearRegression(fit_intercept=False) 
        for _ in range(10)
    ]
    
    test['scores']  = []
    
    models_and_samples = zip(test['models'],
                             test['samples'])

    for model, sample in models_and_samples:
        target = sample[0]
        features = sample[1]
        model.fit(features, target)
        
        mse = MSE(model.predict(features), target)

        test['scores'].append(mse)
    
    test['scores'] = np.array(test['scores'])
    
    results = { 'description' : model_description }
    
    results['bias'] = test['scores'].mean()
    results['variance'] = test['scores'].std()

    return test, results

# COMMAND ----------

# Get the results of running the test on the minimal model

model_1 = "petal_width ~ 1"
test_1, results_1 = test_func(model_1)

# COMMAND ----------

# Display the results of the MSE scores on the minimal model

test_1['scores']

# COMMAND ----------

# Show how we are collecting results

test_1['scores'].mean(), test_1['scores'].std()

# COMMAND ----------

# Display the results returned by the test

results_1

# COMMAND ----------

# Define all models

model_2 = "petal_width ~ 1 + petal_length"
model_3 = "petal_width ~ 1 + sepal_length"
model_4 = "petal_width ~ 1 + sepal_width"
model_5 = "petal_width ~ 1 + petal_length + sepal_width"
model_6 = "petal_width ~ 1 + petal_length + sepal_length"
model_7 = "petal_width ~ 1 + sepal_length + sepal_width"
model_8 = "petal_width ~ 1 + petal_length + sepal_length + sepal_width"

# COMMAND ----------

# Execute the test on the other seven model and collect the results

test_2, results_2 = test_func(model_2)
test_3, results_3 = test_func(model_3)
test_4, results_4 = test_func(model_4)
test_5, results_5 = test_func(model_5)
test_6, results_6 = test_func(model_6)
test_7, results_7 = test_func(model_7)
test_8, results_8 = test_func(model_8)
results = [
    results_1,
    results_2,
    results_3,
    results_4,
    results_5,
    results_6,
    results_7,
    results_8
]

# COMMAND ----------

# Turn the results into a `DataFrame` for ease of display

results = pd.DataFrame(results)

# COMMAND ----------

# Add colors to results 

results['color'] = ['red','orange', 'yellow', 'green', 
                    'blue', 'cyan', 'purple' ,'black']

# COMMAND ----------

# Display results

results

# COMMAND ----------

# Plot the bias versus the variance 

plt.figure(figsize=(10,5))

for i in results.index:
    plt.scatter(
        results.loc[i].bias,
        results.loc[i].variance, 
        c=results.loc[i].color, 
        label=results.loc[i].description
    )

plt.xlabel('Bias')
plt.ylabel('Variance')

plt.legend()

plt.show()
