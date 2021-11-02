# Databricks notebook source
# MAGIC %md
# MAGIC ## Probabilistic Model Selection / In-Sample Model Selection

# COMMAND ----------

# MAGIC %md
# MAGIC From a probabilistic perspective we seek the model estimate $\widehat{f}$ that is most probable given the data. Let $H_f$ be the hypothesis that a specific model is the correct model and $p(D)$ represent the probability of the data, then we seek $\widehat{f}$ that maximizes, that is we seek 
# MAGIC 
# MAGIC $$\widehat{f} = \underset{f}{\mathrm{argmax}}\left(p(H_f|D)\right)$$
# MAGIC 
# MAGIC We can use Bayes' Rule to invert this probability so that
# MAGIC 
# MAGIC $$\widehat{f} = \underset{f}{\mathrm{argmax}}\left(\frac{p(D|H_f)p(H_f)}{p(D)}\right)$$
# MAGIC 
# MAGIC If we consider that each model is equally likely and that the probability of the data $p(D)$ is a constant applied to every calculation, then without loss of generality, we can say 
# MAGIC 
# MAGIC $$\widehat{f} = \underset{f}{\mathrm{argmax}}\left(p(D|H_f)\right)$$

# COMMAND ----------

# MAGIC %md
# MAGIC In the Regression setting, it can be shown that maximizing this equation is equivalent to minimizing the residual sum of squares, 
# MAGIC 
# MAGIC $$\text{RSS} = \sum (y_i - \widehat{y})^2 = (\mathbf{y} -\mathbf{\widehat{y}})^T(\mathbf{y} -\mathbf{\widehat{y}})$$

# COMMAND ----------

# MAGIC %md
# MAGIC #### Generating Model Hypotheses

# COMMAND ----------

# MAGIC %md
# MAGIC For this particular task we will consider the hypothesis space to be all of the linear models possible given our data sets. There will be eight possible models:
# MAGIC 
# MAGIC \begin{align*}
# MAGIC \widehat{f}_1 &= \beta_0 \\
# MAGIC \widehat{f}_2 &= \beta_0 + \beta_1x_1\\
# MAGIC \widehat{f}_3 &= \beta_0 + \beta_1x_2\\
# MAGIC \widehat{f}_4 &= \beta_0 + \beta_1x_3\\
# MAGIC \widehat{f}_5 &= \beta_0 + \beta_1x_1 + \beta_2x_2\\
# MAGIC \widehat{f}_6 &= \beta_0 + \beta_1x_1 + \beta_2x_3\\
# MAGIC \widehat{f}_7 &= \beta_0 + \beta_1x_2 + \beta_2x_3\\
# MAGIC \widehat{f}_8 &= \beta_0 + \beta_1x_1 + \beta_2x_2+ \beta_3x_3\\
# MAGIC \end{align*}

# COMMAND ----------

# MAGIC %md
# MAGIC Essentially we are forming what is known as a Power Set of possible feature combinations. The number of elements in a Power Set is given by $2^p$ where $p$ is the number of elements in the set. 
# MAGIC 
# MAGIC For $p=2$, this is four models.
# MAGIC For $p=3$, this is eight models.
# MAGIC 
# MAGIC For $p=100$, this is a hypothesis space With a dimension of $1.27\times10^{30}$. If we trained one model per second, it would take us $4.02\times10^{22}$ years to search the entire hypothesis space. For perspective, physicists estimate that the universe is approximately $13.8\times10^{9}$ years old.

# COMMAND ----------

# MAGIC %md
# MAGIC In other words, we will rarely be able to exhaustively search a hypothesis space.

# COMMAND ----------

# Load Iris Dataset

iris.data = read.csv("iris.csv", row.names='X')

# COMMAND ----------

# Display the first $n$ rows of the `DataFrame`

head(iris.data)

# COMMAND ----------

# Install the Package `GGally`

install.packages("GGally")

# COMMAND ----------

# Load the library `GGally`

library(GGally)

# COMMAND ----------

# Display `ggpairs` for the Iris Dataset

ggpairs(iris.data)

# COMMAND ----------

# Perform a Logistic Regression on the Species Label

iris.glm = glm("label ~ 1 + 
                        sepal_length + 
                        sepal_width + 
                        petal_length + 
                        petal_width", data = iris.data)

# COMMAND ----------

# Display the `summary` of the Logistic Regression

summary(iris.glm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The Log-Likelihood

# COMMAND ----------

# MAGIC %md
# MAGIC Without going too far into the math, we can think of the log-likelihood as a **likelihood function** telling us how likely a model is given the data. 

# COMMAND ----------

# MAGIC %md
# MAGIC This value is not human interpretable but is useful as a comparison.

# COMMAND ----------

# Calculate the Log-Likelihood of the Logistic Regression

logLik(iris.glm)

# COMMAND ----------

# MAGIC %md
# MAGIC > "All models are wrong, but some are useful." - George Box

# COMMAND ----------

# MAGIC %md
# MAGIC ##### William of Occam

# COMMAND ----------

# MAGIC %md
# MAGIC We might be concerned with one additional property - the **complexity** of the model. **Occam's razor** is the problem-solving principle that, when presented with competing hypothetical answers to a problem, one should select the one that makes the fewest assumptions.

# COMMAND ----------

# MAGIC %md
# MAGIC <include type="image" url="William-of-Ockham---Logica-1341.jpg">
# MAGIC 
# MAGIC William of Occam    
# MAGIC     
# MAGIC ![](../img/William-of-Ockham---Logica-1341.jpg)
# MAGIC     
# MAGIC </include>    

# COMMAND ----------

# MAGIC %md
# MAGIC We can represent this idea of complexity in terms of both the number of features we use and the amount of data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Bayesian Information Criterion

# COMMAND ----------

# MAGIC %md
# MAGIC The BIC is formally defined as
# MAGIC 
# MAGIC $$ \mathrm{BIC} = {\ln(n)k - 2\ln({\widehat L})}. $$
# MAGIC 
# MAGIC where
# MAGIC 
# MAGIC - $\widehat L$ = the maximized value of the likelihood function of the model $M$
# MAGIC - $x$ = the observed data
# MAGIC - $n$ = the number of data points in $x$, the number of observations, or equivalently, the sample size;
# MAGIC - $k$ = the number of parameters estimated by the model. For example, in multiple linear regression, the estimated parameters are the intercept, the $q$ slope parameters, and the constant variance of the errors; thus, $k = q + 2$.
# MAGIC 
# MAGIC 
# MAGIC It might help us to think of it as 
# MAGIC 
# MAGIC $$ \mathrm{BIC} = \text{complexity}-\text{likelihood}$$

# COMMAND ----------

# MAGIC %md
# MAGIC #### Akaike Information Criterion

# COMMAND ----------

# MAGIC %md
# MAGIC The AIC is formally defined as
# MAGIC 
# MAGIC $$ \mathrm{AIC} = {2k - 2\ln({\widehat L})}. $$
# MAGIC 
# MAGIC where
# MAGIC 
# MAGIC - $\widehat L$ = the maximized value of the likelihood function of the model $M$
# MAGIC - $x$ = the observed data
# MAGIC - $n$ = the number of data points in $x$, the number of observations, or equivalently, the sample size;
# MAGIC - $k$ = the number of parameters estimated by the model. For example, in multiple linear regression, the estimated parameters are the intercept, the $q$ slope parameters, and the constant variance of the errors; thus, $k = q + 2$.
# MAGIC 
# MAGIC 
# MAGIC It might help us to think of it as 
# MAGIC 
# MAGIC $$ \mathrm{AIC} = \text{complexity}-\text{likelihood}$$
# MAGIC 
# MAGIC We can think of - likelihood as bias and complexity as variance.

# COMMAND ----------

# Calculate the BIC using R

BIC(iris.glm)

# COMMAND ----------

# Calculate the BIC manually

n = length(iris.glm$fitted.values)
p = length(coefficients(iris.glm))

likelihood = 2 * logLik(iris.glm)
complexity = log(n)*(p+1)

bic = complexity - likelihood
bic

# COMMAND ----------

# Define a function to calcuate the BIC

BIC_of_model = function (model) {
    n = length(model$fitted.values)
    p = length(coefficients(model))

    likelihood = 2 * logLik(model)
    complexity = log(n)*(p+1)

    bic = complexity - likelihood
    return(bic)
}

# COMMAND ----------

# Calculate the BIC with function

BIC_of_model(iris.glm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Selection

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we choose the optimal model by removing features one by one.

# COMMAND ----------

# Define full model

model_1 = "label ~ 1"
model_1 = paste(model_1, "+ sepal_length")
model_1 = paste(model_1, "+ sepal_width")
model_1 = paste(model_1, "+ petal_length")
model_1 = paste(model_1, "+ petal_width")

# COMMAND ----------

# Define four models each with one feature removed

model_2a = gsub('\\+ petal_width','', model_1)
model_2b = gsub('\\+ petal_length', '', model_1)
model_2c = gsub('\\+ sepal_width', '', model_1)
model_2d = gsub('\\+ sepal_length', '', model_1)

# COMMAND ----------

# Display the string defining `model_2a`

cat(model_2a)

# COMMAND ----------

# Perform Logistic Regression on these models

iris.glm.1 = glm(model_1, data=iris.data)
iris.glm.2a = glm(model_2a, data=iris.data)
iris.glm.2b = glm(model_2b, data=iris.data)
iris.glm.2c = glm(model_2c, data=iris.data)
iris.glm.2d = glm(model_2d, data=iris.data)

# COMMAND ----------

# Display the results of the BIC on these models

print(c('model_1', BIC_of_model(iris.glm.1)))
print(c('model_2a', BIC_of_model(iris.glm.2a )))
print(c('model_2b', BIC_of_model(iris.glm.2b )))
print(c('model_2c', BIC_of_model(iris.glm.2c )))
print(c('model_2d', BIC_of_model(iris.glm.2d )))

# COMMAND ----------

# Display the results of the BIC on these models using R

print(c('model_1', BIC(iris.glm.1)))
print(c('model_2a', BIC(iris.glm.2a )))
print(c('model_2b', BIC(iris.glm.2b )))
print(c('model_2c', BIC(iris.glm.2c )))
print(c('model_2d', BIC(iris.glm.2d )))

# COMMAND ----------

# Define three models with two features removed

model_3a = gsub('\\+ petal_width','', model_2c)
model_3b = gsub('\\+ petal_length','', model_2c)
model_3c = gsub('\\+ sepal_length','', model_2c)

# COMMAND ----------

# Perform Logistic Regressions 

iris.glm.3a = glm(model_3a, data=iris.data)
iris.glm.3b = glm(model_3b, data=iris.data)
iris.glm.3c = glm(model_3c, data=iris.data)

# COMMAND ----------

# Display the results of the BIC using R

print(c('model_1', BIC(iris.glm.1)))
print(c('model_2c', BIC(iris.glm.2c )))
print(c('model_3a', BIC(iris.glm.3a )))
print(c('model_3b', BIC(iris.glm.3b )))
print(c('model_3c', BIC(iris.glm.3c )))

# COMMAND ----------

# Display best three feature model

cat(model_2c)

# COMMAND ----------

# Display best two feature model

cat(model_3c)

# COMMAND ----------

# Display Log-Likelihood's of Models

print(c('model_1', logLik(iris.glm.1)))
print(c('model_2c', logLik(iris.glm.2c )))
print(c('model_3a', logLik(iris.glm.3a )))
print(c('model_3b', logLik(iris.glm.3b )))
print(c('model_3c', logLik(iris.glm.3c )))

# COMMAND ----------

# Display the results of the AIC using R

print(c('model_1', AIC(iris.glm.1)))
print(c('model_2a', AIC(iris.glm.2a )))
print(c('model_2b', AIC(iris.glm.2b )))
print(c('model_2c', AIC(iris.glm.2c )))
print(c('model_2d', AIC(iris.glm.2d )))
print(c('model_3a', AIC(iris.glm.3a )))
print(c('model_3b', AIC(iris.glm.3b )))
print(c('model_3c', AIC(iris.glm.3c )))
