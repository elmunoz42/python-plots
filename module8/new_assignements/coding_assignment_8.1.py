#!/usr/bin/env python
# coding: utf-8

# ## Required Codio Assignment 8.1: Scikit-Learn Pipelines
# 
# **Estimated time: 60 minutes**
# 
# **Total Points: 24 Points**
# 
# This activity focuses on using the pipeline functionality of scikit-learn to combine a transformer with an estimator.  Specifically, you will combine the process of generating polynomial features with that of building a linear regression model.  You will use the `Pipeline` functionality from the `sklearn.pipeline` module to construct both a quadratic and cubic model.
# 
# ## Index:
# 
#  - [Problem 1](#Problem-1)
#  - [Problem 2](#Problem-2)
#  - [Problem 3](#Problem-3)
#  - [Problem 4](#Problem-4)
#  - [Problem 5](#Problem-5)
#  - [Problem 6](#Problem-6)

# In[1]:


import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")


# ### The Data
# 
# The data will again be the automobile dataset.  You are to use the pipelines to build quadratic features and linear models using `horsepower` to predict `mpg`.   

# In[ ]:


auto = pd.read_csv('data/auto.csv')


# In[ ]:


auto.head()


# [Back to top](#Index:) 
# 
# ## Problem 1
# 
# ### Creating a `Pipeline`
# 
# **4 Points**
# 
# Use `Pipeline` to create a pipeline object. Inside the pipeline object, define a tuple where the first element is a string identifier `quad_features` and the second element is an instance of `PolynomialFeatures` of degree `2`. Inside the pipeline define another tuple where the first element is a string identifier `quad_model`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `pipe`.

# In[ ]:


### GRADED

pipe = ''

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(pipe))
print(pipe.named_steps)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 2
# 
# ### Fitting the Pipeline
# 
# **4 Points**
# 
# Complete the code below according to the following instructions:
# 
# - Assign to the variable `X` the values of the `horsepower` of `auto`.
# - Assign to the variable `y` the values of the `mpg` of `auto`.
# - Use the function `fit` on `pipe` to train your model on `X` and `y`.
# - Determine the `mean_squared_error` of your model, and assign the value as a float to `quad_pipe_mse` below.  

# In[ ]:


### GRADED

X = ''
y = ''

quad_pipe_mse = ''

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(quad_pipe_mse))
print(quad_pipe_mse)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 3
# 
# ### Examining the Coefficients
# 
# **4 Points**
# 
# Now, to examine the coefficients, use the `.named_steps` attribute on the `pipe` object to extract the regressor.  Assign the model to `quad_reg` below.  
# 
# Extract the coefficients from the model and assign these as an array to the variable `coefs`.

# In[ ]:


### GRADED

quad_reg = '' #regressor from pipeline
coefs = '' #coefficients of regressor

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(quad_reg))
print(coefs)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 4
# 
# ### Considering the Bias 
# 
# **4 Points**
# 
# Not that your coefficients have 3 values.  Your model also contains an intercept term though, and this leads to one more value than expected from a quadratic model with one input feature.  This is due to the inclusion of the bias term using `PolynomialFeatures` and the intercept term added with the `fit_intercept = True` default setting in the regressor.  
# 
# 
# To get the appropriate model coefficients and intercept, you can set `include_bias = False` in the `PolynomialFeatures` transformer.  
# 
# Complete the code according to the instructions below:
# 
# - Use `Pipeline` to create a pipeline object. Inside the pipeline object, define a a tuple where the first element is a string identifier `quad_features` and the second element is an instance of `PolynomialFeatures` of degree `2` with `include_bias = False`. Inside the pipeline define another tuple where the first element is a string identifier `quad_model`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `pipe_no_bias`.
# - Use the `fit` function on `pipe_no_bias` to train your model on `X` and `y`. 
# - Use the `mean_squared_error` function to calculate the MSE between `y` and `pipe_no_bias.predict(X)`. Assign the result as a float `no_bias_mse`.
# 
# 

# In[ ]:


### GRADED

pipe_no_bias = '' #pipeline with no bias in transformer
no_bias_mse = '' #mean squared error of new model

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(pipe_no_bias))
print(no_bias_mse)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 5
# 
# ### Building a Cubic Model with `Pipeline`
# 
# **4 Points**
# 
# Complete the code according to the instructions below:
# 
# - Use `Pipeline` to create a pipeline object. Inside the pipeline object, define a a tuple where the first element is a string identifier `quad_features` and the second element is an instance of `PolynomialFeatures` of degree `3` with `include_bias = False`. Inside the pipeline define another tuple where the first element is a string identifier `quad_model`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `cubic_pipe`.
# - Use the `fit` function on `cubic_pipe` to train your model on `X` and `y`. 
# - Use the `mean_squared_error` function to calculate the MSE between `y` and `cubic_pipe.predict(X)`. Assign the result as a float to `no_bias_mse`.
# 

# In[ ]:


### GRADED

cubic_pipe = '' #pipeline with no bias in 3rd degree transformer
cubic_mse = '' #mean squared error of new model

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(cubic_pipe))
print(cubic_mse)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 6
# 
# ### Making Predictions on New Data
# 
# **4 Points**
# 
# Finally, one of the main benefits derived from using a Pipeline is that you do not need to engineer new polynomial features when predicting with new data.  Use your cubic pipeline to predict the `mpg` for a vehicle with 200 horsepower.  Assign your prediction as a numpy array to `cube_predict` below.

# In[ ]:


### GRADED

cube_predict = '' #cubic pipe prediction

# YOUR CODE HERE
raise NotImplementedError()

# Answer check
print(type(cube_predict))
print(cube_predict)


# In[ ]:





# In[ ]:




