#!/usr/bin/env python
# coding: utf-8

# ### Codio Assignment 9.3: Using StandardScaler
# 
# **Estimated Time: 45 Minutes**
# 
# **Total Points: 40**
# 
# 
# This activity focuses on using the `StandardScaler` to scale the data by converting it to $z$-scores.  To begin, you will scale data using just NumPy functions.  Then, you will use the scikit-learn transformer and incorporate it into a `Pipeline` with a `Ridge` regression model.  

# #### Index
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# 

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ### The Dataset
# 
# For this example, we will use a housing dataset that is part of the scikitlearn datasets module.  The dataset is chosen because we have multiple features on very different scales.  It is loaded and explored below -- your task is to predict `MedHouseVal` using all the other features after scaling and applying regularization with the `Ridge` estimator. 

# In[2]:


cali = fetch_california_housing(as_frame=True)


# In[3]:


cali.frame.head()


# In[4]:


print(cali.DESCR)


# In[5]:


cali.frame.info()


# In[6]:


X = cali.frame.drop('MedHouseVal', axis = 1)
y = cali.frame['MedHouseVal']


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# ### Problem 1
# 
# #### Scaling the Train data
# 
# **10 Points**
# 
# Recall that **standard scaling** consists of subtracting the feature mean from each datapoint and subsequently dividing by the standard deviation of the feature.  Below, you are to scale `X_train` by subtracting the mean and dividing by the standard deviation.  Be sure to use the `numpy` mean and standard deviation functions with default settings.  
# 
# Assign your results to `X_train_scaled` below.  

# In[8]:


### GRADED

X_train_scaled = ( X_train - np.mean(X_train) ) / np.std(X_train)

# Answer check
print(X_train_scaled.mean())
print('-----------------')
print(X_train_scaled.std())


# In[ ]:





# ### Problem 2
# 
# #### Scale the test data
# 
# **10 Points**
# 
# To scale the test data, use the mean and standard deviation of the **training** data.  In practice, you would not have seen the test data, so you would not be able to compute its mean and deviation.  Instead, you assume it is similar to your train data and use what you know to scale it.  
# 
# Assign the response as an array to `X_test_scaled` below.

# In[9]:


### GRADED

X_test_scaled = (X_test - np.mean(X_train)) / np.std(X_train)

# Answer check
print(X_test_scaled.mean())
print('-----------------')
print(X_test_scaled.std())


# In[ ]:





# ### Problem 3
# 
# #### Using `StandardScaler`
# 
# **10 Points**
# 
# - Instantiate a `StandardScaler` transformer. Assign the result to `scaler`.
# - Use the `.fit_transform` method on `scaler` to transform the training data. Assign the result to `X_train_scaled`.
# - Use the `.transform` method on `scaler` to transform the test data. Assign the result to `X_test_scaled`.

# In[10]:


### GRADED

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Answer check
print(scaler.mean_)
print('----------')
print(scaler.scale_)


# In[ ]:





# ### Problem 4
# 
# #### Building a `Pipeline`
# 
# **15 Points**
# 
# Now, construct a pipeline with named steps `scaler` and `ridge` that takes in your data, applies the `StandardScaler`, and fits a `Ridge` model with default settings. Next, use the `fit` function to train this pipeline on `X_train` and `y_train`. Assign your pipeline to `scaled_pipe`.
# 
# Use the `predict` function on `scaled_pipe` to compute the predictions on `X_train`. Assign your result to `train_preds`.
# 
# Use the `predict` function on `scaled_pipe` to compute the predictions on `X_test`. Assign your result to `test_preds`.
# 
# Use the `mean_squared_error` function to compute the MSE between `y_train` and `train_preds`. Assign your result to `train_mse`.
# 
# Use the `mean_squared_error` function to compute the MSE between `y_test` and `test_preds`. Assign your result to `test_mse`.

# In[11]:


### GRADED

scaled_pipe = Pipeline([
    ('scale', StandardScaler()),
    ('ridge', Ridge())
]) 
scaled_pipe.fit(X_train, y_train)
train_preds = scaled_pipe.predict(X_train)
test_preds = scaled_pipe.predict(X_test)
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

# Answer check
print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')


# In[ ]:





# In[ ]:




