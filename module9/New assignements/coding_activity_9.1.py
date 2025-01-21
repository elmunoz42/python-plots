#!/usr/bin/env python
# coding: utf-8

# ### Codio Assignment 9.1: Cross-Validation with SequentialFeatureSelector
# 
# **Expected Time: 45 Minutes**
# 
# **Total Points: 60**
# 
# This activity uses the same example as in Video 9.3. That example uses the `shuffle` and `np.split` functions to create the train/test split indices. Then, these indices are passed to the `cv` argument in the `SequentialFeatureSelector`. Here, rather than using a Pipeline, you are to conduct the transformations by hand and feed the resulting selected features into a `LinearRegression` estimator.
# 
# 
# #### Index
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from random import shuffle, seed

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error


# ### The Dataset
# 
# For this exercise, the built-in diabetes dataset will be used.  The features are blood measurements and demographic information, and the target is a numeric measurement of diabetes progression.  The data is loaded and displayed below.  

# In[2]:


diabetes = load_diabetes(as_frame = True)


# In[3]:


print(diabetes.DESCR)


# In[4]:


df = diabetes.frame


# In[5]:


df.head()


# In[6]:


df.info()


# [Back to top](#Index)
# 
# ### Problem 1
# 
# #### Create a list of indicies and shuffle them
# 
# **10 Points**
# 
# To begin, create a list of the indices of the DataFrame `df`.  Assign this list to `all_indices`.  Then, use the `shuffle` function to shuffle the list in place.  In order to be consistent, set a `seed(42)` before calling the `shuffle` function.  

# In[7]:


### GRADED
# Set the seed for reproducibility
seed(42)

# Create a list of indices from 0 to the length of the DataFrame minus one
all_indices = list(range(len(df)))

# Shuffle the list in place
shuffle(all_indices)

# Answer check
all_indices[:5] #Should be [170, 226, 231, 345, 337]


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 2
# 
# **10 Points**
# 
# #### Split indices to train and test values
# 
# Use the `np.split` function to split the `all_indices` data based on the first 350 values.  Assign these as arrays to `train_idx` and `test_idx` below.  

# In[8]:


### GRADED
train_idx, test_idx = np.split(all_indices, [320])

# Answer check
print(train_idx[:5])#Should be [170 226 231 345 337]
print(test_idx[:5])#Should be [311  36 331 349 136]


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 3
# 
# #### Creating `SequentialFeatureSelector` object
# 
# **20 Points**
# 
# Create a `SequentialFeatureSelector` object named `selector` below that:
# 
# - Uses `LinearRegression` estimator to select features.
# - Selects four features using `n_features_to_select`.
# - Uses `train_idx` and `test_idx` inside the `cv` argument.
# - Uses `neg_mean_squared_error` for the `scoring` argument.
# 
# Then, use the `fit_transform` function on `selector`  to transform the data `X` and `y` given below. Assign the transformed data as an array to `Xt` below.

# In[10]:


X = diabetes.frame.drop('target', axis = 1)
y = diabetes.frame.target


# In[11]:


### GRADED
selector = SequentialFeatureSelector(LinearRegression(), n_features_to_select=4, cv=[[train_idx, test_idx]], scoring='neg_mean_squared_error')
Xt = selector.fit_transform(X, y)

### ANSWER CHECK
print(Xt.shape)#should have four columns


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 4
# 
# #### Using selected features in a model
# 
# **20 Points**
# 
# Follow the instructions below to complete the code:
# 
# - Instantiate a `LinearRegression` classifier. To it, chain a `fit` function to train the model on `X` and `y`. Assign this result to `lr`.
# - Use the `predict` function on `lr` to make your predictions on `Xt`. Assign this result to `model_preds`.
# - Finally, use the `mean_squared_error` function to calculate the error between `y` and `model_preds`. Assign your result to `mse`.
# 
# 

# In[12]:


### GRADED
lr = LinearRegression().fit(Xt,y)
model_preds = lr.predict(Xt)
mse = mean_squared_error(y, model_preds)

# Answer check
print(mse)


# In[ ]:





# In[ ]:




