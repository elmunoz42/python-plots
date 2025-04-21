#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 20.2: Implementing Bagging
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 50**
# 
# This activity focuses on using the `BaggingClassifier`.  You will use the Scikit-Learn implementation to compare performance on the fetal health dataset to that of the other models in the module -- Random Forests, Adaptive Boosting, and Gradient Boosting. The `BaggingClassifier` is a meta estimator that will aggregate estimators built on samples of the data.  You are to specify certain estimators and samples to become familiar with the functionality of the estimator and the variations you can produce with important arguments.  
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ### Data and Documentation
# 
# Below the data is loaded and prepared.  For this exercise, you will be expected to consult the documentation on the `BaggingClassifier` [here](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator).  The vocabulary in each problem can be found in the documentation and you are expected to use the correct settings for the arguments as a result of reading the documentation.  For each model, be sure to set `random_state = 42`.    

# In[6]:


df = pd.read_csv('data/fetal.zip', compression = 'zip')
X, y = df.drop('fetal_health', axis = 1), df['fetal_health']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Aggregating bootstrap models
# 
# **10 Points**
# 
# To start, create an ensemble of a decision tree using the `BaggingClassifier` function with `random_state = 42`. Fit this model to the training data `X_train` and `y_train`. Assign this moel to `bagged_model`.
# 
# Next, use the `score` function on `bagged_model` to calculate the performance on the test data. Assign this value to `bagged_score`.
# 

# In[7]:


### GRADED
bagged_model = BaggingClassifier(DecisionTreeClassifier(), random_state=42).fit(X_train, y_train)
bagged_score = bagged_model.score(y_test, bagged_model.predict(X_test))

### ANSWER CHECK
print(bagged_score)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Pasting vs. Bagging
# 
# **10 Points**
# 
# 
# 
# Create an ensemble of a decision tree using the `BaggingClassifier` function with `random_state = 42`. Consult the documentation [here](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) and adjust the appropriate argument of `BaggingClassifier` to change from **bagging** to **pasting**. Fit this model to the training data `X_train` and `y_train`. Assign this moel to `pasted_model`.
# 
# Next, use the `score` function on `pasted_model` to calculate the performance on the test data. Assign this value to `pasted_score`.
# 

# In[ ]:


### GRADED
pasted_model = ''
pasted_score = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(pasted_score)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Random Subspaces
# 
# **10 Points**
# 
# 
# 
# Create an ensemble of a decision tree using the `BaggingClassifier` function with `random_state = 42`. Consult the documentation [here](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) and adjust the appropriate argument to change from **bagging** to **random subspaces** with at most 10 features sampled. Fit this model to the training data `X_train` and `y_train`. Assign this moel to `subspace_model`.
# 
# Next, use the `score` function on `subspace_model` to calculate the performance on the test data. Assign this value to `subspace_score`.
# 
# 

# In[ ]:


### GRADED
subspace_model = ''
subspace_score = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(subspace_score)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Random Patches
# 
# **10 Points**
# 
# Create an ensemble of a decision tree using the `BaggingClassifier` function with `random_state = 42`. Consult the documentation [here](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) and adjust the appropriate argument to change from **bagging** to **random patches**. Use no more than 30% of the data and no more than 10 features in your samples. Assign this moel to `patches_model`.
# 
# Next, use the `score` function on `patches_model` to calculate the performance on the test data. Assign this value to `patches_score`.
# 
# 
# 

# In[ ]:


### GRADED
patches_model = ''
patches_score = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(patches_score)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Nature of the Tree Models
# 
# **10 Points**
# 
# Consult the documentation [here](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator) and observe whether or not bagging typically works with simple or complex tree models.  Enter your answer as `simple` or `complex` as a string to `ans5`. 

# In[ ]:


### GRADED
ans5 = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(ans5)


# In[ ]:





# In[ ]:




