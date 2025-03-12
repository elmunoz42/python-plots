#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 14.2: Preventing Overfitting by Limiting Growth
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 50**
# 
# This activity focuses on using the hyperparameters in the scikit-learn model that restrict the depth of the tree.  You will compare different setting combinations of these hyperparameters to determine the best parameters using a test set for evaluating your classifier.  
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV


# ### The Data
# 
# For this exercise, you will use the credit card default dataset.  Again, the goal is to predict credit card default.  Below, the data is loaded, cleaned, and split for you.

# In[2]:


default = pd.read_excel('data/Default.xls', skiprows = 1)
default.head()


# In[3]:


default = default.rename({'default payment next month': 'default'}, axis = 1)


# In[4]:


default.info()


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(default.drop('default', axis = 1), default.default, 
                                                   random_state = 42)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Unlimited Growth
# 
# **10 Points**
# 
# Use a  `DecisionTreeClassifier` with `random_state=42` to fit the estimator on the training data `X_train` and `y_train`. Assign the estimator ato the variable `dtree`.
# 
# Compare the training and test set accuracy score and assign them as floats to `train_acc` and `test_acc`, respectively.  
# 
# Examine the depth of the tree with the `.get_depth()` method.  Assign this to `depth_1`.  
# 
# <div class="alert alert-block alert-info"><b>Note: </b> Use <code>random_state = 42</code> for all estimators in this assignment!</div>

# In[7]:





# In[8]:


### GRADED
dtree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
depth_1 = dtree.get_depth()
train_acc = dtree.score(X_train, y_train)
test_acc = dtree.score(X_test,y_test)

### Answer Check
print(f'Training Accuracy: {train_acc: .2f}')
print(f'Trest Accuracy: {test_acc: .2f}')
print(f'Depth of tree: {depth_1}')


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# ### `min_samples_split`
# 
# Setting the `min_samples_split` argument will control splitting nodes with either a number of samples or percent of the data as valued.  From the estimators docstring:
# 
# ```
# min_samples_split : int or float, default=2
#     The minimum number of samples required to split an internal node:
# 
#     - If int, then consider `min_samples_split` as the minimum number.
#     - If float, then `min_samples_split` is a fraction and
#       `ceil(min_samples_split * n_samples)` are the minimum
#       number of samples for each split.
# ```
# 
# Inside the `DecisionTreeClassifier` estimator, use this argument with value `0.05` to limit the trees growth to nodes with more than 5% of the samples. Fit this estimator to the training data `X_train` and `y_train` and assign the estimator to `dtree_samples`.
# 
# 
# Evaluate the train and test accuracy as floats and assign the results to `samples_train_acc` and `samples_test_acc`, respectively.  Assign the depth of the tree with the `.get_depth()` method and assign the result to `depth_2` below.  Remember to set `random_state = 42` in your estimator.
# 
# **10 Points**
# 

# In[9]:


### GRADED
dtree_samples = DecisionTreeClassifier(min_samples_split=0.05, random_state=42).fit(X_train, y_train)
depth_2 = dtree_samples.get_depth()
samples_train_acc = dtree_samples.score(X_train, y_train)
samples_test_acc = dtree_samples.score(X_test, y_test)

### Answer Check
print(f'Training Accuracy: {samples_train_acc: .2f}')
print(f'Trest Accuracy: {samples_test_acc: .2f}')
print(f'Depth of tree: {depth_2}')


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### `max_depth`
# 
# Below, create a tree that grows to a maximum depth of 5 and fit it to the training data.  Assign the estimator as `depth_tree`. Be sure to set `random_state = 42`. 
# 
# Calculate the accuracy on the train and test set as floats to `depth_train_acc` and `depth_test_acc` respectively.  
# 
# 
# 
# **10 Points**
# 

# In[10]:


### GRADED
depth_tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
depth_train_acc = depth_tree.score(X_train, y_train)
depth_test_acc = depth_tree.score(X_test, y_test)

### Answer Check
print(f'Training Accuracy: {depth_train_acc: .2f}')
print(f'Trest Accuracy: {depth_test_acc: .2f}')


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# ### `min_impurity_decrease`
# 
# **10 Points**
# 
# The argument `min_impurity_decrease` stops splitting when there is less than a given amount of impurity decrease. 
# 
# Below, define a decision tree called `imp_tree` with a `min_impurity_decrease = 0.01` and `random_state=42` and fit it to the training data. Calculate its depth as `depth_4`. 
# 
# Finally, calculate the train and test scores as floats to `imp_training_acc` and `imp_test_acc` respectively. 
# 
# 
# 

# In[11]:


### GRADED
imp_tree = DecisionTreeClassifier(min_impurity_decrease = 0.01, random_state=42).fit(X_train, y_train)
imp_train_acc = imp_tree.score(X_train, y_train)
imp_test_acc = imp_tree.score(X_test, y_test)
depth_4 = imp_tree.get_depth()

### Answer Check
print(f'Training Accuracy: {imp_train_acc: .2f}')
print(f'Trest Accuracy: {imp_test_acc: .2f}')
print(f'Depth of tree: {depth_4}')


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# ###  Grid Searching parameters
# 
# **10 Points**
# 
# 
# Finally, consider the parameters defined below to perform a grid search with a decision tree. 
# 
# Below, define a decision tree called `grid` with `param_grid=params` and `random_state=42` and fit it to the training data. 
# 
# Calculate the train and test scores as floats to `grid_training_acc` and `grid_test_acc` respectively. 
# 
# Finally, use the method `best_params_` on `grid` to derive the best parameters for this tree. Assign the results to `best_params`.
# 

# In[12]:


params = {'min_impurity_decrease': [0.01, 0.02, 0.03, 0.05],
         'max_depth': [2, 5, 10],
         'min_samples_split': [0.1, 0.2, 0.05]}


# In[15]:


### GRADED
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=params).fit(X_train, y_train)
grid_train_acc = grid.best_estimator_.score(X_train, y_train)
grid_test_acc = grid.best_estimator_.score(X_test,y_test)
best_params = grid.best_params_

### Answer Check
print(f'Training Accuracy: {grid_train_acc: .2f}')
print(f'Trest Accuracy: {grid_test_acc: .2f}')
print(f'Best parameters of tree: {best_params}')


# In[ ]:





# Note how long the basic grid search takes.  You likely don't want to try to be too exhaustive with the parameters due to the time for training cost. 

# In[ ]:




