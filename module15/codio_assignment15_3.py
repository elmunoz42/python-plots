#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 15.3: Stochastic Gradient Descent
# 
# **Expected Time = 30 minutes**
# 
# **Total Points = 20**
# 
# This activity explores the use of the `SGDRegressor` from scikitlearn.  While there is not a standard gradient descent estimator, the more efficient example of stochastic gradient descent is available.  You will use the earlier credit dataset to explore its use and learn an important lesson about scaling your data with gradient descent methods. 
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


credit = pd.read_csv('data/Credit.csv', index_col=0)


# In[3]:


credit.head()


# In[4]:


X = credit[['Income', 'Limit']]
y = credit['Balance']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Fitting a basic Linear Regression model
# 
# **5 Points**
# 
# To begin, use the `LinearRegression` estimator with all default parameters to build a model. Fit this model  on the training data `X_train` and `y_train`. Assign this model to the variable `lr`.
# 
# Next, assign the training and testing errors to `train_mse` and `test_mse` respectively.

# In[6]:


# mean_squared_error?


# In[7]:


### GRADED
lr = LinearRegression().fit(X_train, y_train)
train_mse = mean_squared_error(y_train, lr.predict(X_train))
test_mse = mean_squared_error(y_test, lr.predict(X_test))

### ANSWER CHECK
print(train_mse)
print(test_mse)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Fitting a basic SGD model
# 
# **10 Points**
# 
# Define define an `SGDRegressor` estimator with `random_state = 42` and fit to the training data `X_train` and `y_train`. Assign this model to `sgd_defaults`.
# 
# Next, assign the training and testing errors to `train_mse` and `test_mse` respectively.

# In[8]:


get_ipython().run_line_magic('pinfo', 'SGDRegressor')


# In[9]:


### GRADED
sgd_defaults = SGDRegressor(random_state=42).fit(X_train, y_train)
train_mse = mean_squared_error(y_train, sgd_defaults.predict(X_train))
test_mse = mean_squared_error(y_test, sgd_defaults.predict(X_test))

### ANSWER CHECK
print(sgd_defaults)
print(train_mse)
print(test_mse)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Scaling the Data
# 
# **10 Points**
# 
# An important message in gradient descent methods is that scaling the data and using regularization helps to constrain the solution path.  
# 
# You should be able to see the effect of providing the `SGDRegressor` scaled data below.  
# 
# Define define an `SGDRegressor` estimator with `random_state = 42` and fit to the training data `X_tr_scaled` and `y_train`. Assign this model to `sgd_scaled`.
# 
# Next, assign the training and testing errors to `train_mse` and `test_mse` respectively.

# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_ts_scaled = scaler.transform(X_test)


# In[13]:


### GRADED
sgd_scaled = SGDRegressor(random_state=42).fit(X_tr_scaled, y_train)
train_mse = mean_squared_error(y_train, sgd_scaled.predict(X_tr_scaled))
test_mse = mean_squared_error(y_test, sgd_scaled.predict(X_ts_scaled))
### ANSWER CHECK
print(sgd_scaled)
print(train_mse)
print(test_mse)


# In[ ]:





# You can return to your earlier examples and see if scaling your data made any differences in the gradient descent convergence, this is important to all models using a gradient descent method and you will see this again with neural networks.

# In[ ]:




