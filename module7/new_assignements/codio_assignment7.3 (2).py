#!/usr/bin/env python
# coding: utf-8

# ## Required Codio Assignment 7.3: Multiple Linear Regression
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 20**
# 
# This assignment focuses on building a regression model using multiple features.  Using a dataset from the `seaborn` library, you are to build and evaluate regression models with one, two, and three features.
# 
# ## Index:
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ### The Dataset
# 
# Below, a dataset containing information on diamonds is loaded and displayed.  Your task is to build a regression model that predicts the price of the diamond given different features as inputs.  

# In[4]:


diamonds = sns.load_dataset('diamonds')


# In[5]:


diamonds.head()


# [Back to top](#Index:) 
# 
# ## Problem 1
# 
# ### Regression with single feature
# 
# **5 Points**
# 
# Use sklearn's `LinearRegression` estimator with argument `fit_intercept` equal to `False` to build a regression model. Next, chain a `fit()` function using the `carat` column as the feature and the `price` column as the target.  
# 
# Assign your result to the variable `lr_1_feature` below.

# In[11]:


### GRADED
features = diamonds[['carat']]
price = diamonds['price']

lr_1_feature = LinearRegression(fit_intercept = False).fit(features, price)

# Answer check
print(lr_1_feature)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 2
# 
# ### Regression with two features
# 
# **5 Points**
# 
# Use sklearn's `LinearRegression` estimator with argument `fit_intercept` equal to `False` to build a regression model. Next, chain a `fit()` function using the `carat` and `depth` columns as the feature and the `price` column as the target.  
# 
# Assign your result to the variable `lr_2_feature` below.

# In[12]:


### GRADED
features = diamonds[['carat', 'depth']]
price = diamonds['price']
lr_2_features = LinearRegression(fit_intercept = False).fit(features, price)

# Answer check
print(lr_2_features)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 3
# 
# ### Regression with three features
# 
# **5 Points**
# 
# Use sklearn's `LinearRegression` estimator with argument `fit_intercept` equal to `False` to build a regression model. Next, chain a `fit()` function using the `carat`, `delth`, and `table` columns as the feature and the `price` column as the target.  
# 
# Assign your result to the variable `lr_3_feature` below.

# In[13]:


### GRADED
features = diamonds[['carat', 'depth', 'table']]
price = diamonds['price']
lr_3_features =  LinearRegression(fit_intercept = False).fit(features, price)


# Answer check
print(lr_3_features)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 4
# 
# ### Computing MSE and MAE
# 
# **5 Points**
# 
# For each of your models, compute the mean squared error and mean absolute errors.  Create a DataFrame to match the structure below:
# 
# | Features | MSE | MAE |
# | ----- | ----- | ----- |
# | 1 Feature |  -  | - |
# | 2 Features | -  | -  |
# | 3 Features | - | - |
# 
# Assign your solution as a DataFrame to `error_df` below.  Note that the `Features` column should be the index column in your DataFrame.

# In[16]:


### GRADED
predict1 = lr_1_feature.predict(diamonds[['carat']])
predict2 = lr_2_features.predict(diamonds[['carat', 'depth']])
predict3 = lr_3_features.predict(diamonds[['carat', 'depth', 'table']])

MSE_1 = mean_squared_error(price, predict1 )
MAE_1 = mean_absolute_error(price, predict1)

MSE_2 = mean_squared_error(price, predict2 )
MAE_2 = mean_absolute_error(price, predict2 )

MSE_3 = mean_squared_error(price, predict3 )
MAE_3 = mean_absolute_error(price, predict3 )

error_df =pd.DataFrame({
    'Features': ['1 Feature', '2 Features', '3 Features'],
    'MSE': [MSE_1, MSE_2, MSE_3],
    'MAE': [MAE_1, MAE_2, MAE_3]
})
error_df.set_index('Features')
# Answer check
error_df


# In[ ]:





# In[ ]:




