#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 8.2: Comparing Complexity and Variance
# 
# **Expected Time: 60 Minutes**
# 
# **Total Points: 35**
# 
# In this activity, you will explore the effect of model complexity on the variance in predictions.  Continuing with the automotive data, you will build models on a subset of 10 vehicles.  You will compare the model error when used on the entire dataset and investigate how variance changes with model complexity.
# 
# #### Index:
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import plotly.express as px


# In[2]:


auto = pd.read_csv('data/auto.csv')


# In[3]:


auto.head()


# ### The Sample
# 
# Below, a sample of ten vehicles from the data is extracted.  These data will form our **training** data.  The data is subsequently split into `X_train` and `y_train`.  You are to use this smaller dataset to build your models on and explore their performance using the entire dataset.

# In[4]:


X = auto.loc[:,['horsepower']]
y = auto['mpg']
sample = auto.sample(10, random_state = 22)
X_train = sample.loc[:, ['horsepower']]
y_train = sample['mpg']


# In[5]:


X_train


# In[6]:


y_train


# In[7]:


X.shape


# [Back to top](#Index:) 
# 
# ### Problem 1
# 
# #### Iterate on Models
# 
# **20 Points**
# 
# Complete the code below according to the instructions below:
# 
# - Assign the values in the `horsepower` column of `auto` to the variable `X` below.
# - Assign the values in the `mpg` column of `auto` to the variable `y` below.
# 
# Use a `for` loop to loop over the values from one to ten. For each iteration `i`:
# 
# - Use `Pipeline` to create a pipeline object. Inside the pipeline object, define a a tuple where the first element is a string identifier `quad_features'` and the second element is an instance of `PolynomialFeatures` of degree `i` with `include_bias = False`. Inside the pipeline define another tuple where the first element is a string identifier `quad_model`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `pipe`.
# - Use the `fit` function on `pipe` to train your model on `X_train` and `y_train`. Assign the result to `preds`.
# - Use the `predict` function to predict the value of `X_train`. Assign the result to `preds`.
# - Assign each `model_predictions` of degree `i` the corresponding `preds` value.

# In[8]:


### GRADED

### YOUR SOLUTION HERE
model_predictions = {f'degree_{i}': None for i in range(1, 11)}

def predictions_for_range_of_degrees(X_train, y_train, X_pred, range_start, range_stop):
    predictions = []
    #for 1, 2, 3, ..., 10
    for i in range(range_start, range_stop):
        #create pipeline
        pipe = Pipeline([
            ('quad_features', PolynomialFeatures(degree=i, include_bias=False)),
            ('quad_model', LinearRegression())
        ])
        #fit pipeline on training data
        pipe.fit(X_train, y_train)
        #make predictions on all data
        preds = pipe.predict(X_pred)
        #assign to model_predictions
        predictions.append(preds)
        
    return predictions

predictions = predictions_for_range_of_degrees(X_train, y_train, X_train, 1, 11)

for key, value in zip(model_predictions.keys(), predictions):
    model_predictions[key] = value
    
# Answer check
model_predictions['degree_1'][:10]


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 2
# 
# #### DataFrame of Predictions
# 
# **5 Points**
# 
# Use the `model_predictions` dictionary to create a DataFrame of the 10 models predictions.  Assign your solution to `pred_df` below as a DataFrame. 

# In[9]:


### GRADED

### YOUR SOLUTION HERE
pred_df = pd.DataFrame(model_predictions)

# Answer check
print(type(pred_df))
print(pred_df.head())


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 3
# 
# #### DataFrame of Errors
# 
# **5 Points**
# 
# Now, determine the error for each model and create a DataFrame of these errors.  One way to do this is to use your prediction DataFrame's `.subtract` method to subtract `y` from each feature.  Assign the DataFrame of errors as `error_df` below.  

# In[11]:


### GRADED

### YOUR SOLUTION HERE
error_df = pred_df.subtract(y, axis=0)

# Answer check
print(type(error_df))
print(error_df.head())


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 4
# 
# #### Mean and Variance of Model Errors
# 
# **5 Points**
# 
# 
# Using the DataFrame of errors, examine the mean and variance of each model's error.  What degree model has the highest variance?  Assign your response as an integer to `highest_var_degree` below.

# In[12]:


### GRADED

## Exploration of the data statistics
error_stats = error_df.describe()
# Calculate variance and create a DataFrame
var_df = pd.DataFrame(error_df.var()).T
var_df.index = ['var']

# Concatenate the original DataFrame with the variance DataFrame
error_stats = pd.concat([error_stats, var_df])

print(error_stats)

### YOUR SOLUTION HERE
highest_degree_name = error_df.var().idxmax()
print(highest_degree_name)
highest_var_degree = 3

# Answer check
print(type(highest_var_degree))
print(highest_var_degree)


# In[ ]:





# In[ ]:




