#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 15.1: Gradient Descent and Linear Regression
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 50**
# 
# In this activity you will use gradient descent to identify the parameter $\theta_0$ that minimizes the Mean Squared Error of predictions using the model $y = \theta_0  x$.  In this example, you will use a dataset containing information from a credit card company on customers.  Your goal will be to build a linear model to predict the balance using the credit rating. 
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


credit = pd.read_csv('data/Credit.csv', index_col=0)


# In[3]:


credit.head()


# In[4]:


sns.heatmap(credit.corr()[['Balance']].sort_values(by = 'Balance', ascending = False), annot = True)


# In[5]:


X = credit[['Rating']]
y = credit['Balance']


# In[6]:


sns.scatterplot(data = credit, x = 'Rating', y = 'Balance')


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### A Basic Model
# 
# **10 Points**
# 
# To begin, complete the function `mse` below that takes in a value for $\theta$ and returns the mean squared error based on the model 
# 
# $$\text{Balance} = \theta \times \text{Rating}$$

# In[7]:


### GRADED
def mse(theta):
    '''
    This function takes in a float for theta and 
    returns the mean squared error according to the
    mean of the formula (y - theta*credit['Rating'])**2.
    
    Arguments
    ---------
    theta: float
          coefficient of linear model
          
    Returns
    -------
    mse: float
         Mean Squared Error of Linear model against y
    '''
    return np.mean(credit['Balance']-theta*credit['Rating'])

### ANSWER CHECK
mse(10)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Considering Many $\theta$'s
# 
# **10 Points**
# 
# Now, consider the array of thetas given below as `thetas`.  
# 
# Use a `for` loop to iterate over the array of thetas and compute the **Mean Squared Error** for the each given $\theta$.  Store these values in the list `mses` below.  
# 
# Observe the plot below to view the results.

# In[ ]:


thetas = np.linspace(-10, 13, 50)


# In[ ]:


### GRADED
mses = []
for theta in thetas:
    pass
    
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
mses[-5:]


# In[ ]:


plt.plot(thetas, mses, '--o')
plt.xlabel(r'$\theta$')
plt.ylabel('MSE');
plt.title(r'Mean Squared Error for given $\theta$');


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Finding the $\theta$ that minimizes MSE
# 
# **10 Points**
# 
# Now, you are to use gradient descent with parameters `xs` and `step_size` given below to identify the $\theta$ that minimizes the `mse` function.  
# 
# 
# ```
# - xs = 10
# - step_size = 0.1
# ```
# 
# You are given a function `df` to approximate the derivative of the `mse` function.
# 
# Use a `for` loop with 200 iterations to calculate the gradient descent of the fucntion `df` given to you below. 
# 
# Did the algorithm converge to a reasonable value on the last step?  Assign this value to `theta_big_step` as a float below.

# In[ ]:


def df(x):
    return (mse(x + 0.001) - mse(x))/0.001


# In[ ]:


### GRADED
xs = [10]
for i in range(200):
    pass
    
theta_big_step = ''   
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
xs[-5:]


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### A smaller step size
# 
# **10 Points**
# 
# 
# Repeat the exercise above but now try using 1000 iterations of the gradient descent algorithm with:
# 
# ```
# x0 = 10
# step_size = 0.000001
# ```
# 
# Did the algorithm converge to a reasonable value on the last step?  Assign this value to `theta_small_step` as a float below.

# In[ ]:


### GRADED
xs = [10]
for i in range(1000):
    pass
    
theta_small_step = ''   
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
xs[-5:]


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Checking against `sklearn`
# 
# **10 Points**
# 
# Finally, you are to compare your solution using the small step size to that obtained from scikitlearn's `LinearRegression` estimator.  Be sure to set `fit_intercept = False`. 
# 
# After fitting the estimator on `X` and `y`, determine the absolute difference between your solution for $\theta$ from [Problem 4](#-Problem-4) and the `.coef_` attribute of the fit sklearn model.  Assign the result as a float to `error`. 

# In[ ]:


### GRADED
lr = ''
error = ''
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(error)


# In[ ]:





# In[ ]:




