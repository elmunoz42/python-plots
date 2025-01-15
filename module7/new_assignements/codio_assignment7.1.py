#!/usr/bin/env python
# coding: utf-8

# ## Required Codio Assignment 7.1: Using SciPy Optimize To Optimize L2 Loss
# 
# **Expected Time = 45 minutes**
# 
# **Total Points = 15**
# 
# This assignment focuses on using `scipy.optimize` to minimize the mean squared error for a linear model.  For this example,  a synthetic dataset is created using `sklearn`.  
# 
# ## Index:
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ### Creating Data
# 
# To create the dataset, a linear function with a known slope is created, and Gaussian noise is added to each point at random.  This allows comparison to the results and helps to see if the minimum solution is reasonable. 
# 
# $$y = 4.2x + \sigma$$

# In[2]:


np.random.seed(42)
x = np.linspace(0, 1, 100)
y = 4.2*x + np.random.normal(size = 100)
plt.scatter(x, y)


# [Back to top](#Index:) 
# 
# ## Problem 1
# 
# ### Array of $\theta$'s
# 
# **5 Points**
# 
# Below, create an array of possible $\theta$ values using `np.linspace`.  Create 100 values starting at 3 and ending at 5.  Assign your solution as an array to `thetas` below.

# In[3]:


### GRADED

thetas = np.linspace(3,5,100)
print(thetas)
# Answer check
print(type(thetas))
print(thetas.shape)


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 2
# 
# ### Loss Function
# 
# **5 Points**
# 
# Now, complete the function `l2_loss` below that accepts a single `theta` value as input and calculates the mean squared error based on the true y-values and the given theta.
# 
# 
# The function should return a single float value representing the mean squared error.

# In[4]:


### GRADED
def l2_loss(theta):
    """
    This function accepts a single theta value
    and calculates the mean squared error based
    on (theta*x - y)^2

    Arguments
    ---------
    theta: float
    The value to use for the parameter of the
    regression model.

    Returns
    -------
    mse: float
    Mean Squared Error
    """

    return np.mean((theta*x - y)**2)

mses = l2_loss(8)
print(mses)
print(type(mses))


# In[ ]:





# [Back to top](#Index:) 
# 
# ## Problem 3
# 
# ### Using `scipy` to minimize `l2_loss`
# 
# **5 Points**
# 
# Use the `minimize` function that has been imported from `scipy.optimize` to find the minimum value of `l2_loss` using `x0 = 4`.  Assign your results to the `minimum_theta` variable below.  
# 
# Next, use the `minimum_theta.x` attribute to examine the solution and assign as a numpy array to `theta_solution` below.

# In[5]:


### GRADED

minimum_theta = minimize(l2_loss, x0=4)
theta_solution = minimum_theta.x

# Answer check
print(type(theta_solution))
print(theta_solution)


# In[ ]:





# Now that you have found the minimum value, you can uncomment the code below and visualize the mean squared error along with the minimum value based on `scipy`.  

# In[6]:


plt.plot(thetas, [l2_loss(i) for i in thetas])
plt.plot(theta_solution, l2_loss(theta_solution), 'ro', label = f'solution: {np.round(theta_solution[0], 3)}')
plt.legend();
plt.title(r'Minimizing Mean Squared Error given $\theta$');
plt.xlabel(r'$\theta$')
plt.ylabel('MSE');


# In[ ]:




