#!/usr/bin/env python
# coding: utf-8

# ### Required Codio assignment 20.3: Implementing Random Forests
# 
# **Expected Time = 60 minutes** 
# 
# **Total Points = 30** 
# 
# This activity focuses on building models using the `RandomForestClassifier` from Scikit-Learn.  You will explore the estimator, and how the number of trees in the model affect the performance. To evaluate your model you will look to the out of bag data rather than a test set. 
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/fetal.zip', compression = 'zip')


# In[3]:


df.head()


# In[4]:


X, y = df.drop('fetal_health', axis = 1), df['fetal_health']


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Buildilng the Model
# 
# **10 Points**
# 
# Below, create an instance of the `RandomForestClassifier` estimtor with `random_state = 42` and
# `oob_score = True`  and fit to X and Y. Assign this model to `forest_1` below.
# 
# Next, use the `oob_score_` method on `forest_1` to calculate the oob score of your model and assign the result to `score`.

# In[5]:


### GRADED
forest_1 = RandomForestClassifier(random_state=42, oob_score = True).fit(X,y)
score = forest_1.oob_score_

### ANSWER CHECK
print(score)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Changing the number of trees
# 
# **10 Points**
# 
# The `n_trees` list below defined the different numbers of trees to use.
# 
# In the code cell below, complete the `for` loop to iterate over the different number of trees and keep track of the oob score with the list `oob_scores`.   Ensure to set `random_state = 42`, `oob_score = True`, and `n_estimators` equal to the number of trees.
# 
# 

# In[6]:


n_trees = [1, 10, 100, 500, 1000, 2000]


# In[7]:


### GRADED
oob_scores = []
for i in n_trees:
    forrest_i = RandomForestClassifier(random_state=42, oob_score=True, n_estimators=i).fit(X,y)
    oob_scores.append(forrest_i.oob_score_)

### ANSWER CHECK
print(oob_scores)


# In[ ]:





# In[8]:


plt.plot(n_trees, oob_scores, '--o')
plt.grid()
plt.title('Number of trees and oob score')
plt.xlabel('Number of Trees')
plt.ylabel("oob score");


# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Changing the trees themselves
# 
# **10 Points**
# 
# The `RandomForestClassifier` contains most of the same parameters as the `DecisionTreeClassifier` including `max_depth` and `ccp_alpha` that control the geometry of the individual trees.  
# 
# While searching over many parameters of a forest might seem like a good idea, in this context it is too computationally complex to be exhaustive.  
# 
# Below, compare trees with 200 trees in the model, and explore if the depth of these trees effects the out of bag score.  Use the list `depths` below, and use the list `depth_oob` to keep track of the scores.
# 
# 

# In[9]:


depths = [1, 2, 3, 4, 5, None]


# In[10]:


### GRADED
depth_oobs = []
for d in depths:
    forrest_i = RandomForestClassifier(random_state=42, oob_score=True, n_estimators=200, max_depth=d).fit(X,y)
    depth_oobs.append(forrest_i.oob_score_)

### ANSWER CHECK
print(depth_oobs)


# In[11]:


plt.plot(depths, depth_oobs, '--o')
plt.grid()
plt.title('Max Depth of Trees and oob score')
plt.xlabel('Max Depth')
plt.ylabel("oob score");


# In[ ]:





# The Random Forest estimator is a powerful example of ensembling.  From situation to situation it is important to recall that there is no one model that will always do better.  In the next activities, you will see two more ensembling techniques that will be important options to consider in your work.

# In[ ]:




