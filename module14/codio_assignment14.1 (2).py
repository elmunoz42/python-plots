#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 14.1: Decision Trees in Scikit-Learn
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 50**
# 
# This activity introduces using the `DecisionTreeClassifier` from the `sklearn.tree` module.  You will build some basic models and explore hyperparameters available.  Using the results of the model, you will explore decision boundaries determined by the estimator. 
# 
# #### Index 
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)
# 
# 

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import set_config

set_config("diagram")


# ### The Data
# 
# For this activity, you will again use the `penguins` data from Seaborn.  You will target the two most important features to determining between `Adelie` and `Gentoo`.  

# In[3]:


penguins = sns.load_dataset('penguins').dropna()


# In[4]:


penguins.head()


# In[5]:


X = penguins.select_dtypes(['float'])
y = penguins.species


# In[6]:


sns.pairplot(data = penguins, hue = 'species')


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Fitting a model
# 
# To being, build a `DecisionTreeClassifier` with the parameter `max_depth = 1`.  Fit the model on the training data `X` and `y` and assign it to the variable `dtree` below.
# 
# **10 Points**
# 
# 

# In[9]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[10]:


### GRADED

dtree = DecisionTreeClassifier(max_depth=1).fit(X,y)

# Answer check
print(dtree)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Examining the Decision
# 
# To examine a basic text representation of the fit tree `dtree`, use the `export_text` function and set the argument `feature_names = list(X.columns)`.   Assign the result to `depth_1`.
# 
# **10 Points**

# In[14]:


### GRADED

depth_1 = export_text(dtree, feature_names=list(X.columns))

### ANSWER CHECK
print(depth_1)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Two Features
# 
# **10 Points**
# 
# Now to make it simplier only a subset of features `flipper_length_mm` and `bill_length_mm` are considered.
# 
# Below, instantiate a `DecisionTreeClassifier` instance with `max_depth = 2`. Fit this classifier to the data `X2` and `y`.
# 
# Next, use the function `export_text` with arguments `dtree` and `feature_names = list(X2.columns)` and assign your result to `tree2`
# 
# 
# <center>
#     <img src = 'images/p3.png' />
# </center>
# 
# 

# In[15]:


### GRADED

X2 = X[['flipper_length_mm', 'bill_length_mm']]
dtree2 = DecisionTreeClassifier(max_depth = 2).fit(X2, y)
tree2 = export_text(dtree2, feature_names = list(X2.columns))

### ANSWER CHECK
print(tree2)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Evaluating the tree
# 
# **10 Points**
# 
# Again, the default metric of the classifier is accuracy.  Evaluate the accuracy of the estimator `DecisionTreeClassifier` defined in the previous question and assign as a float to `acc_depth_2` below.  As you see there are a few points misclassified in the image of the decision boundaries.

# In[23]:


# accuracy_score?
# dtree2.score?


# In[30]:


### GRADED

acc_depth_2 = dtree2.score(X2, y)

### ANSWER CHECK
print(acc_depth_2)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### A Deeper Tree
# 
# **10 Points**
# 
# Finally, consider a tree with `max_depth = 3`.  Print the results and and use them to decide a prediction for the following penguin:
# 
# | flipper_length_mm | bill_length_mm |
# | ----------------- | -------------  |
# | 209 | 41.2 |
# 
# Assign your results as a string `Adelie`, `Chinstrap`, or `Gentoo` to `prediction` below.

# In[33]:


### GRADED

dtree3 = DecisionTreeClassifier(max_depth=3).fit(X2, y)
tree3 = export_text(dtree3, feature_names=list(X2.columns))
print(tree3)  # Print the tree to help understand the decision path

# Create a properly formatted test sample
X_test = pd.DataFrame({'flipper_length_mm': [209], 'bill_length_mm': [41.2]})

# Make prediction
prediction_array = dtree3.predict(X_test)
prediction = prediction_array[0]

# Answer check
print(prediction)


# In[49]:


get_ipython().run_line_magic('pinfo', 'plot_tree')


# In[47]:


# Feature names
feature_names = X2.columns

# Get the classes from the model in the correct order
class_names = list(dtree3.classes_)

plt.figure(figsize=(15, 10))
plot_tree(dtree3,  feature_names=feature_names, class_names=class_names, rounded=True, filled=True)


# In[48]:


# SADLY THIS GRAPHVIZ LIBRARY DOESN"T SEEM TO BE WORKING....

# !pip install graphviz


# In[43]:


# import graphviz
# from sklearn.tree import export_graphviz

# # Feature names
# feature_names = X2.columns

# # Get the classes from the model in the correct order
# class_names = list(dtree3.classes_)

# # Create the visualization
# dot_data = export_graphviz(
#     dtree3, 
#     out_file=None, 
#     feature_names=feature_names, 
#     class_names=class_names,
#     filled=True,
#     rounded=True,
#     special_characters=True
# )

# # Render the graph
# graph = graphviz.Source(dot_data)
# graph.render("penguin_tree", format="png", cleanup=True)
# display(graph)


# In[ ]:





# In[ ]:




