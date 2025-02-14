#!/usr/bin/env python
# coding: utf-8

# ### Codio Assignment 12.2: Decision Boundaries
# 
# **Estimated Time: 60 Minutes**
# 
# **Total Points: 55**
# 
# This activity focuses on the effect of changing your decision threshold and the resulting predictions.  Again, you will use the `KNeighborsClassifier` ,but this time, you will explore the `predict_proba` method of the fit estimator to change the thresholds for classifying observations.  You will explore the results of changing the decision threshold on the false negative rate of the classifier for the insurance data.  Here, we suppose the important thing is to not make the mistake of predicting somebody would not default when they really do.  
# 
# #### Index
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)
# - [Problem 6](#Problem-6)

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn import set_config

set_config(display="diagram")


# ### The Dataset
# 
# You continue to use the default example, and the data is again loaded and split for you below. 

# In[3]:


default = pd.read_csv('data/default.csv')


# In[4]:


default.head(2)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(default.drop('default', axis = 1), 
                                                    default['default'],
                                                   random_state=42)


# In[6]:


transformer = make_column_transformer((OneHotEncoder(drop = 'if_binary'), ['student']),
                                     remainder = StandardScaler())


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Basic Pipeline
# 
# **10 Points**
# 
# Use the `Pipeline` function to create a pipeline `base_pipe` with steps `transformer` and `knn`. Assign `transformer` to `'transformer'` and assign a `KNeighborsClassifier()` with `n_neighbors = 10` to `'knn'`. 

# In[7]:


### GRADED

base_pipe = Pipeline([
    ('transformer', transformer),
    ('knn', KNeighborsClassifier(n_neighbors = 10))
])

# Answer check
base_pipe


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Accuracy of KNN with 50% probability boundary
# 
# **10 Points**
# 
# - Use the `fit` function to train `base_pipe` on `X_train` and `y_train`.
# - Use the `score` function to calculate the performance of `base_pipe` on the test sets. Assign the result to `base_acc`.
# - Use the `predict` function on `base_pipe` to make predictions on `X_test`. Assign the reusl to `preds`.
# - Initialize the `base_fn` variable to `0`.
# - Use a `for` loop to loop over `zip(preds, y_test)`. Inside the `for` loop:
#     - Use an `if` block to determine the accuracy for this default setting and assign it to `base_acc`. Also, consider the proportion of false negatives here.  Assign these as `base_fn`.  

# In[33]:


### GRADED
# Fit the pipeline
base_pipe.fit(X_train, y_train)
# Get accuracy score
base_acc = base_pipe.score(X_test, y_test)
# Get predictions 
preds = base_pipe.predict(X_test)
def calculate_false_negative_proportion(preds, y_test):
    """
    Calculate the number and proportion of false negatives in a binary classification model.
    
    A false negative occurs when the model predicts 'No' when the actual value
    is 'Yes'.
    """
    # Initialize counter inside function
    fn_count = 0
    total_count = len(preds)
    
    for pred, actual in zip(preds, y_test):
        # False negative is when we predict 0 (no default) but actual is 1 (default)
        if pred == 'No' and actual == 'Yes':
            fn_count += 1  
    
    return fn_count, fn_count / total_count

# Calculate false negative proportion
base_fn, base_fn_proportion  = calculate_false_negative_proportion(preds, y_test)
        
# Answer check
print(base_acc)
print(base_fn)
print(base_fn_proportion)


# In[24]:





# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Prediction probabilities
# 
# **10 Points**
# 
# As demonstrated in Video 12.5, your fit estimator has a `predict_proba` method that will output a probability for each observation.  
# 
# 
# Use the `predict_proba` function on `base_pipe` to predict the probabilities on `X_test`. Assign the predicted probabilities as an array using the test data to `base_probs` below. 

# In[19]:


### GRADED

base_probs = base_pipe.predict_proba(X_test)

# Answer check
pd.DataFrame(base_probs[:5], columns = ['p_no', 'p_yes'])


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### A Stricter `default` estimation
# 
# **10 Points**
# 
# As discussed in the previous assignment, if you aim to minimize the number of predictions that miss default observations you may consider increasing the probability threshold to make such a classification.  Accordingly, use your probabilities from the last problem to only predict 'No' if you have a higher than 70% probability that this is the label.  Assign your new predictions as an array to `strict_preds`.  Determine the number of false negative predictions here and assign them to `strict_fn` below.  

# In[25]:


### GRADED
# Get first column probabilities (probability of "No" default)
no_default_probs = base_probs[:, 0]

# Make strict predictions using 70% threshold
strict_preds = np.where(no_default_probs > 0.7, 'No', 'Yes')
# This means: if probability of "No" > 0.7, predict 0 (No default)
#            otherwise predict 1 (Yes default)

# Use your false negative calculation function from before
strict_fn, strict_fn_proportion = calculate_false_negative_proportion(strict_preds, y_test)

# Answer check
print(strict_fn)
print(strict_fn_proportion)


# In[39]:


# Make strict predictions using 70% threshold
stricter_preds = np.where(no_default_probs > 0.9, 'No', 'Yes')
# This means: if probability of "No" > 0.9, predict 0 (No default)
#            otherwise predict 1 (Yes default)

# Use your false negative calculation function from before
stricter_fn, stricter_fn_proportion = calculate_false_negative_proportion(stricter_preds, y_test)
print(stricter_fn)
print(stricter_fn_proportion)


# In[ ]:





# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Minimizing False Negatives
# 
# **10 Points**
# 
# Consider a 50%, 70%, and 90% decision boundary for predicting "No".  Which of these minimizes the number of false negatives?  Assign your solution as an integer -- 50, 70, or 90 -- to `ans5` below.
# 
# 

# In[35]:


### GRADED

ans5 = 90

# Answer check
print(ans5)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 6
# 
# #### Visualizing decision boundaries
# 
# **5 Points**
# 
# For this exercise, a visualization of the decision boundary using a synthetic dataset is created and plotted below.  Which of these would you choose to minimize the number of false negatives?  Enter your choice as an integer -- 1, 20, or 50 -- to `ans6` below.
# 
# <center>
#     <img src = images/dbounds.png />
# </center>

# In[ ]:


### GRADED

ans6 = 1

# Answer check
print(ans6)


# In[ ]:





# In[ ]:




