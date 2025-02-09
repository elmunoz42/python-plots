#!/usr/bin/env python
# coding: utf-8

# ### Codio Assignment 12.1: Identifying the Best K
# 
# This activity focuses on identifying the "best" number of neighbors that optimize the accuracy of a `KNearestNeighbors` estimator. The ideal number of neighbors will be selected through cross-validation and a grid search over the `n_neighbors` parameter.  Again, before building the model, you will want to scale the data in a `Pipeline`.
# 
# **Expected Time: 60 Minutes**
# 
# **Total Points: 50**
# 
# #### Index
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)
# - [Problem 6](#Problem-6)
# 

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline


# ### The Dataset
# 
# Again, you will use the credit default dataset to predict default -- yes or no.  The data is loaded and split into train and test sets for you below.  You will again build a column transformer to encode the `student` feature.  Note that scikit-learn handles a string target features in the `KNeighborsClassifier`, and we do not need to encode this column.

# In[31]:


df = pd.read_csv('data/default.csv', index_col=0)


# In[32]:


df.head(2)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('default', axis = 1), 
                                                    df['default'],
                                                   random_state=42)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Baseline for Models
# 
# **5 Points**
# 
# Before starting the modeling process, you should have a baseline to determine whether your model is any good. 
# 
# Consider the `default` column of `df`. Perform a `value_counts` operation with the argument `normalize` equal to `True`. 
# 
# What would the accuracy of such a classifier be?  Enter your answer as a float to `baseline` below.
# 
# 

# In[34]:


### GRADED

baseline = df['default'].value_counts(normalize=True)['No']

# Answer check
print(baseline)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Column transforms and KNN
# 
# **10 Points**
# 
# Use the `make_column_transformer` to create a column `transformer`. Inside the `make_column_transformer` specify an instance of the `OneHotEncoder` transformer from scikit-learn. Inside `OneHotEncoder` set `drop` equal to `'if_binary'`. Apply this transformation to the `student` column. On the `remainder` columns, apply a `StandardScaler()` transformation.
#  
# 
# Next, build a `Pipeline` named `knn_pipe` with  steps `transform` and `knn`. Set `transform` equal to `transformer` and `knn` equal to `KNeighborsClassifier()`. Be sure to leave all the settings in `knn` as the default.  

# In[46]:


### GRADED

remainder = ['balance', 'income']

transformer = make_column_transformer(
    (OneHotEncoder(drop='if_binary'), ['student']),
    (StandardScaler(), remainder) 
)

# Create pipeline
knn_pipe = Pipeline([
    ('transform', transformer),
    ('knn', KNeighborsClassifier())
])

# Answer check
knn_pipe


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Parameter grid
# 
# **10 Points**
# 
# Now that your pipeline is ready, you are to construct a parameter grid to search over.  Consider two things:
# 
# - You will not be able to predict on a test dataset where `n_neigbors > len(test_data)`.  This will limit our upper bound on `k`.  In this example, too high a `k` will slow down the computation, so only consider `k = [1, 3, 5, ..., 21]`. 
# - Ties in voting are decided somewhat arbitrarily and for speed and clarity you should consider only odd values for number of neighbors
# 
# Creating a dictionary called `params` that specifies hyperparameters for the KNN classifier. 
# 
# - The key of your dictionary will be `knn__n_neighbors`
# - The values in your dictionary will be `list(range(1, 22, 2))`
# 
# 

# In[47]:


### GRADED
k = list(range(1,22,2))
params = {'knn__n_neighbors': k}

# Answer check
list(params.values())[0]

print(X_train.columns)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Grid search `k`
# 
# **10 Points**
# 
# - Use `GridSearchCV` with the `knn_pipe` and `param_grid` equal to `params`. Assign the result to `knn_grid`.
# - Use the `fit` function on `knn_grid` to train your model on `X_train` and `y_train`.
# - Retrieve the best value for the hyperparameter `k` from the `best_params_` attribute of the grid search object `knn_grid`. Assign the result to `best_k`.
# - Use the `score` function to calculate the accuracy of the `knn_grid` classifier on a test dataset. Assign your best models accuracy on the test data as a float to `best_acc`
# 
# 

# In[48]:


### GRADED
knn_grid = GridSearchCV(knn_pipe, param_grid=params).fit(X_train, y_train)
best_k = knn_grid.best_params_
best_acc = knn_grid.score(X_test,y_test)

# Answer check
print(best_acc)
print(best_k)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Other parameters to consider
# 
# **10 Points**
# 
# The number of neighbors is not the only parameter in the implementation from scikit-learn.  For example, you can also consider different weightings of points based on their distance, change the distance metric, and search over alternative versions of certain metrics like Minkowski.  See the docstring from `KNeighborsClassifier` below. 
# 
# ```
# weights : {'uniform', 'distance'} or callable, default='uniform'
#     Weight function used in prediction.  Possible values:
# 
#     - 'uniform' : uniform weights.  All points in each neighborhood
#       are weighted equally.
#     - 'distance' : weight points by the inverse of their distance.
#       in this case, closer neighbors of a query point will have a
#       greater influence than neighbors which are further away.
#     - [callable] : a user-defined function which accepts an
#       array of distances, and returns an array of the same shape
#       containing the weights.
#       
# ===========================
# 
# p : int, default=2
#     Power parameter for the Minkowski metric. When p = 1, this is
#     equivalent to using manhattan_distance (l1), and euclidean_distance
#     (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
#     
# ```
# 
# Create a new parameter grid and consider both weightings as well as `p = [1, 2]`.  Assign this as a dictionary to `params2` below.  
# 
# Search over these parameters in your `knn_pipe` with a `GridSearchCV` named `weight_grid` below. Also, consider `n_neighbors` as in [Problem 4](#-Problem-4).  Did your new grid search results perform better than earlier?  Assign this grid's accuracy to `weights_acc` below.

# In[56]:


### GRADED

params2 = {
    'knn__weights': ['uniform', 'distance'],  # Added 'knn__' prefix
    'knn__p': [1, 2]                         # Added 'knn__' prefix
}

weight_grid = GridSearchCV(knn_pipe, param_grid=params2).fit(X_train, y_train)
weights_acc = weight_grid.score(X_test, y_test)

# Answer check
print(weights_acc)


# In[57]:


get_ipython().run_line_magic('pinfo', 'KNeighborsClassifier')


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 6
# 
# #### Further considerations
# 
# **5 Points**
# 
# When performing your grid search you want to also be sensitive to the amount of parameters you are searching and the number of different models being built.  How many models were constructed in [Problem 5](#-Problem-5)?  Enter your answer as an integer to `ans6` below.  You might use the grids `.cv_results_` attribute to determine this.

# In[59]:


### GRADED

ans6 = len(weight_grid.cv_results_['params'])

# Answer check
print(ans6)


# In[ ]:





# In[60]:


def print_cv_results(grid):
    results = grid.cv_results_
    
    print("\nPARAMETER COMBINATIONS TESTED:")
    print("-" * 30)
    for i, params in enumerate(results['params']):
        print(f"\nModel {i+1}:")
        print(f"Parameters: {params}")
        print(f"Mean Test Score: {results['mean_test_score'][i]:.4f}")
        print(f"Standard Deviation: {results['std_test_score'][i]:.4f}")
        print(f"Rank: {results['rank_test_score'][i]}")
        print(f"Average Fit Time: {results['mean_fit_time'][i]:.4f} seconds")

# Use it on your grid search results
print_cv_results(weight_grid)


# In[ ]:




