#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 20.1: Basic Aggregating of Models
# 
# This activity focuses on combining models in an ensemble to make predictions.  You will first create an ensemble on your own and then be introduced to the `VotingClassifier` from `scikit-learn` to implement these ensembles.  You will consider a classification problem and use Logistic Regression, KNN, and Support Vector Machines to build your ensemble.  
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)
# - [Problem 6](#-Problem-6)
# 
# 

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### The Data
# 
# 
# The data was retrieved from [kaggle](https://www.kaggle.com/) and contains information from fetal Cardiotocogram exams that were classified into three categories:
# 
# 
# - Normal
# - Suspect
# - Pathological
# 

# In[2]:


df = pd.read_csv('data/fetal.zip', compression = 'zip')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df['fetal_health'].value_counts()


# In[6]:


X = df.drop('fetal_health', axis = 1)
y = df['fetal_health']


# In[7]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Model Predictions
# 
# **10 Points**
# 
# Given the models below and the starter code, scale that data and train the models on the data, assigning the predictions as an array to the given dictionary.  

# In[8]:


models = [LogisticRegression(), KNeighborsClassifier(), SVC()]


# In[12]:


### GRADED
results = {'logistic': [],
          'knn': [],
          'svc': []}
# Get the keys as a list
keys = list(results.keys())
i = 0
X_train, X_test, y_train, y_test = train_test_split(X,y)
for model in models:
    #fit the model
    model.fit(X_train, y_train)
    #make predictions
    predictions = model.predict(X_test)
    #track predictions with predictions -- should have three 
    results[keys[i]] = predictions
    i += 1

### ANSWER CHECK
results


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Majority Vote
# 
# **10 Points**
# 
# Using your dictionary of predictions, create a DataFrame called `prediction_df` and add a column to the DataFrame named `ensemble_prediction` based on the majority vote of your predictions.

# In[15]:


import scipy.stats as stats
get_ipython().run_line_magic('pinfo', 'stats.mode')


# In[19]:


### GRADED
prediction_df = pd.DataFrame()
prediction_df['logistic'] = results['logistic']
prediction_df['knn'] = results['knn']
prediction_df['svc'] = results['svc']
preds_ensemple = np.array()
for i in results:
    preds_i = np.array()
    preds_i = [results['logistic'][i], results['knn'][i], results['svc'][i]]
    preds_ensemble.append(stats.mode(preds_i)
prediction_df['ensemble_prediction'] = stats.mode(preds_ensemple)

### ANSWER CHECK
prediction_df.head()


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# 
# #### Accuracy of Classifiers
# 
# **10 Points**
# 
# 
# Create a list of accuracy scores for each of the classifiers.  Use this list with the columns to create a DataFrame named `results_df` to hold the accuracy scores of the classifiers.  What rank was your ensemble?
# 

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


### GRADED
accuracies = []
#for col in prediction_df.columns:
    # put your answer here
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
accuracies


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Using the Voting Classifier
# 
# **10 Points**
# 
# Use the documentation and User Guide [here](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier) to create a voting ensemble using the `VotingClassifier` based on the majority vote using the same three classifiers `svc`, `lgr`, and `knn`.  Assign the accuracy of the ensemble to `vote_accuracy` below.

# In[ ]:


### GRADED
voter = ''
vote_accuracy = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
vote_accuracy


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Voting based on probabilities
# 
# **10 Points**
# 
# Consult the user guide and create a new ensemble that makes predictions based on the probabilities of the estimators.  **HINT**: This has to do with the `voting` parameter.  Assign the ensemble as `soft_voter` and the accuracy as `soft_accuracy`. 

# In[ ]:


### GRADED
soft_voter = ''
soft_accuracy = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
soft_accuracy


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 6
# 
# #### Using different weights 
# 
# **10 Points**
# 
# Finally, consider weighing the classifiers differently.  Use the Logistic Regression estimator as .5 of the weight in predicting based on majority votes, and the SVC and KNN as 0.25 each.  Assign the accuracy of these predictions on the test data to `weighted_acc`.  

# In[ ]:


### GRADED
weighted_voter = ''
weighted_score = ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
weighted_score


# In[ ]:





# In[ ]:




