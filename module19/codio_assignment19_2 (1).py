#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 19.2: Using SURPRISE
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 50**
# 
# This activity focuses on using the `Surprise` library to predict user ratings.  You will use a dataset derived from the movieLens data -- a common benchmark for recommendation algorithms.  Using `Surprise` you will load the data, create a train set and test set, make predictions for a test set, and cross validate the model on the dataset. 
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# In[17]:


from surprise import Dataset, Reader, SVD
import pandas as pd
import numpy as np

# DOCUMENTATION FOR SURPRISE
# https://surprise.readthedocs.io/en/stable/getting_started.html


# ### The Data
# 
# The data is derived from the MovieLens data [here](https://grouplens.org/datasets/movielens/).  The original dataset has been sampled so the processing is faster.
# 
# The dataframe contain information about the user, movie, and the associated ratings when they exist.

# In[8]:


movie_ratings = pd.read_csv('data/movie_ratings.csv', index_col=0)


# In[9]:


movie_ratings.head()


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Loading a Dataset
# 
# **10 Points**
# 
# Extract the columns `userId`, `title`, and `rating` from the `movie_ratings` dataframe and assign them to the variable `a`.
# 
# Initialize a `Reader` object, specifying that the ratings are on a scale from 0 to 5 and assign this result to `reader `. Next, use the `Dataset` object to convert the selected dataframe `a` into the format expected by `Surprise` using the `reader` object. Assign this result to `sf`.
# 
# Finally, use the `build_full_trainset` function on `sf` to build the full training set from the dataset, making it ready for training a recommendation algorithm. Assign this result to `train`.
# 

# In[6]:


# Reader?
# Dataset?


# In[11]:


### GRADED

# 1. Select only the required columns
# 2. Reorder them to match Surprise's expectations (user, item, rating)
surprise_df = movie_ratings[['userId', 'movieId', 'rating']]

# Create a Reader object
reader = Reader(rating_scale=(0,5))

# Load the data into Surprise
sf = Dataset.load_from_df(surprise_df, reader)

# Build train set
train = sf.build_full_trainset()

### ANSWER CHECK
print(type(sf))
print(type(train))


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Instantiate the `SVD` model
# 
# **10 Points**
# 
# Below, create an `SVD` object with 2 factors and assign it as `model` below.

# In[13]:


# SVD?


# In[14]:


### GRADED

## THIS IS FOR THE SAKE OF EXERCISE ONLY NORMALLY YOU'D USE 50-200 n_factors !!!!!!!)
model = SVD(n_factors=2)

### ANSWER CHECK
print(model.n_factors)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# ### Fitting the Model
# 
# **10 Points**
# 
# Below, fit `model` on the training data `train`. 

# In[15]:


### GRADED
#fit your model below. No variable needs to be assigned.
model.fit(train)

### ANSWER CHECK
print(model)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# ### Making Predictions
# 
# **10 Points**
# 
# Use the `build_testset` function on `train` to build a testset named `test`. Next, use `test` to create a list of predictions for the testset.  Assign the result to `predictions_list` below.

# In[23]:


### GRADED

## NOTE THIS SEEMS WRONG SHOULDN"T IT BE BUILDING A TEST SET ON sf (the whole dataset) ????
test = train.build_testset()
predictions_list = []
for uid, iid, _ in test:
    prediction = model.predict(uid, iid)
    predictions_list.append(prediction)

### ANSWER CHECK
print(predictions_list[:5])


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Cross Validate the Model
# 
# **10 Points**
# 
# You may use the test data to evaluate the model, as well as also cross validate the model using the data object `sf`. 
# 
# In the code cell below, use the `cross_validate` function to calculate the RMSE of the model. Assign the result to `cross_val_results` below. 

# In[24]:


from surprise.model_selection import cross_validate


# In[25]:


### GRADED
cross_val_results = cross_validate(model, sf, measures=['RMSE'])

### ANSWER CHECK
print(cross_val_results)


# In[ ]:





# In[ ]:




