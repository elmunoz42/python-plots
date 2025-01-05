#!/usr/bin/env python
# coding: utf-8

# ## Required Codio Assignment 6.3: Running PCA with Clustering
# 
# **Expected Time = 120 minutes**
# 
# **Total Points = 36**
# 
# Now that you've seen how to use PCA to reduce the dimensionality of data while maintaining important information, it is time to see how we can use these ideas applied to a real dataset. In this activity you will use a dataset related to marketing campaigns with the task being to identify groups of similar customers.  Once the cluster labels are assigned, you will briefly explore inside of each cluster for patterns that help identify characteristics of customers.
# 
# ## Index:
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)
# - [Problem 6](#Problem-6)
# - [Problem 7](#Problem-7)
# - [Problem 8](#Problem-8)
# - [Problem 9](#Problem-9)
# 

# In[8]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings


# In[9]:


warnings.filterwarnings("ignore")


# ### The Dataset
# 
# More information on the dataset can be found [here](https://www.kaggle.com/imakash3011/customer-personality-analysis).  Below the data is loaded, the info is displayed, describe the continuous features, and the first five rows of the data are displayed.

# In[10]:


df = pd.read_csv('data/marketing_campaign.csv', sep = '\t')


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.head()


# [Back to top](#Index:)
# 
# ## Problem 1
# 
# ### Preparing the Data
# 
# **4 Points**
# 
# Before starting to build cluster models, the data needs to be limited to numeric representations.  How many non-numeric columns are there, and what are their names?  Assign you solution as a list of strings to `object_cols` below.  The names should match the column names in the DataFrame exactly.  

# In[14]:


### GRADED
print(df.info())

object_cols = ['Education','Marital_Status','Dt_Customer']

# Answer check
print(object_cols)
print(type(object_cols))


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 2
# 
# ### Dropping the `object` columns 
# 
# **4 Points**
# 
# To simplify things, eliminate the columns containing `object` datatypes.  Assign your new DataFrame to `df_numeric` below.

# In[15]:


### GRADED

df_numeric = df.drop(object_cols, axis=1)

# Answer check
print(df_numeric.shape)
df_numeric.info()


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 3
# 
# ### Dropping non-informative columns
# 
# **4 Points**
# 
# Two columns, `Z_CostContact`, and `Z_Revenue` have one unique value. Also, the `ID` column is basically an index. These will not add any information to our problem. Drop the columns `Z_CostContact`, `Z_Revenue`, and `ID` and save your all numeric data without these two columns as a DataFrame to `df_clean` below.

# In[16]:


### GRADED

df_clean = df_numeric.drop(['Z_CostContact','Z_Revenue','ID'], axis=1)

# Answer check
print(df_clean.shape)
df_clean.info()


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 4
# 
# ### Dropping the missing data
# 
# **4 Points**
# 
# Note that the `Income` column is missing data.  This will cause issues for `PCA` and clustering algorithms.  Drop the missing data using pandas `.dropna` method on `df_clean`, and assign your non-missing dataset as a DataFrame to `df_clean_nona` below. 

# In[17]:


### GRADED

df_clean_nona = df_clean.dropna()

# Answer check
print(df_clean_nona.shape)
df_clean_nona.info()


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 5
# 
# ### Scaling the Data
# 
# **4 Points**
# 
# As earlier with the PCA models, the data needs to be mean centered.  
# 
# 
# Below, scale the `df_clean_nona` by subtracting its mean and by dividing it by its standard deviation.  Assign your results as a DataFrame to `df_scaled` below.  

# In[18]:


### GRADED

df_scaled = (df_clean_nona - df_clean_nona.mean()) / df_clean_nona.std()

# Answer check
print(df_scaled.shape)
print(type(df_scaled))


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 6
# 
# ### PCA
# 
# **4 Points**
# 
# With the data cleaned and scaled, you are ready to perform PCA.  Below, use the `PCA` transformer from `sklearn` to transform your data and select the top three principal components.  First, create an instance of the `PCA` that limits the number of components to 3 using the `n_components` argument.  Also, set the argument `random_state = 42`  and assign your instance as `pca` below.

# In[19]:


### GRADED

pca = PCA(n_components=3, random_state=42)

# Answer check
print(pca)
print(pca.n_components)


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 7
# 
# ### Extracting the Components
# 
# **4 Points**
# 
# Use the `.fit_transform` method with argument equal to `df_scaled` on `pca` to extract the three principal components.  Save these components as an array to the variable `components` below.  

# In[20]:


### GRADED

components = pca.fit_transform(df_scaled)

# Answer check
print(type(components))
print(components.shape)


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 8
# 
# ### `KMeans`
# 
# **4 Points**
# Complete the code below according to the instructions below:
# 
# 
# - To the `kmeans` variable, assign the `KMeans` clusterer with argument `n_clusters` equal to `3` and argument `random_state` equal to `42`. To this, chain the `fit()` method with argument equal to `components`.
# - Copy the code line that reads the data  in your solution code.
# - Copy the code to drop the missing value in your solution. Here, inside the `dropna()` function, set the argument `subset` equal to `['Income']`.
# - Inside `df_clustered`, create a new column `cluster`. To this column, assign `kmeans.labels_`.
# 

# In[21]:


### GRADED

kmeans = KMeans(n_clusters=3, random_state=42).fit(components)
df = pd.read_csv('data/marketing_campaign.csv', sep = '\t')

# Drop rows where 'Income' is NaN
df_clustered = df.dropna(subset=['Income'])

# Assign cluster labels
df_clustered['cluster'] = kmeans.labels_

# Answer check
print(type(df_clustered))
print(df_clustered.shape)


# In[ ]:





# [Back to top](#Index:)
# 
# ## Problem 9
# 
# ### Examining the Results
# 
# **4 Points**
# 
# The image below shows a `boxenplot` of the clusters based on amounts spent on meat products.  If you were marketing a meat sale and there is a cost involved in advertisiting per customer.  If you were to select only one cluster to market to, which cluster would you target? Assign your response as an integer to `target_cluster` below.
# 
# ![](images/meats.png)

# In[22]:


### GRADED

target_cluster = 1

# Answer check
print(type(target_cluster))
print(target_cluster)


# In[ ]:





# While this is a start, there is much more work to be done.  We glossed over perhaps one of the most important parts of the task -- feature engineering.  Some of the columns that were objects could be represented numerically.  Also, we could try different numbers of components from PCA and numbers of clusters.  In a business setting, it is important to keep the number of clusters small so that the groups can be distinguished in meaningful ways, so we don't want to let the number of clusters get too large.  

# In[ ]:




