#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 8.3: Evaluating Multiple Models
# 
# **Estimated Time: 120 Minutes**
# 
# **Total points: 100**
# 
# This assignment focuses on solving a specific regression problem using basic cross-validation with a train/test/validation split.  In addition to using the methods explored, this assignment also aims to familiarize you with further utilities for data transformation including, the `OneHotEncoder` and `OrdinalEncoder` along with their use in a `make_column_transformer`.  
# 
# The operations of encoding categorical features will be introduced using `sklearn`.  This will allow you to streamline your model-building pipelines.  Depending on whether a string type feature is **ordinal** or **categorical** we want to encode differently.  The `OrdinalEncoder` will be used to encode features that do not need to be binarized due to an underlying order, and `OneHotEncoder` for categorical features (as a similar approach to that of the `.get_dummies()` method in pandas).  By the end of the assignment, you will see how to chain multiple feature encoding methods together, including the earlier `PolynomialFeatures` for numeric features. 
# 
# <center>
#     <img src = images/pipes.png width = 50% />
# </center>

# #### Index
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
# - [Problem 10](#Problem-10)
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, make_column_selector

from sklearn import set_config

set_config(display="diagram") #setting this will display your pipelines as seen above


# ### The Data: Ames Housing
# 
# This dataset is a popular beginning dataset used in teaching regression.  The task is to use specific features of houses to predict the price of the house.  In addition to this, as discussed in video 8.10 -- this dataset is available for use in an ongoing competition where you can use the `test.csv` to submit your model's predictions.  Accordingly, the two data files are identical with the exception of the `test.csv` file not containing the target feature.
# 
# The data contains 81 columns of different information on the individual houses and their sale price.  A full description of the data is attached [here](data/data_description.txt).  In this assignment, you will use a small subset of the features to begin modeling with that includes ordinal, categorical, and numeric features. As an optional exercise, you are encouraged to continue engineering additional features and attempt to improve the performance of your model including submitting the predictions on Kaggle. 

# In[2]:


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[3]:


train.info()


# In[4]:


train.head()


# In[5]:


#note the difference in one column from train to test
[i for i in train.columns if i not in test.columns]


# [Back to top](#Index:) 
# 
# ### Problem 1
# 
# #### Train/Test split
# 
# **5 Points**
# 
# Despite having a test dataset, you want to create a holdout set to assess your model's performance.  To do so, use sklearn's `train_test_split` to split `X` and `y` with arguments:
# 
# - `test_size = 0.3`
# - `random_state = 22`
# 
# Assign your results to `X_train, X_test, y_train, y_test`.
# 

# In[6]:


X = train.drop('SalePrice', axis = 1)
y = train['SalePrice']


# In[7]:


### GRADED

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=22)

# Answer check
print(X_train.shape)
print(X_test.shape)
print(type(X_train), type(y_train))#should be DataFrame and Series


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 2
# 
# #### Baseline Predictions
# 
# **10 Points**
# 
# Before building a regression model, you should set a baseline to compare your later models to.  One way to do this is to guess the mean of the `SalePrice` column.  For the variables `baseline_train` and `baseline_test`, create arrays of same shape as `y_train` and `y_test` respectively.  The variable `baseline_train` should contain `y_train.mean()`. The variable `baseline_test` should contain `y_test.mean()`.
# 
# 
# Use the  `mean_squared_error` function to calculate the error between `baseline_train` and `y_train`, Assign the result to `mse_baseline_train`.
# 
# Use the  `mean_squared_error` function to calculate the error between `baseline_test` and `y_test`, Assign the result to `mse_baseline_test`.
# 

# In[8]:


### GRADED

# Create arrays filled with the mean value
baseline_train = np.full(shape=y_train.shape, fill_value=y_train.mean())
baseline_test = np.full(shape=y_test.shape, fill_value=y_test.mean())
# Calculate MSE
mse_baseline_train = mean_squared_error(baseline_train, y_train)
mse_baseline_test = mean_squared_error(baseline_test, y_test)

# Answer check
print(baseline_train.shape, baseline_test.shape)
print(f'Baseline for training data: {mse_baseline_train}')
print(f'Baseline for testing data: {mse_baseline_test}')


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 3
# 
# #### Examining the Correlations
# 
# **5 Points**
# 
# What feature has the highest positive correlation with `SalePrice`?  Assign your answer as a string matching the column name exactly to `highest_corr` below.  

# In[9]:


### GRADED

def find_highest_correlation_against_feature(data_frame, feature):
    # Compute the correlation matrix
    correlation_matrix = data_frame.corr()

    # Get the correlation values for a given feature
    feature_correlation = correlation_matrix[feature]

    # Drop the original feature to avoid self-correlation
    feature_correlation = feature_correlation.drop(feature)

    # Sort the correlations
    sorted_correlation = feature_correlation.sort_values(ascending=False)
    
    # Return the feature name of the highest correlating feature and the correlation value
    return sorted_correlation.index[0], sorted_correlation.iloc[0]

highest_corr_name, highest_corr_value = find_highest_correlation_against_feature(train, 'SalePrice')
highest_corr = highest_corr_name
# Answer check
print(highest_corr)


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 4
# 
# #### Simple Model
# 
# **10 Points**
# 
# Complete the code below according to the instructions below:
# 
# - Define a variable `X1` and assign to it the values in the column `OverallQual`.
# - Instantiate a `LinearRegression` model and use the `fit` function to train it using `X1` and `y_train`. Assing your result to `lr`.
# - Use the  `mean_squared_error` function to calculate the error between `y_train` and `lr.predict(X1)`. Assign the result to `model_1_train_mse`.
# - Use the  `mean_squared_error` function to calculate the error between `y_test` and `lr.predict(X_test[['OverallQual']]`. Assign the result to `model_1_test_mse`.

# In[10]:


### GRADED
X1 = X_train[['OverallQual']]
lr = LinearRegression()
lr.fit(X1, y_train)
model_1_train_mse = mean_squared_error(y_train, lr.predict(X1))
model_1_test_mse = mean_squared_error(y_test, lr.predict(X_test[['OverallQual']]))

# Answer check
print(f'Train MSE: {model_1_train_mse: .2f}')
print(f'Test MSE: {model_1_test_mse: .2f}')


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 5
# 
# #### Using `OneHotEncoder`
# 
# **10 Points**
# 
# Similar to the `pd.get_dummies()` method earlier encountered, scikit-learn has a utility for encoding categorical features in the same way.  Below, the `OneHotEncoder` is demonstrated in the `CentralAir` column.  You are to use these results to build a model where the only feature is the `CentralAir` column.  Note the two arguments are used in the `OneHotEncoder`:
# 
# - `sparse = False`: returns an array that we can investigate vs with `sparse = True` you are returned a sparse matrix -- a memory saving representation
# - `drop = if_binary`: returns a single column for any binary categories.  This avoids redundant features in our regression model.
# 
# In the code cell below, instantiate a `LinearRegression` model and use the `fit` function to train it using `model_2_train` and `y_train`. Assing your result to `model_2`. 

# In[15]:


#extract the features
central_air_train = X_train[['CentralAir']]
central_air_test = X_test[['CentralAir']]


# In[16]:


#a categorical feature
central_air_train.head()


# In[17]:


#Instantiate a OHE object
#sparse = False returns an array so we can view
ohe = OneHotEncoder(sparse = False, drop='if_binary')
print(ohe.fit_transform(central_air_train)[:5])


# In[18]:


model_2_train = ohe.fit_transform(central_air_train)
model_2_test = ohe.transform(central_air_test)


# In[19]:


### GRADED

lr2 = LinearRegression()
lr2.fit(model_2_train, y_train)
model_2 = lr2

# Answer check
print(model_2.coef_)


# In[ ]:





# 
# 
# To build a model using both the `OverallQual` column and the `CentralAir` column, you could use the `OneHotEncoder` to transform `CentralAir`, and then concatenate the results back into a DataFrame or numpy array.  To streamline this process, the `make_column_transformer` can be used to seperate specific columns for certain transformations.  Below, a `make_column_transformer` has been created for you to do just this.  
# 
# 
# The arguments are tuples of the form `(transformer, columns)` that specify a transformation to perform on the given column.  Further, the `remainder = passthrough` argument says to just pass the other columns through.  You are returned a numpy array with the `CentralAir` column binarized and concatenated to the `OverallQual` feature.
# 
# 
# For an example using the `make_column_transformer` see [here](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py).
# 

# In[21]:


col_transformer = make_column_transformer((OneHotEncoder(drop = 'if_binary'), ['CentralAir']), 
                                          remainder='passthrough')


# In[22]:


col_transformer.fit_transform(X_train[['OverallQual', 'CentralAir']])


# [Back to top](#Index:) 
# 
# ### Problem 6
# 
# #### Using `make_column_transformer`
# 
# **10 Points**
# 
# 
# Complete the code below according to the instructions below:
# 
# 
# - Use `Pipeline` to create a pipeline object. Inside the pipeline object, define a a tuple where the first element is a string identifier `col_transformer` and the second element is an instance of `col_transformer`. Inside the pipeline define another tuple where the first element is a string identifier `linreg`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `pipe_1`.
# - Use the `fit` function on `pipe_1` to train your model on `X_train[['OverallQual', 'CentralAir']]` and `y_train`. 

# In[23]:


### GRADED

pipe_1 = Pipeline([
    ('col_transformer', col_transformer),
    ('lin_reg', LinearRegression())
])
pipe_1.fit(X_train[['OverallQual','CentralAir']], y_train)
# Answer check
print(pipe_1.named_steps)#col_transformer and linreg should be keys
pipe_1


# In[ ]:





# 
# 
# Not all columns warrant binarization as done on the `CentralAir` column.  For example, consider the `HeatingQC` feature -- representing the quality of the heating in the house.  From the data description, the unique values are described as:
# 
# ```
# HeatingQC: Heating quality and condition
# 
#        Ex	Excellent
#        Gd	Good
#        TA	Average/Typical
#        Fa	Fair
#        Po	Poor
# ```

# These are ordered values, and rather than binarizing them a numeric value representing the scale can be used.  For example, using a scale of 0 - 4 you may associate the categories with an order in a list from least to greatest as:
# 
# ```
# ['Po', 'Fa', 'TA', 'Gd', 'Ex']
# ```
# 
# Creating an `OrdinalEncoder` with these categories will transform the `HeatingQC` feature mapping each category as
# 
# ```
# Po: 0
# Fa: 1
# TA: 2
# Gd: 3
# Ex: 4
# ```
# 
# This is demonstrated below, and in a similar manner, the use of the `make_column_transformer` is shown using the three columns `['OverallQual', 'CentralAir', 'HeatingQC']`, applying the appropriate transformations to each column and passing the remaining numeric feature through.  

# In[24]:


oe = OrdinalEncoder(categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']])


# In[25]:


oe.fit_transform(X_train[['HeatingQC']])


# In[26]:


X_train['HeatingQC'].head()


# In[27]:


ordinal_ohe_transformer = make_column_transformer((OneHotEncoder(drop = 'if_binary'), ['CentralAir']),
                                          (OrdinalEncoder(categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ['HeatingQC']),
                                          remainder='passthrough')


# In[28]:


ordinal_ohe_transformer.fit_transform(X_train[['OverallQual', 'CentralAir', 'HeatingQC']])[:5]


# In[29]:


X_train[['OverallQual', 'CentralAir', 'HeatingQC']].head()


# [Back to top](#Index:) 
# 
# ### Problem 7
# 
# #### Using `OrdinalEncoder`
# 
# **10 Points**
# 
# 
# Complete the code below according to the instructions below:
# 
# 
# - Use `Pipeline` to create a pipeline object. Inside the pipeline object define a tuple where the first element is a string identifier `transformer` and the second element is an instance of `ordinal_ohe_transformer`. Inside the pipeline define another tuple where the first element is a string identifier `linreg`, and the second element is an instance of `LinearRegression`. Assign the pipeline object to the variable `pipe_2`.
# - Use the `fit` function on `pipe_2` to train your model on `X_train[['OverallQual', 'CentralAir', 'HeatingQC']]` and `y_train`. 
# - Use the `predict` function on `pipe_2` to make your predictions of `X_train[['OverallQual', 'CentralAir', 'HeatingQC']]`. Assign the result to `pred_train`.
# - - Use the `predict` function on `pipe_2` to make your predictions of `X_test[['OverallQual', 'CentralAir', 'HeatingQC']]`. Assign the result to `pred_test`.
# - Use the `mean_squared_error` function to calculate the MSE between `y_train` and `pred_train`. Assign the result to `pipe_2_train_mse`.
# - Use the `mean_squared_error` function to calculate the MSE between `y_test` and `pred_test`. Assign the result to `pipe_2_test_mse`.

# In[ ]:


### GRADED

pipe_2 = Pipeline([
    ('transformer', ordinal_ohe_transformer),
    ('linreg', LinearRegression())
])
pipe_2.fit(X_train[['OverallQual', 'CentralAir', 'HeatingQC']], y_train)
pred_train = pipe_2.predict(X_train[['OverallQual', 'CentralAir', 'HeatingQC']])
pred_test = pipe_2.predict(X_test[['OverallQual', 'CentralAir', 'HeatingQC']])
pipe_2_train_mse = mean_squared_error(pred_train, y_train)
pipe_2_test_mse = mean_squared_error(pred_test, y_test)

# Answer check
print(pipe_2.named_steps)
print(f'Train MSE: {pipe_2_train_mse: .2f}')
print(f'Test MSE: {pipe_2_test_mse: .2f}')
pipe_2


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 8
# 
# #### Including `PolynomialFeatures`
# 
# **10 Points**
# 
# Finally, the earlier transformation of continuous columns using the `PolynomialFeatures` with `degree = 2` can be implemented alongside the `OneHotEncoder` and `OrdinalEncoder`.  
# 
# The `make_column_transformer` is again used, and you are to create a `Pipeline` with steps `transformer` and `linreg`.  
# 
# The `Pipeline` is fit on the training data using features `['OverallQual', 'CentralAir', 'HeatingQC']`.  
# 
# - Use the `predict` function on `pipe_3` to predict the values of `X_train[['OverallQual', 'CentralAir', 'HeatingQC']]`. Assign your result to `quad_train_preds`.
# - Use the `predict` function on `pipe_3` to predict the values of `X_test[['OverallQual', 'CentralAir', 'HeatingQC']]`. Assign your result to `quad_test_preds`.
# - Use the `mean_squared_error` function to calculate the MSE between `y_train` and `quad_train_preds`. Assign the result to `quad_train_mse`.
# - Use the `mean_squared_error` function to calculate the MSE between `y_test` and `quad_test_preds`. Assign the result to `quad_test_mse`.

# In[30]:


poly_ordinal_ohe = make_column_transformer((OrdinalEncoder(categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ['HeatingQC']),
                                           (OneHotEncoder(drop = 'if_binary'), ['CentralAir']),
                                           (PolynomialFeatures(include_bias = False, degree = 2), ['OverallQual']))
pipe_3 = Pipeline([('transformer', poly_ordinal_ohe), 
                  ('linreg', LinearRegression())])


# In[31]:


pipe_3.fit(X_train[['OverallQual', 'CentralAir', 'HeatingQC']], y_train)


# In[32]:


### GRADED

quad_train_preds = pipe_3.predict(X_train[['OverallQual', 'CentralAir', 'HeatingQC']])
quad_test_preds = pipe_3.predict(X_test[['OverallQual', 'CentralAir', 'HeatingQC']])
quad_train_mse = mean_squared_error(quad_train_preds, y_train)
quad_test_mse = mean_squared_error(quad_test_preds, y_test)

# Answer check
print(f'Train MSE: {quad_train_mse: .2f}')
print(f'Test MSE: {quad_test_mse: .2f}')


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 9
# 
# #### Including More Features
# 
# **20 Points**
# 
# Use the following features to build a new `make_column_transformer` and fit 5 different models of degree 1 - 5 using the `degree` argument in your `PolynomialFeatures` transformer.  Keep track of the subsequent train mean squared error and test set mean squared error with the lists `train_mses` and `test_mses` respectively.  
# 
# The `poly_ordinal_ohe` object contains the different transformers needed.  Note that rather than passing a list of columns to the `PolynomialFeatures` transformer, the `make_column_selector` function is used to select any numeric feature.  For more information on the `make_column_selector` see [here](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html).
# 
# 

# In[34]:


features = ['CentralAir', 'HeatingQC', 'OverallQual', 'GrLivArea', 'KitchenQual', 'FullBath']


# In[35]:


X_train[features].head()


# In[36]:


poly_ordinal_ohe = make_column_transformer((PolynomialFeatures(), make_column_selector(dtype_include=np.number)),
                                           (OrdinalEncoder(categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ['HeatingQC', 'KitchenQual']),
                                               (OneHotEncoder(drop = 'if_binary', sparse = False), ['CentralAir']))


# In[37]:


### GRADED

train_mses = []
test_mses = []
#for degree in 1 - 5
for i in range(1, 6):
    #create pipeline with PolynomialFeatures degree i 
    #ADD APPROPRIATE ARGUMENTS IN POLYNOMIALFEATURES
    poly_ordinal_ohe = make_column_transformer((PolynomialFeatures(degree = i), make_column_selector(dtype_include=np.number)),
                                           (OrdinalEncoder(categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]), ['HeatingQC']),
                                               (OneHotEncoder(drop = 'if_binary'), ['CentralAir']))
    
    pipe = Pipeline([
        ('transformer', poly_ordinal_ohe),
        ('line_reg', LinearRegression())
    ])
    #fit on train
    pipe.fit(X_train[features], y_train)
    #predict on train and test
    train_pred = pipe.predict(X_train[features])
    test_pred = pipe.predict(X_test[features])
    #compute mean squared errors
    train_mse = mean_squared_error(train_pred, y_train)
    test_mse = mean_squared_error(test_pred, y_test)
    #append to train_mses and test_mses respectively
    train_mses.append(train_mse)
    test_mses.append(test_mse)

# Answer check
print(train_mses)
print(test_mses)
pipe


# In[ ]:





# [Back to top](#Index:) 
# 
# ### Problem 10
# 
# #### Optimal Model Complexity 
# 
# **10 Points**
# 
# Based on your model's mean squared error on the testing data in **Problem 9** above, what was the optimal complexity?  Assign your answer as an integer to `best_complexity` below.  Compute the **MEAN SQUARED ERROR** of this model and assign it to `best_mse` as a float. 

# In[39]:


### GRADED
best_complexity =  test_mses.index(min(test_mses)) + 1
best_mse = float(min(test_mses))

# Answer check
print(f'The best degree polynomial model is:  {best_complexity}')
print(f'The smallest mean squared error on the test data is : {best_mse: .2f}')


# In[ ]:





# ### Further Exploration
# 
# This activity was meant to introduce you to a more streamlined modeling process using the `sklearn` library.  While your models should be performing better than the baseline, it is likely that with a bit more feature engineering and cross-validation you would be able to further improve the performance.  You are encouraged to explore further feature engineering and encoding, particularly with handling missing values.  
# 
# Additionally, other transformations on the data may be appropriate.  For example, if you look at the distribution of errors in your model, you will note that they are slightly skewed.  An assumption of a Linear Regression model is that these should be roughly normally distributed.  By building a model on the logarithm of the target column and evaluating the model on the logarithm of the testing data, you will improve towards this assumption.  Note that the actual Kaggle exercise is judged on the **ROOT MEAN SQUARED ERROR** of the logarithm of the target feature. 
# 
# If interested, scikitlearn also provides a function `TransformedTargetRegressor` that will accomplish this transformation and can easily be added to a pipeline. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) for more information on this transformer. 

# In[ ]:




