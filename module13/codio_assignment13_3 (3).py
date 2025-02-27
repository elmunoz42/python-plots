#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 13.3: End-to-End Classification Problem
# 
# **Expected Time = 90 minutes**
# 
# **Total Points = 85**
# 
# This example leads you through an end-to-end analysis of a classification algorithm using `LogisticRegression`. You will perform some brief exploratory data analysis (EDA). Then, you will construct a feature engineering, selection, and model pipeline. Finally, you will explore the mistakes your models make and compare different classification metrics.

# ### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)
# - [Problem 6](#-Problem-6)
# - [Problem 7](#-Problem-7)
# - [Problem 8](#-Problem-8)
# - [Problem 9](#-Problem-9)
# - [Problem 10](#-Problem-10)
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ### The Data
# 
# This data is originally from the IBM and contains information on a telecommunications company customer subscriptions.  Your task is to predict the customers who will Churn.  The data is loaded, displayed, and split below.

# In[3]:


df = pd.read_csv('data/wa_churn.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype('float')


# In[7]:


df = df.dropna()


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(['Churn', 'customerID'], axis = 1), df['Churn'], random_state = 442,
                                                   stratify = df['Churn'])


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# ### `make_column_selector`
# 
# **5 Points**
# 
# To begin, you may want to incorporate many of the categorical features here.  Rather than writing a list of names, you can use the `make_column_selector` to select features by datatype.  For example:
# 
# ```python
# make_column_selector(dtype_include=object)
# ```
# 
# will select all columns with `object` datatype.  This selector will replace the list of column names in the `make_column_transformer`.  
# 
# Create a selector object to select the columns with `object` datatype below.  Assign this to `selector`.

# In[11]:


# make_column_selector?


# In[12]:


### GRADED

selector = make_column_selector(dtype_include=object)

# Answer check
selector


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Column Transformer
# 
# **5 Points**
# 
# Use the `make_column_transformer` function on the the columns selected by `selector`. To these columns, apply the `OneHotEncoder` with `drop = first`. To the `remainder` columns, apply `StandardScaler()`
# 
# Assign the result to `transformer` below.
# 
# 

# In[13]:


# make_column_transformer?


# In[14]:


### GRADED
transformer = make_column_transformer(
    (OneHotEncoder(drop = 'first'), selector),
    remainder = StandardScaler()
)

# Answer check
transformer


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Feature Extractor
# 
# **5 Points**
# 
# Just as in our earlier assignment you can use `LogisticRegression` with `l1` penalty to select features for the model.
# 
# Below, create a `SelectFromModel` object that uses a `LogisticRegression` estimator with `penalty = 'l1'`, solver of `liblinear` and `random_state = 42`.  Assign your transformer as `extractor` below.

# In[15]:


# SelectFromModel?


# In[16]:


### GRADED
extractor = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42))

# Answer check
extractor


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Pipeline with Estimator
# 
# **10 Points**
# 
# Now, build a pipeline `lgr_pipe` with named steps `transformer`, `selector`, and `lgr` that implement the column transformer, feature selector from above and a `LogisticRegression` estimator with `random_state = 42` and `max_iter = 1000`.  
# 
# Fit the pipeline on the training data and determine the score on the test data.  
# 
# Finally, use the function `score` to calculate the accuracy as a float to `pipe_1_acc` below. 

# In[17]:


# Pipeline?
# LogisticRegression?


# In[18]:


### GRADED

lgr_pipe = Pipeline([
    ('transformer', transformer),
    ('selector', extractor),
    ('lgr', LogisticRegression(random_state = 42, max_iter = 1000))
])
lgr_pipe.fit(X_train, y_train)
pipe_1_acc = lgr_pipe.score(X_test, y_test)
print(f"pipe accuracy: {pipe_1_acc}")
# Answer check
lgr_pipe


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Comparison to Baseline
# 
# **10 Points**
# 
# Use the `value_counts` method to determine the baseline score by choosing the majority class as your predictions.  Did your pipeline outperform the baseline model?  Answer `yes` or `no` as a string to `ans5` below.

# In[19]:


### GRADED
# First, look at the distribution of the target variable
majority_class_distribution = y_test.value_counts(normalize=True)
print(majority_class_distribution)

# The baseline accuracy would be the percentage of the majority class
baseline_accuracy = majority_class_distribution.max()
print(f"Baseline accuracy: {baseline_accuracy}")

# Compare with your pipeline accuracy
print(f"Pipeline accuracy: {pipe_1_acc}")

# Determine if your pipeline outperformed the baseline
ans5 = "yes" if pipe_1_acc > baseline_accuracy else "no"

### ANSWER TEST
print(ans5)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 6
# 
# ### Confusion Matrix and ROC Curve
# 
# **10 Points**
# 
# Examine both the confusion matrix and ROC curve using the cell below.  
# 
# Create a 1 row by 2 column subplot object and place the confusion matrix on the left and ROC curve on the right.
# 
# 
# 
# Use these to determine the number of false positives and false negatives on the test data.  Assign as an integer to `fp` and `fn` below.  Also, use the `RocCurveDisplay` legend to determine the AUC score.  Assign this as a float with two decimal places to `auc` below. 

# In[26]:


# np.where?
# ConfusionMatrixDisplay?
# confusion_matrix?


# In[21]:


### GRADED
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Get predictions
y_pred = lgr_pipe.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(
    lgr_pipe, X_test, y_test, 
    display_labels=["No", "Yes"],
    ax=ax[0]
)
ax[0].set_title("Confusion Matrix")

# For the second subplot - ROC curve
y_prob = lgr_pipe.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, _ = roc_curve(y_test == "Yes", y_prob)
roc_auc = auc(fpr, tpr)

# Round to 2 decimal places
auc = round(roc_auc, 2)
print(f"Area Under the Curve: {auc}")

# Display ROC curve
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc).plot(ax=ax[1])
ax[1].set_title('ROC Curve')
plt.tight_layout()
plt.show()

### ANSWER CHECK
fp, fn, auc


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 7
# 
# #### What Really Matters
# 
# **10 Points**
# 
# You see above that you should have 194 False Negatives and 126 False Positives.  Suppose you want to implement an intervention to attempt turning over customers.  To use your classifier, this means being sure about targeting the customers you expect to churn -- in other words minimize the False Negatives.  Use the `predict_proba` method to select the probabilities of the `No` class.  Assign this as an array to `no_probs` below.

# In[27]:


### GRADED

no_probs = lgr_pipe.predict_proba(X_test)[:,0]

### ANSWER CHECK
no_probs[:5]


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 8
# 
# #### Only target customers with high probability
# 
# **10 Points**
# 
# Even though our classifier is doing better than the baseline, it is still making a high number of mistakes.  Let's only look at the labels for `No` where you are better than 80% sure they are `No`'s.  Select these from your `no_probs` and assign as an array to `high_prob_no` below.

# In[31]:


### GRADED

high_prob_no = no_probs[no_probs >= 0.8]

### ANSWER CHECK
high_prob_no[:5]


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 9
# 
# #### Number of Predictions
# 
# **10 Points**
# 
# How many datapoints had probability greater than 80% of `No`?  What percent of the test data is this?  What percent of the original test data set `No` values is this?  Assign your answer as a float to `percent_of_test_data` and `percent_of_no` below. 
# 

# In[33]:


### GRADED

# Count of samples with "No" probability ≥ 0.8
high_prob_count = len(high_prob_no)

# Total test samples
total_test_samples = len(y_test)

# Count of actual "No" instances in test set
actual_no_count = (y_test == "No").sum()

# Calculate percentages
percent_of_test_data = high_prob_count / total_test_samples
percent_of_no = high_prob_count / actual_no_count

### ANSWER CHECK
print(percent_of_test_data)
print(percent_of_no)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 10
# 
# #### Important Features
# 
# **10 Points**
# 
# Now, let us explore the coefficients of the model.  Because the data were scaled, we can think about the coefficients as speaking to a relative feature importance.  Extract the coefficients from your model and sort their absolute values from greatest to least.  Create a DataFrame called `coef_df` that contains the feature name and coefficient. The results begin as shown below:
# 
# <table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>feature</th>      <th>coefs</th>    </tr>  </thead>  <tbody>    <tr>      <th>14</th>      <td>Contract_Two year</td>      <td>1.321160</td>    </tr>    <tr>      <th>20</th>      <td>tenure</td>      <td>1.301754</td>    </tr>    <tr>      <th>9</th>      <td>TechSupport_No internet service</td>      <td>0.753071</td>    </tr>    <tr>      <th>13</th>      <td>Contract_One year</td>      <td>0.701108</td>    </tr>    <tr>      <th>5</th>      <td>InternetService_Fiber optic</td>      <td>0.679121</td>    </tr>  </tbody></table>

# In[42]:


# lgr_pipe?
# pd.DataFrame?


# In[58]:


### GRADED
# Get coefficients from the final estimator
coefficients = lgr_pipe.named_steps['lgr'].coef_[0]

# Get feature names after transformation
transformed_feature_names = lgr_pipe.named_steps['transformer'].get_feature_names_out()

# Get the indices of features selected by the SelectFromModel step
selected_indices = lgr_pipe.named_steps['selector'].get_support()

# Filter the feature names to only include those that were selected
selected_feature_names = transformed_feature_names[selected_indices]

# Clean up feature names - remove prefixes like "onehotencoder__"
cleaned_feature_names = []
for name in selected_feature_names:
    # Remove prefixes and just keep the meaningful part
    if 'onehotencoder__' in name:
        cleaned_name = name.replace('onehotencoder__', '')
        cleaned_feature_names.append(cleaned_name)
    elif 'remainder__' in name:
        cleaned_name = name.replace('remainder__', '')
        cleaned_feature_names.append(cleaned_name)
    else:
        cleaned_feature_names.append(name)

# Create DataFrame with only the selected features and their coefficients
coef_df = pd.DataFrame({
    'feature': cleaned_feature_names,
    'coefs': coefficients
})

# Sort by absolute coefficient values (descending)
coef_df = coef_df.sort_values(by='coefs', key=lambda x: abs(x), ascending=False)
### ANSWER CHECK
coef_df.head()


# In[ ]:





# Notice that you should have a higher percentage of No values in your predictions than how much of the data it is comprised of.  In other words, if you randomly selected 50% of the data, you would expect 50% of the No.  Here, by ranking our predictions by probabilities and only selecting those with higher probability we are able to identify almost 70% of the No.  This notion of *LIFT* is an alternative method to that of ROC for understanding the quality of predictions, particularly if you have finite resources to expend.  If you are interested read more [here](https://www.ibm.com/docs/en/spss-statistics/24.0.0?topic=overtraining-cumulative-gains-lift-charts) and the `skplot` library has a straightforward visualization of the lift curve [here](https://scikit-plot.readthedocs.io/en/stable/metrics.html#scikitplot.metrics.plot_cumulative_gain).
