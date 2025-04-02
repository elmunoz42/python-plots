#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 18.2: Stemming and Lemmatization
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 25**
# 
# In this activity, you will stem and lemmatize a text to normalize a given text.  Here, you will review using the lemmatizer and stemmer on a basic list and then turn to data in a DataFrame, writing a function to apply the lemmatization and stemming operations to a column of text data.  The data is the WhatsApp status dataset from kaggle, and you will focus on the `content` feature.
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# In[1]:


import pandas as pd
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')


# #### The Data
# 
# The text data again comes from [kaggle](https://www.kaggle.com/datasets/sankha1998/emotion?select=Emotion%28sad%29.csv) and is related to classifying WhatsApp status. We load in only the "angry" sentiment below.
# 

# In[2]:


angry = pd.read_csv('data/Emotion(angry).csv')


# In[3]:


angry.head()


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Stemming a list of words
# 
# **5 Points**
# 
# Use `PorterStemmer` to stem the different variations on the word "compute" in the list `C` below.  Assign your results to the list `stemmed_words` below. 

# In[4]:


C = ['computer', 'computing', 'computed', 'computes', 'computation', 'compute']


# In[7]:


# PorterStemmer?


# In[11]:


### GRADED
stemmer = PorterStemmer(mode="NLTK_EXTENSIONS")
stemmed_words = stemmer.stem(C)


### ANSWER CHECK
print(type(stemmed_words))
print(stemmed_words)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Lemmatizing a list of words
# 
# **5 Points**
# 
# Use `WordNetLemmatizer` to lemmatize the different variations on the word "compute" in the list `C` below.  Assign your results to the list `lemmatized_words` below. 

# In[ ]:


### GRADED
lemma = ''
lemmatized_words = ''

    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(type(lemmatized_words))
print(lemmatized_words)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Which performed better
# 
# **5 Points**
# 
# Assuming we wanted all the words in `C` to be normalized to the same word, which worked better to this goal -- stemming or lemmatizing.  Assign your response as a string -- `stem` or `lemmatize` -- to `ans3` below.

# In[ ]:


### GRADED
ans3 = ''

    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(ans3)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### A function for stemming
# 
# **5 Points**
# 
# Use `PorterStemmer` to complete the function `stemmer` below. This function should take in a string of text and return a string of stemmed text. Note that you will need to tokenize the text before stemming. This function should return a single string.
# 
# Hint: Use the `join` method

# In[ ]:


### GRADED
def stemmer(text):
    '''
    This function takes in a string of text and returns
    a string of stemmed text.

    Arguments
    ---------
    text: str
        string of text to be stemmed

    Returns
    -------
    str
       string of stemmed words from the text input
    '''
    return ''
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
text = 'The computer did not compute the answers correctly.'
print(text)
print(stemmer(text))#should return --> the comput did not comput the answer correctli .


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Using the stemmer on a DataFrame
# 
# **5 Points**
# 
# Use your function `stemmer` to apply to the `content` feature of the DataFrame `angry`.  Assign the resulting series to `stemmed_content` below.
# 
# Hint: use the `.apply` method

# In[ ]:


### GRADED
stemmed_content = ''

    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(type(stemmed_content))
print(stemmed_content.head())


# In[ ]:





# In[ ]:




