#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 18.1: Tokenization
# 
# **Expected Time = 60 minutes** 
# 
# **Total Points = 60**
# 
# This activity focuses on tokenizing text.  You will use `nltk` to tokenize words and sentences of given documents.  In general, tokenizing a text refers to the operation of splitting the text apart into chunks.  Here, our chunks can be individual "words" and "sentences".  These are not necessarily meant to refer to proper grammatical structure or meaning, however splitting entities based on white space, periods, or other punctuation.  
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)
# - [Problem 6](#-Problem-6)

# #### The Data
# 
# We use both a single piece of text in the form of a lead paragraph from Isaac Newton's *Principia* and a dataset including text data from [kaggle](https://www.kaggle.com/datasets/sankha1998/emotion?select=Emotion%28sad%29.csv) related to classifying WhatsApp status. 

# In[5]:


principia = '''
Since the ancients (as we are told by Pappus), made great account of the science of mechanics in the investigation of natural things; and the moderns, laying aside substantial forms and occult qualities, have endeavoured to subject the phænomena of nature to the laws of mathematics, I have in this treatise cultivated mathematics so far as it regards philosophy. The ancients considered mechanics in a twofold respect; as rational, which proceeds accurately by demonstration: and practical. To practical mechanics all the manual arts belong, from which mechanics took its name. But as artificers do not work with perfect accuracy, it comes to pass that mechanics is so distinguished from geometry, that what is perfectly accurate is called geometrical, what is less so, is called mechanical. But the errors are not in the art, but in the artificers. He that works with less accuracy is an imperfect mechanic; and if any could work with perfect accuracy, he would be the most perfect mechanic of all; for the description if right lines and circles, upon which geometry is founded, belongs to mechanics. Geometry does not teach us to draw these lines, but requires them to be drawn; for it requires that the learner should first be taught to describe these accurately, before he enters upon geometry; then it shows how by these operations problems may be solved. To describe right lines and circles are problems, but not geometrical problems. The solution of these problems is required from mechanics; and by geometry the use of them, when so solved, is shown; and it is the glory of geometry that from those few principles, brought from without, it is able to produce so many things. Therefore geometry is founded in mechanical practice, and is nothing but that part of universal mechanics which accurately proposes and demonstrates the art of measuring. But since the manual arts are chiefly conversant in the moving of bodies, it comes to pass that geometry is commonly referred to their magnitudes, and mechanics to their motion. In this sense rational mechanics will be the science of motions resulting from any forces whatsoever, and of the forces required to produce any motions, accurately proposed and demonstrated. This part of mechanics was cultivated by the ancients in the five powers which relate to manual arts, who considered gravity (it not being a manual power), no otherwise than as it moved weights by those powers. Our design not respecting arts, but philosophy, and our subject not manual but natural powers, we consider chiefly those things which relate to gravity, levity, elastic force, the resistance of fluids, and the like forces, whether attractive or impulsive; and therefore we offer this work as the mathematical principles if philosophy; for all the difficulty of philosophy seems to consist in this—from the phænomena of motions to investigate the forces of nature, and then from these forces to demonstrate the other phænomena; and to this end the general propositions in the first and second book are directed. In the third book we give an example of this in the explication of the System of the World; for by the propositions mathematically demonstrated in the former books, we in the third derive from the celestial phenomena the forces of gravity with which bodies tend to the sun and the several planets. Then from these forces, by other propositions which are also mathematical, we deduce the motions of the planets, the comets, the moon, and the sea. I wish we could derive the rest of the phænomena of nature by the same kind of reasoning from mechanical principles; for I am induced by many reasons to suspect that they may all depend upon certain forces by which the particles of bodies, by some causes hitherto unknown, are either mutually impelled towards each other, and cohere in regular figures, or are repelled and recede from each other; which forces being unknown, philosophers have hitherto attempted the search of nature in vain; but I hope the principles here laid down will afford some light either to this or some truer method of philosophy.'''


# In[6]:


print(principia)


# In[7]:


import nltk
import pandas as pd


# In[8]:


from nltk import word_tokenize, sent_tokenize
nltk.download('punkt')


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Word Tokenizing a String
# 
# **10 Points**
# 
# Use the `word_tokenize` function to split the string `principia` into individual elements of the text.  Assign your results as a list to `ans1` below. 

# In[10]:


# word_tokenize?


# In[11]:


### GRADED
ans1 = word_tokenize(principia)

### ANSWER CHECK
print(type(ans1))
print(ans1[:5])


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Sentence Tokenization of a string
# 
# **10 Points**
# 
# Rather than breaking a text apart into individual tokens or "words" you can split based on sentences using the `sent_tokenize` function. Split the principia text into sentences and assign your answer as a list to `ans2` below.

# In[13]:


# sent_tokenize?


# In[14]:


### GRADED
ans2 = sent_tokenize(principia)

### ANSWER CHECK
print(type(ans2))
print(ans2[:2])


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Unique Words with `set`
# 
# **10 Points**
# 
# The tokenization does not yield unique words.  To create a collection of unique words, use the `set` function along with `word_tokenize` to create a mathematical set object of the words from the principia.  Assign your solution to `ans3` below.  

# In[15]:


get_ipython().run_line_magic('pinfo', 'set')


# In[16]:


### GRADED

ans3 = set(word_tokenize(principia))

### ANSWER CHECK
print(type(ans3))
print(ans3)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Counts of words
# 
# **10 Points**
# 
# Determine the number of words in the principia text using `word_tokenize` and the `len` function.  Assign your answer as an integer to `ans4` below.

# In[ ]:


### GRADED
ans4 = ''

    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(type(ans4))
print(ans4)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Lexical Diversity
# 
# **10 Points**
# 
# The lexical diversity of a text is the ratio of unique words to the total words.  Compute the lexical diversity of the principia text and assign your answer as a float to `ans5` below. 
# 
# Hint: Use the `length` function to find the numerial amount of unique and non-unique words

# In[ ]:


### GRADED
ans5 = ''
ans1 = len(set(word_tokenize(principia)))
ans2 = len(word_tokenize(principia))
    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK
print(type(ans5))
print(ans5)
print(ans1)
print(ans2)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 6
# 
# #### Text in a DataFrame
# 
# **10 Points**
# 
# To this point, we have been dealing with a block of text. How do you work with multiple lines of text in a DataFrame?
# 
# You can use the `set` function to determine the number of unique words (as above), but this will only provide a result PER ITEM, not for the entire DataFrame. To determine the total amount of words in a DataFrame, first use the `word_tokenize` function with the `.apply` method, and sum the resulting column to get a non-unique list of words. 
# 
# Use your work above to determine the number of non-unique words (using `len`) from `happy_df` in the `content` feature given below.  Assign your answer as an integer to `ans6` below.

# In[ ]:


happy_df = pd.read_csv('data/Emotion(happy).csv')
happy_df.head()


# In[ ]:


### GRADED
ans6 = ''

    
# YOUR CODE HERE
raise NotImplementedError()

### ANSWER CHECK

print(type(ans6))
print(ans6)


# In[ ]:





# In[ ]:




