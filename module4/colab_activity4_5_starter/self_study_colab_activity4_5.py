#!/usr/bin/env python
# coding: utf-8

# ### Self-Study Colab Activity 4.5: Analyzing a Real-World Dataset 
# 
# #### Exploring Credit Risks
# 
# This activity is another open exploration of a dataset using both cleaning methods and visualizations.  The data describes customers as good or bad credit risks based on a small set of features specified below.  Your task is to create a Jupyter notebook with an exploration of the data using both your `pandas` cleaning and analysis skills and your visualization skills using `matplotlib`, `seaborn`, and `plotly`.  Your final notebook should be formatted with appropriate headers and markdown cells with written explanations for the code that follows. 
# 
# Post your notebook file in Canvas, as well as a brief (3-4 sentence) description of what you found through your analysis. Respond to your peers with reflections on their analysis. 
# 
# 
# 

# ##### Data Description
# 
# ```
# 1. Status of an existing checking account, in Deutsche Mark.
# 2. Duration in months
# 3. Credit history (credits taken, paid back duly, delays, critical accounts)
# 4. Purpose of the credit (car, television,...)
# 5. Credit amount
# 6. Status of savings account/bonds, in Deutsche Mark.
# 7. Present employment, in number of years.
# 8. Installment rate in percentage of disposable income
# 9. Personal status (married, single,...) and sex
# 10. Other debtors / guarantors
# 11. Present residence since X years
# 12. Property (e.g., real estate)
# 13. Age in years
# 14. Other installment plans (banks, stores)
# 15. Housing (rent, own,...)
# 16. Number of existing credits at this bank
# 17. Job
# 18. Number of people being liable to provide maintenance for
# 19. Telephone (yes, no)
# 20. Foreign worker (yes, no)
# ```

# In[9]:


import pandas as pd
import plotly.express as px


# In[3]:


df = pd.read_csv('data/dataset_31_credit-g.csv')


# In[4]:


df.head(3)


# In[5]:


df.info()


# ### Exploration of the "purpose" feature
# Use the value counts function to examine different columns to see what the values look like and if there's any issues.

# In[12]:


df["purpose"].value_counts()


# ### Observation: 
# 
# In the purpose column the single quotes are used inconsistently, this is also true in other columns.

# ### Cleanup: 
# Remove single quotes from multiple columns

# In[7]:


# Load data
df = pd.read_csv('data/dataset_31_credit-g.csv')

# Convert all object columns to string
df = df.astype({col: str for col in df.select_dtypes(['object'])})

# Then apply the replace operation to all string columns
for col in df.select_dtypes(['object']):
    df[col] = df[col].str.replace("\'", "")

df.head()


# ### Analysis Question: Based on the purpose of the purchase how much credit do people take out and what is their checking account status?

# In[13]:


px.bar(df, x = "purpose", y = "credit_amount", color="checking_status")


# ### Analysis Findings: 
# 
# Interestingly it seems that for 'radio/tv', 'furniture/equipment', 'new car', and 'used car' individuals are more willing to take out credit
# even though they have very little amount of money in their checking account. One surprising number here is the "no checking" which makes me wonder if 
# what is meant is that there is no checking account information for the user because they didn't provide any.

# In[ ]:





# In[ ]:




