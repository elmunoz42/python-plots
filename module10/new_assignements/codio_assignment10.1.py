#!/usr/bin/env python
# coding: utf-8

# ### Codio Assignment 10.1: Computing Autocorrelation
# 
# 
# **Expected Time: 45 Minutes**
# 
# **Total Points: 50**
# 
# 
# This activity focuses on computing the autocorrelation of a time series dataset.  You will use `statsmodels` to compute autocorrelation and determine whether or not the series is stationary.  Finally, you are to difference the data and see if the resulting series is itself stationary.  

# #### Index
# 
# - [Problem 1](#Problem-1)
# - [Problem 2](#Problem-2)
# - [Problem 3](#Problem-3)
# - [Problem 4](#Problem-4)
# - [Problem 5](#Problem-5)

# In[2]:


import numpy as np
import pandas as pd
from statsmodels.tsa import arima_process
import matplotlib.pyplot as plt
import warnings

from statsmodels.datasets import nile
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
warnings.filterwarnings("ignore")


# [Back to top](#Index)
# 
# ### Problem 1
# 
# **10 Points**
# 
# #### Creating a dataset with `ArmaProcess`
# 
# Following video 10.3, create an `arima_process` using the arguments:
# 
# - `ar = [.9, -0.3]`
# - `ma = [2]`
# 
# Assign this as an `ArmaProcess` object to `process` below.

# In[3]:


### GRADED

process = arima_process.ArmaProcess(ar=[.9,-0.3], ma=[2])

# Answer check
print(process)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# **10 Points**
# 
# #### Generating a sample
# 
# <center>
#     <img src = images/arma1.png/>
# </center>
# 
# 
# Next, you are to generate a sample of size 100 from the arima_process created in [Problem 1](#Problem-1).  To assure consistent results, make sure to leave the `np.random.seed(32)`.  This assures the same sample will be generated time after time.

# In[4]:


### GRADED

np.random.seed(32)#dont

sample = process.generate_sample(nsample=100)

### Answer check
print(sample[:5])


# In[5]:


# Uncomment to plot the sample
plt.plot(sample, '--o')
plt.title('Arma Process Sample Data')
plt.grid();


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 3
# 
# **10 Points**
# 
# 
# #### Computing the autocorrelation
# 
# Use the `sample` created above together with the `acf` function from statsmodels to compute the autocorrelation values for the sample.  Assign these values to `auto_corr` as an array below.  **Note**: Set `fft = True` in the acf function to avoid a warning.
# 
# <center>
#     <img src = 'images/arma2.png'/>
# </center>

# In[6]:


### GRADED

# Compute the autocorrelation of the data
auto_corr = acf(sample, fft=True)

# Difference the data to make it stationary
sample_diff = np.diff(sample)

# Compute the autocorrelation of the differenced data
auto_corr_diff = acf(sample_diff, fft=True)
print(auto_corr_diff)

# Plot the autocorrelation of the original data
fig, ax = plt.subplots(2, 1, figsize=[10, 8])
plot_acf(sample, lags=20, ax=ax[0])
ax[0].set_title('Autocorrelation of Original Data')

# Plot the autocorrelation of the differenced data
plot_acf(sample_diff, lags=20, ax=ax[1])
ax[1].set_title('Autocorrelation of Differenced Data')

plt.show()
# Answer check
print(auto_corr[:5])
print(type(auto_corr))


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 4
# 
# **10 Points**
# 
# #### Using `acf` to compute autocorrelation
# 
# Below, a dataset relating the volume of flow in the Nile river from statsmodels is loaded  and visualized.  Use the `acf` function from statsmodels to compute the autocorrelation values of the `volume` feature. Assign your results as an array to `nile_acf` below.  
# 
# Visualizing the autocorrelation data using the `plot_acf` function from statsmodels generates:
# 
# <center>
#     <img src = 'images/ar4.png' />
# </center>
# 
# Does this suggest the data is stationary?  Why or why not?

# In[7]:


nile_df = nile.load_pandas().data
nile_df.head()


# In[8]:


plt.plot(nile_df['year'], nile_df['volume'], '--o')
plt.title('Nile River flows at Ashwan 1871 - 1970.')
plt.grid();


# In[10]:


# Reload the DataFrame
nile_df = nile.load_pandas().data

# Calculate Auto correlation for undiferentiated data
nile_acf = acf(nile_df['volume'], fft = True)

# Convert the 'year' column to integer
nile_df['year'] = nile_df['year'].astype(int)

# Convert the 'year' column to datetime
nile_df['year'] = pd.to_datetime(nile_df['year'], format='%Y')

# Set the 'year' column as the index
nile_df.set_index('year', inplace=True)

# Print the DataFrame to verify
print(nile_df.head())

# Difference the 'volume' column to make it stationary
nile_df['volume_diff'] = nile_df['volume'].diff()

# Drop the first row with NaN value resulting from differencing
nile_diff = nile_df.dropna()

# Compute the autocorrelation of the differenced data
nile_acf_diff = acf(nile_diff['volume_diff'], nlags=20, fft=True)

# Plot the autocorrelation w/ differentiated
fig, ax = plt.subplots(figsize=[10, 4])
plot_acf(nile_df['volume'], lags=20, ax=ax)
plt.show()

# Plot the autocorrelation w/ differentiated
fig, ax = plt.subplots(figsize=[10, 4])
plot_acf(nile_diff['volume_diff'], lags=20, ax=ax)
plt.title('Autocorrelation for Differentiated Data')
plt.show()

# Answer check
print(nile_acf[:5])


# In[ ]:





# [Back to top](#Index)
# 
# ### Problem 5
# 
# **10 Points**
# 
# #### Tesla and stationarity
# 
# Below, stock data from Tesla corporation are loaded from the beginning of the year 2020.  The Adjusted Closing price is plotted below.  You are to use the autocorrelation plots to determine which version of the data is stationary.  Assign one of the following strings to `ans5` below:
# 
# - `original`: the original adjusted closing price is stationary
# - `first_diff`: the first difference of the adjusted closing price is stationary
# - `neither`: neither the original time series or its first difference are stationary
# 
# 

# In[12]:


tsla = pd.read_csv('data/tsla.csv', index_col='Date')
tsla.head()


# In[13]:


plt.plot(tsla['Adj Close'])
plt.grid()
#plt.xticks(rotation = 40)
plt.title('TSLA Adjusted Closing Price 2020 - 2021');


# In[14]:


### GRADED

# Difference the 'Adj Close' column to make it stationary
tsla['Adj Close_diff'] = tsla['Adj Close'].diff()

# Drop the first row with NaN value resulting from differencing
tsla_diff = tsla.dropna()

# Compute the autocorrelation of the differenced data
tsla_acf = acf(tsla_diff['Adj Close_diff'], nlags=20, fft=True)

# Plot the autocorrelation of the differenced data
fig, ax = plt.subplots(figsize=[10, 4])
plot_acf(tsla['Adj Close'], lags=20, ax=ax)
ax.set_title('Autocorrelation of Adjusted Closing Price')
plt.show()

# Plot the autocorrelation of the differenced data
fig, ax = plt.subplots(figsize=[10, 4])
plot_acf(tsla_diff['Adj Close_diff'], lags=20, ax=ax)
ax.set_title('Autocorrelation of Differenced Adjusted Closing Price')
plt.show()

ans5 = 'first_diff'

# Answer check
print(ans5)


# In[ ]:




