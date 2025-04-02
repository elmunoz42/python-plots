#!/usr/bin/env python
# coding: utf-8

# ### Required Codio Assignment 16.2: Tuning the 'SVC' Classifier
# 
# **Expected Time = 60 minutes**
# 
# **Total Points = 40**
# 
# This activity focuses on tuning the `SVC` classifier parameters to improve its performance using the wine data.  Typically, the `SVC` will need some parameter tuning.  In practice, you will want to be deliberate about the tuning parameters and not be too exhaustive as the grid searches can be energy intensive.  Here, you will compare different kernels and the `gamma` parameter of the classifier.
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_wine


# In[3]:


X, y = load_wine(return_X_y=True, as_frame=True)


# In[4]:


y.value_counts(normalize = True)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                   random_state = 42)


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Baseline for Classifier
# 
# **10 Points**
# 
# Below, determine the baseline score for the classifier by using the `DummyClassifier` to fit the the training data `X_train` and `y_train`.  Assign this to the variable `dummy_clf`.
# 
# Next, use the `score` function on `dummy_clf` with arguments `X_test` and `y_test` and assign the result to `baseline_score`.
# 
# 
# **Note**: The `DummyClassifier` works just as all other estimators you have encountered and has a `.fit` and `.score` method.

# In[6]:


# DummyClassifier?


# In[7]:


### GRADED
dummy_clf = DummyClassifier().fit(X_train, y_train)
baseline_score = dummy_clf.score(X_test, y_test)


### ANSWER CHECK
print(baseline_score)


# ## Explanation
# 
# The baseline score of 0.4 (or 40% accuracy) from your DummyClassifier means that your baseline model correctly classified 40% of the samples in your test set without learning any real patterns from your data.
# 
# This 0.4 result provides important context:
# 
# 1. **Class distribution**: This suggests you likely have an imbalanced classification problem with the majority class representing approximately 40% of your data. The dummy classifier is probably using the "most_frequent" strategy (the default), which always predicts the most common class in the training set.
# 
# 2. **Performance benchmark**: Any meaningful model you build should achieve accuracy significantly higher than 0.4. If your actual model only reaches, say, 0.45 accuracy, it's barely better than guessing the majority class.
# 
# 3. **Multi-class problem**: Since the accuracy is 0.4, you might be working with a multi-class classification problem (likely with approximately 2-3 classes), rather than a binary classification where we'd expect closer to 0.5 accuracy for a balanced dataset.
# 
# 4. **Evaluation context**: When you evaluate your actual models, you now know that getting 70% accuracy would represent a 30 percentage point improvement over the baseline, which is substantial.
# 
# This baseline helps you avoid the pitfall of being impressed by a model that achieves, for example, 50% accuracy, when that's only marginally better than what you could achieve with no real learning.

# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Default Settings with `SVC`
# 
# **10 Points**
# 
# Now, define an `SVC` estimator with default parameters and fit it to the training data `X_train` and `y_train`. Assign this estimator to `svc` below.
# 
# Next, use the function `score` on `svc` with arguments `X_test` and `y_test`. Assign your answer as a float to `svc_defaults` below.

# In[8]:


### GRADED
svc = SVC().fit(X_train, y_train)
svc_defaults = svc.score(X_test, y_test)

### ANSWER CHECK
print(svc_defaults)


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Grid Searching with `SVC`
# 
# **10 Points**
# 
# While your `svc` should improve upon the baseline score, there is possible room for improvement.  Below, use `GridSearchCV` to grid search the different kernels available with the `SVC` estimator and some different parameters defined by the `params` dictionary below. Fit this estimator to the training data. Assign this result to `grid`.
# 
# Next, use the function `score` on `grid` with arguments `X_test` and `y_test`. Assign your answer as a float to `grid_score` below.
# 

# In[10]:


params = {'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
         'gamma': [0.1, 1.0, 10.0, 100.0],}


# In[16]:


get_ipython().run_line_magic('pinfo', 'GridSearchCV')


# In[17]:


### GRADED
grid = GridSearchCV(estimator=SVC(), param_grid=params).fit(X_train, y_train)
grid_score = grid.score(X_test, y_test)

### ANSWER CHECK
print(grid_score)


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Extract the results from the grid search
results = pd.DataFrame(grid.cv_results_)

# Create a figure with subplots for each kernel
plt.figure(figsize=(20, 15))

# Define the kernels and gamma values
kernels = ['rbf', 'poly', 'linear', 'sigmoid']
gamma_values = [0.1, 1.0, 10.0, 100.0]

# Plot 1: Bar chart comparing all combinations
plt.subplot(2, 2, 1)
# Reshape the data for easier plotting
plot_data = []
for i, row in results.iterrows():
    params = row['params']
    plot_data.append({
        'kernel': params['kernel'],
        'gamma': params['gamma'] if params['kernel'] != 'linear' else 'N/A',
        'score': row['mean_test_score']
    })
    
plot_df = pd.DataFrame(plot_data)

# Create a categorical plot
sns.barplot(x='kernel', y='score', hue='gamma', data=plot_df)
plt.title('Test Scores by Kernel and Gamma', fontsize=14)
plt.xlabel('Kernel', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0.4, 1.0)  # Start from baseline score
plt.legend(title='Gamma')

# Plot 2: Line plots for each kernel (except linear) showing gamma effect
plt.subplot(2, 2, 2)
for kernel in [k for k in kernels if k != 'linear']:
    kernel_data = plot_df[plot_df['kernel'] == kernel]
    plt.plot(kernel_data['gamma'], kernel_data['score'], marker='o', label=kernel)

plt.title('Effect of Gamma on Different Kernels', fontsize=14)
plt.xlabel('Gamma Value', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xscale('log')  # Log scale for gamma values
plt.ylim(0.4, 1.0)  # Start from baseline score
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Heatmap visualization
plt.subplot(2, 2, 3)
# Create a pivot table for the heatmap
heatmap_data = plot_df.pivot_table(
    index='kernel', 
    columns='gamma', 
    values='score',
    aggfunc='mean'
)
sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=0.4, vmax=1.0)
plt.title('Score Heatmap by Kernel and Gamma', fontsize=14)

# Plot 4: Best model parameters and comparison with baseline
plt.subplot(2, 2, 4)
best_params = grid.best_params_
best_score = grid.best_score_

text = f"Best Model:\nKernel: {best_params['kernel']}\n"
if best_params['kernel'] != 'linear':
    text += f"Gamma: {best_params['gamma']}\n"
text += f"Best CV Score: {best_score:.4f}\n"
text += f"Test Score: {grid_score:.4f}\n\n"
text += f"Baseline Score: {baseline_score:.4f}\n"
text += f"Improvement: {grid_score - baseline_score:.4f} (+{(grid_score - baseline_score) * 100:.1f}%)"

plt.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.show()

# Additional visualization: Individual plots for each kernel
plt.figure(figsize=(15, 10))
for i, kernel in enumerate(kernels):
    plt.subplot(2, 2, i+1)
    
    if kernel == 'linear':
        # Linear kernel doesn't use gamma
        linear_score = plot_df[plot_df['kernel'] == 'linear']['score'].values[0]
        plt.bar(['linear'], [linear_score])
        plt.axhline(y=baseline_score, color='r', linestyle='--', label='Baseline')
        plt.title(f'Linear Kernel (gamma N/A)', fontsize=14)
        plt.ylim(0.4, 1.0)
    else:
        kernel_data = plot_df[plot_df['kernel'] == kernel]
        plt.plot(kernel_data['gamma'], kernel_data['score'], marker='o')
        plt.axhline(y=baseline_score, color='r', linestyle='--', label='Baseline')
        plt.title(f'{kernel.capitalize()} Kernel', fontsize=14)
        plt.xlabel('Gamma Value', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xscale('log')
        plt.ylim(0.4, 1.0)
        plt.grid(True, alpha=0.3)
    
    plt.legend()

plt.tight_layout()
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Extract the results from grid search
results = pd.DataFrame(grid.cv_results_)

# Create a dataframe with all parameter combinations and their scores
plot_data = []
for i, row in results.iterrows():
    params = row['params']
    # Handle the fact that linear kernel doesn't use gamma
    gamma_value = params.get('gamma', 'N/A')
    if params['kernel'] == 'linear':
        gamma_value = 'N/A'
    
    plot_data.append({
        'kernel': params['kernel'],
        'gamma': gamma_value,
        'score': row['mean_test_score']
    })

plot_df = pd.DataFrame(plot_data)

# Find the best score for each kernel
best_by_kernel = plot_df.loc[plot_df.groupby('kernel')['score'].idxmax()]

# Create a figure
plt.figure(figsize=(14, 10))

# 1. Bar chart of best kernel/gamma combinations
plt.subplot(2, 1, 1)
best_by_kernel = best_by_kernel.sort_values('score', ascending=False)
bars = plt.bar(
    range(len(best_by_kernel)), 
    best_by_kernel['score'],
    color=sns.color_palette("viridis", len(best_by_kernel))
)

# Add baseline reference line
plt.axhline(y=baseline_score, color='red', linestyle='--', label=f'Baseline Score: {baseline_score:.3f}')

# Add data labels on top of bars
for i, bar in enumerate(bars):
    kernel = best_by_kernel.iloc[i]['kernel']
    gamma = best_by_kernel.iloc[i]['gamma']
    score = best_by_kernel.iloc[i]['score']
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height() + 0.01, 
        f'{score:.3f}', 
        ha='center', fontsize=11
    )
    plt.text(
        bar.get_x() + bar.get_width()/2, 
        bar.get_height()/2, 
        f"{kernel}\nγ={gamma}", 
        ha='center', va='center', 
        color='white', fontweight='bold', fontsize=11
    )

plt.xticks([])  # Hide x-ticks as labels are in bars
plt.ylabel('Score', fontsize=12)
plt.title('Best Kernel and Gamma Combinations', fontsize=14, fontweight='bold')
plt.ylim(0.3, 1.0)  # Start slightly below baseline
plt.legend()

# 2. Heatmap of all kernel/gamma combinations
plt.subplot(2, 1, 2)

# Create a pivot table for the heatmap
# Convert gamma to string for proper sorting in heatmap
plot_df['gamma_str'] = plot_df['gamma'].astype(str)
heatmap_data = plot_df.pivot_table(
    index='kernel', 
    columns='gamma_str', 
    values='score',
    aggfunc='mean'
)

# Sort columns numerically (except 'N/A')
cols = sorted([c for c in heatmap_data.columns if c != 'N/A'], 
              key=lambda x: float(x) if x != 'N/A' else 0)
if 'N/A' in heatmap_data.columns:
    cols.append('N/A')
heatmap_data = heatmap_data[cols]

# Create the heatmap
sns.heatmap(
    heatmap_data, 
    annot=True, 
    cmap='viridis', 
    vmin=baseline_score, 
    vmax=np.ceil(plot_df['score'].max() * 10) / 10,
    fmt='.3f',
    linewidths=0.5
)

# Highlight the best score in each kernel
for i, kernel in enumerate(heatmap_data.index):
    kernel_data = plot_df[plot_df['kernel'] == kernel]
    best_gamma_str = kernel_data.loc[kernel_data['score'].idxmax()]['gamma_str']
    if best_gamma_str in heatmap_data.columns:
        j = list(heatmap_data.columns).index(best_gamma_str)
        plt.text(j + 0.5, i + 0.5, '★', 
                 ha='center', va='center', color='white', fontsize=20)

plt.title('Score Heatmap by Kernel and Gamma\n(★ indicates best gamma for each kernel)', fontsize=14)

# Add text about the overall best model
best_params = grid.best_params_
best_score = grid.best_score_

text_x = 0.5
text_y = -0.15
plt.figtext(text_x, text_y, 
            f"Best Model: Kernel={best_params['kernel']}, " + 
            (f"Gamma={best_params['gamma']}" if 'gamma' in best_params else "Gamma=N/A") + 
            f", CV Score={best_score:.4f}, Test Score={grid_score:.4f}\n" +
            f"Improvement over baseline: +{(grid_score - baseline_score):.4f} absolute " +
            f"(+{(grid_score - baseline_score)/baseline_score*100:.1f}% relative)",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the text
plt.show()


# In[ ]:





# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Optimal Kernel Function
# 
# **10 Points**
# 
# Based on your grid search above what is the best performing kernel function?  Assign your answer as a string -- `linear`, `poly`, `rbf`, or `sigmoid` -- to `best_kernel` below.  

# In[20]:


### GRADED
best_kernel = 'poly'

### ANSWER CHECK
print(best_kernel)


# In[ ]:





# In[ ]:




