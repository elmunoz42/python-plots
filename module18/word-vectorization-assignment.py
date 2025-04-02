#!/usr/bin/env python
# coding: utf-8

# ### Required Assignment: Word Vectorization Techniques
# 
# **Expected Time = 60 minutes** 
# 
# **Total Points = 50**
# 
# This activity focuses on converting text into numerical vectors, which is essential for machine learning with text data. You will use different techniques to transform text into vector representations, including:
# 1. Bag of Words (CountVectorizer)
# 2. TF-IDF (Term Frequency-Inverse Document Frequency)
# 3. Word2Vec (Word Embeddings)
# 
# #### Index
# 
# - [Problem 1](#-Problem-1)
# - [Problem 2](#-Problem-2)
# - [Problem 3](#-Problem-3)
# - [Problem 4](#-Problem-4)
# - [Problem 5](#-Problem-5)

# #### Setup and Data Loading

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')


# In[2]:


# Load the data
happy_df = pd.read_csv('data/Emotion(happy).csv')
angry_df = pd.read_csv('data/Emotion(angry).csv')

# Display the first few rows
print("Happy dataset:")
print(happy_df.head())
print("\nAngry dataset:")
print(angry_df.head())


# [Back to top](#-Index)
# 
# ### Problem 1
# 
# #### Creating a Bag of Words Representation
# 
# **10 Points**
# 
# Use the `CountVectorizer` from scikit-learn to create a bag of words representation for the texts in the 'content' column of the happy_df DataFrame. Set the minimum document frequency to 2 to filter out very rare words.
# 
# Complete the function `create_bow` below that takes a pandas Series of text and returns:
# 1. The vectorizer object
# 2. The feature names (vocabulary)
# 3. The bag of words matrix

# In[3]:


### GRADED
def create_bow(text_series):
    """
    Create a bag of words representation from a series of text.
    
    Parameters:
    -----------
    text_series : pandas.Series
        A series containing text documents
        
    Returns:
    --------
    vectorizer : CountVectorizer
        The fitted vectorizer object
    feature_names : list
        The list of feature names (vocabulary)
    bow_matrix : scipy.sparse.csr.csr_matrix
        The bag of words matrix
    """
    # Initialize the vectorizer with min_df=2 to filter out rare words
    vectorizer = CountVectorizer(min_df=2)
    
    # Fit and transform the text series
    bow_matrix = vectorizer.fit_transform(text_series)
    
    # Get the feature names (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    
    return vectorizer, feature_names, bow_matrix

# Test the function
vectorizer, feature_names, bow_matrix = create_bow(happy_df['content'])

# Display results
print(f"Vocabulary size: {len(feature_names)}")
print(f"First 10 words in vocabulary: {feature_names[:10]}")
print(f"Shape of bag of words matrix: {bow_matrix.shape}")
print(f"Type of the matrix: {type(bow_matrix)}")


# [Back to top](#-Index)
# 
# ### Problem 2
# 
# #### Creating TF-IDF Vectors
# 
# **10 Points**
# 
# Now use the `TfidfVectorizer` from scikit-learn to create TF-IDF representations for the same texts. Again, set the minimum document frequency to 2.
# 
# Complete the function `create_tfidf` below that takes a pandas Series of text and returns:
# 1. The TF-IDF vectorizer object
# 2. The feature names (vocabulary)
# 3. The TF-IDF matrix

# In[4]:


### GRADED
def create_tfidf(text_series):
    """
    Create a TF-IDF representation from a series of text.
    
    Parameters:
    -----------
    text_series : pandas.Series
        A series containing text documents
        
    Returns:
    --------
    vectorizer : TfidfVectorizer
        The fitted vectorizer object
    feature_names : list
        The list of feature names (vocabulary)
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        The TF-IDF matrix
    """
    # YOUR CODE HERE
    return vectorizer, feature_names, tfidf_matrix

# Test the function
vectorizer, feature_names, tfidf_matrix = create_tfidf(happy_df['content'])

# Display results
print(f"Vocabulary size: {len(feature_names)}")
print(f"First 10 words in vocabulary: {feature_names[:10]}")
print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")


# [Back to top](#-Index)
# 
# ### Problem 3
# 
# #### Comparing Document Representations
# 
# **10 Points**
# 
# Use the cosine similarity to compare the first document to all other documents in the `happy_df` DataFrame, using both the bag of words and TF-IDF representations. 
# 
# Complete the function `compare_documents` below that:
# 1. Computes the cosine similarity between the first document and all others
# 2. Returns the indices of the top 3 most similar documents (excluding itself)

# In[5]:


### GRADED
def compare_documents(document_matrix, doc_index=0, top_n=3):
    """
    Compare a document to all other documents using cosine similarity.
    
    Parameters:
    -----------
    document_matrix : scipy.sparse.csr.csr_matrix
        Matrix of document vectors (bow or tfidf)
    doc_index : int
        Index of the document to compare
    top_n : int
        Number of top similar documents to return
        
    Returns:
    --------
    top_indices : numpy.ndarray
        Indices of the top_n most similar documents
    similarities : numpy.ndarray
        Similarity scores of the top_n most similar documents
    """
    # Extract the vector for the specified document
    
    # Compute cosine similarity between this document and all others
    
    # Get the indices of the top N most similar documents
    # (excluding the document itself)
    
    # YOUR CODE HERE
    return top_indices, similarities

# Compare using bag of words
bow_top_indices, bow_similarities = compare_documents(bow_matrix)
print("Top similar documents using BOW:")
for i, idx in enumerate(bow_top_indices):
    print(f"Document {idx}: Similarity {bow_similarities[i]:.4f}")
    print(f"Content: {happy_df['content'].iloc[idx][:100]}...")

# Compare using TF-IDF
tfidf_top_indices, tfidf_similarities = compare_documents(tfidf_matrix)
print("\nTop similar documents using TF-IDF:")
for i, idx in enumerate(tfidf_top_indices):
    print(f"Document {idx}: Similarity {tfidf_similarities[i]:.4f}")
    print(f"Content: {happy_df['content'].iloc[idx][:100]}...")


# [Back to top](#-Index)
# 
# ### Problem 4
# 
# #### Training a Word2Vec Model
# 
# **10 Points**
# 
# Now let's explore Word2Vec, which creates dense vector representations of words based on their context. These embeddings capture semantic relationships between words.
# 
# Complete the function `train_word2vec` below that:
# 1. Tokenizes each document in the text series
# 2. Trains a Word2Vec model on the tokenized documents
# 3. Returns the trained model

# In[6]:


### GRADED
def train_word2vec(text_series, vector_size=100, window=5, min_count=2):
    """
    Train a Word2Vec model on a series of text documents.
    
    Parameters:
    -----------
    text_series : pandas.Series
        A series containing text documents
    vector_size : int
        Dimension of the word vectors
    window : int
        Maximum distance between the current and predicted word
    min_count : int
        Ignores all words with total frequency lower than this
        
    Returns:
    --------
    model : Word2Vec
        The trained Word2Vec model
    """
    # Tokenize each document
    
    # Train the Word2Vec model
    
    # YOUR CODE HERE
    return model

# Train Word2Vec on the combined dataset for better coverage
combined_text = pd.concat([happy_df['content'], angry_df['content']])
w2v_model = train_word2vec(combined_text)

# Test the model by finding similar words
test_words = ['happy', 'good', 'sad', 'angry', 'love']
for word in test_words:
    if word in w2v_model.wv:
        print(f"\nMost similar words to '{word}':")
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        for similar_word, similarity in similar_words:
            print(f"  {similar_word}: {similarity:.4f}")
    else:
        print(f"\nWord '{word}' not in vocabulary")


# [Back to top](#-Index)
# 
# ### Problem 5
# 
# #### Visualizing Word Embeddings
# 
# **10 Points**
# 
# Finally, visualize the word embeddings created by Word2Vec in a 2D space using PCA.
# 
# Complete the function `visualize_embeddings` below that:
# 1. Extracts word vectors for the specified words
# 2. Reduces their dimensionality to 2D using PCA
# 3. Plots the words in a scatter plot

# In[7]:


### GRADED
def visualize_embeddings(model, words, figsize=(12, 8)):
    """
    Visualize word embeddings in 2D space using PCA.
    
    Parameters:
    -----------
    model : Word2Vec
        The trained Word2Vec model
    words : list
        List of words to visualize
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    None (displays the plot)
    """
    # Filter words to those present in the model's vocabulary
    
    # Extract the word vectors
    
    # Reduce dimensionality with PCA
    
    # Create a plot
    
    # YOUR CODE HERE
    return

# Define words to visualize (emotions and related words)
words_to_visualize = [
    'happy', 'joy', 'love', 'smile', 'laugh', 'excited', 'fun',
    'sad', 'angry', 'upset', 'mad', 'hate', 'disappointment',
    'good', 'bad', 'better', 'worse', 'best', 'terrible',
    'friend', 'family', 'work', 'life', 'time', 'day'
]

# Visualize word embeddings
visualize_embeddings(w2v_model, words_to_visualize)
