from sklearn.cluster import KMeans
import numpy as np


def check_inertia_of_range_of_clusters(X, upper_n_limit, init_value='k-means++', random_state=None):
    """
    Calculate K-means inertia scores for a range of cluster numbers to help determine optimal k.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data to cluster
    upper_n_limit : int
        Upper limit of number of clusters to test (exclusive)
    init_value : str or array-like, default='k-means++' 
        Method for initialization:
        'k-means++' : selects initial cluster centers using k-means++ algorithm
        'random': choose n_clusters observations (rows) at random from data
    random_state : int, optional (default=None)
        Determines random number generation for centroid initialization
        Use an int for reproducible results
    
    Returns:
    --------
    list
        Inertia scores for each number of clusters from 1 to upper_n_limit-1
    
    Example:
    --------
    >>> inertias = check_inertia_of_range_of_clusters(X, 11, 'k-means++', 42)
    >>> plt.plot(range(1,11), inertias)  # Create elbow plot
    """
    
    # Input validation
    if not isinstance(upper_n_limit, int) or upper_n_limit < 2:
        raise ValueError("upper_n_limit must be an integer greater than 1")
    
    if init_value not in ['k-means++', 'random'] and not isinstance(init_value, np.ndarray):
        raise ValueError("init_value must be 'k-means++', 'random', or array-like")
        
    inertia_array = [] 
    for n in range(1, upper_n_limit):
        if random_state is not None:
            kmeans_temp = KMeans(n_clusters=n, init=init_value, random_state=random_state).fit(X)
        else:
            kmeans_temp = KMeans(n_clusters=n, init=init_value).fit(X)
        i = kmeans_temp.inertia_
        inertia_array.append(i)
        
    return inertia_array




def exampine_number_of_clusters_for_array_of_eps(epsilons, X):
    n_clusters_list = []
    for eps in epsilons:
        db = DBSCAN(eps = eps).fit(X)
        n_clusters = len(np.unique(db.labels_))
        n_clusters_list.append(n_clusters)
    plt.plot(epsilons, n_clusters_list)
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Clusters')
    plt.title('How the Number of Clusters varies with eps')
    
    
exampine_number_of_clusters_for_array_of_eps(epsilons)