import pandas as pd
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



############################################################################

# Clustering

############################################################################



def create_cluster(train, X, k):
    '''
    takes in train, X df with variables to cluster on, and k.
    It scales the X, calculates the clusters, 
    returns train (with clusters), 
    the Scaled dataframe,
    the scaler,
    kmeans object, 
    and unscaled centroids as a dataframe
    '''
    #scale X
    scaler = MinMaxScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    #calculate clusters
    kmeans = KMeans(n_clusters = k, random_state = 123)
    kmeans.fit(X_scaled)
    kmeans.predict(X_scaled)
    train['cluster'] = kmeans.predict(X_scaled)
    train['cluster'] = 'cluster_' + train.cluster.astype(str)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return train, X_scaled, scaler, kmeans, centroids


def cluster_scatter_plot(x,y,train,kmeans, X_scaled, scaler):
    '''
    takes in x and y variable names as strings, 
    along with returned objects from previous create_cluster function,
    and creates a plot
    '''
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x = x, y = y, data = train, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')                        
            
          