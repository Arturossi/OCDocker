#!/usr/lib/python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to cluster molecules based on their
rmsd.

They are imported as:

import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
'''

# Imports
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from scipy.cluster.hierarchy import ClusterWarning
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score

from typing import Dict, List, Union
from warnings import simplefilter

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.Initialise import *

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##
def get_medoids(data: Union[Dict[str, Dict[str, float]], pd.DataFrame], algorithm: str = 'agglomerativeClustering', outputPlot: str = "") -> Union[List[str], int]:
    '''Cluster molecules based on their rmsd.

    Parameters
    ----------
    data : Union[Dict[str, Dict[str, float]], pd.DataFrame]
        The rmsd matrix.
    algorithm : str, optional
        The clustering algorithm to be used. The default is 'agglomerativeClustering'. The options are: 'agglomerativeClustering'.
    outputPlot : str, optional
        The path to the output plot. The default is "". If it is "", the plot is not saved.

    Returns
    -------
    List[str] | int
        The medoids of the clusters if no errros occur. Otherwise, the error code.
    '''

    # Check if the data is a dict
    if isinstance(data, dict):
        # Convert the dict to a DataFrame
        data = pd.DataFrame(data)
    
    # If the shape[0] is 1, return it
    if data.shape[0] == 1:
        # Print the warning
        ocprint.print_warning(f"The shape of the data is {data.shape}. There is no need to cluster it.")
        # Return the only column
        return data.columns.tolist()

    # Convert the dataframe into numpy arrays to be used by the clustering algorithm
    npdata = data.to_numpy()

    # Check if the algorithm is agglomerativeClustering
    if algorithm.lower() == 'agglomerativeclustering':
        # Ignore the cluster warning (the matrices are too small, thus the warning keeps popping up)
        simplefilter("ignore", ClusterWarning)

        # Initialise the results and scores
        results = []
        scores = []

        # Use elbow method to find the best number of clusters
        for i in range(2, npdata.shape[0]):
            # Perform the clustering
            analysis = AgglomerativeClustering(n_clusters = i).fit_predict(npdata)
            # Get the silhouette score
            silhouette_avg = silhouette_score(npdata, analysis)

            # Append the results and scores
            results.append(analysis)
            scores.append(silhouette_avg)
        
        # If the outputPlot is not ""
        if outputPlot != "":
            # Create a dendrogram for visualization
            linkage_matrix = sch.linkage(npdata, method='ward')  # Adjust the linkage method as needed
            _ = sch.dendrogram(linkage_matrix)
            plt.title('Agglomerative Clustering Dendrogram')
            plt.xlabel('Data Points')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.savefig(outputPlot)
        
        # Get the best result
        clusters = results[scores.index(max(scores))]

        # Get the unique clusters
        unique_clusters = np.unique(clusters)

        # Initialize a list to store medoids
        medoids = []

        # Calculate medoid for each cluster
        for cluster in unique_clusters:
            # Select data points belonging to the current cluster
            cluster_data = data[clusters == cluster]

            # Calculate pairwise distances within the cluster
            distances = pairwise_distances(cluster_data, metric='euclidean')

            # Calculate the sum of distances for each data point
            sum_distances = np.sum(distances, axis=1)
            
            # Find the index of the data point with the smallest sum of distances
            medoid_index = np.argmin(sum_distances)

            # Get the index name
            medoid_index_label = cluster_data.index[medoid_index]
            
            # Append the medoid to the list of medoids
            medoids.append(medoid_index_label)

        # Return the medoid paths
        return medoids

    else:
        return errors.unsupported_clustering_algorithm(f"The clustering algorithm '{algorithm}' is not supported. Currently the supported algorithms are: 'agglomerativeClustering'.")
