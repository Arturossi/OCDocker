#!/usr/bin/env python3

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
import matplotlib

matplotlib.use('agg')

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
def get_medoids(data: Union[Dict[str, Dict[str, float]], pd.DataFrame], clusters: np.ndarray, onlyBiggest: bool = True) -> List[str]:
    '''Get the medoids of the clusters.

    Parameters
    ----------
    data : Union[Dict[str, Dict[str, float]], pd.DataFrame]
        The rmsd matrix.
    clusters : np.ndarray
        The clusters.
    onlyBiggest : bool, optional
        If True, only the medoid of the biggest clusters are returned. The default is True.

    Returns
    -------
    List[str]
        The paths to the medoids.
    '''

    # Check if the data is a dict
    if isinstance(data, dict):
        # Convert the dict to a DataFrame
        data = pd.DataFrame(data)

    if isinstance(clusters, int):
        print(clusters)
    
    # Check if the clusters is an int or is not empty or invalid
    if isinstance(clusters, int) or clusters.size == 0 or np.any(clusters < 0):
        return []
    
    # If onlyBiggest is True
    if onlyBiggest:
        # Get the size of each cluster
        cluster_sizes = np.bincount(clusters)

        # Get the label of the biggest clusters (may be more than one)
        unique_clusters = np.where(cluster_sizes == np.max(cluster_sizes))[0]
    else:
        # Get the unique clusters
        unique_clusters = np.unique(clusters)

    # Initialize a list to store medoids
    medoids = []

    # Calculate medoid for each cluster
    for cluster in unique_clusters:
        # Select data points belonging to the current cluster
        cluster_data = data[clusters == cluster]

        # Check if the cluster is empty
        if cluster_data.empty:
            _ = ocerror.Error.empty_cluster(f"The cluster {cluster} is empty.") # type: ignore
            continue

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

def cluster_rmsd(data: Union[Dict[str, Dict[str, float]], pd.DataFrame], algorithm: str = 'agglomerativeClustering', max_distance_threshold: float = 20.0, min_distance_threshold: float = 10.0, threshold_step: float = 0.1, outputPlot: str = "") -> Union[np.ndarray, int]:
    '''Cluster molecules based on their rmsd.

    Parameters
    ----------
    data : Union[Dict[str, Dict[str, float]], pd.DataFrame]
        The rmsd matrix.
    algorithm : str, optional
        The clustering algorithm to be used. The default is 'agglomerativeClustering'. The options are: 'agglomerativeClustering'.
    min_distance_threshold : float, optional
        The minimum distance threshold for the agglomerative clustering. The default is 10.0.
    max_distance_threshold : float, optional
        The maximum distance threshold for the agglomerative clustering. The default is 20.0.
    threshold_step : float, optional
        The step to perform the distance threshold search. The default is 0.1.
    outputPlot : str, optional
        The path to the output plot. The default is "". If it is "", the plot is not saved.

    Returns
    -------
    np.ndarray | int
        The clusters or the error code. IMPORTANT: The error code 751 means that the cluster could not determine any consensus among the poses. This means that the poses are too different from each other. In this case, the poses should be discarded.
    '''

    # Check if max_distance_threshold is smaller than min_distance_threshold
    if max_distance_threshold < min_distance_threshold:
        # Return the value error
        return ocerror.Error.value_error(f"The max_distance_threshold ({max_distance_threshold}) is smaller than the min_distance_threshold ({min_distance_threshold}).") # type: ignore

    # Check if the data is a dict
    if isinstance(data, dict):
        # Convert the dict to a DataFrame
        data = pd.DataFrame(data)
    
    # If the shape[0] is 1, return it
    if data.shape[0] == 1:
        # Print the warning
        ocprint.print_warning(f"The shape of the data is {data.shape}. There is no need to cluster it.")
        # Return the only column as a single cluster (np.array with 0.0)
        return np.array([0.0])

    # Convert the dataframe into numpy arrays to be used by the clustering algorithm
    npdata = data.to_numpy()

    # Check if the algorithm is agglomerativeClustering
    if algorithm.lower() == 'agglomerativeclustering':
        # Ignore the cluster warning (the matrices are too small, thus the warning keeps popping up)
        simplefilter("ignore", ClusterWarning)

        # Define the scores and distance_threshold as -1
        scores = -1
        distance_threshold = -1

        # Define the last computed result
        last_result = np.array([])

        # Create the loop to iterate from max_distance_threshold to min_distance_threshold using step threshold_step
        for distance_threshold in np.arange(max_distance_threshold, min_distance_threshold, -threshold_step):
            # Perform the clustering
            results = AgglomerativeClustering(n_clusters = None, distance_threshold = distance_threshold).fit_predict(npdata)

            # Get the number oe elements in each cluster
            cluster_sizes = np.bincount(results)

            # Get the unique clusters
            unique_clusters = np.unique(results)

            # If the length of the unique clusters is the same as the shape of the data (every element is a cluster)
            if len(unique_clusters) == data.shape[0]:
                # If last_result is not empty
                if last_result.size != 0:
                    # Set the results to the last result
                    results = last_result
                    # Break the loop
                    break
                else:
                    # Print the message, returning the error code
                    return ocerror.Error.cluster_not_converged(f"The clustering algorithm did not converge. The distance threshold is {distance_threshold}.") # type: ignore

            # Find the biggest cluster (may be more than one)
            biggest_cluster = np.where(cluster_sizes == np.max(cluster_sizes))[0]

            # If the biggest cluster is 1
            if len(biggest_cluster) == 1:
                # If there is only one cluster, do not perform the silhouette score
                if len(unique_clusters) > 1:
                    # Get the silhouette score
                    scores = silhouette_score(npdata, results)
                    # Break the loop
                    break
            else:
                # Set the last result to the current result
                last_result = results

        # If the scores is -1
        if scores == -1:
            # Print the message, returning the error code
            return ocerror.Error.cluster_not_converged(f"The clustering algorithm did not converge. The distance threshold is {distance_threshold}.") # type: ignore

        # If the outputPlot is not ""
        if outputPlot != "":
            # Create a dendrogram for visualization
            linkage_matrix = sch.linkage(npdata, method='ward')  # Adjust the linkage method as needed
            _ = sch.dendrogram(linkage_matrix)
            plt.title('Agglomerative Clustering Dendrogram')
            plt.xlabel('Data Points')
            plt.ylabel('Distance')
            # Extend the y-axis limits, adding a bit of buffer at the top to allow the text to fit
            plt.ylim(0, max(linkage_matrix[:, 2]) * 1.2)
            # Add a red line at the distance threshold
            plt.axhline(y=distance_threshold, color='r', linestyle='--')
            # Add the silhouette score (left, top) rounded to 2 decimals
            plt.text(0.05, 0.95, f"Silhouette Score: ~{round(scores, 2)}", transform=plt.gca().transAxes, size=10, verticalalignment='top', horizontalalignment='left')
            # Add a label to the distance threshold below the silhouette score
            plt.text(0.05, 0.9, f"Distance Threshold: {round(distance_threshold, 2)}", transform=plt.gca().transAxes, size=10, verticalalignment='top', horizontalalignment='left')
            plt.savefig(outputPlot)
        
        # Return the results
        return results # type: ignore
    
    else:
        return ocerror.Error.unsupported_clustering_algorithm(f"The clustering algorithm '{algorithm}' is not supported. Currently the supported algorithms are: 'agglomerativeClustering'.") # type: ignore
