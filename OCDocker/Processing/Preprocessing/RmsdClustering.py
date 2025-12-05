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

import OCDocker.Error as ocerror

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
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


def cluster_rmsd(data: Union[Dict[str, Dict[str, float]], pd.DataFrame], algorithm: str = 'agglomerativeClustering', max_distance_threshold: float = 20.0, min_distance_threshold: float = 10.0, threshold_step: float = 0.1, outputPlot: str = "", molecule_name: str = "") -> Union[np.ndarray, int]:
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
    molecule_name : str, optional
        The name of the molecule to include in the plot title. The default is "".

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

        # If the scores is -1 (clustering didn't converge)
        if scores == -1:
            # Check if last_result has any clusters with more than 1 member
            if last_result.size > 0:
                cluster_sizes_last = np.bincount(last_result)
                # Check if any cluster has more than 1 member
                if np.any(cluster_sizes_last > 1):
                    # Find the maximum cluster size (clusters with most members)
                    max_cluster_size = np.max(cluster_sizes_last)
                    
                    # Get clusters with the maximum size
                    max_size_clusters = np.where(cluster_sizes_last == max_cluster_size)[0]
                    
                    # Find the cluster with the least difference among its members
                    # (minimum maximum pairwise distance within cluster)
                    # Only consider clusters with the maximum number of members
                    unique_clusters_last = np.unique(last_result)
                    min_max_distance = np.inf
                    best_cluster = -1
                    
                    for cluster in unique_clusters_last:
                        # Only consider clusters with the maximum size
                        if cluster in max_size_clusters:
                            # Get members of this cluster
                            cluster_indices = np.where(last_result == cluster)[0]
                            
                            # Only consider clusters with more than 1 member
                            if len(cluster_indices) > 1:
                                # Get pairwise distances within this cluster
                                cluster_data = npdata[cluster_indices]
                                cluster_distances = pairwise_distances(cluster_data, metric='euclidean')
                                # Maximum distance within cluster (diameter)
                                max_distance_in_cluster = np.max(cluster_distances)
                                
                                # Track cluster with smallest maximum distance
                                if max_distance_in_cluster < min_max_distance:
                                    min_max_distance = max_distance_in_cluster
                                    best_cluster = cluster
                    
                    # If we found a cluster with multiple members, use it
                    if best_cluster >= 0:
                        ocprint.print_warning(f"Clustering did not fully converge. Using cluster {best_cluster} (size: {max_cluster_size}) with smallest internal variance (max pairwise distance: {min_max_distance:.2f}).")
                        return last_result
                    
            # If all clusters have only 1 member, fail
            ocprint.print_warning("All clusters have only 1 member. Clustering failed.")
            # Print the message, returning the error code
            return ocerror.Error.cluster_not_converged(f"The clustering algorithm did not converge. The distance threshold is {distance_threshold}.") # type: ignore

        # If the outputPlot is not ""
        if outputPlot != "":
            try:
                # Create a dendrogram for visualization
                linkage_matrix = sch.linkage(npdata, method='ward')
                
                # Get cluster assignments at the distance threshold
                clusters_at_threshold = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit_predict(npdata)
                unique_clusters = np.unique(clusters_at_threshold)
                n_clusters = len(unique_clusters)
                
                # Debug: Print cluster information
                ocprint.printv(f"Dendrogram: {len(clusters_at_threshold)} data points form {n_clusters} clusters at threshold {distance_threshold}")
                for cluster_id in unique_clusters:
                    cluster_members = np.where(clusters_at_threshold == cluster_id)[0]
                    ocprint.printv(f"  Cluster {cluster_id}: {len(cluster_members)} members (indices: {cluster_members.tolist()})")
                
                # Get medoids (representative elements) for highlighting
                medoids = get_medoids(data, results, onlyBiggest=True)  # type: ignore
                medoid_indices = set()
                if isinstance(data, pd.DataFrame):
                    for medoid_path in medoids:
                        if medoid_path in data.index:
                            medoid_indices.add(data.index.get_loc(medoid_path))
                
                # Define colors for clusters using colorblind-friendly palette
                # Use Set3 (colorblind-friendly, no blue shades) or Set2 for small clusters
                import matplotlib.cm as cm
                cluster_colors = []
                
                # Set3 is colorblind-friendly and doesn't use blue shades
                if n_clusters <= 12:
                    cmap = cm.get_cmap('Set3')
                    cluster_colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
                elif n_clusters <= 20:
                    # For more clusters, use Set2 (8 colors) and cycle
                    cmap = cm.get_cmap('Set2')
                    # Create more colors by cycling
                    base_colors = [cmap(i / 7.0) for i in range(8)]
                    for i in range(n_clusters):
                        cluster_colors.append(base_colors[i % 8])
                else:
                    # For many clusters, use a custom palette avoiding blue
                    # Use colors from Set1, Set2, Set3, and Pastel1
                    colors1 = [cm.get_cmap('Set1')(i / 8.0) for i in range(9)]
                    colors2 = [cm.get_cmap('Set2')(i / 7.0) for i in range(8)]
                    colors3 = [cm.get_cmap('Set3')(i / 11.0) for i in range(12)]
                    # Combine and filter out blue shades (colors with high blue component)
                    all_colors = colors1 + colors2 + colors3
                    # Filter out colors that are too blue (blue component > 0.6)
                    filtered_colors = [c for c in all_colors if c[2] < 0.6]  # RGB: c[2] is blue
                    if len(filtered_colors) < n_clusters:
                        # If not enough colors, add some from Pastel1
                        pastel_colors = [cm.get_cmap('Pastel1')(i / 8.0) for i in range(9)]
                        filtered_colors.extend([c for c in pastel_colors if c[2] < 0.6])
                    cluster_colors = filtered_colors[:n_clusters]
                
                cluster_color_map = {int(cluster_id): cluster_colors[i] for i, cluster_id in enumerate(unique_clusters)}
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create dendrogram - this will create collections for each cluster
                # Ensure all leaves are shown by setting count_sort and distance_sort
                dendro_dict = sch.dendrogram(
                    linkage_matrix,
                    color_threshold=distance_threshold,
                    above_threshold_color='gray',  # Use gray instead of blue for colorblind-friendliness
                    ax=ax,
                    count_sort=False,  # Don't sort by count
                    distance_sort=False,  # Don't sort by distance
                    show_leaf_counts=True,  # Show leaf counts if needed
                    no_plot=False  # Ensure plotting happens
                )
                
                # Get leaf order from dendrogram
                leaf_order = dendro_dict['leaves']
                n_leaves = len(leaf_order)
                
                # Verify we have all data points as leaves
                if n_leaves != len(clusters_at_threshold):
                    ocprint.print_warning(f"Dendrogram shows {n_leaves} leaves but expected {len(clusters_at_threshold)} data points. Some points may be merged at distance 0.")
                    ocprint.print_warning(f"Leaf order: {leaf_order}, Expected indices: {list(range(len(clusters_at_threshold)))}")
                
                # Create a mapping from original index to cluster ID
                original_to_cluster = {i: int(clusters_at_threshold[i]) for i in range(len(clusters_at_threshold))}
                
                # Build a mapping from each internal node to its cluster ID
                # by checking which cluster all leaves under that node belong to
                def get_node_cluster(node_id, n):
                    """Get cluster ID for a dendrogram node."""
                    if node_id < n:
                        # Leaf node - return its cluster
                        return int(clusters_at_threshold[node_id])
                    else:
                        # Internal node - check linkage matrix
                        link_idx = node_id - n
                        if link_idx < len(linkage_matrix):
                            merge_dist = linkage_matrix[link_idx, 2]
                            # If merge is above threshold, return -1 (blue)
                            if merge_dist > distance_threshold:
                                return -1
                            # Get children clusters
                            child1 = int(linkage_matrix[link_idx, 0])
                            child2 = int(linkage_matrix[link_idx, 1])
                            cluster1 = get_node_cluster(child1, n)
                            cluster2 = get_node_cluster(child2, n)
                            # If both children are in same cluster, return that cluster
                            if cluster1 == cluster2 and cluster1 >= 0:
                                return cluster1
                            # Different clusters or above threshold
                            return -1
                        return -1
                
                # Map each collection to its cluster by finding the topmost node in that collection
                # and determining its cluster
                # IMPORTANT: Only color collections that are ENTIRELY below the threshold
                n = len(clusters_at_threshold)
                collection_to_cluster = {}
                
                for i, collection in enumerate(ax.collections):
                    paths = collection.get_paths()
                    if not paths:
                        continue
                    
                    # Check if this collection is ENTIRELY below threshold
                    # (all y-coordinates must be <= threshold)
                    max_y = -np.inf
                    min_y = np.inf
                    for path in paths:
                        vertices = path.vertices
                        if len(vertices) > 0 and vertices.shape[1] >= 2:
                            y_coords = vertices[:, 1]
                            max_y = max(max_y, np.max(y_coords))
                            min_y = min(min_y, np.min(y_coords))
                    
                    # Only color if the ENTIRE collection is below the threshold
                    # (max_y must be <= threshold, and we want to ensure it doesn't cross)
                    if max_y <= distance_threshold and min_y <= distance_threshold:
                        # Find the topmost node for this collection
                        # The topmost node corresponds to the highest y-coordinate
                        top_y = -np.inf
                        top_x = None
                        for path in paths:
                            vertices = path.vertices
                            if len(vertices) > 0 and vertices.shape[1] >= 2:
                                y_coords = vertices[:, 1]
                                x_coords = vertices[:, 0]
                                max_idx = np.argmax(y_coords)
                                if y_coords[max_idx] > top_y:
                                    top_y = y_coords[max_idx]
                                    top_x = x_coords[max_idx]
                        
                        # Find which internal node this corresponds to
                        # by checking linkage matrix for nodes at this distance
                        cluster_id = -1
                        for link_idx in range(len(linkage_matrix)):
                            if abs(linkage_matrix[link_idx, 2] - top_y) < 0.01:  # Small tolerance
                                node_id = n + link_idx
                                cluster_id = get_node_cluster(node_id, n)
                                if cluster_id >= 0:
                                    break
                        
                        # If we couldn't find by distance, try finding by leaf membership
                        if cluster_id < 0:
                            leaf_positions_in_collection = set()
                            for path in paths:
                                vertices = path.vertices
                                if len(vertices) > 0 and vertices.shape[1] >= 2:
                                    x_coords = vertices[:, 0]
                                    for x in x_coords:
                                        leaf_pos = int(round(x))
                                        if 0 <= leaf_pos < n_leaves:
                                            leaf_positions_in_collection.add(leaf_pos)
                            
                            if leaf_positions_in_collection:
                                original_indices = [leaf_order[pos] for pos in leaf_positions_in_collection]
                                cluster_ids = [original_to_cluster.get(idx, -1) for idx in original_indices]
                                cluster_ids = [c for c in cluster_ids if c >= 0]
                                if cluster_ids:
                                    cluster_id = max(set(cluster_ids), key=cluster_ids.count)
                        
                        collection_to_cluster[i] = cluster_id
                    else:
                        # Above threshold or crosses threshold - must be blue
                        collection_to_cluster[i] = -1
                
                # Now apply colors to collections
                # Only collections entirely below threshold get colored, all others are blue
                # NOTE: The number of colored branches equals the number of clusters at the threshold,
                # not the number of data points. If multiple points merge below threshold, they form one colored branch.
                colored_count = 0
                blue_count = 0
                for i, collection in enumerate(ax.collections):
                    cluster_id = collection_to_cluster.get(i, -1)
                    if cluster_id >= 0 and cluster_id in cluster_color_map:
                        # Only apply color if collection is entirely below threshold
                        collection.set_color(cluster_color_map[cluster_id])
                        colored_count += 1
                    else:
                        # Above threshold or crosses threshold - use gray instead of blue for colorblind-friendliness
                        collection.set_color('gray')
                        blue_count += 1
                
                # Debug output
                ocprint.printv(f"Dendrogram: {len(clusters_at_threshold)} data points, {n_clusters} clusters at threshold {distance_threshold:.2f}")
                ocprint.printv(f"  Colored branches (clusters): {colored_count}, Blue branches (above threshold): {blue_count}")
                ocprint.printv(f"  Total collections: {len(ax.collections)}")
                
                # Highlight representative elements (medoids) with a marker
                # Find leaf positions and x-coordinates for medoids in the dendrogram
                medoid_x_coords = []
                if isinstance(data, pd.DataFrame):
                    # Get the actual file paths/names of medoids
                    medoid_paths = get_medoids(data, results, onlyBiggest=True)  # type: ignore
                    # Find their positions in the dendrogram
                    for medoid_path in medoid_paths:
                        if medoid_path in data.index:
                            original_idx = data.index.get_loc(medoid_path)
                            # Find this index in the leaf_order
                            if original_idx in leaf_order:
                                leaf_pos = list(leaf_order).index(original_idx)
                                # Get the actual x-coordinate from the dendrogram's icoord
                                # icoord contains x-coordinates for each internal node
                                # We need to find the leaf's x-coordinate
                                # Leaves are at positions 0 to n_leaves-1 in the plot
                                # The x-coordinates are spaced evenly: 5, 10, 15, ... (5 * (i+1))
                                x_coord = 5 * (leaf_pos + 1)
                                medoid_x_coords.append(x_coord)
                
                # Highlight medoids with a prominent star marker
                if medoid_x_coords:
                    # Calculate a visible y-position (slightly below 0, but within the plot area)
                    max_distance = max(linkage_matrix[:, 2])
                    marker_y = -max_distance * 0.08  # Position marker below the dendrogram
                    
                    # Adjust y-axis limits to accommodate the marker
                    current_ylim = ax.get_ylim()
                    ax.set_ylim(marker_y * 1.2, current_ylim[1])
                    
                    # Draw star markers for each medoid
                    for i, x_coord in enumerate(medoid_x_coords):
                        # Draw a large, prominent star marker with red color and black edge
                        ax.plot(x_coord, marker_y, 'r*', markersize=25, markeredgewidth=4, 
                               markeredgecolor='black', markerfacecolor='red', zorder=100,
                               label='Representative' if i == 0 else '')
                    
                    # Add a legend if we have medoids
                    if len(medoid_x_coords) > 0:
                        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
                
                # Set title with molecule name if provided
                title = 'Agglomerative Clustering Dendrogram'
                if molecule_name:
                    title = f'{title} - {molecule_name}'
                ax.set_title(title)
                ax.set_xlabel('Data Points')
                ax.set_ylabel('Distance (Å)')
                # Extend the y-axis limits, adding a bit of buffer at the top to allow the text to fit
                # If we have medoids, the y-axis was already adjusted, so preserve that
                if not medoid_x_coords:
                    ax.set_ylim(0, max(linkage_matrix[:, 2]) * 1.2)
                else:
                    # Ensure the top limit is still extended for text
                    current_ylim = ax.get_ylim()
                    ax.set_ylim(current_ylim[0], max(linkage_matrix[:, 2]) * 1.2)
                # Add a red line at the distance threshold
                ax.axhline(y=distance_threshold, color='r', linestyle='--', linewidth=2, label='Distance Threshold')
                # Add the silhouette score (left, top) rounded to 2 decimals
                ax.text(0.05, 0.95, f"Silhouette Score: ~{round(scores, 2)}", transform=ax.transAxes, size=10, verticalalignment='top', horizontalalignment='left')
                # Add a label to the distance threshold below the silhouette score
                ax.text(0.05, 0.9, f"Distance Threshold: {round(distance_threshold, 2)}", transform=ax.transAxes, size=10, verticalalignment='top', horizontalalignment='left')
                plt.tight_layout()
                plt.savefig(outputPlot, dpi=150)
                plt.close()

                # Also save an index-to-name mapping with representative flags (medoids)
                try:
                    # Determine representative structures (medoids) using the computed clusters
                    medoids = set(get_medoids(data, results))  # type: ignore[arg-type]
                    labels = [str(x) for x in data.index.tolist()]
                    map_path = (
                        f"{outputPlot.rsplit('.', 1)[0]}_labels.txt" if "." in outputPlot else f"{outputPlot}_labels.txt"
                    )
                    with open(map_path, 'w') as mf:
                        mf.write("# Index\tName\tRepresentative\n")
                        for i, name in enumerate(labels):
                            rep = "YES" if name in medoids else "NO"
                            mf.write(f"{i}\t{name}\t{rep}\n")
                except (OSError, IOError, PermissionError):
                    # Non-fatal: mapping is best-effort for users of the dendrogram
                    pass
            except Exception as e:
                # If plotting fails, log the error but don't fail the entire clustering
                ocprint.print_warning(f"Failed to generate clustering plot: {e}")
                # Try to create a simple plot as fallback
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    linkage_matrix = sch.linkage(npdata, method='ward')
                    _ = sch.dendrogram(linkage_matrix, ax=ax)
                    title = 'Agglomerative Clustering Dendrogram'
                    if molecule_name:
                        title = f'{title} - {molecule_name}'
                    ax.set_title(title)
                    ax.set_xlabel('Data Points')
                    ax.set_ylabel('Distance')
                    plt.axhline(y=distance_threshold, color='r', linestyle='--', linewidth=2)
                    plt.tight_layout()
                    plt.savefig(outputPlot, dpi=150)
                    plt.close()
                except Exception as e2:
                    ocprint.print_warning(f"Failed to generate fallback plot: {e2}")
        
        # Return the results
        return results # type: ignore
    
    else:
        return ocerror.Error.unsupported_clustering_algorithm(f"The clustering algorithm '{algorithm}' is not supported. Currently the supported algorithms are: 'agglomerativeClustering'.") # type: ignore
