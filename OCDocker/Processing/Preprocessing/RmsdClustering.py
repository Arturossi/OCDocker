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

from typing import Dict, List, Union, Optional
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


def cluster_rmsd(data: Union[Dict[str, Dict[str, float]], pd.DataFrame], algorithm: str = 'agglomerativeClustering', max_distance_threshold: float = 20.0, min_distance_threshold: float = 10.0, threshold_step: float = 0.1, outputPlot: str = "", molecule_name: str = "", pose_engine_map: Optional[Dict[str, str]] = None, engine_colors: Optional[Dict[str, str]] = None) -> Union[np.ndarray, int]:
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
    pose_engine_map : Dict[str, str], optional
        Mapping from pose file paths to engine names ('vina', 'smina', 'plants'). Used for coloring labels in the plot.
    engine_colors : Dict[str, str], optional
        Mapping from engine names to colors. Default: {'plants': 'green', 'vina': '#9B59B6', 'smina': 'blue'}.
        Engine names should be lowercase.

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
                    # Generate a plot even if clustering failed, then return error code
                    if outputPlot != "":
                        try:
                            fig, ax = plt.subplots(figsize=(14, 9))
                            linkage_matrix = sch.linkage(npdata, method='ward')
                            _ = sch.dendrogram(linkage_matrix, ax=ax)
                            title = 'Pose consensus'
                            if molecule_name:
                                title = f'{molecule_name} pose consensus'
                            ax.set_title(title, fontsize=16)
                            ax.set_xlabel('Data Points', fontsize=14)
                            ax.set_ylabel('Distance (Å)', fontsize=14)
                            ax.tick_params(axis='both', which='major', labelsize=12)
                            # Add warning text
                            ax.text(0.5, 0.5, 'Clustering did not converge.\nAll poses are too different.', 
                                   transform=ax.transAxes, fontsize=14, ha='center', va='center',
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                            plt.tight_layout()
                            plt.savefig(outputPlot, dpi=150)
                            plt.close()
                            ocprint.print_warning(f"Generated plot for failed clustering: {outputPlot}")
                        except Exception as e:
                            ocprint.print_warning(f"Failed to generate plot for non-converged clustering: {e}")
                    # Print the message, returning the error code
                    return ocerror.Error.cluster_not_converged(f"The clustering algorithm did not converge. The distance threshold is {distance_threshold}.") # type: ignore

            # Find the biggest cluster (may be more than one)
            biggest_cluster = np.where(cluster_sizes == np.max(cluster_sizes))[0]

            # If the biggest cluster is 1
            if len(biggest_cluster) == 1:
                # If there is only one cluster, accept it as valid (single big cluster)
                if len(unique_clusters) == 1:
                    # Single cluster is valid - set scores to 0 (no need for silhouette with one cluster)
                    scores = 0
                    # Store results for plotting
                    results = results
                    # Break the loop
                    break
                elif len(unique_clusters) > 1:
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
                        # Set results and scores for plotting, then continue to plot generation
                        results = last_result
                        scores = 0  # Set to 0 to indicate we're using fallback result
                        # Use the last distance threshold from the loop, or calculate a reasonable default
                        if distance_threshold == -1:
                            # Calculate a reasonable threshold from the linkage matrix
                            linkage_matrix = sch.linkage(npdata, method='ward')
                            distance_threshold = np.max(linkage_matrix[:, 2]) * 0.8  # 80% of max distance
                        # Continue to plot generation (don't return early)
                    else:
                        # No valid cluster found, will fall through to error case
                        pass
                else:
                    # All clusters have only 1 member
                    pass
            else:
                # last_result is empty
                pass
            
            # If we still have scores == -1, clustering truly failed
            if scores == -1:
                # If all clusters have only 1 member, fail
                ocprint.print_warning("All clusters have only 1 member. Clustering failed.")
                # Generate a plot even if clustering failed, then return error code
                if outputPlot != "":
                    try:
                        fig, ax = plt.subplots(figsize=(14, 9))
                        linkage_matrix = sch.linkage(npdata, method='ward')
                        _ = sch.dendrogram(linkage_matrix, ax=ax)
                        title = 'Pose consensus'
                        if molecule_name:
                            title = f'{molecule_name} pose consensus'
                        ax.set_title(title, fontsize=16)
                        ax.set_xlabel('Data Points', fontsize=14)
                        ax.set_ylabel('Distance (Å)', fontsize=14)
                        ax.tick_params(axis='both', which='major', labelsize=12)
                        # Add warning text
                        ax.text(0.5, 0.5, 'Clustering did not converge.\nAll poses are too different.', 
                               transform=ax.transAxes, fontsize=14, ha='center', va='center',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                        plt.tight_layout()
                        plt.savefig(outputPlot, dpi=150)
                        plt.close()
                        ocprint.print_warning(f"Generated plot for failed clustering: {outputPlot}")
                    except Exception as e:
                        ocprint.print_warning(f"Failed to generate plot for non-converged clustering: {e}")
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
                # Use Set2 (colorblind-friendly, no blue or yellow shades) for small clusters
                import matplotlib.cm as cm
                cluster_colors = []
                
                def is_light_grey(color):
                    """Check if a color is light grey (too close to white/grey)."""
                    r, g, b = color[0], color[1], color[2]
                    # Light grey has all RGB components relatively high and similar
                    # Check if all components are > 0.6 and within 0.15 of each other
                    if min(r, g, b) > 0.6:
                        if max(r, g, b) - min(r, g, b) < 0.15:
                            return True
                    return False
                
                # Set2 is colorblind-friendly and doesn't use blue or yellow shades
                if n_clusters <= 8:
                    cmap = cm.get_cmap('Set2')
                    all_colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
                    # Filter out light grey
                    cluster_colors = [c for c in all_colors if not is_light_grey(c)]
                    # If we filtered out too many, add back some colors
                    while len(cluster_colors) < n_clusters:
                        # Try to get more colors from the colormap
                        for i in range(n_clusters, n_clusters + 10):
                            c = cmap(i / max(n_clusters + 9, 1))
                            if not is_light_grey(c) and c not in cluster_colors:
                                cluster_colors.append(c)
                                if len(cluster_colors) >= n_clusters:
                                    break
                        if len(cluster_colors) < n_clusters:
                            break
                    cluster_colors = cluster_colors[:n_clusters]
                elif n_clusters <= 12:
                    # Use Set1 but filter out yellow, blue, and light grey
                    cmap = cm.get_cmap('Set1')
                    all_colors = [cmap(i / 8.0) for i in range(9)]
                    # Filter out yellow (high red+green, low blue), blue (high blue component), and light grey
                    filtered_colors = []
                    for c in all_colors:
                        # Skip yellow (high red and green, low blue), blue (high blue), and light grey
                        is_yellow = c[0] > 0.7 and c[1] > 0.7 and c[2] < 0.3
                        is_blue = c[2] > 0.6
                        if not (is_yellow or is_blue or is_light_grey(c)):
                            filtered_colors.append(c)
                    # Cycle through filtered colors
                    for i in range(n_clusters):
                        cluster_colors.append(filtered_colors[i % len(filtered_colors)])
                else:
                    # For many clusters, use a custom palette avoiding blue, yellow, and light grey
                    # Use colors from Set1, Set2, and Pastel1, filtering out yellow, blue, and light grey
                    colors1 = [cm.get_cmap('Set1')(i / 8.0) for i in range(9)]
                    colors2 = [cm.get_cmap('Set2')(i / 7.0) for i in range(8)]
                    colors3 = [cm.get_cmap('Pastel1')(i / 8.0) for i in range(9)]
                    # Combine and filter out blue, yellow, and light grey shades
                    all_colors = colors1 + colors2 + colors3
                    filtered_colors = []
                    for c in all_colors:
                        # Skip yellow (high red and green, low blue), blue (high blue), and light grey
                        is_yellow = c[0] > 0.7 and c[1] > 0.7 and c[2] < 0.3
                        is_blue = c[2] > 0.6
                        if not (is_yellow or is_blue or is_light_grey(c)):
                            filtered_colors.append(c)
                    if len(filtered_colors) < n_clusters:
                        # If not enough colors, cycle
                        while len(filtered_colors) < n_clusters:
                            filtered_colors.extend(filtered_colors[:min(len(filtered_colors), n_clusters - len(filtered_colors))])
                    cluster_colors = filtered_colors[:n_clusters]
                
                cluster_color_map = {int(cluster_id): cluster_colors[i] for i, cluster_id in enumerate(unique_clusters)}
                
                # Create figure and axis with larger size to accommodate text
                fig, ax = plt.subplots(figsize=(14, 9))
                
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
                clusters_actually_colored = set()  # Track which clusters are actually colored in the plot
                for i, collection in enumerate(ax.collections):
                    cluster_id = collection_to_cluster.get(i, -1)
                    if cluster_id >= 0 and cluster_id in cluster_color_map:
                        # Only apply color if collection is entirely below threshold
                        collection.set_color(cluster_color_map[cluster_id])
                        clusters_actually_colored.add(cluster_id)
                        colored_count += 1
                    else:
                        # Above threshold or crosses threshold - use gray instead of blue for colorblind-friendliness
                        collection.set_color('gray')
                        blue_count += 1
                
                # Debug output
                ocprint.printv(f"Dendrogram: {len(clusters_at_threshold)} data points, {n_clusters} clusters at threshold {distance_threshold:.2f}")
                ocprint.printv(f"  Colored branches (clusters): {colored_count}, Blue branches (above threshold): {blue_count}")
                ocprint.printv(f"  Total collections: {len(ax.collections)}")
                ocprint.printv(f"  Clusters actually colored in plot: {sorted(clusters_actually_colored)}")
                
                # Highlight representative elements (medoids) with a marker
                # Find leaf positions for medoids in the dendrogram
                medoid_leaf_positions = []  # Store leaf positions directly
                if isinstance(data, pd.DataFrame):
                    # Get the actual file paths/names of medoids
                    medoid_paths = get_medoids(data, results, onlyBiggest=True)  # type: ignore
                    
                    # Create a set of medoid paths for quick lookup
                    medoid_set = set(medoid_paths)
                    
                    # Find their positions in the dendrogram
                    # The leaf_order is a list where leaf_order[i] gives the original index
                    # that appears at leaf position i in the dendrogram
                    # We need to iterate through leaf_order to find which leaf position
                    # corresponds to each medoid
                    for leaf_pos, original_idx in enumerate(leaf_order):
                        # Get the path at this original index position in data.index
                        if original_idx < len(data.index):
                            path_at_idx = data.index[original_idx]
                            # Check if this path is a medoid
                            if path_at_idx in medoid_set:
                                # Store the leaf position directly (this corresponds to the label index)
                                medoid_leaf_positions.append(leaf_pos)
                                ocprint.printv(f"Medoid found: {path_at_idx} (original_idx={original_idx}, leaf_pos={leaf_pos})")
                
                # Define engine color mapping (use provided or default)
                if engine_colors is None:
                    engine_colors = {
                        'plants': 'green',
                        'vina': '#9B59B6',
                        'smina': 'blue'
                    }
                
                # Color leaf labels according to engine (before drawing boxes)
                if pose_engine_map and isinstance(data, pd.DataFrame):
                    tick_labels = ax.get_xticklabels()
                    for i, label in enumerate(tick_labels):
                        if i < len(leaf_order):
                            original_idx = leaf_order[i]
                            if original_idx < len(data.index):
                                path_at_idx = data.index[original_idx]
                                # Try to find engine for this pose
                                engine = None
                                # Check direct match
                                if path_at_idx in pose_engine_map:
                                    engine = pose_engine_map[path_at_idx].lower()
                                else:
                                    # Try to match by basename or partial path
                                    for pose_path, eng in pose_engine_map.items():
                                        if path_at_idx in pose_path or pose_path in path_at_idx:
                                            engine = eng.lower()
                                            break
                                
                                if engine and engine in engine_colors:
                                    label.set_color(engine_colors[engine])
                
                # Render figure first to ensure labels are accessible and positioned
                fig.canvas.draw()
                
                # Note: Representative poses are now shown as text below Distance Threshold
                # No visual marking on the plot itself
                
                # Set title with ligand name
                if molecule_name:
                    title = f'{molecule_name} pose consensus'
                else:
                    title = 'Pose consensus'
                ax.set_title(title, fontsize=16)
                ax.set_xlabel('Data Points', fontsize=14)
                ax.set_ylabel('Distance (Å)', fontsize=14)
                
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=12)
                # Extend the y-axis limits, adding a bit of buffer at the top to allow the text to fit
                # Always start at 0 (no space below for markers - box is in axes coordinates)
                ax.set_ylim(0, max(linkage_matrix[:, 2]) * 1.2)
                
                # Add a red dashed line at the distance threshold
                ax.axhline(y=distance_threshold, color='red', linestyle='--', linewidth=2, label='Distance Threshold', zorder=50)
                
                # Build legend entries
                legend_handles = []
                legend_labels = []
                
                # Add threshold line to legend
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2))
                legend_labels.append('Distance Threshold')
                
                # Add cluster numbers and colors to legend (only clusters actually colored in the plot)
                if cluster_color_map and len(clusters_actually_colored) > 1:
                    # Get unique colors and their corresponding cluster IDs for actually colored clusters only
                    color_to_clusters = {}
                    for cluster_id in clusters_actually_colored:
                        if cluster_id in cluster_color_map:
                            cluster_color = cluster_color_map[cluster_id]
                            # Convert color to tuple for comparison (handles both string and RGB tuple colors)
                            if isinstance(cluster_color, str):
                                color_key = cluster_color
                            else:
                                color_key = tuple(cluster_color) if hasattr(cluster_color, '__iter__') else cluster_color
                            
                            if color_key not in color_to_clusters:
                                color_to_clusters[color_key] = []
                            color_to_clusters[color_key].append(cluster_id)
                    
                    # If only one unique color, show just one entry
                    if len(color_to_clusters) == 1:
                        color_key = list(color_to_clusters.keys())[0]
                        cluster_color = cluster_color_map[sorted(color_to_clusters[color_key])[0]]
                        legend_handles.append(Line2D([0], [0], color=cluster_color, linestyle='-', linewidth=3))
                        legend_labels.append('Cluster')
                    else:
                        # Multiple unique colors - show each unique color
                        for color_key, cluster_ids in color_to_clusters.items():
                            cluster_color = cluster_color_map[sorted(cluster_ids)[0]]  # Get color from first cluster with this color
                            legend_handles.append(Line2D([0], [0], color=cluster_color, linestyle='-', linewidth=3))
                            legend_labels.append(f'Cluster {sorted(cluster_ids)[0]}')
                
                # Add engine colors to legend if pose_engine_map is provided
                if pose_engine_map:
                    for engine_name, color in engine_colors.items():
                        if any(eng.lower() == engine_name for eng in pose_engine_map.values()):
                            legend_handles.append(Line2D([0], [0], color=color, linestyle='-', linewidth=2, marker='o', markersize=8))
                            legend_labels.append(engine_name.capitalize())
                
                # Add legend inside the plot area
                if legend_handles:
                    # Place legend inside the plot, in upper right corner
                    # Use a more compact position to avoid overlap
                    legend = ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=11, framealpha=0.9, 
                                      bbox_to_anchor=(0.98, 0.98), handlelength=2, handletextpad=0.5, 
                                      columnspacing=1.0, borderpad=0.5)
                    # Adjust layout to ensure legend fits inside
                    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
                else:
                    # No legend, use standard layout
                    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
                
                # Add the silhouette score (left, top) rounded to 2 decimals
                # For single cluster, silhouette score is not meaningful, show N/A
                if scores == 0 and len(np.unique(results)) == 1:
                    ax.text(0.02, 0.98, "Silhouette Score: N/A (single cluster)", transform=ax.transAxes, size=12, verticalalignment='top', horizontalalignment='left')
                else:
                    ax.text(0.02, 0.98, f"Silhouette Score: ~{round(scores, 2)}", transform=ax.transAxes, size=12, verticalalignment='top', horizontalalignment='left')
                # Add a label to the distance threshold below the silhouette score
                ax.text(0.02, 0.94, f"Distance Threshold: {round(distance_threshold, 2)} Å", transform=ax.transAxes, size=12, verticalalignment='top', horizontalalignment='left')
                
                # Add representative pose information below the distance threshold
                if medoid_leaf_positions and isinstance(data, pd.DataFrame):
                    # Get the original indices for medoids (matching medoids_labels.txt format)
                    medoid_labels = []
                    for leaf_pos in medoid_leaf_positions:
                        # The leaf_pos corresponds to the position in the dendrogram
                        # leaf_order[leaf_pos] gives the original index in the data
                        if leaf_pos < len(leaf_order):
                            original_idx = leaf_order[leaf_pos]
                            # Display the original index to match medoids_labels.txt
                            medoid_labels.append(str(original_idx))
                    
                    # Create representative text with data point numbers
                    if medoid_labels:
                        rep_text = f"Representative: {', '.join(medoid_labels)}"
                        ax.text(0.02, 0.90, rep_text, transform=ax.transAxes, size=12, verticalalignment='top', horizontalalignment='left')
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
                import traceback
                ocprint.print_warning(f"Failed to generate clustering plot: {e}")
                ocprint.print_warning(f"Traceback: {traceback.format_exc()}")
                # Try to create a simple plot as fallback
                try:
                    fig, ax = plt.subplots(figsize=(14, 9))
                    linkage_matrix = sch.linkage(npdata, method='ward')
                    _ = sch.dendrogram(linkage_matrix, ax=ax)
                    title = 'Pose consensus'
                    if molecule_name:
                        title = f'{molecule_name} pose consensus'
                    ax.set_title(title, fontsize=16)
                    ax.set_xlabel('Data Points', fontsize=14)
                    ax.set_ylabel('Distance (Å)', fontsize=14)
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    plt.axhline(y=distance_threshold, color='r', linestyle='--', linewidth=2, label='Distance Threshold')
                    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
                    plt.tight_layout()
                    plt.savefig(outputPlot, dpi=150)
                    plt.close()
                    ocprint.print_warning(f"Generated fallback plot for {outputPlot}")
                except Exception as e2:
                    ocprint.print_warning(f"Failed to generate fallback plot: {e2}")
                    ocprint.print_warning(f"Fallback traceback: {traceback.format_exc()}")
        
        # Return the results
        return results # type: ignore
    
    else:
        return ocerror.Error.unsupported_clustering_algorithm(f"The clustering algorithm '{algorithm}' is not supported. Currently the supported algorithms are: 'agglomerativeClustering'.") # type: ignore
