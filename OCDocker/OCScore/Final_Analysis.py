import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    HDBSCAN, # type: ignore
    MeanShift,
    AgglomerativeClustering,
    SpectralClustering,
    OPTICS,
    Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr, spearmanr
import matplotlib.patheffects as path_effects

# Load the updated CSV file with new columns
file_path = 'ablation.csv'
data = pd.read_csv(file_path)

# Add a new column to the data to highlight the best combined features
data['highlight'] = 'none'

# For each row in the data
for index, row in data.iterrows():
    # Get the first 16 characters of the best_combined_features string, which represent the features
    features = row['best_combined_features'][:16]

    # If the best_combined_features string is made up of all 1s
    if features == '1' * len(features):
        # Highlight the row as 'all_1'
        data.at[index, 'highlight'] = 'all_1'
    # If the best_combined_features string is made up of all 0s
    elif features == '0' * len(features):
        # Highlight the row as 'all_0'
        data.at[index, 'highlight'] = 'all_0'
    
# Set the highlight in the best_combined_metric to 'best' for the best combined value (lowest)
data.loc[data['best_combined_metric'] == data['best_combined_metric'].min(), 'highlight'] = 'best'

# Select relevant features for clustering (AUC and ERROR)
features = data[['best_combined_auc', 'best_combined_value']]

# One-hot encode the first 16 characters of the best_combined_metric column (as integers)
#one_hot_features = pd.get_dummies(data['best_combined_metric'].apply(lambda x: list(x[:16])), prefix='feature')
#one_hot_features = pd.DataFrame(data['best_combined_features'].apply(lambda x: list(x[:16])).to_list(), columns=[f'feature_{i}' for i in range(16)])
one_hot_features = pd.DataFrame(
    data['best_combined_features'].apply(lambda x: [int(c) for c in x[:16]]).to_list(),
    columns=[f'feature_{i}' for i in range(16)]
)

# Concatenate the one-hot encoded features with the original features
features = pd.concat([features, one_hot_features], axis=1)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Elbow Method
def run_elbow(normalized_features: np.ndarray, max_clusters: int = 15, plot: bool = True) -> list:
    ''' Run the Elbow Method to determine the optimal number of clusters for K-Means Clustering.
    
    Parameters
    ----------
    normalized_features : np.array
        The normalized features to be used for clustering
    max_clusters : int
        The maximum number of clusters to test
    plot : bool
        Whether to plot the Elbow Method graph or not

    Returns
    -------
    list
        A list of the Within-Cluster Sum of Squares (WCSS) for each number of clusters.
    '''

    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(normalized_features)
        wcss.append(kmeans.inertia_)
    
    if plot:
        plt.figure(figsize=(10, 8))
        plt.plot(range(1, max_clusters+1), wcss, marker='o')
        plt.title('Elbow Method for Determining Optimal Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
        plt.grid(True)
        plt.savefig('Elbow.png')
        plt.close()
    return wcss

def run_kmeans(normalized_features: np.ndarray, optimal_clusters: int = 4) -> dict:
    ''' Run K-Means Clustering with the optimal number of clusters and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    optimal_clusters : int, optional
        The optimal number of clusters to use for K-Means Clustering. Default is 4.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(normalized_features)
    
    # Add cluster labels to the original data
    data['cluster'] = clusters
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])

    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=optimal_clusters)

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('K-Means Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('Kmeans_with_density.png')
    plt.close()
    
    return correlation_dict

def run_dbscan(normalized_features: np.ndarray) -> dict:
    ''' Run DBSCAN Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    dbscan = DBSCAN(eps=0.01, min_samples=10)
    data['cluster'] = dbscan.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('DBSCAN Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('DBSCAN.png')
    plt.close()
    
    return correlation_dict

def run_hdbscan(normalized_features: np.ndarray, min_samples: int = 10, min_cluster_size: int = 10) -> dict:
    ''' Run HDBSCAN Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    min_samples : int, optional
        The minimum size of clusters. Default is 10.
    min_cluster_size : int, optional
        The minimum cluster size. Default is 10.

    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    hdbscan = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
    data['cluster'] = hdbscan.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('HDBSCAN Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('HDBSCAN.png')
    plt.close()
    
    return correlation_dict

def run_meanshift(normalized_features: np.ndarray) -> dict:
    ''' Run MeanShift Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    meanshift = MeanShift()
    data['cluster'] = meanshift.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('MeanShift Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('MeanShift.png')
    plt.close()
    
    return correlation_dict

def run_agglomerative(normalized_features: np.ndarray) -> dict:
    ''' Run Agglomerative Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    agglomerative = AgglomerativeClustering()
    data['cluster'] = agglomerative.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('Agglomerative Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('Agglomerative.png')
    plt.close()
    
    return correlation_dict

def run_spectral(normalized_features: np.ndarray, n_clusters: int = 4) -> dict:
    ''' Run Spectral Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_clusters : int, optional
        The number of clusters to form. Default is 4.
    
    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
    data['cluster'] = spectral.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=n_clusters)

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('Spectral Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('Spectral.png')
    plt.close()
    
    return correlation_dict

def run_ward(normalized_features: np.ndarray, n_clusters: int = 4) -> dict:
    ''' Run Ward Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_clusters : int, optional
        The number of clusters to form. Default is 4.
    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    data['cluster'] = ward.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=n_clusters)

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('Ward Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('Ward.png')
    plt.close()
    
    return correlation_dict

def run_optics(normalized_features: np.ndarray) -> dict:
    ''' Run OPTICS Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    optics = OPTICS(min_samples=10)
    data['cluster'] = optics.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('OPTICS Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('OPTICS.png')
    plt.close()
    
    return correlation_dict

def run_birch(normalized_features: np.ndarray, n_clusters: int = 4) -> dict:
    ''' Run Birch Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_clusters : int, optional
        The number of clusters to form. Default is 4.
    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    birch = Birch(n_clusters=n_clusters)
    data['cluster'] = birch.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=n_clusters)

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('Birch Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('Birch.png')
    plt.close()
    
    return correlation_dict

def run_gaussian_mixture(normalized_features: np.ndarray, n_components: int = 4) -> dict:
    ''' Run Gaussian Mixture Model Clustering and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_components : int, optional
        The number of mixture components (clusters). Default is 4.
    
    Returns
    -------
    dict
        A dictionary containing the Spearman correlation and p-value between the AUC and ERROR values.
    '''

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    data['cluster'] = gmm.fit_predict(normalized_features[:, :2])
    
    correlation_dict = {
        'Pearson': {},
        'Spearman': {}
    }

    # Calculate correlations
    correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['best_combined_auc'], data['best_combined_value'])
    correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['best_combined_auc'], data['best_combined_value'])
    
    # Define the color palette based on the clusters
    palette = sns.color_palette('viridis', as_cmap=False, n_colors=n_components)

    # Create a JointGrid with scatterplot
    g = sns.JointGrid(data=data, x="best_combined_auc", y="best_combined_value", height=8)

    # Plot the scatterplot for clusters
    sns.scatterplot(x='best_combined_auc', y='best_combined_value', hue='cluster', data=data, palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    # Plot the scatterplot for highlights with different markers and sizes
    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        # Explicitly set the color based on the cluster
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(
                x='best_combined_auc', 
                y='best_combined_value', 
                data=cluster_data, 
                color=palette[cluster],  # Use specific color for the cluster
                edgecolor='black', 
                s=sizes[highlight], 
                marker=marker, 
                ax=g.ax_joint, 
                legend=False  # Disable legend here to avoid multiple entries
            )

    # Add custom legend for shapes
    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    # Plot the marginal densities on the axes with matching colors
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_auc'], ax=g.ax_marg_x, color=palette[idx], fill=True)
        #sns.kdeplot(data=data[data['cluster'] == cluster]['best_combined_value'], ax=g.ax_marg_y, color=palette[idx], fill=True, vertical=True)
        sns.kdeplot(
            y=data[data['cluster'] == cluster]['best_combined_value'], 
            ax=g.ax_marg_y, 
            color=palette[idx], 
            fill=True
        )
    
    # Add the regression line
    sns.regplot(x='best_combined_auc', y='best_combined_value', data=data, scatter=False, color='cyan', ax=g.ax_joint)
    
    # Annotate the Spearman correlation
    spearman_corr_text = g.ax_joint.text(0.5, 0.43, f'Spearman Correlation: {correlation_dict["Spearman"]["correlation"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    spearman_pval_text = g.ax_joint.text(0.5, 0.47, f'Spearman p-value: {correlation_dict["Spearman"]["p_value"]:.2f}', color='cyan', fontsize=14, weight='bold', transform=g.ax_joint.transAxes)
    
    for s in [spearman_corr_text, spearman_pval_text]:
        s.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

    # Hide the legends for the marginal density plots if they exist
    if g.ax_marg_x.legend_:
        g.ax_marg_x.legend_.remove()
    if g.ax_marg_y.legend_:
        g.ax_marg_y.legend_.remove()

    # Adjust layout to make space for the title
    plt.subplots_adjust(top=0.9)

    # Set the title with space
    g.figure.suptitle('Gaussian Mixture Clustering with Density Plots', fontsize=16, weight='bold')

    # Save the figure
    plt.savefig('GaussianMixture.png')
    plt.close()
    
    return correlation_dict


# Execute functions
print('Running Elbow Method...')
wcss = run_elbow(normalized_features, max_clusters=15, plot=True)

# Set the optimal number of clusters based on the elbow method
optimal_clusters = 4

print('Running K-Means Clustering...')
k_means_corr = run_kmeans(normalized_features, optimal_clusters=optimal_clusters)

print('Running DBSCAN Clustering...')
dbscan_corr = run_dbscan(normalized_features)

print('Running HDBSCAN Clustering...')
hdbscan_corr = run_hdbscan(normalized_features, min_samples=10, min_cluster_size=10)

#print('Running MeanShift Clustering...')
#meanshift_corr = run_meanshift(normalized_features)

print('Running Agglomerative Clustering...')
agglomerative_corr = run_agglomerative(normalized_features)

print('Running Spectral Clustering...')
spectral_corr = run_spectral(normalized_features, n_clusters=optimal_clusters)

print('Running Ward Clustering...')
ward_corr = run_ward(normalized_features, n_clusters=optimal_clusters)

print('Running OPTICS Clustering...')
optics_corr = run_optics(normalized_features)

print('Running Birch Clustering...')
birch_corr = run_birch(normalized_features, n_clusters=optimal_clusters)

print('Running Gaussian Mixture Clustering...')
gmm_corr = run_gaussian_mixture(normalized_features, n_components=optimal_clusters)

print('Done!')
