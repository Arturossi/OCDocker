import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import optuna

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, spearmanr, kendalltau, chi2_contingency
from typing import Any
from urllib.parse import quote_plus

import matplotlib.patheffects as path_effects


ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"
base_models_folder: str = f"{base_path}/models"

study_name = f"NN_Ablation_Optimization_1"
study = optuna.load_study(study_name = study_name, storage = storage)

# Filter the trials to only include the ones that are complete
trials = study.trials_dataframe()
data = trials[trials['state'] == 'COMPLETE']

# Reset data index
data = data.reset_index(drop=True)

# Rename the columns
# value is the RMSE
# user_attrs_Feature_Mask is the Feature Mask
# user_attrs_AUC is the AUC
data = data.rename(columns={
        'value': 'RMSE',
        'user_attrs_Feature_Mask': 'Feature_Mask',
        'user_attrs_AUC': 'AUC'
    }
)

# Check for duplicates Feature_Masks
repeated_feature_masks = data[data.duplicated(subset=['Feature_Mask'], keep=False)]

# If there are repeated Feature_Masks
if not repeated_feature_masks.empty:
    # Warn the user
    print('There are repeated Feature Masks in the data.')

# Drop the columns datetime_start, datetime_complete, number, state, duration
data = data.drop(columns=['datetime_start', 'datetime_complete', 'number', 'state', 'duration'])

# Add a new column to the data to highlight the best combined features
data['highlight'] = 'none'

# Make the score column (RMSE - AUC)
data['score'] = data['RMSE'] - data['AUC']

# For each row in the data
for index, row in data.iterrows():
    # Get the first 16 characters of the Feature_Mask string, which represent the features
    features = row['Feature_Mask'][:16]

    # If the Feature_Mask string is made up of all 1s
    if features == '1' * len(features):
        # Highlight the row as 'all_1'
        data.at[index, 'highlight'] = 'all_1'
    # If the Feature_Mask string is made up of all 0s
    elif features == '0' * len(features):
        # Highlight the row as 'all_0'
        data.at[index, 'highlight'] = 'all_0'
    
# Set the highlight in the score to 'best' for the best combined value (lowest)
data.loc[data['score'] == data['score'].min(), 'highlight'] = 'best'

# Select relevant features for clustering (AUC, RMSE and score)
features = data[['AUC', 'RMSE', 'score']]

# One-hot encode the first 16 characters of the Feature_Mask string
one_hot_features = pd.DataFrame(
    data['Feature_Mask'].apply(
            lambda x: [int(c) for c in x[:16]]
        ).to_list(),
        columns=[f'feature_{i}' for i in range(16)]
)

# Concatenate the one-hot encoded features with the original features
features = pd.concat([features, one_hot_features], axis=1)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Define the feature names to be converted later
features_names=["SMINA_VINA", "SMINA_SCORING_DKOES", "SMINA_VINARDO", "SMINA_OLD_SCORING_DKOES", "SMINA_FAST_DKOES", "SMINA_SCORING_AD4", "VINA_VINA", "VINA_VINARDO", "PLANTS_CHEMPLP", "PLANTS_PLP", "PLANTS_PLP95", "ODDT_RFSCORE_V1", "ODDT_RFSCORE_V2", "ODDT_RFSCORE_V3", "ODDT_PLECRF_P5_L1_S65536", "ODDT_NNSCORE"]

# Rename the columns from feature_{i} to the actual feature names
features.columns = ['AUC', 'RMSE', 'score'] + features_names


# Helper functions
def compute_correlations(data: pd.DataFrame, correlation_types: list) -> dict:
    '''Compute specified correlations between 'AUC' and 'RMSE'.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing the 'AUC' and 'RMSE' columns.
    correlation_types : list
        List of correlation types to compute.

    Returns
    -------
    dict
        A dictionary with the computed correlations and p-values.
    '''

    correlation_dict = {corr_type: {} for corr_type in correlation_types}

    if 'Pearson' in correlation_types:
        correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(data['AUC'], data['RMSE'])
    if 'Spearman' in correlation_types:
        correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(data['AUC'], data['RMSE'])
    if 'Kendall' in correlation_types:
        correlation_dict['Kendall']['correlation'], correlation_dict['Kendall']['p_value'] = kendalltau(data['AUC'], data['RMSE'])
    if 'Distance' in correlation_types:
        correlation_dict['Distance']['correlation'] = dcor.distance_correlation(data['AUC'], data['RMSE'])
        correlation_dict['Distance']['p_value'] = None  # Distance correlation doesn't have a p-value
    if 'MutualInfo' in correlation_types:
        correlation_dict['MutualInfo']['correlation'] = mutual_info_score(data['AUC'], data['RMSE'])
        correlation_dict['MutualInfo']['p_value'] = None  # Mutual information doesn't have a p-value

    return correlation_dict

def create_joint_grid(data: pd.DataFrame, title: str, correlation_dict: dict, correlation_types: list) -> sns.JointGrid:
    '''Create a JointGrid plot with the specified title and correlation values.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing the 'AUC', 'RMSE', 'cluster', and 'highlight' columns.
    title : str
        The title of the plot.
    correlation_dict : dict
        A dictionary containing the computed correlations and p-values.
    correlation_types : list
        List of correlation types to display in the plot.
    
    Returns
    -------
    sns.JointGrid
        A JointGrid object with the created plot.
    '''

    palette = sns.color_palette('viridis', as_cmap=False, n_colors=len(data['cluster'].unique()))
    g = sns.JointGrid(data=data, x="AUC", y="RMSE", height=10)  # Increase plot size

    sns.scatterplot(x='AUC', y='RMSE', hue='cluster', data=data, palette=palette,
                    edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint)

    markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
    sizes = {'all_1': 100, 'all_0': 100, 'best': 100}

    for highlight, marker in markers.items():
        highlighted_data = data[data['highlight'] == highlight]
        for cluster in highlighted_data['cluster'].unique():
            cluster_data = highlighted_data[highlighted_data['cluster'] == cluster]
            sns.scatterplot(x='AUC', y='RMSE', data=cluster_data, color=palette[cluster],
                            edgecolor='black', s=sizes[highlight], marker=marker,
                            ax=g.ax_joint, legend=False)

    for highlight, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=highlight, s=sizes[highlight])

    g.ax_joint.legend(loc='best')

    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(data=data[data['cluster'] == cluster]['AUC'], ax=g.ax_marg_x,
                    color=palette[idx], fill=True)
        sns.kdeplot(y=data[data['cluster'] == cluster]['RMSE'], ax=g.ax_marg_y,
                    color=palette[idx], fill=True)

    sns.regplot(x='AUC', y='RMSE', data=data, scatter=False, color='cyan', ax=g.ax_joint)

    correlation_text = "\n".join(
        [f"{corr_type} Correlation: {round(correlation_dict[corr_type]['correlation'], 3)}, "
         f"p-value: {round(correlation_dict[corr_type]['p_value'], 3)}"
         for corr_type in correlation_types]
    )

    g.figure.set_size_inches(12, 10)  # Enlarged figure
    g.figure.subplots_adjust(bottom=0.17, top=0.92)  # Adjusted top for reduced title space

    g.figure.text(0.5, 0.07, correlation_text, ha='center', va='center', fontsize=12,
                  weight='bold', wrap=True, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=5))

    g.figure.suptitle(title, fontsize=18, weight='bold')
    plt.savefig(f'{title}.png', bbox_inches='tight')
    return g

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
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(normalized_features)
        wcss.append(kmeans.inertia_)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot WCSS values on the first plot
        ax1.plot(range(1, max_clusters + 1), wcss, marker='o', label='WCSS')
        for i, value in enumerate(wcss):
            ax1.text(i + 1, value, f"{value:.2f}", ha='center', va='bottom')
        ax1.set_title('WCSS for Each Number of Clusters')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.grid(True)
        ax1.legend()

        # Calculate and plot absolute WCSS differences on the second plot
        wcss_diff = np.abs(np.diff(wcss))
        ax2.plot(range(2, max_clusters + 1), wcss_diff, marker='x', linestyle='--', color='r', label='Absolute WCSS Difference')
        for i, value in enumerate(wcss_diff):
            ax2.text(i + 2, value, f"{value:.2f}", ha='center', va='bottom')
        ax2.set_title('Absolute Difference Between Consecutive WCSS Values')
        ax2.set_xlabel('Number of Clusters (Interval)')
        ax2.set_ylabel('WCSS Difference')
        ax2.grid(True)
        ax2.legend()

        # Set custom x-axis labels for the lower plot
        interval_labels = [f"{i} to {i+1}" for i in range(1, max_clusters)]
        ax2.set_xticks(range(2, max_clusters + 1))
        ax2.set_xticklabels(interval_labels, rotation=45)

        # Adjust y-axis to fit the differences
        ax2.set_ylim(0, max(wcss_diff) + 0.1 * max(wcss_diff))

        plt.tight_layout()
        plt.savefig('Elbow.png')
        plt.close()

    return wcss

# K-Means Clustering
def run_kmeans(normalized_features: np.ndarray, optimal_clusters: int = 4, 
               correlation_types: list = ['Pearson', 'Spearman', 'Kendall']) -> dict:
    '''Run K-Means Clustering with the optimal number of clusters and plot the results.
    
    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    optimal_clusters : int, optional
        The optimal number of clusters to use. Default is 4.
    correlation_types : list, optional
        The types of correlations to compute. Default is ['Pearson', 'Spearman', 'Kendall'].
    
    Returns
    -------
    dict
        A dictionary containing the computed correlations and p-values.
    '''

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    data['cluster'] = kmeans.fit_predict(normalized_features)
    correlation_dict = compute_correlations(data, correlation_types)
    create_joint_grid(data, 'K-Means Clustering', correlation_dict, correlation_types)
    return correlation_dict

# DBSCAN Clustering
def run_dbscan(normalized_features: np.ndarray) -> dict:
    ''' Run DBSCAN Clustering and plot the results.

    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation
        and p-value between the AUC and RMSE values.
    '''

    dbscan = DBSCAN(eps=0.01, min_samples=10)
    data['cluster'] = dbscan.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'DBSCAN Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# HDBSCAN Clustering
def run_hdbscan(normalized_features: np.ndarray, min_samples: int = 10, 
                min_cluster_size: int = 10) -> dict:
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
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'HDBSCAN Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# MeanShift Clustering
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
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'MeanShift Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# Agglomerative Clustering
def run_agglomerative(normalized_features: np.ndarray) -> dict:
    ''' Run Agglomerative Clustering and plot the results.

    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation
        and p-value between the AUC and ERROR values.
    '''

    agglomerative = AgglomerativeClustering()
    data['cluster'] = agglomerative.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'Agglomerative Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# OPTICS Clustering
def run_optics(normalized_features: np.ndarray) -> dict:
    ''' Run OPTICS Clustering and plot the results.

    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation
        and p-value between the AUC and ERROR values.
    '''

    optics = OPTICS(min_samples=10)
    data['cluster'] = optics.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'OPTICS Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# Birch Clustering
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
        A dictionary containing the Spearman correlation
        and p-value between the AUC and ERROR values.
    '''

    birch = Birch(n_clusters=n_clusters)
    data['cluster'] = birch.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'Birch Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# Spectral Clustering
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
        A dictionary containing the Spearman correlation
        and p-value between the AUC and ERROR values.
    '''

    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
    data['cluster'] = spectral.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'Spectral Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# Gaussian Mixture Clustering
def run_gaussian_mixture(normalized_features: np.ndarray, n_components: int = 4) -> dict:
    ''' Run Gaussian Mixture Clustering and plot the results.

    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_components : int, optional
        The number of components to form. Default is 4.

    Returns
    -------
    dict
        A dictionary containing the Spearman correlation
        and p-value between the AUC and ERROR values.
    '''

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    data['cluster'] = gmm.fit_predict(normalized_features[:, :2])
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])
    create_joint_grid(data, 'Gaussian Mixture Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])
    return correlation_dict

# Ward Clustering
def run_ward(normalized_features: np.ndarray, n_clusters: int = 4) -> dict:
    '''
    Run Ward Clustering and plot the results.

    Parameters
    ----------
    normalized_features : np.ndarray
        The normalized features to be used for clustering.
    n_clusters : int, optional
        The number of clusters to form. Default is 4.

    Returns
    -------
    dict
        A dictionary containing the Pearson, Spearman, and Kendall correlations and p-values.
    '''

    # Initialize and run Ward Clustering
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    data['cluster'] = ward.fit_predict(normalized_features[:, :2])

    # Compute correlations
    correlation_dict = compute_correlations(data, ['Pearson', 'Spearman', 'Kendall'])

    # Create and save the plot
    create_joint_grid(data, 'Ward Clustering', correlation_dict, ['Pearson', 'Spearman', 'Kendall'])

    return correlation_dict

def individual_contributions_analysis(
    features: pd.DataFrame, 
    verbose: bool = False, 
    features_names: list = [], 
    grid_layout: tuple = (2, 2)
) -> pd.DataFrame:
    ''' 
    Perform an analysis of the individual contributions of each feature to the RMSE and AUC. 
    
    Parameters
    ----------
    features : pd.DataFrame
        The features to analyze.
    verbose : bool, optional
        Whether to print the results or not. Default is False.
    features_names : list, optional
        Custom labels for the x-axis. If None, the feature names will be used.
    grid_layout : tuple, optional
        Grid layout for the plots. Default is (2, 2).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the RMSE and AUC statistics for each feature.
    '''

    if not features_names:
        # Initialize results list dynamically for all binary feature columns
        feature_cols = [col for col in features.columns if 'feature_' in col]
    else:
        feature_cols = features_names

    results = []

    # Loop through each feature column dynamically
    for feature_name in feature_cols:
        on_data = features[features[feature_name] == 1]
        off_data = features[features[feature_name] == 0]

        # Store aggregated results for On and Off
        for label, data_subset in zip(['On', 'Off'], [on_data, off_data]):
            results.append({
                'Feature': feature_name,
                'On/Off/Difference': label,
                'RMSE Mean': data_subset['RMSE'].mean(),
                'RMSE Median': data_subset['RMSE'].median(),
                'RMSE Std': data_subset['RMSE'].std(),
                'AUC Mean': data_subset['AUC'].mean(),
                'AUC Median': data_subset['AUC'].median(),
                'AUC Std': data_subset['AUC'].std()
            })

        # Store the differences
        results.append({
            'Feature': feature_name,
            'On/Off/Difference': 'Difference',
            'RMSE Mean': on_data['RMSE'].mean() - off_data['RMSE'].mean(),
            'RMSE Median': on_data['RMSE'].median() - off_data['RMSE'].median(),
            'RMSE Std': on_data['RMSE'].std() - off_data['RMSE'].std(),
            'AUC Mean': on_data['AUC'].mean() - off_data['AUC'].mean(),
            'AUC Median': on_data['AUC'].median() - off_data['AUC'].median(),
            'AUC Std': on_data['AUC'].std() - off_data['AUC'].std()
        })

        if verbose:
            print(f"\nFeature: {feature_name}")
            print(f"On Data - RMSE: Mean: {on_data['RMSE'].mean():.3f}, Median: {on_data['RMSE'].median():.3f}, Std: {on_data['RMSE'].std():.3f}")
            print(f"On Data - AUC: Mean: {on_data['AUC'].mean():.3f}, Median: {on_data['AUC'].median():.3f}, Std: {on_data['AUC'].std():.3f}")
            print(f"Off Data - RMSE: Mean: {off_data['RMSE'].mean():.3f}, Median: {off_data['RMSE'].median():.3f}, Std: {off_data['RMSE'].std():.3f}")
            print(f"Off Data - AUC: Mean: {off_data['AUC'].mean():.3f}, Median: {off_data['AUC'].median():.3f}, Std: {off_data['AUC'].std():.3f}")

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('individual_analysis.csv', index=False)

    # Set x-axis labels if provided, otherwise use feature names
    if not features_names:
        features_names = feature_cols
    elif len(features_names) != len(feature_cols):
        raise ValueError("The length of features_names must match the number of features.")

    # Create dynamic grid layout for plots
    fig, axes = plt.subplots(*grid_layout, figsize=(15, 12), sharex=True)

    # Flatten the axes for easier indexing
    axes = axes.flatten()

    # Plot for RMSE On/Off values
    on_off_data = results_df[results_df['On/Off/Difference'].isin(['On', 'Off'])]
    sns.barplot(x='Feature', y='RMSE Mean', hue='On/Off/Difference', data=on_off_data, ax=axes[0])
    axes[0].set_title('On/Off States')
    axes[0].set_ylabel('Mean RMSE')
    axes[0].legend(loc='upper right')

    # Plot for RMSE Differences
    diff_data = results_df[results_df['On/Off/Difference'] == 'Difference']
    sns.barplot(x='Feature', y='RMSE Mean', data=diff_data, ax=axes[1], color='coral')
    axes[1].set_title('Differences Between On and Off States')
    axes[1].set_ylabel('RMSE Difference')

    # Plot for AUC On/Off values
    sns.barplot(x='Feature', y='AUC Mean', hue='On/Off/Difference', data=on_off_data, ax=axes[2])
    #axes[2].set_title('Mean AUC for On/Off States')
    axes[2].set_ylabel('Mean AUC')
    axes[2].legend(loc='upper right')

    # Plot for AUC Differences
    sns.barplot(x='Feature', y='AUC Mean', data=diff_data, ax=axes[3], color='skyblue')
    #axes[3].set_title('Mean AUC Differences Between On and Off States')
    axes[3].set_ylabel('AUC Difference')

    # Set custom x-axis labels
    for ax in axes[2:]:
        ax.set_xticklabels(features_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('individual_analysis.png')
    plt.close()

    return results_df

# Plotting 

def plot_feature_mask_correlation_heatmap(features: pd.DataFrame, 
                                          features_names: list = [], 
                                          correlation_types: list = ['Pearson', 'Spearman', 'Kendall']) -> None:
    '''Generate a Pearson correlation heatmap between RMSE, AUC, and feature mask bits.'''
    # Extract feature columns
    feature_cols = [col for col in features.columns]

    # Drop the score column if it exists
    if 'score' in feature_cols:
        features = features.drop(columns=['score'])
        # Remove the score from the feature columns list
        feature_cols.remove('score')

    # Set x-axis labels if provided, otherwise use feature names
    if features_names:
        if len(features_names) != (len(feature_cols) - 2):
            raise ValueError("The length of features_names must match the number of features.")
        features.columns = ['RMSE', 'AUC'] + features_names

    for corr_type in correlation_types:
        # Compute correlation matrix based on the type
        if corr_type == 'Pearson':
            correlation_matrix = features.corr(method='pearson')
        elif corr_type == 'Spearman':
            correlation_matrix = features.corr(method='spearman')
        elif corr_type == 'Kendall':
            correlation_matrix = features.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation type: {corr_type}")

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Set up the matplotlib figure
        plt.figure(figsize=(12, 8))

        # Generate the heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            cbar=True,
            linewidths=0.5
        )

        # Add title
        plt.title(f"{corr_type} Correlation Heatmap with RMSE and AUC", fontsize=16, weight='bold')
        plt.xticks(rotation=45, ha='right')

        # Save and display the heatmap
        plt.savefig(f'{corr_type.lower()}_correlation_heatmap.png', bbox_inches='tight')
        plt.close()
    
    return None

# Chi-Square Analysis

def chi_square_analysis(features: pd.DataFrame, 
                        metric: str = 'AUC', 
                        feature_bits: list = None, 
                        split: str = 'binary') -> pd.DataFrame:
    '''
    Perform chi-square tests for independence between feature mask bits 
    and a discretized performance metric (binary or ternary).

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing the 'AUC' or 'RMSE' column and the binary feature bits.
    metric : str, optional
        The metric column to analyze ('AUC' or 'RMSE'). Default is 'AUC'.
    feature_bits : list, optional
        List of feature bit column names. If None, will infer from column names in `features`.
    split : str, optional
        Choose either 'binary' or 'ternary' discretization. Default is 'binary'.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the chi-square test statistics and p-values for each feature bit.
    '''
    
    # Check for valid split option
    if split not in ['binary', 'ternary']:
        raise ValueError("Invalid split option. Choose 'binary' or 'ternary'.")

    # Discretize the metric based on the chosen split method
    if split == 'binary':
        # Binary: High vs. Low based on the median
        features[f'{metric}_category'] = np.where(
            features[metric] > features[metric].median(), 'high', 'low'
        )
    elif split == 'ternary':
        # Ternary: Low, Medium, High based on quartiles
        q1 = features[metric].quantile(0.33)
        q2 = features[metric].quantile(0.66)
        features[f'{metric}_category'] = pd.cut(
            features[metric], 
            bins=[-np.inf, q1, q2, np.inf], 
            labels=['low', 'medium', 'high']
        )

    # Determine feature bit columns if not provided
    if feature_bits is None:
        feature_bits = [col for col in features.columns if col.startswith('feature_')]

    # Store results in a DataFrame
    results = {'Feature': [], 'Chi2 Statistic': [], 'p-value': []}

    # Run chi-square test for each feature bit
    for bit in feature_bits:
        contingency_table = pd.crosstab(features[bit], features[f'{metric}_category'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        results['Feature'].append(bit)
        results['Chi2 Statistic'].append(chi2)
        results['p-value'].append(p)

    # Compile results
    chi_square_results = pd.DataFrame(results)

    return chi_square_results

def visualize_chi_square_comparison(auc_df: pd.DataFrame, rmse_df: pd.DataFrame):
    '''Visualize chi-square statistics and p-values for both AUC and RMSE analyses.

    Parameters
    ----------
    auc_df : pd.DataFrame
        DataFrame containing chi-square results for AUC.
    rmse_df : pd.DataFrame
        DataFrame containing chi-square results for RMSE.
    '''
    
    # Sort by Chi2 Statistic for ordered bar plots
    auc_df = auc_df.sort_values(by='Chi2 Statistic', ascending=False)
    rmse_df = rmse_df.sort_values(by='Chi2 Statistic', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # AUC: Bar Plot of Chi2 Statistic
    sns.barplot(
        data=auc_df,
        y='Feature',
        x='Chi2 Statistic',
        palette="magma",
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Chi-Square Test Statistics by Feature - AUC')
    axes[0, 0].set_xlabel('Chi-Square Statistic')
    axes[0, 0].set_ylabel('Feature')

    # RMSE: Bar Plot of Chi2 Statistic
    sns.barplot(
        data=rmse_df,
        y='Feature',
        x='Chi2 Statistic',
        palette="cividis",
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Chi-Square Test Statistics by Feature - RMSE')
    axes[0, 1].set_xlabel('Chi-Square Statistic')
    axes[0, 1].set_ylabel('')

    # AUC: Scatter Plot of Chi2 Statistic vs. -Log10(P-Value)
    sns.scatterplot(
        data=auc_df,
        x='Chi2 Statistic',
        y=-np.log10(auc_df['p-value']),
        hue='Feature',
        palette="magma",
        s=100,
        legend=None,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Chi-Square Statistic vs. -Log10(P-Value) - AUC')
    axes[1, 0].set_xlabel('Chi-Square Statistic')
    axes[1, 0].set_ylabel('-Log10(P-Value)')
    axes[1, 0].axhline(-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
    
    # RMSE: Scatter Plot of Chi2 Statistic vs. -Log10(P-Value)
    sns.scatterplot(
        data=rmse_df,
        x='Chi2 Statistic',
        y=-np.log10(rmse_df['p-value']),
        hue='Feature',
        palette="cividis",
        s=100,
        legend=None,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Chi-Square Statistic vs. -Log10(P-Value) - RMSE')
    axes[1, 1].set_xlabel('Chi-Square Statistic')
    axes[1, 1].set_ylabel('')
    axes[1, 1].axhline(-np.log10(0.05), color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('chi_square_comparison.png')
    plt.close()

    return None

# Chi-Square Post-Hoc Analysis

def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list, float]:
    ''' Apply Bonferroni correction to a list of p-values.
    
    Parameters
    ----------
    p_values : list[float]
        List of p-values to correct.
    alpha : float, optional
        The significance level. Default is 0.05.

    Returns
    -------
    list
        List of boolean values indicating significance after correction.
    float
        Corrected alpha level.
    '''

    # Apply Bonferroni correction
    corrected_alpha = alpha / len(p_values)

    # Check significance after correction
    return [p < corrected_alpha for p in p_values], corrected_alpha

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    ''' Apply Benjamini-Hochberg correction to a list of p-values.
    
    Parameters
    ----------
    p_values : list[float]
        List of p-values to correct.
    alpha : float, optional
        The significance level. Default is 0.05.
    
    Returns
    -------
    list
        List of boolean values indicating significance after correction.
    '''

    # Convert p-values to numpy array
    p_values_np = np.array(p_values)
    
    # Sort p-values in ascending order
    n = len(p_values_np)
    sorted_indices = np.argsort(p_values_np)

    # Compute adjusted alpha levels
    adjusted_alpha = alpha * np.arange(1, n + 1) / n

    # Check significance after correction
    significant = p_values_np[sorted_indices] <= adjusted_alpha

    # Reorder the significance values
    return significant[sorted_indices.argsort()]

# Feature importance using Random Forest

def random_forest_feature_engineering(
    df: pd.DataFrame, 
    target_column: str = 'AUC', 
    feature_columns: list = [], 
    n_estimators: int = 100, 
    random_state: int = 42
) -> pd.DataFrame:
    '''
    Perform feature engineering using Random Forest to determine feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the features and target.
    target_column : str, optional
        The target column for classification ('AUC_category', 'RMSE_category', etc.). Default is 'AUC_category'.
    feature_columns : list, optional
        List of feature column names to include. If an empty list is provided, all feature columns will be used.
    n_estimators : int, optional
        Number of trees in the forest. Default is 100.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importance scores sorted in descending order.
    '''

    # Check if the target column is AUC or RMSE
    if target_column not in ['AUC', 'RMSE']:
        raise ValueError("Invalid target column. Choose 'AUC' or 'RMSE'.")
    else:
        target_column = f'{target_column}_category'

    # Define feature columns if not provided
    if not feature_columns:
        feature_columns = [col for col in df.columns if col.startswith('feature_')]

    # Prepare features and target
    X = df[feature_columns]
    y = df[target_column]

    # Train-test split for validation (test_size = 0.0 because we are not training a model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=random_state)

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)

    # Compute feature importances
    feature_importances = rf.feature_importances_

    # Create a DataFrame to store the results
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Normalize importances to sum to 1 (optional)
    feature_importance_df['Normalized Importance'] = feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()

    return feature_importance_df

def plot_feature_importance_comparison(importance_df_auc: pd.DataFrame, 
                                       importance_df_rmse: pd.DataFrame) -> None:
    '''
    Plot feature importance for AUC and RMSE in separate bar plots.

    Parameters
    ----------
    importance_df_auc : pd.DataFrame
        DataFrame containing feature importance for AUC.
    importance_df_rmse : pd.DataFrame
        DataFrame containing feature importance for RMSE.
    '''

    # Sort DataFrames by importance
    importance_df_auc = importance_df_auc.sort_values(by='Importance', ascending=False)
    importance_df_rmse = importance_df_rmse.sort_values(by='Importance', ascending=False)

    # Set up the plot with two independent y-axes
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # AUC Plot
    sns.barplot(
        y='Feature', 
        x='Importance', 
        hue='Feature',  # Assign 'Feature' to hue to avoid the warning
        data=importance_df_auc, 
        ax=axes[0], 
        palette='mako', 
        dodge=False,  # Prevents multiple bars for each feature
        legend=False  # We don't need the legend
    )
    axes[0].set_title('Feature Importance for AUC', fontsize=16, weight='bold')
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_ylabel('Feature', fontsize=12)

    # RMSE Plot
    sns.barplot(
        y='Feature', 
        x='Importance', 
        hue='Feature',  # Assign 'Feature' to hue to avoid the warning
        data=importance_df_rmse, 
        ax=axes[1], 
        palette='crest', 
        dodge=False,  # Prevents multiple bars for each feature
        legend=False  # We don't need the legend
    )
    axes[1].set_title('Feature Importance for RMSE', fontsize=16, weight='bold')
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()

    return None

# Relationships the number of features and the AUC/RMSE values
def analyze_feature_effects(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Analyze the impact of the number of features turned on 
    on AUC and RMSE, and visualize the results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing AUC, RMSE, and binary feature columns.
        
    Returns
    -------
    dict
        Dictionary containing the correlation coefficients and p-values.
    """

    # Calculate the number of features turned on 
    df['Features_Turned_On'] = df[feature_columns].sum(axis=1)

    # Group by the number of features turned on and calculate mean AUC and RMSE
    results = df.groupby('Features_Turned_On').agg({'AUC': 'mean', 'RMSE': 'mean'}).reset_index()

    # Plot AUC and RMSE trends against the number of features turned on
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # For each bin of features turned on, calculate the standard deviation
    auc_std = df.groupby('Features_Turned_On')['AUC'].std().values
    rmse_std = df.groupby('Features_Turned_On')['RMSE'].std().values

    # Change NaN values to 0
    auc_std[np.isnan(auc_std)] = 0.0
    rmse_std[np.isnan(rmse_std)] = 0.0

    # AUC plot
    color = 'tab:blue'
    ax1.set_xlabel('Number of Features Turned On')
    ax1.set_ylabel('AUC', color=color)
    ax1.plot(results['Features_Turned_On'], results['AUC'], color=color)
    ax1.fill_between(results['Features_Turned_On'], results['AUC'] - auc_std, results['AUC'] + auc_std, color=color, alpha=0.2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(axis='y', linestyle=':', alpha=0.6, color=color)

    # Create second y-axis for RMSE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('RMSE', color=color)
    ax2.plot(results['Features_Turned_On'], results['RMSE'], color=color)
    ax2.fill_between(results['Features_Turned_On'], results['RMSE'] - rmse_std, results['RMSE'] + rmse_std, color=color, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(axis='y', linestyle='--', alpha=0.6, color=color)

    # Adjust layout and show plot
    plt.xticks(np.arange(0, 17, 1))
    plt.xlim([0, 16])  # Ensure the x-axis starts at 0 and ends at 16
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.title('Effect of Features Turned On: AUC vs RMSE')
    fig.tight_layout()
    plt.savefig('features_on_auc_rmse.png', bbox_inches='tight')
    plt.close()

    # Calculate correlations
    auc_corr, auc_p = pearsonr(df['Features_Turned_On'], df['AUC'])
    rmse_corr, rmse_p = pearsonr(df['Features_Turned_On'], df['RMSE'])

    # Print correlations
    print(f'Correlation between Features Turned On and AUC: {auc_corr:.3f} (p-value: {auc_p:.3f})')
    print(f'Correlation between Features Turned On and RMSE: {rmse_corr:.3f} (p-value: {rmse_p:.3f})')

    # Visualize the distribution of features turned on
    fig = plt.figure(figsize=(10, 6))
    # Plot a histogram of the number of features turned on adding the standard deviation
    plt.hist(df['Features_Turned_On'], bins=np.arange(0, 17) - 0.5, alpha=0.6, color='b', edgecolor='black', align='mid')
    plt.xlabel('Number of Features Turned On')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 17, 1))
    plt.xlim([0, 16])  # Ensure the x-axis starts at 0 and ends at 16
    plt.grid(alpha = 0.3)
    plt.title('Distribution of Features Turned On')
    fig.tight_layout()
    plt.savefig('features_on_distribution.png', bbox_inches='tight')
    plt.close()

    # Return the correlation results
    return {
        'AUC_Correlation': (auc_corr, auc_p),
        'RMSE_Correlation': (rmse_corr, rmse_p)
    }

# Execute functions
print('Running Elbow Method...')
wcss = run_elbow(normalized_features, max_clusters=25, plot=True)

# Set the optimal number of clusters based on the elbow method
optimal_clusters = 5

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

print('Performing individual contributions analysis for each feature...')
results_df = individual_contributions_analysis(features, verbose=True, features_names=features_names)

print('Plotting feature mask correlation heatmap...')
plot_feature_mask_correlation_heatmap(features, features_names=features_names)

print('Performing Chi-Square test for feature selection...')
chi_square_results_auc = chi_square_analysis(features, metric='AUC', feature_bits=features_names, split='ternary')
chi_square_results_rmse = chi_square_analysis(features, metric='RMSE', feature_bits=features_names, split='ternary')

print('Plotting Chi-Square test results...')
visualize_chi_square_comparison(chi_square_results_auc, chi_square_results_rmse)

print("Applying Bonferroni correction to the p-values...")
chi_square_results_auc['Bonferroni_Significant_AUC'], corrected_alpha_auc = bonferroni_correction(chi_square_results_auc['p-value'].tolist())
chi_square_results_rmse['Bonferroni_Significant_RMSE'], corrected_alpha_rmse = bonferroni_correction(chi_square_results_rmse['p-value'].tolist())

print("Applying Benjamini-Hochberg correction to the p-values...")
chi_square_results_auc['BH_Significant_AUC'] = benjamini_hochberg(chi_square_results_auc['p-value'].tolist())
chi_square_results_rmse['BH_Significant_RMSE'] = benjamini_hochberg(chi_square_results_rmse['p-value'].tolist())

print('Performing feature importance analysis...')
feature_importance_df_auc = random_forest_feature_engineering(features, target_column='AUC', feature_columns=features_names)
feature_importance_df_rmse = random_forest_feature_engineering(features, target_column='RMSE', feature_columns=features_names)

print('Plotting feature importance analysis...')
plot_feature_importance_comparison(feature_importance_df_auc, feature_importance_df_rmse)

print('Analyzing the impact of the number of features turned on...')
feature_effects = analyze_feature_effects(features, feature_columns=features_names)

print('Done!')
