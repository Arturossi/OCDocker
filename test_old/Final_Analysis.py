import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import optuna

import pingouin as pg

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import (
    pearsonr, 
    spearmanr, 
    kendalltau, 
    chi2_contingency, 
    probplot, 
    shapiro, 
    anderson, 
    kruskal
)
from typing import Optional
from urllib.parse import quote_plus

import matplotlib.patheffects as path_effects


ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"
method: str = "kmeans"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"
base_models_folder: str = f"{base_path}/models"

study_name = f"NN_Ablation_Optimization_1"
#study_name = f"NN_Seed_Ablation_Optimization_2"
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

# if user_attrs_random_seed is in the data, rename it to seed
if 'user_attrs_random_seed' in data.columns:
    data = data.rename(columns={'user_attrs_random_seed': 'seed'})

# if user_attrs_pr_auc is in the data, rename it to pr_auc
if 'user_attrs_pr_auc' in data.columns:
    data = data.rename(columns={'user_attrs_pr_auc': 'PR_AUC'})

# if user_attrs_mae is in the data, rename it to mae
if 'user_attrs_mae' in data.columns:
    data = data.rename(columns={'user_attrs_mae': 'MAE'})

# if user_attrs_log_loss is in the data, rename it to log_loss
if 'user_attrs_log_loss' in data.columns:
    data = data.rename(columns={'user_attrs_log_loss': 'log_loss'})

# If there is Seed in the study name
if 'Seed' in study_name:
    # Check for duplicates seeds
    repeated_seeds = data[data.duplicated(subset=['seed'], keep=False)]

    # If there are repeated seeds
    if not repeated_seeds.empty:
        # Warn the user
        print('There are repeated seeds in the data.')
else:
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

# Define the feature names to be converted later
features_names=["SMINA_VINA", "SMINA_SCORING_DKOES", "SMINA_VINARDO", "SMINA_OLD_SCORING_DKOES", "SMINA_FAST_DKOES", "SMINA_SCORING_AD4", "VINA_VINA", "VINA_VINARDO", "PLANTS_CHEMPLP", "PLANTS_PLP", "PLANTS_PLP95", "ODDT_RFSCORE_V1", "ODDT_RFSCORE_V2", "ODDT_RFSCORE_V3", "ODDT_PLECRF_P5_L1_S65536", "ODDT_NNSCORE"]

# Select relevant features for clustering (AUC, RMSE and score)
if "Seed" in study_name:
    features_fake = data[['AUC', 'RMSE', 'score', 'PR_AUC', "seed"]]
else:
    features_fake = data[['AUC', 'RMSE', 'score']]

features = data[['AUC', 'RMSE']]

# One-hot encode the first 16 characters of the Feature_Mask string
one_hot_features = pd.DataFrame(
    data['Feature_Mask'].apply(
            lambda x: [int(c) for c in x[:16]]
        ).to_list(),
        columns=[f'feature_{i}' for i in range(16)]
)

# Concatenate the one-hot encoded features with the original features
features = pd.concat([features, one_hot_features], axis=1)
features_fake = pd.concat([features_fake, one_hot_features], axis=1)

# Normalize the features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Select relevant features for clustering (AUC, RMSE and score)
if "Seed" in study_name:
    features_fake.columns = ['AUC', 'RMSE', 'score', 'PR_AUC', 'seed'] + features_names
else:
    features_fake.columns = ['AUC', 'RMSE', 'score'] + features_names

features.columns = ['AUC', 'RMSE'] + features_names

def plot_metrics_with_highlights_old(df: pd.DataFrame, plot_type: str = 'box', use_labels: bool = True, mark_seed_id: int = 0, add_metrics: list = []) -> None:
    '''
    Plots RMSE, AUC, and Score distributions using boxplots or violin plots,
    with highlighted points for 'all_0' and 'all_1'. Adds subplot labels A, B, C.

    Parameters:
    ------------
    df : pandas.DataFrame
        Data containing columns ['RMSE', 'AUC', 'score', 'highlight'].
    plot_type : str
        Type of plot to use: 'box' or 'violin'. Default is 'box'.
    use_labels : bool
        Whether to add subplot labels A, B, C. Default is True.
    mark_seed_id : int, optional
        Seed to mark in the plot. Default is 0 (which marks no seed).
    add_metrics : list, optional
        Additional metrics to add to the plot. Default is an empty list.
    '''

    if plot_type not in ['box', 'violin']:
        raise ValueError("plot_type must be either 'box' or 'violin'")

    # Count the number of additional metrics
    num_additional_metrics = len(add_metrics)

    # Divide it by 3 and apply ceil to determine the number of rows needed
    if num_additional_metrics > 0:
        num_rows = (num_additional_metrics) // 3
        # Set the extra labels for additional metrics following the alphabet letter sequence
        more_labels = [chr(68 + i) for i in range(num_additional_metrics)]
    else:
        num_rows = 0
        more_labels = []
    
    metrics = ['RMSE', 'AUC', 'score'] + add_metrics
    labels = ['A', 'B', 'C'] + more_labels

    fig, axes = plt.subplots(1 + num_rows, 3, figsize=(20, 6 + 4 * num_rows), sharey=True)

    for ax, metric, label in zip(axes, metrics, labels):
        if plot_type == 'box':
            sns.boxplot(y=df[metric], ax=ax, color='lightgray')
        elif plot_type == 'violin':
            sns.violinplot(y=df[metric], ax=ax, color='lightgray', inner='box')

        # Highlight 'all_0'
        sns.stripplot(
            y=df[df['highlight'] == 'all_0'][metric],
            ax=ax,
            color='red',
            size=8,
            marker='X',
            label='all_0'
        )

        # Highlight 'all_1'
        sns.stripplot(
            y=df[df['highlight'] == 'all_1'][metric],
            ax=ax,
            color='blue',
            size=8,
            marker='D',
            label='all_1'
        )

        # Highlight 'best'
        sns.stripplot(
            y=df[df['highlight'] == 'best'][metric],
            ax=ax,
            color='green',
            size=8,
            marker='^',
            label='best'
        )

        if mark_seed_id > 0:
            # Highlight the specific seed ID
            seed_row = df[df['seed'] == mark_seed_id]
            if not seed_row.empty:
                sns.stripplot(
                    y=seed_row[metric],
                    ax=ax,
                    color='orange',
                    size=10,
                    marker='o',
                    label=f'Seed {mark_seed_id}'
                )

        plt.rcParams.update({'font.size': 14})
        ax.set_title(f'Distribuição do {metric}')
        ax.legend()

        if use_labels:
            # Add label A, B, C
            ax.text(
                -0.1, 1.02, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='left'
            )

    plt.tight_layout()
    plt.savefig('metrics_boxplots.png', bbox_inches='tight')

def plot_metrics_with_highlights(df: pd.DataFrame, plot_type: str = 'box', use_labels: bool = True, mark_seed_id: int = 0, add_metrics: list = [], columns: int = 3) -> None:
    '''
    Plots RMSE, AUC, and Score distributions using boxplots or violin plots,
    with highlighted points for 'all_0', 'all_1', and 'best'. Adds subplot labels A, B, C.
    Additionally, you can highlight a specific seed.

    Parameters:
    ------------
    df : pandas.DataFrame
        Data containing columns ['RMSE', 'AUC', 'score', 'highlight', 'seed'].
    plot_type : str
        Type of plot to use: 'box' or 'violin'. Default is 'box'.
    use_labels : bool
        Whether to add subplot labels A, B, C. Default is True.
    mark_seed_id : int, optional
        Seed to highlight in the plot. Default is 0 (no seed marked).
    add_metrics : list, optional
        Additional metrics to add to the plot. Default is an empty list.
    columns : int, optional
        Number of columns for the subplots. Default is 3.
    '''

    if plot_type not in ['box', 'violin']:
        raise ValueError("plot_type must be either 'box' or 'violin'")

    # Count the number of additional metrics
    num_additional_metrics = len(add_metrics)

    # Divide by columns and apply ceil to determine the number of rows needed
    num_rows = (num_additional_metrics + columns - 1) // columns  # Adjust rows calculation for better placement
    more_labels = [chr(68 + i) for i in range(num_additional_metrics)] if num_additional_metrics > 0 else []

    metrics = ['RMSE', 'AUC', 'score'] + add_metrics
    labels = ['A', 'B', 'C'] + more_labels

    fig, axes = plt.subplots(1 + num_rows, columns, figsize=(7 * columns, 6 + 4 * num_rows))

    # Flatten axes array to iterate over all subplots
    axes = axes.flat

    for ax, metric, label in zip(axes, metrics, labels):
        if plot_type == 'box':
            sns.boxplot(y=df[metric], ax=ax, color='lightgray')
        elif plot_type == 'violin':
            sns.violinplot(y=df[metric], ax=ax, color='lightgray', inner='box')

        # Highlight 'all_0'
        sns.stripplot(
            y=df[df['highlight'] == 'all_0'][metric],
            ax=ax,
            color='red',
            size=8,
            marker='X',
            label='all_0'
        )

        # Highlight 'all_1'
        sns.stripplot(
            y=df[df['highlight'] == 'all_1'][metric],
            ax=ax,
            color='blue',
            size=8,
            marker='D',
            label='all_1'
        )

        # Highlight 'best'
        sns.stripplot(
            y=df[df['highlight'] == 'best'][metric],
            ax=ax,
            color='green',
            size=8,
            marker='^',
            label='best'
        )

        # Highlight specific seed if provided
        if mark_seed_id > 0:
            seed_row = df[df['seed'] == mark_seed_id]
            if not seed_row.empty:
                sns.stripplot(
                    y=seed_row[metric],
                    ax=ax,
                    color='orange',
                    size=10,
                    marker='o',
                    label=f'Seed {mark_seed_id}'
                )

        # Adjusting axis titles and labels
        ax.set_title(f'Distribuição do {metric}')
        ax.legend()

        # Add subplot labels A, B, C, etc.
        if use_labels:
            ax.text(
                -0.1, 1.02, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='left'
            )

    plt.tight_layout()
    plt.savefig('metrics_boxplots.png', bbox_inches='tight')
    plt.close()

if "Seed" in study_name:
    plot_metrics_with_highlights(data, plot_type='violin', use_labels=True, mark_seed_id=42, add_metrics=['PR_AUC', 'MAE', 'log_loss'], columns = 2)
else:
    plot_metrics_with_highlights(data, plot_type='violin', use_labels=True, columns = 2)

# Helper functions
def compute_correlations(
    data: pd.DataFrame,
    correlation_types: list,
    x_var: str,
    y_var: str
) -> dict:
    """
    Compute specified correlations between X and Y variables.

    Returns
    -------
    dict
        Dictionary with correlations and p-values.
    """
    correlation_dict = {corr_type: {} for corr_type in correlation_types}

    x = data[x_var]
    y = data[y_var]

    if 'Pearson' in correlation_types:
        correlation_dict['Pearson']['correlation'], correlation_dict['Pearson']['p_value'] = pearsonr(x, y)

    if 'Spearman' in correlation_types:
        correlation_dict['Spearman']['correlation'], correlation_dict['Spearman']['p_value'] = spearmanr(x, y)

    if 'Kendall' in correlation_types:
        correlation_dict['Kendall']['correlation'], correlation_dict['Kendall']['p_value'] = kendalltau(x, y)

    if 'Distance' in correlation_types:
        correlation_dict['Distance']['correlation'] = dcor.distance_correlation(x, y)
        correlation_dict['Distance']['p_value'] = None

    if 'MutualInfo' in correlation_types:
        correlation_dict['MutualInfo']['correlation'] = mutual_info_score(x, y)
        correlation_dict['MutualInfo']['p_value'] = None

    return correlation_dict

def create_joint_grid(
    data: pd.DataFrame,
    title: str,
    correlation_dict: dict,
    correlation_types: list,
    x_var: str,
    y_var: str,
    alpha: float = 0.8,
    add_all_markers: bool = True
) -> sns.JointGrid:
    """
    Create a JointGrid plot with KDE marginals and scatterplot clusters.

    Returns
    -------
    sns.JointGrid
        The JointGrid object.
    """

    palette = sns.color_palette('viridis', n_colors=data['cluster'].nunique())
    g = sns.JointGrid(data=data, x=x_var, y=y_var, height=10)

    sns.scatterplot(
        data=data, x=x_var, y=y_var, hue='cluster',
        palette=palette, edgecolor='black', s=20, alpha=0.5, ax=g.ax_joint
    )

    if add_all_markers == True:
        markers = {'all_1': 'D', 'all_0': 's', 'best': '^'}
        sizes = {'all_1': 100, 'all_0': 100, 'best': 100}
    else: 
        markers = {'best': '^'}
        sizes = {'best': 100}

    for label, marker in markers.items():
        subset = data[data['highlight'] == label]
        for cluster in subset['cluster'].unique():
            cluster_data = subset[subset['cluster'] == cluster]
            sns.scatterplot(
                data=cluster_data, x=x_var, y=y_var, color=palette[cluster],
                edgecolor='black', s=sizes[label], marker=marker, ax=g.ax_joint, legend=False
            )

    # Add empty markers for legend
    for label, marker in markers.items():
        g.ax_joint.scatter([], [], c='k', marker=marker, label=label, s=sizes[label])

    g.ax_joint.legend(loc='best')

    # KDE Marginals
    for idx, cluster in enumerate(sorted(data['cluster'].unique())):
        sns.kdeplot(
            data=data[data['cluster'] == cluster][x_var], ax=g.ax_marg_x,
            color=palette[idx], fill=True
        )
        sns.kdeplot(
            y=data[data['cluster'] == cluster][y_var], ax=g.ax_marg_y,
            color=palette[idx], fill=True
        )

    # Regression line
    sns.regplot(data=data, x=x_var, y=y_var, scatter=False, color='cyan', ax=g.ax_joint)

    # Correlation Text
    correlation_text = "\n".join(
        [f"{c}: {correlation_dict[c]['correlation']:.3g}"
         + (f", p={correlation_dict[c]['p_value']:.3g}" if correlation_dict[c]['p_value'] is not None else "")
         for c in correlation_types]
    )

    g.figure.set_size_inches(12, 10)
    g.figure.subplots_adjust(bottom=0.17, top=0.92)

    g.figure.text(
        0.5, 0.07, correlation_text,
        ha='center', va='center', fontsize=12,
        weight='bold',
        bbox=dict(facecolor='white', alpha=alpha, edgecolor='black', pad=5)
    )

    g.figure.suptitle(title, fontsize=18, weight='bold')

    g.ax_joint.set_xlabel(x_var, fontsize=14)
    g.ax_joint.set_ylabel(y_var, fontsize=14)
    g.ax_joint.tick_params(axis='both', labelsize=12)
    g.ax_marg_x.tick_params(axis='x', labelsize=12)
    g.ax_marg_y.tick_params(axis='y', labelsize=12)

    plt.savefig(f'{title}.png', bbox_inches='tight')
    plt.close()

    return g
# Elbow Method
def run_elbow(
    data: pd.DataFrame,
    normalized_features: np.ndarray,
    max_clusters: int = 15,
    plot: bool = True,
    stacked: bool = False,
    use_labels: bool = True,
    plot_labels: bool = False
) -> list:
    
    wcss = []
    subplot_labels = ['A', 'B']

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
        kmeans.fit(normalized_features)
        wcss.append(kmeans.inertia_)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 12)) if stacked else plt.subplots(1, 2, figsize=(16, 6))
        ax1, ax2 = axes

        # Plot WCSS
        ax1.plot(range(1, max_clusters + 1), wcss, marker='o', label='WCSS')
        if plot_labels:
            for i, value in enumerate(wcss):
                ax1.text(i + 1, value, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
        ax1.set_title('WCSS for Each Number of Clusters')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('WCSS')
        ax1.set_xticks(range(1, max_clusters + 1))
        ax1.grid(True)
        ax1.legend()

        # Add subplot label 'A'
        if use_labels:
            ax1.text(-0.1, 1.05, subplot_labels[0], transform=ax1.transAxes,
                     fontsize=16, fontweight='bold', va='top', ha='right')

        # Plot absolute WCSS differences
        wcss_diff = np.abs(np.diff(wcss))
        x_pos = np.arange(2, max_clusters + 1)

        ax2.plot(x_pos, wcss_diff, marker='x', linestyle='--', color='r', label='Absolute WCSS Difference')
        if plot_labels:
            for i, value in enumerate(wcss_diff):
                ax2.text(x_pos[i], value, f"{value:.2f}", ha='center', va='bottom', fontsize=8)

        ax2.set_title('Absolute Difference Between Consecutive WCSS Values')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('WCSS Difference')
        ax2.grid(True)
        ax2.legend()

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([str(i) for i in x_pos])

        ax2.set_ylim(0, max(wcss_diff) + 0.1 * max(wcss_diff))

        # Add subplot label 'B'
        if use_labels:
            ax2.text(-0.1, 1.05, subplot_labels[1], transform=ax2.transAxes,
                     fontsize=16, fontweight='bold', va='top', ha='right')

        plt.tight_layout()
        plt.savefig('Elbow.png')
        plt.close()

    return wcss

# K-Means Clustering
def run_kmeans2(normalized_features: np.ndarray, optimal_clusters: int = 4, 
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

def run_kmeans(
    data: pd.DataFrame,
    normalized_features: np.ndarray,
    optimal_clusters: int = 3,
    correlation_types: Optional[list] = None,
    x_var: str = 'AUC',
    y_var: str = 'RMSE',
    alpha: float = 0.4,
    add_all_markers: bool = True
) -> dict:
    """
    Run K-Means Clustering with the optimal number of clusters and plot the results.

    Parameters
    ----------
    data : pd.DataFrame
        Data containing the variables for plotting and highlight.
    normalized_features : np.ndarray
        Normalized features for clustering.
    optimal_clusters : int
        Number of clusters.
    correlation_types : list, optional
        Correlations to compute. Default is ['Pearson', 'Spearman', 'Kendall'].
    x_var : str
        Variable for the X-axis. Default is 'AUC'.
    y_var : str
        Variable for the Y-axis. Default is 'RMSE'.
    alpha : float
        Transparency level for the plot. Default is 0.4.
    add_all_markers : bool
        Whether to add markers for 'all_0', 'all_1', and 'best' highlights. Default is True.

    Returns
    -------
    dict
        Dictionary with correlations and p-values.
    """

    if correlation_types is None:
        correlation_types = ['Pearson', 'Spearman', 'Kendall']

    assert x_var in data.columns and y_var in data.columns, \
        f"Columns '{x_var}' and '{y_var}' must be present in the data."

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init='auto')
    data['cluster'] = kmeans.fit_predict(normalized_features)

    correlation_dict = compute_correlations(data, correlation_types, x_var, y_var)
    create_joint_grid(data, 'K-Means Clustering', correlation_dict, correlation_types, x_var, y_var, alpha=alpha, add_all_markers=add_all_markers)

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
    #if features_names:
    #    if len(features_names) != (len(feature_cols) - 2):
    #        raise ValueError("The length of features_names must match the number of features.")
    #    features.columns = ['RMSE', 'AUC'] + features_names

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

def show_discretization_categories(data: pd.DataFrame) -> None:
    """
    Display and plot the discretization of AUC and RMSE 
    into binary, ternary, and quaternary categories in a single image.
    """

    metrics = ['AUC', 'RMSE']
    splits = ['binary', 'ternary', 'quaternary']
    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for row_idx, metric in enumerate(metrics):
        for col_idx, split in enumerate(splits):
            idx = row_idx * 3 + col_idx
            ax = axes[idx]

            # Plot
            if split == 'binary':
                median = data[metric].median()
                data['binary'] = np.where(data[metric] > median, 'high', 'low')
                sns.histplot(data, x=metric, hue='binary', kde=True, multiple="stack",
                             palette='Set2', ax=ax)
                ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.3f}')
                ax.set_title(f'{metric} - Binary')
                ax.legend()

                print(f"\n--- Binary Split {metric} ---")
                print(f"Median: {median}")
                print(data['binary'].value_counts())

            elif split == 'ternary':
                q1 = data[metric].quantile(0.33)
                q2 = data[metric].quantile(0.66)
                data['ternary'] = pd.cut(
                    data[metric],
                    bins=[-np.inf, q1, q2, np.inf],
                    labels=['low', 'medium', 'high']
                )
                sns.histplot(data, x=metric, hue='ternary', kde=True, multiple="stack",
                             palette='Set3', ax=ax)
                ax.axvline(q1, color='red', linestyle='--', label=f'33%: {q1:.3f}')
                ax.axvline(q2, color='blue', linestyle='--', label=f'66%: {q2:.3f}')
                ax.set_title(f'{metric} - Ternary')
                ax.legend()

                print(f"\n--- Ternary Split {metric}  ---")
                print(f"Quantiles: 33%={q1}, 66%={q2}")
                print(data['ternary'].value_counts())

            elif split == 'quaternary':
                q = data[metric].quantile([0.25, 0.5, 0.75])
                data['quaternary'] = pd.cut(
                    data[metric],
                    bins=[-np.inf, q[0.25], q[0.5], q[0.75], np.inf],
                    labels=['very low', 'low', 'high', 'very high']
                )
                sns.histplot(data, x=metric, hue='quaternary', kde=True, multiple="stack",
                             palette='tab10', ax=ax)
                ax.axvline(q[0.25], color='red', linestyle='--', label=f'25%: {q[0.25]:.3f}')
                ax.axvline(q[0.5], color='blue', linestyle='--', label=f'50%: {q[0.5]:.3f}')
                ax.axvline(q[0.75], color='green', linestyle='--', label=f'75%: {q[0.75]:.3f}')
                ax.set_title(f'{metric} - Quaternary')
                ax.legend()

                print(f"\n--- Quaternary Split {metric} ---")
                print(f"Quantiles: 25%={q[0.25]}, 50%={q[0.5]}, 75%={q[0.75]}")
                print(data['quaternary'].value_counts())

            # Add Panel Label (A, B, C...)
            ax.text(-0.05, 1.05, panel_labels[idx], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')

    plt.tight_layout()
    plt.savefig('chi_square_discretization.png')

def chi_square_analysis_old(
        features: pd.DataFrame,
        metric: str = 'AUC',
        feature_bits: Optional[list] = None,
        split: str = 'quaternary',
        invert_metric: bool = False,
        invert_feature: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chi-square tests for independence between feature mask bits
    and a discretized performance metric (binary, ternary, or quaternary),
    with optional inversion.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing 'AUC' or 'RMSE' and feature bits.
    metric : str
        Metric column to analyze ('AUC' or 'RMSE').
    feature_bits : list
        List of feature bit column names. If None, inferred.
    split : str
        'binary', 'ternary', or 'quaternary'. Default is 'quaternary'.
    invert_metric : bool
        Invert the metric categories. Default is False.
    invert_feature : bool
        Invert the presence/absence of feature bits. Default is False.

    Returns
    -------
    pd.DataFrame
        Chi-square statistics, p-values, and Cramér's V.
    pd.DataFrame
        DataFrame with the discretized metric categories.
    """

    data = features.copy()

    if split not in ['binary', 'ternary', 'quaternary']:
        raise ValueError("split must be 'binary', 'ternary', or 'quaternary'.")

    # Metric discretization
    if split == 'binary':
        median = data[metric].median()
        data[f'{metric}_category'] = np.where(data[metric] > median, 'high', 'low')
        if invert_metric:
            data[f'{metric}_category'] = data[f'{metric}_category'].map({'high': 'low', 'low': 'high'})
    elif split == 'ternary':
        q1 = data[metric].quantile(0.33)
        q2 = data[metric].quantile(0.66)
        labels = ['low', 'medium', 'high']
        if invert_metric:
            labels = labels[::-1]
        data[f'{metric}_category'] = pd.cut(
            data[metric],
            bins=[-np.inf, q1, q2, np.inf],
            labels=labels
        )
    elif split == 'quaternary':
        q = data[metric].quantile([0.25, 0.5, 0.75])
        labels = ['very low', 'low', 'high', 'very high']
        if invert_metric:
            labels = labels[::-1]
        data[f'{metric}_category'] = pd.cut(
            data[metric],
            bins=[-np.inf, q[0.25], q[0.5], q[0.75], np.inf],
            labels=labels
        )

    if feature_bits is None:
        feature_bits = [col for col in features.columns if col.startswith('feature_')]

    results = {
        'Feature': [],
        'Chi2 Statistic': [],
        'p-value': [],
        "Cramér's V": []
    }

    for bit in feature_bits:
        bit_data = data[bit]
        if invert_feature:
            bit_data = 1 - bit_data  # Invert 1 <-> 0

        contingency_table = pd.crosstab(bit_data, data[f'{metric}_category'])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        n = contingency_table.sum().sum()
        k = min(contingency_table.shape)

        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan

        results['Feature'].append(bit)
        results['Chi2 Statistic'].append(chi2)
        results['p-value'].append(p)
        results["Cramér's V"].append(cramers_v)

    return pd.DataFrame(results), data

def plot_cramers_comparison(auc_df: pd.DataFrame, rmse_df: pd.DataFrame) -> None:
    """
    Plota dois gráficos de barras horizontais lado a lado
    mostrando Cramér's V para AUC e RMSE, com faixas coloridas
    de força de associação e anotações dos valores.

    Parâmetros
    ----------
    auc_df : pd.DataFrame
        DataFrame contendo 'Feature' e "Cramér's V" para AUC.
    rmse_df : pd.DataFrame
        DataFrame contendo 'Feature' e "Cramér's V" para RMSE.
    """

    # Renomeia a coluna "Cramér's V" para "CramerV" para consistência
    auc_df = auc_df.rename(columns={"Cramér's V": "CramerV"})
    rmse_df = rmse_df.rename(columns={"Cramér's V": "CramerV"})

    # Assegura que a coluna "CramerV" seja numérica
    auc_df["CramerV"] = pd.to_numeric(auc_df["CramerV"], errors='coerce')
    rmse_df["CramerV"] = pd.to_numeric(rmse_df["CramerV"], errors='coerce')

    # Prepara os dados ordenados
    auc_sorted = auc_df.sort_values(by="CramerV", ascending=True)
    rmse_sorted = rmse_df.sort_values(by="CramerV", ascending=True)

    # Configura o layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)

    min = 0
    max = 0.7

    # Definir limites e faixas de fundo
    limits = (min, max)

    regions = [
        (min, 0.1, 'lightgray', 'No association'),
        (0.1, 0.2, '#d0f0c0', 'Weak'),
        (0.2, 0.3, '#fef3b7', 'Moderate'),
        (0.3, 0.5, '#fdd9b5', 'Strong'),
        (0.5, max, '#fbb4ae', 'Very strong')
    ]

    # Função para plotar cada gráfico
    def plot_single(ax, data, title):
        # Faixas coloridas
        for start, end, color, label in regions:
            ax.axvspan(start, end, color=color, alpha=0.4, zorder=0)

        # Barras
        sns.barplot(
            data=data,
            x="CramerV",
            y="Feature",
            color='steelblue',
            ax=ax
        )

        # Anotações dos valores
        for index, row in data.iterrows():
            ax.text(
                row["CramerV"] + 0.01,  # deslocamento lateral
                row["Feature"],
                f"{row['CramerV']:.3f}",
                va='center'
            )

        ax.set_title(title)
        ax.set_xlim(limits)
        ax.set_xlabel("Cramér's V")
        ax.set_ylabel("")

    # Plota AUC
    plot_single(axes[0], auc_sorted, "Cramér's V - AUC")

    # Plota RMSE
    plot_single(axes[1], rmse_sorted, "Cramér's V - RMSE")

    plt.tight_layout()
    plt.savefig('cramer.png')
    plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def visualize_chi_square_comparison_old(
    auc_df: pd.DataFrame,
    rmse_df: pd.DataFrame,
    outfile: str = "chi_square_comparison_combined.png",
    palette_name: str = "tab20",   # pode trocar para "husl", "Set3", etc.
    pastel_factor: float = 0.0,    # 0 = cores vivas, 1 = pastel
    x_jitter_frac: float = 0.03    # deslocamento horizontal em RMSE
    ) -> None:
    """
    Visualiza estatísticas de qui-quadrado para AUC e RMSE:
      - Barras unificadas ordenadas por AUC (eixo X log).
      - Scatter único com AUC (círculo) e RMSE (estrela), mesmas cores por feature.
      - Contorno preto nos marcadores.
      - Legenda no modo 'best'.
    """

    auc_df = auc_df.copy()
    rmse_df = rmse_df.copy()

    # Ordenar por Chi² AUC
    order_auc = (
        auc_df[["Feature", "Chi2 Statistic"]]
        .sort_values("Chi2 Statistic", ascending=False)["Feature"]
        .tolist()
    )

    # Mapear valores
    auc_map = dict(zip(auc_df["Feature"], auc_df["Chi2 Statistic"]))
    rmse_map = dict(zip(rmse_df["Feature"], rmse_df["Chi2 Statistic"]))

    features = order_auc
    n = len(features)

    auc_vals = np.array([auc_map.get(f, np.nan) for f in features], dtype=float)
    rmse_vals = np.array([rmse_map.get(f, np.nan) for f in features], dtype=float)

    # Função para -log10(p)
    def neglog10_p(series):
        adjusted = np.clip(series.to_numpy(dtype=float), a_min=1e-300, a_max=1.0)
        return -np.log10(adjusted)

    auc_logp = neglog10_p(auc_df["p-value"])
    rmse_logp = neglog10_p(rmse_df["p-value"])

    # Gerar cores
    base_palette = sns.color_palette(palette_name, n_colors=n)
    if pastel_factor > 0:
        base_palette = sns.color_palette(
            [(np.array(c) + pastel_factor) / (1 + pastel_factor) for c in base_palette]
        )
    color_by_feature = {feat: col for feat, col in zip(features, base_palette)}

    # ===== Figura =====
    plt.rcParams.update({"font.size": 14})
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 2]}
    )

    # Top: Barras
    y = np.arange(n)
    bar_h = 0.36
    offset = bar_h / 2.0

    auc_bar_colors = [color_by_feature[f] for f in features]
    rmse_bar_colors = auc_bar_colors

    ax_top.barh(
        y + offset, auc_vals, height=bar_h,
        color=auc_bar_colors, edgecolor="black", linewidth=0.6, label="AUC"
    )
    ax_top.barh(
        y - offset, rmse_vals, height=bar_h,
        color=rmse_bar_colors, edgecolor="black", linewidth=0.6,
        hatch="///", label="RMSE"
    )

    ax_top.set_yticks(y, features)
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Chi-Square Statistic (log scale)")
    ax_top.set_ylabel("Feature")
    ax_top.set_title("Chi-Square Test Statistics by Feature")
    ax_top.grid(axis="x", linestyle=":", alpha=0.35)

    top_handles = [
        Patch(facecolor="#BBBBBB", edgecolor="black", label="AUC"),
        Patch(facecolor="#BBBBBB", edgecolor="black", hatch="///", label="RMSE"),
    ]
    ax_top.legend(handles=top_handles, loc="lower right", frameon=False)

    # Bottom: Scatter
    rmse_x = rmse_df["Chi2 Statistic"].to_numpy(dtype=float) * (1.0 + x_jitter_frac)

    auc_point_colors = [color_by_feature[f] for f in auc_df["Feature"]]
    rmse_point_colors = [color_by_feature[f] for f in rmse_df["Feature"]]

    ax_bot.scatter(
        auc_df["Chi2 Statistic"], auc_logp,
        label="AUC", marker="o", s=95,
        facecolors=auc_point_colors, edgecolors="black", linewidth=1.0
    )

    ax_bot.scatter(
        rmse_x, rmse_logp,
        label="RMSE", marker="*", s=160,
        facecolors=rmse_point_colors, edgecolors="black", linewidth=1.0
    )

    sig_y = -np.log10(0.05)
    ax_bot.axhline(sig_y, linestyle="--", linewidth=1.0, color="#9E9E9E")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="AUC",
               markerfacecolor="#888888", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="*", color="w", label="RMSE",
               markerfacecolor="#888888", markeredgecolor="black", markersize=12),
        Line2D([0], [0], color="#9E9E9E", lw=1.0, ls="--", label="p = 0.05"),
    ]
    ax_bot.legend(handles=legend_handles, loc="best", frameon=False)

    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Chi-Square Statistic (log scale)")
    ax_bot.set_ylabel("-log10(p-value)")
    ax_bot.set_title("Chi-Square vs. -log10(p-value)")
    ax_bot.grid(axis="both", linestyle=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

def visualize_chi_square_comparison(
    auc_df: pd.DataFrame,
    rmse_df: pd.DataFrame,
    outfile: str = "chi_square_comparison_combined.png",
    palette_name: str = "tab20",   # you can change to "husl", "Set3", etc.
    pastel_factor: float = 0.0,    # 0 = vivid colors, 1 = pastel
    x_jitter_frac: float = 0.03    # small horizontal offset for RMSE markers
    ) -> None:
    """
    Visualize chi-square statistics for AUC and RMSE:
      - Unified horizontal bars ordered by AUC (log X-axis).
      - Single scatter with AUC (circle) and RMSE (star), same color per feature.
      - Black outline on markers.
      - Legend in 'best' position.
      - Alternating background stripes behind bars to improve readability.
    """

    auc_df = auc_df.copy()
    rmse_df = rmse_df.copy()

    # Sort by AUC Chi²
    order_auc = (
        auc_df[["Feature", "Chi2 Statistic"]]
        .sort_values("Chi2 Statistic", ascending=False)["Feature"]
        .tolist()
    )

    # Build quick lookups
    auc_map = dict(zip(auc_df["Feature"], auc_df["Chi2 Statistic"]))
    rmse_map = dict(zip(rmse_df["Feature"], rmse_df["Chi2 Statistic"]))

    features = order_auc
    n = len(features)

    auc_vals = np.array([auc_map.get(f, np.nan) for f in features], dtype=float)
    rmse_vals = np.array([rmse_map.get(f, np.nan) for f in features], dtype=float)

    # Helper: -log10(p)
    def neglog10_p(series):
        adjusted = np.clip(series.to_numpy(dtype=float), a_min=1e-300, a_max=1.0)
        return -np.log10(adjusted)

    auc_logp = neglog10_p(auc_df["p-value"])
    rmse_logp = neglog10_p(rmse_df["p-value"])

    # Colors
    base_palette = sns.color_palette(palette_name, n_colors=n)
    if pastel_factor > 0:
        base_palette = sns.color_palette(
            [(np.array(c) + pastel_factor) / (1 + pastel_factor) for c in base_palette]
        )
    color_by_feature = {feat: col for feat, col in zip(features, base_palette)}

    # ===== Figure =====
    plt.rcParams.update({"font.size": 14})
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [2, 2]}
    )

    # ----- Top: Bars with zebra background -----
    y = np.arange(n)
    bar_h = 0.36
    offset = bar_h / 2.0

    auc_bar_colors = [color_by_feature[f] for f in features]
    rmse_bar_colors = auc_bar_colors

    # Alternating background stripes (behind the bars)
    # Each stripe spans one "row" centered at integer y ticks.
    for i in range(n):
        face = "white" if (i % 2 == 0) else "#DADADA"
        ax_top.axhspan(i - 0.5, i + 0.5, facecolor=face, alpha=1.0, zorder=0)

    # Bars on top of the background
    ax_top.barh(
        y + offset, auc_vals, height=bar_h,
        color=auc_bar_colors, edgecolor="black", linewidth=0.6, label="AUC", zorder=3
    )
    ax_top.barh(
        y - offset, rmse_vals, height=bar_h,
        color=rmse_bar_colors, edgecolor="black", linewidth=0.6,
        hatch="///", label="RMSE", zorder=3
    )

    ax_top.set_yticks(y, features)
    ax_top.set_ylim(-0.5, n - 0.5)  # ensure full stripes are visible
    ax_top.set_xscale("log")
    ax_top.set_xlabel("Chi-Square Statistic (log scale)")
    ax_top.set_ylabel("Feature")
    ax_top.set_title("Chi-Square Test Statistics by Feature")
    ax_top.set_axisbelow(True)  # grid above background, below bars
    ax_top.grid(axis="x", linestyle=":", alpha=0.35, zorder=2)

    top_handles = [
        Patch(facecolor="#BBBBBB", edgecolor="black", label="AUC"),
        Patch(facecolor="#BBBBBB", edgecolor="black", hatch="///", label="RMSE"),
    ]
    ax_top.legend(handles=top_handles, loc="lower right", frameon=False)

    # ----- Bottom: Scatter -----
    rmse_x = rmse_df["Chi2 Statistic"].to_numpy(dtype=float) * (1.0 + x_jitter_frac)
    auc_point_colors = [color_by_feature[f] for f in auc_df["Feature"]]
    rmse_point_colors = [color_by_feature[f] for f in rmse_df["Feature"]]

    ax_bot.scatter(
        auc_df["Chi2 Statistic"], auc_logp,
        label="AUC", marker="o", s=95,
        facecolors=auc_point_colors, edgecolors="black", linewidth=1.0
    )
    ax_bot.scatter(
        rmse_x, rmse_logp,
        label="RMSE", marker="*", s=160,
        facecolors=rmse_point_colors, edgecolors="black", linewidth=1.0
    )

    sig_y = -np.log10(0.05)
    ax_bot.axhline(sig_y, linestyle="--", linewidth=1.0, color="#9E9E9E")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="AUC",
               markerfacecolor="#888888", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="*", color="w", label="RMSE",
               markerfacecolor="#888888", markeredgecolor="black", markersize=12),
        Line2D([0], [0], color="#9E9E9E", lw=1.0, ls="--", label="p = 0.05"),
    ]
    ax_bot.legend(handles=legend_handles, loc="best", frameon=False)

    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Chi-Square Statistic (log scale)")
    ax_bot.set_ylabel("-log10(p-value)")
    ax_bot.set_title("Chi-Square vs. -log10(p-value)")
    ax_bot.grid(axis="both", linestyle=":", alpha=0.35)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


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

# Residual Analysis

def compute_standardized_residuals(contingency: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standardized residuals for a contingency table.
    Uses: (O - E) / sqrt(E * (1 - row_prop) * (1 - col_prop))
    """
    contingency = contingency.astype(float)
    total = contingency.values.sum()
    row_sum = contingency.sum(axis=1).values[:, None]
    col_sum = contingency.sum(axis=0).values[None, :]
    expected = (row_sum @ col_sum) / total

    # avoid divide-by-zero for degenerate cells
    row_prop = row_sum / total
    col_prop = col_sum / total
    denom = np.sqrt(expected * (1 - row_prop) * (1 - col_prop))
    denom[denom == 0] = np.nan

    residuals = (contingency.values - expected) / denom
    return pd.DataFrame(residuals, index=contingency.index, columns=contingency.columns)

def melt_residuals(res_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Long-format residuals with columns:
    ['Feature','FeatureLevel','MetricCategory','StdResidual']
    """
    long = res_df.reset_index().melt(id_vars=res_df.index.name or 'index',
                                     var_name='MetricCategory',
                                     value_name='StdResidual')
    long = long.rename(columns={res_df.index.name or 'index': 'FeatureLevel'})
    long.insert(0, 'Feature', feature_name)
    return long

def plot_residuals_heatmap(residuals_df: pd.DataFrame, title: str, outpath: str) -> None:
    """
    Save a heatmap of standardized residuals for a single feature.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(residuals_df, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, cbar_kws={'label': 'Standardized residual'})
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_residuals_grid(residuals_dict: dict[str, pd.DataFrame],
                        title_prefix: str,
                        cols: int = 3,
                        outpath: str = "residuals_grid.png") -> None:
    """
    Grid of heatmaps for multiple features (quick overview).
    """
    if not residuals_dict:
        return
    n = len(residuals_dict)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flatten():
        ax.axis('off')

    for ax, (feat, res_df) in zip(axes.flatten(), residuals_dict.items()):
        sns.heatmap(res_df, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, cbar=False, ax=ax)
        ax.set_title(f"{title_prefix}: {feat}")
        ax.set_xlabel("Metric category")
        ax.set_ylabel("Feature level")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def chi_square_analysis(
        features: pd.DataFrame,
        metric: str = 'AUC',
        feature_bits: Optional[list] = None,
        split: str = 'quaternary',
        invert_metric: bool = False,
        invert_feature: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Returns:
    - results_df
    - data (with metric categories)
    - residuals_dict  {feature: DataFrame residuals}
    - residuals_long  long-format residuals across all features
    - contingency_dict {feature: contingency table}
    """
    data = features.copy()

    if split not in ['binary', 'ternary', 'quaternary']:
        raise ValueError("split must be 'binary', 'ternary', or 'quaternary'.")

    # Metric discretization (como você já tinha)
    if split == 'binary':
        median = data[metric].median()
        data[f'{metric}_category'] = np.where(data[metric] > median, 'high', 'low')
        if invert_metric:
            data[f'{metric}_category'] = data[f'{metric}_category'].map({'high': 'low', 'low': 'high'})
    elif split == 'ternary':
        q1 = data[metric].quantile(0.33)
        q2 = data[metric].quantile(0.66)
        labels = ['low', 'medium', 'high'][::-1] if invert_metric else ['low', 'medium', 'high']
        data[f'{metric}_category'] = pd.cut(data[metric], bins=[-np.inf, q1, q2, np.inf], labels=labels)
    elif split == 'quaternary':
        q = data[metric].quantile([0.25, 0.5, 0.75])
        labels = ['very low', 'low', 'high', 'very high'][::-1] if invert_metric else ['very low', 'low', 'high', 'very high']
        data[f'{metric}_category'] = pd.cut(
            data[metric],
            bins=[-np.inf, q[0.25], q[0.5], q[0.75], np.inf],
            labels=labels
        )

    if feature_bits is None:
        feature_bits = [col for col in features.columns if col.startswith('feature_')]

    results = {'Feature': [], 'Chi2 Statistic': [], 'p-value': [], "Cramér's V": []}
    residuals_dict: dict[str, pd.DataFrame] = {}
    contingency_dict: dict[str, pd.DataFrame] = {}
    residuals_long_list: list[pd.DataFrame] = []

    for bit in feature_bits:
        bit_data = data[bit]
        if invert_feature:
            bit_data = 1 - pd.to_numeric(bit_data, errors='coerce')

        contingency = pd.crosstab(bit_data, data[f'{metric}_category'])
        contingency_dict[bit] = contingency

        # guarda NaN quando a tabela é degenerada
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            chi2, p, cramers_v = np.nan, np.nan, np.nan
            res_df = pd.DataFrame()
        else:
            chi2, p, _, _ = chi2_contingency(contingency)
            n = contingency.values.sum()
            r, c = contingency.shape
            denom = min(r - 1, c - 1)
            cramers_v = np.sqrt(chi2 / (n * denom)) if denom > 0 else np.nan

            # resíduos padronizados
            res_df = compute_standardized_residuals(contingency)

        results['Feature'].append(bit)
        results['Chi2 Statistic'].append(chi2)
        results['p-value'].append(p)
        results["Cramér's V"].append(cramers_v)

        if not res_df.empty:
            res_df.index.name = bit
            residuals_dict[bit] = res_df
            residuals_long_list.append(melt_residuals(res_df, bit))

    residuals_long = (pd.concat(residuals_long_list, ignore_index=True)
                      if residuals_long_list else
                      pd.DataFrame(columns=['Feature','FeatureLevel','MetricCategory','StdResidual']))

    return pd.DataFrame(results), data, residuals_dict, residuals_long, contingency_dict

# Feature importance using Random Forest

def random_forest_feature_engineering(
    df: pd.DataFrame, 
    target_column: str = 'AUC', 
    feature_columns: list = [], 
    n_estimators: int = 100, 
    random_state: int = 42,
    seed_ablation: bool = True
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
    seed_ablation : bool, optional
        If True, will use a fixed random state for feature importance calculation. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature importance scores sorted in descending order.
    '''

    # Check if the target column is AUC or RMSE
    if target_column not in ['AUC', 'RMSE', 'score']:
        raise ValueError("Invalid target column. Choose 'AUC', 'RMSE', or 'score'.")
    #else:
    #    target_column = f'{target_column}_category'

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
    #rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
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
                                       importance_df_rmse: pd.DataFrame,
                                       importance_df_score: Optional[pd.DataFrame]) -> None:
    '''
    Plot feature importance for AUC and RMSE in separate bar plots.

    Parameters
    ----------
    importance_df_auc : pd.DataFrame
        DataFrame containing feature importance for AUC.
    importance_df_rmse : pd.DataFrame
        DataFrame containing feature importance for RMSE.
    importance_df_score : pd.DataFrame, optional
        DataFrame containing feature importance for score. If provided, it will be plotted alongside AUC and RMSE.
    '''

    # Sort DataFrames by importance
    importance_df_auc = importance_df_auc.sort_values(by='Importance', ascending=False)
    importance_df_rmse = importance_df_rmse.sort_values(by='Importance', ascending=False)

    if importance_df_score is not None:
        importance_df_score = importance_df_score.sort_values(by='Importance', ascending=False)
        nplots = 3
    else:
        nplots = 2

    # Set up the plot with two independent y-axes
    fig, axes = plt.subplots(1, nplots, figsize=(10 * nplots, 10))

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

    if importance_df_score is not None:
        # Score Plot
        sns.barplot(
            y='Feature', 
            x='Importance', 
            hue='Feature',  # Assign 'Feature' to hue to avoid the warning
            data=importance_df_score, 
            ax=axes[2], 
            palette='rocket', 
            dodge=False,  # Prevents multiple bars for each feature
            legend=False  # We don't need the legend
        )
        axes[2].set_title('Feature Importance for Score', fontsize=16, weight='bold')
        axes[2].set_xlabel('Importance', fontsize=12)
        axes[2].set_ylabel('Feature', fontsize=12)

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

# Statistical Analysis and Visualization
import matplotlib.pyplot as plt
from scipy.stats import probplot, shapiro, anderson

def qq_normality_test(data, columns):
    """
    Gera QQ plots lado a lado para duas colunas e realiza testes de normalidade.
    Usa Shapiro-Wilk para N <= 5000 e Anderson-Darling para N > 5000.
    
    Parâmetros:
    - data: pandas DataFrame.
    - columns: lista com dois nomes de colunas [col1, col2].

    Retorna:
    - dicionário com os resultados dos testes para ambas as variáveis.
    """
    assert len(columns) == 2, "Você deve passar exatamente duas colunas."

    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    letters = ['A', 'B']

    for i, col in enumerate(columns):
        values = data[col].dropna()
        n = len(values)

        ax = axes[i]
        probplot(values, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot - {col} (n={n})")
        ax.set_xlabel("Quantis Teóricos")
        ax.set_ylabel("Quantis da Amostra")
        ax.grid(True)

        # Anotação fora do gráfico (letras A, B)
        fig.text(0.05 + i * 0.45, 0.95, f'{letters[i]}', fontsize=14, fontweight='bold')

        print(f"\nAnálise de normalidade para: {col}")
        print(f"Tamanho da amostra: {n}")

        res = {'sample_size': n}

        if n <= 5000:
            stat, p = shapiro(values)
            p_text = f"p = {p:.2e}"
            print(f"Shapiro-Wilk Test:\n  W = {stat:.4f}, p = {p:.4e}")
            if p < 0.05:
                conclusion = "Reject normality"
            else:
                conclusion = "Do not reject normality"
            res.update({
                'test': 'Shapiro-Wilk',
                'statistic': stat,
                'p_value': p,
                'conclusion': conclusion
            }) # type: ignore

        else:
            result = anderson(values, dist='norm')
            stat = getattr(result, 'statistic')
            critical_values = getattr(result, 'critical_values')
            significance_levels = getattr(result, 'significance_level')

            # Como o Anderson-Darling não fornece p-value diretamente, você pode indicar rejeição ou não:
            if stat > critical_values[2]:  # 5%
                reject = True
            else:
                reject = False
            
            p_text = f"A² = {stat:.4f}"

            print(f"Anderson-Darling Test Statistic: {stat:.4f}")
            for cv, sig in zip(critical_values, significance_levels):
                print(f"  Critério para {sig:.1f}%: {cv:.4f}")

            res.update({
                'test': 'Anderson-Darling',
                'statistic': stat,
                'critical_values': critical_values.tolist(),
                'significance_levels': significance_levels.tolist(),
                'reject': reject
            }) # type: ignore

        # Adiciona anotação do p-valor no canto inferior direito do subplot
        ax.text(0.95, 0.05, p_text,
                fontsize=10, transform=ax.transAxes,
                ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.8))

        results[col] = res

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig('qq_plot_dual.png', bbox_inches='tight')
    plt.close()

    return results

def nonparametric_group_comparison(data, value_col, group_col, k):
    """
    Realiza comparação não paramétrica entre grupos.
    
    - Se dois grupos → Mann-Whitney U Test.
    - Se mais de dois grupos → Kruskal-Wallis + Dunn post-hoc (se Kruskal for significativo).

    Parâmetros:
    - data: DataFrame com os dados.
    - value_col: string, nome da coluna com os valores (ex.: 'AUC').
    - group_col: string, nome da coluna com os grupos (ex.: 'cluster').
    - k: número de grupos (usado para calcular epsilon squared).

    Retorna:
    - Dicionário com resultados dos testes.
    """

    grupos = data[group_col].dropna().unique()
    grupos.sort()
    print(f"\n### Comparação de {value_col} por {group_col} ###")
    print(f"Número de grupos: {k}")
    print(f"Teste a ser realizado: {'Mann-Whitney U Test' if k == 2 else 'Kruskal-Wallis'}")

    results = {}

    if k == 2:
        # Mann-Whitney U Test
        group1, group2 = grupos[0], grupos[1]
        x = data[data[group_col] == group1][value_col].dropna()
        y = data[data[group_col] == group2][value_col].dropna()

        print(f"\n→ Executando Mann-Whitney para {group1} vs {group2}")
        res = pg.mwu(x, y, alternative='two-sided')

        print(res)

        results = {
            'test': 'Mann-Whitney',
            'groups': (group1, group2),
            'result': res
        }

    elif k > 2:
        # Kruskal-Wallis
        print("\n→ Executando Kruskal-Wallis")

        kruskal_res = pg.kruskal(dv=value_col, between=group_col, data=data)
        H = kruskal_res['H'].values[0]
        p = kruskal_res['p-unc'].values[0]
        N = data['cluster'].notna().sum()
        epsilon_sq = (H - k + 1) / (N - k)
        epsilon_sq = max(0, epsilon_sq)  # Para evitar valores negativos por arredondamento

        print(f"Kruskal-Wallis H = {H:.4f}, p = {p:.4e}, epsilon_squared = {epsilon_sq:.4f}")

        results = {
            'test': 'Kruskal-Wallis',
            'kruskal': kruskal_res,
            'epsilon_squared': epsilon_sq
        }

        if p < 0.05:
            print("→ Kruskal significativo. Executando Dunn post-hoc...")
            dunn_res = pg.pairwise_tests(
                dv=value_col,
                between=group_col,
                data=data,
                padjust='bonf'
            )
            dunn_res['significant'] = dunn_res['p-corr'] < 0.05

            print(dunn_res[['A', 'B', 'p-unc', 'p-corr', 'significant']])
            results['dunn'] = dunn_res
        else:
            print("→ Kruskal não significativo. Dunn post-hoc não realizado.")
            results['dunn'] = None

    else:
        raise ValueError("Menos de dois grupos encontrados. Comparação não possível.")

    return results

# Execute functions
print('Running Elbow Method...')
wcss = run_elbow(data, normalized_features, max_clusters=10, plot=True)

# Set the optimal number of clusters based on the elbow method
if "Seed" in study_name:
    optimal_clusters = 2
else:
    optimal_clusters = 3

if method == "kmeans":
    print('Running K-Means Clustering...')
    k_means_corr = run_kmeans(data, normalized_features, optimal_clusters=optimal_clusters, alpha=0.4, add_all_markers=False)
    #k_means_corr = run_kmeans(data, normalized_features, optimal_clusters=optimal_clusters, alpha=0.4)
elif method == "dbscan":
    print('Running DBSCAN Clustering...')
    dbscan_corr = run_dbscan(normalized_features)
elif method == "hdbscan":
    print('Running HDBSCAN Clustering...')
    hdbscan_corr = run_hdbscan(normalized_features, min_samples=10, min_cluster_size=10)
elif method == "meanshift":
    print('Running MeanShift Clustering...')
    meanshift_corr = run_meanshift(normalized_features)
elif method == "agglomerative":
    print('Running Agglomerative Clustering...')
    agglomerative_corr = run_agglomerative(normalized_features)
elif method == "spectral":
    print('Running Spectral Clustering...')
    spectral_corr = run_spectral(normalized_features, n_clusters=optimal_clusters)
elif method == "ward":
    print('Running Ward Clustering...')
    ward_corr = run_ward(normalized_features, n_clusters=optimal_clusters)
elif method == "optics":
    print('Running OPTICS Clustering...')
    optics_corr = run_optics(normalized_features)
elif method == "birch":
    print('Running Birch Clustering...')
    birch_corr = run_birch(normalized_features, n_clusters=optimal_clusters)
elif method == "gmm":
    print('Running Gaussian Mixture Clustering...')
    gmm_corr = run_gaussian_mixture(normalized_features, n_components=optimal_clusters)
elif method == "affinity_propagation":
    print('Performing individual contributions analysis for each feature...')
    results_df = individual_contributions_analysis(normalized_features, verbose=True, features_names=features_names)
else:
    raise ValueError(f"Unknown clustering method: {method}")

print("Performing QQ normality test for AUC and RMSE...")
qq_test_results = qq_normality_test(data, ['AUC', 'RMSE'])

# Comparar AUC entre os grupos do cluster
resultado_auc = nonparametric_group_comparison(data, value_col='AUC', group_col='cluster', k=optimal_clusters)
# Comparar RMSE entre os grupos do cluster
resultado_rmse = nonparametric_group_comparison(data, value_col='RMSE', group_col='cluster', k=optimal_clusters)

print('Plotting feature mask correlation heatmap...')
plot_feature_mask_correlation_heatmap(features, features_names=features_names)

print('Performing Chi-Square test for feature selection...')
chi_square_results_auc, auc_df, auc_resid_dict, auc_resid_long, cont_auc = chi_square_analysis(features, metric='AUC', feature_bits=features_names, split='quaternary', invert_metric=True, invert_feature=False)
chi_square_results_rmse, rmse_df, rmse_resid_dict, rmse_resid_long, cont_rmse = chi_square_analysis(features, metric='RMSE', feature_bits=features_names, split='quaternary', invert_metric=True, invert_feature=False)

print('Plotting Cramér\'s V comparison...')
plot_cramers_comparison(chi_square_results_auc, chi_square_results_rmse)

print('Plotting Chi-Square test results...')
visualize_chi_square_comparison(
    chi_square_results_auc,
    chi_square_results_rmse,
    palette_name="tab20",
    pastel_factor=0.1,
    x_jitter_frac=0.0
)

print("Applying Bonferroni correction to the p-values...")
chi_square_results_auc['Bonferroni_Significant_AUC'], corrected_alpha_auc = bonferroni_correction(chi_square_results_auc['p-value'].tolist())
chi_square_results_rmse['Bonferroni_Significant_RMSE'], corrected_alpha_rmse = bonferroni_correction(chi_square_results_rmse['p-value'].tolist())

print("Applying Benjamini-Hochberg correction to the p-values...")
chi_square_results_auc['BH_Significant_AUC'] = benjamini_hochberg(chi_square_results_auc['p-value'].tolist())
chi_square_results_rmse['BH_Significant_RMSE'] = benjamini_hochberg(chi_square_results_rmse['p-value'].tolist())

print('Performing feature importance analysis...')
if "Seed" in features_names:
    feature_importance_df_auc = random_forest_feature_engineering(features, target_column='AUC', feature_columns=["seed"], seed_ablation = True)
    feature_importance_df_rmse = random_forest_feature_engineering(features, target_column='RMSE', feature_columns=["seed"], seed_ablation = True)
    feature_importance_df_score = random_forest_feature_engineering(features, target_column='score', feature_columns=["seed"], seed_ablation = True)
else:
    feature_importance_df_auc = random_forest_feature_engineering(features, target_column='AUC', feature_columns=features_names, seed_ablation = True)
    feature_importance_df_rmse = random_forest_feature_engineering(features, target_column='RMSE', feature_columns=features_names, seed_ablation = True)
    feature_importance_df_score = random_forest_feature_engineering(features, target_column='score', feature_columns=features_names, seed_ablation = True)

print('Plotting feature importance analysis...')
plot_feature_importance_comparison(feature_importance_df_auc, feature_importance_df_rmse, feature_importance_df_score)

print('Analyzing the impact of the number of features turned on...')
feature_effects = analyze_feature_effects(features, feature_columns=features_names)

print('Done!')

# Make pearson correlation results for the data df, columns user_attrs_pr_auc and user_attrs_AUC
corr, p_value = pearsonr(data['user_attrs_pr_auc'], data['user_attrs_AUC'])

# Print the results
print(f"Pearson correlation: {corr}")
print(f"P-value: {p_value}")

# Make spearman correlation results for the data df, columns user_attrs_pr_auc and user_attrs_AUC
corr, p_value = spearmanr(data['user_attrs_pr_auc'], data['user_attrs_AUC'])

# Print the results
print(f"Spearman correlation: {corr}")
print(f"P-value: {p_value}")

# Make kendall correlation results for the data df, columns user_attrs_pr_auc and user_attrs_AUC
corr, p_value = kendalltau(data['user_attrs_pr_auc'], data['user_attrs_AUC'])

# Print the results
print(f"Kendall correlation: {corr}")
print(f"P-value: {p_value}")

