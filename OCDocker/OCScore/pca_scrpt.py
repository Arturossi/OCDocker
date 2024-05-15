
import os
import pandas as pd
import pickle
import seaborn as sns
import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

import torch

def save_object(obj, filename):
    """
    Save an object to a file using pickle.

    Parameters:
    obj (any): The object to be pickled.
    filename (str): The name of the file where the object will be stored.
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    """
    Load an object from a file using pickle.

    Parameters:
    filename (str): The name of the file from which to load the object.

    Returns:
    The unpickled object.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)
    
def load_data(file_name) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Parameters
    ----------
    file_name: str
        Name of the CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.
    """

    return pd.read_csv(file_name)

def invert_values_conditionally(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inverts the values of specific columns in a DataFrame. The inversion is applied to columns
    that start with 'VINA' or 'SMINA', as well as the column named 'experimental'.

    This function multiplies the values in these columns by -1, effectively inverting them. It's 
    particularly useful in scenarios where the sign of these values needs to be reversed for 
    analysis or data processing.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with inverted values, ensuring not to modify the original DataFrame.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_modified = df.copy()

    # Get the columns to invert
    invert_columns = [col for col in df_modified.columns if col.startswith("VINA") or col.startswith("SMINA") or col == "experimental"]

    # For each column, multiply the values by -1
    for col in invert_columns:
        df_modified.loc[:, col] *= -1

    return df_modified

def norm_data(df: pd.DataFrame, scaler: str = 'standard', inplace: bool = False) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by scaling selected feature columns using a Scaler.
    The metadata columns ('receptor', 'ligand', 'name', 'type', 'db') are preserved.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    scaler: str
        Scaler to use. Options are 'standard' and 'minmax'.
    inplace: bool
        If True, the original DataFrame is modified. If False, a new DataFrame is returned.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized features while preserving metadata.
    """

    # Check the chosen scaler
    if scaler not in ['standard', 'minmax']:
        raise ValueError("Invalid scaler. Please choose 'standard' or 'minmax'")
    
    # Initialize the scaler
    scaler_model = StandardScaler() if scaler == 'standard' else MinMaxScaler()

    # Select columns to be scaled
    feature_columns = df.columns.difference(['receptor', 'ligand', 'name', 'type', 'db'])

    if inplace:
        # Scale only the selected feature columns in the original DataFrame
        df[feature_columns] = scaler_model.fit_transform(df[feature_columns])
        return df
    
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Scale only the selected feature columns
    df_copy[feature_columns] = scaler_model.fit_transform(df_copy[feature_columns])

    return df_copy

def plot_corr_matrix(df: pd.DataFrame, columns: list = [], scaler: str = "") -> None:
    """
    Plots a correlation matrix for a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the features to plot the correlation matrix for.
    columns: list, optional
        List of columns to plot the correlation matrix for. If empty, all columns except metadata are used.
    scaler: str, optional
        Name of the scaler used to normalize the data. This is used to name the output file.
    """

    # If the scaler is not specified, set it as "raw"
    if not scaler:
        scaler = "raw"

    # If no columns are specified, use all columns except metadata
    if not columns:
        # Extract features from the DataFrame dropping metadata columns if present
        features = df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'], errors = 'ignore')
    
    else:
        # Check if the specified columns are present in the DataFrame
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in DataFrame")
            
        # Extract features from the DataFrame using the specified columns
        features = df[columns]
    
    # Check if there is an experimental column and if it is made of np.nan
    if 'experimental' in features.columns and features['experimental'].isnull().all():
        features = features.drop(columns = 'experimental')

    # Get the db values
    db = df['db'].unique()

    # Check if there are multiple databases
    if len(db) > 1:
        db = "_".join(db)
    else:
        db = db[0]

    # Compute the correlation matrix
    corr_matrix = features.corr()

    # Determine if the annotations should be included
    annotate = corr_matrix.shape[0] <= 20

    # Create the plot
    plt.figure(figsize = (12, 10))
    sns.heatmap(corr_matrix, cmap = "coolwarm", linewidths = 0.5, annot=annotate, vmin = -1, vmax = 1)
    plt.title("Feature Correlation Matrix for " + db + " Database (" + scaler + " scaled)")
    plt.tight_layout()
    plt.savefig(f"corrplot_{db}_{scaler}.png")
    plt.close()

def plot_correlation_similarity(df1: pd.DataFrame, df2: pd.DataFrame, columns: list = [], annot: bool = True, fontsize: Union[float, None] = None, normalize: bool = True) -> None:
    """
    Plots the similarity of correlation matrices from two DataFrames.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame.
    df2 : pd.DataFrame
        The second DataFrame.
    columns : list, optional
        List of columns to compare. If empty, all columns except metadata are used.
    annot : bool, optional
        If True, write the data value in each cell. If False, don't write the data value.
    fontsize : int, optional
        The size of the font for the data value annotations.
    normalize : bool, optional
        If True, normalize the correlation matrices after calculating the similarity.
    """

    # If no columns are specified, use all columns except metadata
    if not columns:
        # Find common columns in both DataFrames
        columns = df1.columns.intersection(df2.columns) # type: ignore

    # Filter both DataFrames to include only common columns
    filtered_df1 = df1[columns]
    filtered_df2 = df2[columns]

    # Calculate the correlation matrices
    corr_matrix_df1 = filtered_df1.corr()
    corr_matrix_df2 = filtered_df2.corr()

    # Calculate the similarity (or difference) matrix
    # This can be customized as needed; here we use simple subtraction
    similarity_matrix = corr_matrix_df1 - corr_matrix_df2

    # Normalize the similarity matrix with min max scaling
    if normalize:
        min_val = similarity_matrix.min().min()
        max_val = similarity_matrix.max().max()
        matrix_shifted = similarity_matrix - min_val
        matrix_scaled = matrix_shifted / (max_val - min_val)
        similarity_matrix = (matrix_scaled * 2) - 1

    # Plot the similarity matrix as a heatmap
    plt.figure(figsize = (10, 8))
    ax = sns.heatmap(similarity_matrix, annot = annot, cmap = 'coolwarm', center = 0, vmin = -1, vmax = 1, linewidths = 0.5, fmt = ".2f")
    plt.title('Heatmap of Correlation Matrix Similarity')

    # Set annotation font size
    if fontsize and annot:
        for text in ax.texts:
            text.set_size(fontsize)

    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
    plt.savefig('correlation_similarity.png')
    plt.close()

    ## Reorder for readability

    # Perform hierarchical clustering to reorder the correlation matrix
    linkage_matrix = linkage(similarity_matrix, method = 'average')
    order = leaves_list(linkage_matrix)

    # Reorder the similarity matrix based on the hierarchical clustering
    similarity_matrix = similarity_matrix.iloc[order, order]

    # Plot the reordered similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(similarity_matrix, annot = True, cmap = 'coolwarm', center = 0, vmin = -1, vmax = 1, linewidths = 0.5, fmt = ".2f")
    plt.title('Reordered Heatmap of Correlation Matrix Similarity')

    # Set annotation font size
    if fontsize and annot:
        for text in ax2.texts:
            text.set_size(fontsize)

    plt.tight_layout()
    plt.savefig('correlation_similarity_sorted.png')
    plt.close()

def plot_roc_curves(df: pd.DataFrame, feature_cols: list, labels: pd.Series, title: str = "ROC") -> None:
    """
    Plots ROC curves for a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the features to plot the ROC curves for.
    feature_cols: list
        List of feature columns to plot ROC curves for.
    labels: pd.Series
        Series containing the labels for the ROC curves.
    title: str, optional
        Title of the plot. Default is "ROC".
    """

    # Get the db values
    db = df['db'].unique()

    # Check if there are multiple databases
    if len(db) > 1:
        db = "_".join(db)
    else:
        db = db[0]

    # Calculate AUC for each feature and store the results
    auc_dict = {}
    for feature in feature_cols:
        fpr, tpr, _ = roc_curve(labels, df[feature])
        roc_auc = auc(fpr, tpr)
        auc_dict[feature] = roc_auc

    # Sort the features by their AUC in descending order
    sorted_features = sorted(auc_dict, key=auc_dict.get, reverse=True) # type: ignore

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Plot ROC curves for each feature, now sorted by AUC
    for feature in sorted_features:
        fpr, tpr, _ = roc_curve(labels, df[feature])
        roc_auc = auc_dict[feature]
        plt.plot(fpr, tpr, lw=2, label=f'{feature} (area = {roc_auc:.2f})')

    # Plot the random line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot parameters
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curves for {db} Dataset Features")

    # Move the legend outside of the plot area
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout for tight fit, so the legend fits within the figure
    plt.tight_layout()

    plt.savefig(f'{title}.png')
    plt.close()

def calculate_metrics(df: pd.DataFrame, selected_columns: list) -> Tuple[pd.DataFrame, list]:
    """
    Calculates additional metrics for a DataFrame. The metrics include average, median, maximum,
    minimum, standard deviation, variance, sum, range, 25th and 75th percentiles.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    selected_columns: list
        List of columns to calculate metrics for.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional metrics.
    list
        List of additional metrics column names.
    """

    # Check if selected columns are present in the DataFrame
    for col in selected_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Calculate metrics
    df['mean'] = df[selected_columns].mean(axis = 1)                  # The mean of the selected columns
    df['median'] = df[selected_columns].median(axis = 1)              # The median of the selected columns
    df['max'] = df[selected_columns].max(axis = 1)                    # The maximum value of the selected columns
    df['min'] = df[selected_columns].min(axis = 1)                    # The minimum value of the selected columns
    df['std'] = df[selected_columns].std(axis = 1)                    # The standard deviation of the selected columns
    df['variance'] = df[selected_columns].var(axis = 1)               # The variance of the selected columns
    df['sum'] = df[selected_columns].sum(axis = 1)                    # The sum of the selected columns
    df['range'] = df['max'] - df['min']                               # The range of the selected columns
    df['quantile_25'] = df[selected_columns].quantile(0.25, axis = 1) # The 25th percentile of the selected columns (lower quartile)
    df['quantile_75'] = df[selected_columns].quantile(0.75, axis = 1) # The 75th percentile of the selected columns (upper quartile)
    df['iqr'] = df['quantile_75'] - df['quantile_25']                 # The interquartile range of the selected columns (IQR)
    df['skewness'] = df[selected_columns].skew(axis = 1)              # The skewness of the selected columns (measure of asymmetry)
    df['kurtosis'] = df[selected_columns].kurtosis(axis = 1)          # The kurtosis of the selected columns (measure of tailedness)

    # Return DataFrame with additional metrics
    return df, ['mean', 'median', 'max', 'min', 'std', 'variance', 'sum', 'range', 'quantile_25', 'quantile_75', 'iqr', 'skewness', 'kurtosis']

def compute_zscore(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Computes the z-score for the specified columns in a DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame.
    columns: list
        List of columns to compute the z-score for.

    Returns
    -------
    pd.DataFrame
        DataFrame with z-score values for the specified columns.
    """

    # Check if the specified columns are present in the DataFrame
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")

    # Compute the z-score for the specified columns
    zscore_df = df.copy()
    zscore_df[["z_" + s for s in columns]] = (zscore_df[columns] - zscore_df[columns].mean()) / zscore_df[columns].std()

    return zscore_df

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    X (pandas.DataFrame): The input features.
    y (pandas.Series): The target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    Returns:
    X_train (pandas.DataFrame): The training input features.
    X_test (pandas.DataFrame): The testing input features.
    y_train (pandas.Series): The training target variable.
    y_test (pandas.Series): The testing target variable.
    """
    
    # Split the data into training and testing sets
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


############################################################################################################

# Load and preprocess data
df = load_data('/data/hd4tb/OCDocker/data/ocdb/predictions/OCDocker_pre.csv.gz')

# Define the score columns
score_columns = df.filter(regex='^(SMINA|VINA|ODDT|PLANTS)').columns.to_list()

# Split DUDEz data from PDBbind
dudez_data = df[df['db'] == 'DUDEz']
pdbbind_data = df[df['db'] == 'PDBbind']

# Inverting values
dudez_data = invert_values_conditionally(dudez_data)

# Inverting values
pdbbind_data = invert_values_conditionally(pdbbind_data)

# Drop the experimental column from DUDEz data
dudez_data = dudez_data.drop(columns = 'experimental')

dudez_standard_norm_df = norm_data(dudez_data, scaler = 'standard')
dudez_minmax_norm_df = norm_data(dudez_data, scaler = 'minmax')
pdbbind_standard_norm_df = norm_data(pdbbind_data, scaler = 'standard')
pdbbind_minmax_norm_df = norm_data(pdbbind_data, scaler = 'minmax')

# Perform PCA on the data
from sklearn.decomposition import PCA

# Define the % of variance to be explained
variance = 0.80

# Create the PCA object
pca = PCA(n_components = variance)

# Perform PCA on the DUDEz dataset (Dropping score columns)
pdbbind_pca_standard = pca.fit_transform(pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns, errors = 'ignore'))

# Convert the variance to string
variance_str = str(variance*100).replace('.0', '')

# Save the PCA object in pickle format
save_object(pca, f'pca{variance_str}.pkl')

### Further testing

# Perform PCA on the all datasets
dudez_pca_standard = pca.fit_transform(dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'] + score_columns, errors = 'ignore'))
dudez_pca_minmax = pca.fit_transform(dudez_minmax_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'] + score_columns, errors = 'ignore'))
pdbbind_pca_standard = pca.fit_transform(pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns, errors = 'ignore'))
pdbbind_pca_minmax = pca.fit_transform(pdbbind_minmax_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns, errors = 'ignore'))

# Create a DataFrame with the PCA results for each dataset then add the score columns back
dudez_pca_standard_df = pd.DataFrame(data = dudez_pca_standard, columns = [f'PC{i+1}' for i in range(dudez_pca_standard.shape[1])])
dudez_pca_minmax_df = pd.DataFrame(data = dudez_pca_minmax, columns = [f'PC{i+1}' for i in range(dudez_pca_minmax.shape[1])])
pdbbind_pca_standard_df = pd.DataFrame(data = pdbbind_pca_standard, columns = [f'PC{i+1}' for i in range(pdbbind_pca_standard.shape[1])])
pdbbind_pca_minmax_df = pd.DataFrame(data = pdbbind_pca_minmax, columns = [f'PC{i+1}' for i in range(pdbbind_pca_minmax.shape[1])])

# Add the metadata columns back
dudez_pca_standard_df = pd.concat([dudez_standard_norm_df[score_columns + ['receptor', 'ligand', 'name', 'type', 'db']], dudez_pca_standard_df], axis = 1)
dudez_pca_minmax_df = pd.concat([dudez_minmax_norm_df[score_columns + ['receptor', 'ligand', 'name', 'type', 'db']], dudez_pca_minmax_df], axis = 1)
pdbbind_pca_standard_df = pd.concat([pdbbind_standard_norm_df[score_columns + ['receptor', 'ligand', 'name', 'type', 'db', 'experimental']], pdbbind_pca_standard_df], axis = 1)
pdbbind_pca_minmax_df = pd.concat([pdbbind_minmax_norm_df[score_columns + ['receptor', 'ligand', 'name', 'type', 'db', 'experimental']], pdbbind_pca_minmax_df], axis = 1)

# Check for NaNs in the PCA datasets
print("==== NaNs in PCA datasets ====")
print("--------------------------------")
print("DUDEz")
print(dudez_pca_standard_df.isnull().sum())
print(dudez_pca_minmax_df.isnull().sum())
print("\nPDBbind")
print(pdbbind_pca_standard_df.isnull().sum())
print(pdbbind_pca_minmax_df.isnull().sum())


# Compare the size of the datasets before and after PCA
print("==== Dataset sizes ====")
print("-----------------------")
print("DUDEZ")
print(f"Before PCA: {dudez_data.shape[1] - 5 - len(score_columns)} features")
print(f"After PCA (standard scaling): {dudez_pca_standard_df.shape[1] - 5} features")
print(f"After PCA (min-max scaling): {dudez_pca_minmax_df.shape[1] - 5} features")

print("\nPDBbind")
print(f"Before PCA: {pdbbind_data.shape[1] - 6 - len(score_columns)} features")
print(f"After PCA (standard scaling): {pdbbind_pca_standard_df.shape[1] - 6} features")
print(f"After PCA (min-max scaling): {pdbbind_pca_minmax_df.shape[1] - 6} features")
