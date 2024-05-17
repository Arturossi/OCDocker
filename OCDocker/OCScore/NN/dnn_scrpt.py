
import os
import pandas as pd
import pickle
import seaborn as sns
import time
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
#dudez_minmax_norm_df = norm_data(dudez_data, scaler = 'minmax')
pdbbind_standard_norm_df = norm_data(pdbbind_data, scaler = 'standard')
#pdbbind_minmax_norm_df = norm_data(pdbbind_data, scaler = 'minmax')

use_pdb_train = True

# Do not use with autoencoder/multiencoder
use_PCA = True
# Set the PCA type 95/90/85/80
pca_type = 85

if use_PCA:
    # Load the PCA
    pca = load_object(f"/data/hd4tb/OCDocker/OCDocker/OCDocker/OCScore/pca{pca_type}.pkl")

    # Transform the data (train/test)
    pdbbind_pca_standard_df = pca.transform(pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns, errors = 'ignore'))

    # Make it a DataFrame
    pdbbind_pca_standard_df = pd.DataFrame(pdbbind_pca_standard_df, columns = [f"PC_{i}" for i in range(pdbbind_pca_standard_df.shape[1])])

    # Create a DataFrame with the metadata and reset the indexes
    metadata_df = pdbbind_standard_norm_df[['receptor', 'ligand', 'name', 'type', 'db', 'experimental'] + score_columns].reset_index(drop=True)

    # Add the scores back to the data in the same order
    pdbbind_standard_norm_df = pd.concat([metadata_df, pdbbind_pca_standard_df], axis=1)

    # Transform the data (validation)
    if use_pdb_train:
        # Transform the data (validation)
        dudez_standard_norm_df_tmp = pca.transform(dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'] + score_columns, errors = 'ignore'))

        # Make it a DataFrame
        dudez_standard_norm_df_tmp = pd.DataFrame(dudez_standard_norm_df_tmp, columns = [f"PC_{i}" for i in range(dudez_standard_norm_df_tmp.shape[1])])

        # Create a DataFrame with the metadata and reset the indexes
        metadata_df = dudez_standard_norm_df[['receptor', 'ligand', 'name', 'type', 'db'] + score_columns].reset_index(drop=True)

        # Add the scores back to the data
        dudez_standard_norm_df = pd.concat([metadata_df, dudez_standard_norm_df_tmp], axis=1)
    
    # Set the study name
    study_name = f"PCA{pca_type}_NN_Optimization"

    # Set the best AO to None
    best_ao_params = None
else:
    # Set the study name
    study_name = f"NN_Optimization"

if use_pdb_train:
    # Split the PDBbind data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'], errors = 'ignore'), pdbbind_standard_norm_df['experimental'], test_size = 0.25, random_state = 42)
    # Split the DUDEz data into validation X and y
    X_val = dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'], errors = 'ignore')
    y_val = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})
else:
    # Set the test size to 0.0 to use the entire dataset for training
    X_train = pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'], errors = 'ignore')
    y_train = pdbbind_standard_norm_df['experimental']

    X_test = dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'], errors = 'ignore')
    y_test = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})

    # Set X and y for validation to None
    X_val = None
    y_val = None


#trainer = NNOptimizer(X_train, y_train, X_test, y_test, X_val, y_val, 1, use_gpu=True, verbose=False)
#trainer.optimize(direction = "minimize", n_trials = 1000, study_name = "NN_Optimization_4_TPE", load_if_exists = True, sampler = TPESampler(), n_jobs = 10)
#trainer.optimize(direction = "minimize", n_trials = 1000, study_name = "NN_Optimization_4_CMA", load_if_exists = True, sampler = CmaEsSampler(), n_jobs = 10)

############################################################################################################

import optuna

from multiprocessing import Pool
from urllib.parse import quote_plus

from AutoencoderOptimizer import AutoencoderOptimizer
from NNOptimizer import NNOptimizer, NeuralNet
from optuna.samplers import TPESampler

def AOworker(
        pid, 
        id,
        X_train,
        X_test, 
        X_val,
        encoding_dims,
        storage,
        models_folder,
        random_seed = 42,
        use_gpu = True, 
        verbose = False, 
        direction = "minimize", 
        n_trials = 250, 
        load_if_exists = True, 
        n_jobs = 10, 
        study_name = "Autoencoder_Optimization"
    ):
    
    print(f"Process {pid} starting optimization")

    # Initialize the trainer
    trainer = AutoencoderOptimizer(
        X_train, 
        X_test, 
        X_val, 
        encoding_dims,
        storage,
        models_folder,
        random_seed = random_seed,
        use_gpu = use_gpu, 
        verbose = verbose
    )

    study = None
    
    for sampler_name, sampler in [("TPE", TPESampler())]:#, ("CMA", CmaEsSampler())]:
        # Run optimization
        study = trainer.optimize(
                direction = direction, 
                n_trials = n_trials, 
                study_name = f"{study_name}_{id}_{sampler_name}", 
                load_if_exists = load_if_exists, 
                sampler = sampler, 
                n_jobs = n_jobs
        )
        print(f"Process {id} completed {sampler_name} optimization")

    return study

def NNworker(
        pid, id,
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        encoder_params = None,
        output_size = 1, 
        random_seed = 42,
        use_gpu = True, 
        verbose = False, 
        direction = "minimize", 
        n_trials = 250, 
        load_if_exists = True, 
        n_jobs = 10, 
        study_name = "NN_Optimization"
    ):
    print(f"Process {pid} starting optimization")

    # Sleep pid seconds before starting
    time.sleep(pid)

    # Initialize the trainer
    trainer = NNOptimizer(
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        encoder_params,
        output_size = output_size, 
        random_seed = random_seed,
        use_gpu = use_gpu, 
        verbose=verbose
    )

    for sampler_name, sampler in [("TPE", TPESampler())]:#, ("CMA", CmaEsSampler())]:
        # Run optimization
        trainer.optimize(
            direction = direction, 
            n_trials = n_trials, 
            study_name = f"{study_name}_{id}_{sampler_name}", 
            load_if_exists = load_if_exists, 
            sampler = sampler, 
            n_jobs = n_jobs
        )
        print(f"Process {id} completed {sampler_name} optimization")

num_processes = 8
storage_id = 46
storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
models_folder = f"/data/hd4tb/OCDocker/data/ocdb/models/autoencoder_{storage_id}"
autoencoder = False
multiencoder = False
run_autoencoder_optimization = False
run_NN_optimization = True
explained_variance = 0.95

# If models folder does not exist, create it
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

if autoencoder:
    if multiencoder:
        # Set the classification for e
        sf = [col for col in X_train.columns if col.startswith("VINA") or col.startswith("SMINA") or col.startswith("ODDT") or col.startswith("PLANTS")]
        ligand = [
            'AUTOCORR2D_1', 'AUTOCORR2D_2', 'AUTOCORR2D_3', 'AUTOCORR2D_4', 'AUTOCORR2D_5', 'AUTOCORR2D_6', 'AUTOCORR2D_7', 'AUTOCORR2D_8', 'AUTOCORR2D_9', 'AUTOCORR2D_10', 'AUTOCORR2D_11', 'AUTOCORR2D_12', 'AUTOCORR2D_13', 'AUTOCORR2D_14', 'AUTOCORR2D_15', 'AUTOCORR2D_16', 'AUTOCORR2D_17', 'AUTOCORR2D_18', 'AUTOCORR2D_19', 'AUTOCORR2D_20', 'AUTOCORR2D_21', 'AUTOCORR2D_22', 'AUTOCORR2D_23', 'AUTOCORR2D_24', 'AUTOCORR2D_25', 'AUTOCORR2D_26', 'AUTOCORR2D_27', 'AUTOCORR2D_28', 'AUTOCORR2D_29', 'AUTOCORR2D_30', 'AUTOCORR2D_31', 'AUTOCORR2D_32', 'AUTOCORR2D_33', 'AUTOCORR2D_34', 'AUTOCORR2D_35', 'AUTOCORR2D_36', 'AUTOCORR2D_37', 'AUTOCORR2D_38', 'AUTOCORR2D_39', 'AUTOCORR2D_40', 'AUTOCORR2D_41', 'AUTOCORR2D_42', 'AUTOCORR2D_43', 'AUTOCORR2D_44', 'AUTOCORR2D_45', 'AUTOCORR2D_46', 'AUTOCORR2D_47', 'AUTOCORR2D_48', 'AUTOCORR2D_49', 'AUTOCORR2D_50', 'AUTOCORR2D_51', 'AUTOCORR2D_52', 'AUTOCORR2D_53', 'AUTOCORR2D_54', 'AUTOCORR2D_55', 'AUTOCORR2D_56', 'AUTOCORR2D_57', 'AUTOCORR2D_58', 'AUTOCORR2D_59', 'AUTOCORR2D_60', 'AUTOCORR2D_61', 'AUTOCORR2D_62', 'AUTOCORR2D_63', 'AUTOCORR2D_64', 'AUTOCORR2D_65', 'AUTOCORR2D_66', 'AUTOCORR2D_67', 'AUTOCORR2D_68', 'AUTOCORR2D_69', 'AUTOCORR2D_70', 'AUTOCORR2D_71', 'AUTOCORR2D_72', 'AUTOCORR2D_73', 'AUTOCORR2D_74', 'AUTOCORR2D_75', 'AUTOCORR2D_76', 'AUTOCORR2D_77', 'AUTOCORR2D_78', 'AUTOCORR2D_79', 'AUTOCORR2D_80', 'AUTOCORR2D_81', 'AUTOCORR2D_82', 'AUTOCORR2D_83', 'AUTOCORR2D_84', 'AUTOCORR2D_85', 'AUTOCORR2D_86', 'AUTOCORR2D_87', 'AUTOCORR2D_88', 'AUTOCORR2D_89', 'AUTOCORR2D_90', 'AUTOCORR2D_91', 'AUTOCORR2D_92', 'AUTOCORR2D_93', 'AUTOCORR2D_94', 'AUTOCORR2D_95', 'AUTOCORR2D_96', 'AUTOCORR2D_97', 'AUTOCORR2D_98', 'AUTOCORR2D_99', 'AUTOCORR2D_100', 'AUTOCORR2D_101', 'AUTOCORR2D_102', 'AUTOCORR2D_103', 'AUTOCORR2D_104', 'AUTOCORR2D_105', 'AUTOCORR2D_106', 'AUTOCORR2D_107', 'AUTOCORR2D_108', 'AUTOCORR2D_109', 'AUTOCORR2D_110', 'AUTOCORR2D_111', 'AUTOCORR2D_112', 'AUTOCORR2D_113', 'AUTOCORR2D_114', 'AUTOCORR2D_115', 'AUTOCORR2D_116', 'AUTOCORR2D_117', 'AUTOCORR2D_118', 'AUTOCORR2D_119', 'AUTOCORR2D_120', 'AUTOCORR2D_121', 'AUTOCORR2D_122', 'AUTOCORR2D_123', 'AUTOCORR2D_124', 'AUTOCORR2D_125', 'AUTOCORR2D_126', 'AUTOCORR2D_127', 'AUTOCORR2D_128', 'AUTOCORR2D_129', 'AUTOCORR2D_130', 'AUTOCORR2D_131', 'AUTOCORR2D_132', 'AUTOCORR2D_133', 'AUTOCORR2D_134', 'AUTOCORR2D_135', 'AUTOCORR2D_136', 'AUTOCORR2D_137', 'AUTOCORR2D_138', 'AUTOCORR2D_139', 'AUTOCORR2D_140', 'AUTOCORR2D_141', 'AUTOCORR2D_142', 'AUTOCORR2D_143', 'AUTOCORR2D_144', 'AUTOCORR2D_145', 'AUTOCORR2D_146', 'AUTOCORR2D_147', 'AUTOCORR2D_148', 'AUTOCORR2D_149', 'AUTOCORR2D_150', 'AUTOCORR2D_151', 'AUTOCORR2D_152', 'AUTOCORR2D_153', 'AUTOCORR2D_154', 'AUTOCORR2D_155', 'AUTOCORR2D_156', 'AUTOCORR2D_157', 'AUTOCORR2D_158', 'AUTOCORR2D_159', 'AUTOCORR2D_160', 'AUTOCORR2D_161', 'AUTOCORR2D_162', 'AUTOCORR2D_163', 'AUTOCORR2D_164', 'AUTOCORR2D_165', 'AUTOCORR2D_166', 'AUTOCORR2D_167', 'AUTOCORR2D_168', 'AUTOCORR2D_169', 'AUTOCORR2D_170', 'AUTOCORR2D_171', 'AUTOCORR2D_172', 'AUTOCORR2D_173', 'AUTOCORR2D_174', 'AUTOCORR2D_175', 'AUTOCORR2D_176', 'AUTOCORR2D_177', 'AUTOCORR2D_178', 'AUTOCORR2D_179', 'AUTOCORR2D_180', 'AUTOCORR2D_181', 'AUTOCORR2D_182', 'AUTOCORR2D_183', 'AUTOCORR2D_184', 'AUTOCORR2D_185', 'AUTOCORR2D_186', 'AUTOCORR2D_187', 'AUTOCORR2D_188', 'AUTOCORR2D_189', 'AUTOCORR2D_190', 'AUTOCORR2D_191', 'AUTOCORR2D_192', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'Chi0', 'Chi0v', 'Chi0n', 'Chi1', 'Chi1v', 'Chi1n', 'Chi2v', 'Chi2n', 'Chi3v', 'Chi3n', 'Chi4v', 'Chi4n', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'EState_VSA11', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'Kappa1', 'Kappa2', 'Kappa3', 'MolLogP', 'MolMR', 'MolWt', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'NPR1', 'NPR2', 'PMI1', 'PMI2', 'PMI3', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SMR_VSA10', 'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10', 'BalabanJ', 'BertzCT', 'ExactMolWt', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomMolWt', 'HeavyAtomCount', 'LabuteASA', 'TPSA', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge', 'qed', 'RingCount', 'Asphericity', 'Eccentricity', 'InertialShapeFactor', 'RadiusOfGyration', 'SpherocityIndex', 'NHOHCount', 'NOCount'
        ]
        receptor = [
            'countA', 'countR', 'countN', 'countD', 'countC', 'countQ', 'countE', 'countG', 'countH', 'countI', 'countL', 'countK', 'countM', 'countF', 'countP', 'countS', 'countT', 'countW', 'countY', 'countV', 'TotalAALength', 'AvgAALength', 'countChain', 'SASA', 'DipoleMoment', 'IsoelectricPoint', 'GRAVY', 'Aromaticity', 'InstabilityIndex'
        ]

        # Extract the data
        sf_train_data = X_train[sf]
        ligand_train_data = X_train[ligand]
        receptor_train_data = X_train[receptor]

        sf_test_data = X_test[sf]
        ligand_test_data = X_test[ligand]
        receptor_test_data = X_test[receptor]

        if X_val is not None:
            sf_val_data = X_val[sf]
            ligand_val_data = X_val[ligand]
            receptor_val_data = X_val[receptor]
        else:
            scoring_functions_val_data = None
            ligand_val_data = None
            receptor_val_data = None
        
        new_X_train = [sf_train_data, ligand_train_data, receptor_train_data]
        new_X_test = [sf_test_data, ligand_test_data, receptor_test_data]
        new_X_val = [sf_val_data, ligand_val_data, receptor_val_data]

        # List to store the best topology for each set
        best_ao_params = []
        
        if run_autoencoder_optimization:
            
            for name, AO_X_train, AO_X_test, AO_X_val in [
                ("SF", sf_train_data, sf_test_data, sf_val_data), 
                ("LIG", ligand_train_data, ligand_test_data, ligand_val_data), 
                ("REC", receptor_train_data, receptor_test_data, receptor_val_data)
            ]:
                # Compute the singular values for AO_X_train
                singular_values = np.linalg.svd(AO_X_train, compute_uv = False)

                # Compute the explained variance ratio
                explained_variance_ratio = singular_values**2 / np.sum(singular_values**2)

                # Compute the cumulative explained variance ratio
                cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

                # Compute the number of components that explain 95% of the variance
                n_components = np.argmax(cumulative_explained_variance_ratio >= explained_variance) + 1

                # Get the number of dimensions for the encoding layer and round up to the nearest power of 2 + 1
                encoding_dims = ( # Size should be the same size or smaller than the number of features to explain the desired variance
                    max(2 ** math.ceil(math.log2(n_components / 2) - 1), 4), # Minimum value
                    n_components
                    #max(2 ** math.ceil(math.log2(AO_X_train.shape[1] / 2) - 1), 4), # Minimum value
                    #max(2 ** math.ceil(math.log2(AO_X_train.shape[1] / 2)) + 1, 8)  # Maximum value
                )

                # Skip SF (for now) TODO: Check if this is necessary
                if name == "SF":
                    continue

                # Create a pool of worker processes
                with Pool(num_processes) as pool:
                    # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
                    pool.starmap(AOworker, [(
                        pid,
                        storage_id, 
                        AO_X_train,
                        AO_X_test,
                        AO_X_val,
                        encoding_dims,
                        storage,
                        models_folder,
                        42,                       # random_seed
                        True,                     # use_gpu
                        False,                    # verbose
                        "minimize",               # direction
                        2500,                     # n_trials 
                        True,                     # load_if_exists
                        1,                        # n_jobs
                        f"AO_Optimization_{name}" # study_name
                        ) for pid in range(num_processes)
                    ])

        for name in ["SF", "LIG", "REC"]:
            if name == "SF":
                best_ao_params.append({'n_layers_encoder': 1, "activation_function_0_encoder": "Identity", "n_units_layer_0_encoder": sf_train_data.shape[1]})
                continue

            # Load the study
            ao_multi_study = optuna.load_study(study_name = f"AO_Optimization_{name}_{storage_id}_TPE", storage = storage)
            ao_multi_df = ao_multi_study.trials_dataframe()
            ao_multi_df['combined_metric'] = abs(ao_multi_df['value'] - ao_multi_df['user_attrs_val_rmse'])

            best_ao_multi_df = ao_multi_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])

            # Recreate the autoencoder object for the best trial based on the best_ao_multi_df
            best_ao_multi_trial = best_ao_multi_df.iloc[0]

            # Select the trial by the best_ao_multi_trial number
            best_ao_multi_trial = ao_multi_study.trials[best_ao_multi_trial.number]

            # Pick the params from the best_ao_multi_trial
            best_ao_params.append(best_ao_multi_trial.params)

    else:
        if run_autoencoder_optimization:
            # Create a pool of worker processes
            with Pool(num_processes) as pool:
                # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
                pool.starmap(AOworker, [(
                    pid,
                    storage_id, 
                    X_train, 
                    X_test, 
                    X_val, 
                    (16, 256),        # encoder dims
                    storage,
                    models_folder,
                    42,               # random_seed
                    True,             # use_gpu
                    False,            # verbose
                    "minimize",       # direction
                    2500,             # n_trials
                    True,             # load_if_exists
                    1,                # n_jobs
                    "AO_Optimization" # study_name
                    ) for pid in range(num_processes)
                ])

        # Load the study
        ao_study = optuna.load_study(study_name = f"AO_Optimization_{storage_id}_TPE", storage = storage)
        ao_df = ao_study.trials_dataframe()

        # Filter the trials to only include the ones that are complete
        ao_df = ao_df[ao_df['state'] == 'COMPLETE']
        
        #ao_df['combined_metric'] = abs(ao_df['value'] - ao_df['user_attrs_val_rmse'])

        #best_ao_df = ao_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])
        best_ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])

        # Recreate the autoencoder object for the best trial based on the best_ao_df
        best_ao_trial = best_ao_df.iloc[0]

        # Select the trial by the best_ao_trial number
        best_ao_trial = ao_study.trials[best_ao_trial.number]

        # Pick the params from the best_ao_trial
        best_ao_params = best_ao_trial.params
        
        new_X_train = X_train
        new_X_test = X_test
        new_X_val = X_val
else:
    new_X_train = X_train
    new_X_test = X_test
    new_X_val = X_val
    best_ao_params = None

if run_NN_optimization:
    with Pool(num_processes) as pool:
        # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
        pool.starmap(NNworker, [(
            pid,
            storage_id, 
            new_X_train, y_train, 
            new_X_test, y_test, 
            new_X_val, y_val, 
            storage,
            best_ao_params,   # encoder
            1,                # output_size
            42,               # random_seed
            True,             # use_gpu
            False,            # verbose
            "minimize",       # direction
            125,               # n_trials
            True,             # load_if_exists
            1,                # n_jobs
            study_name        # study_name
            ) for pid in range(num_processes)
        ])

# Load the study
nn_study = optuna.load_study(study_name = f"{study_name}_{storage_id}_TPE", storage = storage)
nn_df = nn_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
nn_df = nn_df[nn_df['state'] == 'COMPLETE']

nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']

best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])

# Define the number of models to select
n_models = 5

# Get the best n models in the best_nn_df
best_nn_df.head(n_models)

# Build the models
models = []
predictions = []

for i in range(n_models):
    # Get the best trial
    best_trial = nn_study.trials[best_nn_df.iloc[i].number]

    # Pick the params from the best_trial
    best_params = best_trial.params

    # If the new_X_val is a list
    if isinstance(new_X_val, list):
        # Initialize the trainer
        NN_model = NeuralNet(
            [new_X_train[j].shape[0] for j in range(len(new_X_train))],
            1,          
            best_ao_params, 
            best_params,
            random_seed = 42,
            use_gpu = True, 
            verbose = False
        )
    else:
        # Initialize the trainer
        NN_model = NeuralNet(
            new_X_train.shape[1],
            1,          
            best_ao_params, 
            best_params,
            random_seed = 42,
            use_gpu = True, 
            verbose = False
        )

    # Reset the random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)

    # Train the model
    model = NN_model.train_model(
        new_X_train, 
        y_train, 
        new_X_test, 
        y_test, 
        new_X_val, 
        y_val
    )

    # Reset the random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)
    
    # Append the model to the list
    models.append(NN_model)

    # Make predictions
    if isinstance(new_X_val, list):
        predictions.append(
            NN_model.NN(
                [torch.tensor(
                    np.asarray(new_X_val[j]), 
                    dtype=torch.float32
                ).to(torch.device('cuda')) for j in range(len(new_X_val))]
            ).cpu().detach().numpy()
        )
    else:
        predictions.append(
            NN_model.NN(
                torch.tensor(
                    np.asarray(new_X_val), 
                    dtype=torch.float32
                ).to(torch.device('cuda'))
            ).cpu().detach().numpy()
        )

    # Save the model
    torch.save(model, f'/data/hd4tb/OCDocker/data/ocdb/models/NN_model_{i}.pt')

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(np.asarray(predictions).reshape(len(predictions), predictions[0].shape[0]).T)

def select_values(row):
    sorted_row = sorted(row)
    return pd.Series({
        'max': max(row),
        '2nd_highest': sorted_row[-2],
        'median': sorted_row[2],
        '4th_highest': sorted_row[1],
        'min': min(row)
    })

selected_values_df = predictions_df.apply(select_values, axis=1)
full_selected_values_df = pd.concat([predictions_df, selected_values_df], axis=1)
                                    
# Calculate the mean, median, std, min, max, and range for the predictions
full_selected_values_df['std'] = predictions_df.std(axis = 1)
full_selected_values_df['range'] = full_selected_values_df['max'] - full_selected_values_df['min']

# For each column in the full_selected_values_df, calculate the AUC
auc_dict = {}
for col in full_selected_values_df.columns:
    fpr, tpr, _ = roc_curve(y_val, full_selected_values_df[col]) # type: ignore
    auc_dict[col] = auc(fpr, tpr)

# make AUC_dict a DataFrame
auc_df = pd.DataFrame(auc_dict, index = ['AUC']).T

# Save the full_selected_values_df values to csv
full_selected_values_df.to_csv('full_selected_values_df_NN.csv')

'''
# Get the RMSE for each scoring function
rmse_dict = {}

# Get the AUC for each scoring function
auc_dict = {}

for col in score_columns:
    error = pdbbind_standard_norm_df[col] - pdbbind_standard_norm_df['experimental']
    rmse_dict[col] = np.sqrt(np.mean(error**2))

    # Map the Ligand and Decoy values to 1 and 0 respectively
    dudez_standard_norm_df['type_cat'] = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})

    fpr, tpr, _ = roc_curve(dudez_standard_norm_df['type_cat'], dudez_standard_norm_df[col])
    auc_dict[col] = auc(fpr, tpr)
'''
