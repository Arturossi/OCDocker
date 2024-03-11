
import os
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import ParameterGrid, train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

def plot_correlation_similarity(df1: pd.DataFrame, df2: pd.DataFrame, columns: list = [], annot: bool = True, fontsize: float = None, normalize: bool = True) -> None:
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
        columns = df1.columns.intersection(df2.columns)

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
    sorted_features = sorted(auc_dict, key=auc_dict.get, reverse=True)

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

def calculate_metrics(df: pd.DataFrame, selected_columns: list) -> pd.DataFrame and list:
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

def plot_top_n_feature_importances(xgb_regressor, n_features=10):
    """
    Plot the top N feature importances from an XGBRegressor model along with the percentage
    of total importance that these features represent and the cumulative importance.

    Parameters:
    xgb_regressor: The trained XGBRegressor model.
    n_features: Number of top features to plot (default is 10).
    """

    # Get feature importance dictionary from the xgb_regressor
    f_importance = xgb_regressor.get_booster().get_score(importance_type='weight')
    
    # Calculate the total importance for normalization
    total_importance = sum(f_importance.values())
    
    # Sort the feature importance in descending order and select top n_features
    sorted_f_importance = sorted(f_importance.items(), key=lambda item: item[1], reverse=True)[:n_features]
    
    # Calculate the percentage of total importance and cumulative importance
    cumulative_importance = 0
    for i, (feature, importance) in enumerate(sorted_f_importance):
        percentage = (importance / total_importance) * 100
        cumulative_importance += importance
        sorted_f_importance[i] = (feature, importance, percentage)
    
    # Create a DataFrame
    df_importance = pd.DataFrame(sorted_f_importance, columns=['Feature', 'F Score', 'Percentage']).sort_values(by='F Score', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, n_features // 2 + 2))  # Adjust the size as necessary
    plt.barh(df_importance['Feature'], df_importance['F Score'], color='skyblue')
    plt.xlabel('F Score')
    
    # Annotate with the percentage of total importance
    for index, (importance, percentage) in enumerate(zip(df_importance['F Score'], df_importance['Percentage'])):
        plt.text(importance, index, f' {percentage:.2f}%', va='center')
    
    # Add total and cumulative importance to the title
    plt.title(f'Top {n_features} Feature Importances\nTotal Importance: {total_importance:.2f}, '
              f'Cumulative Importance of Top {n_features}: {cumulative_importance:.2f} ({cumulative_importance / total_importance * 100:.2f}%)')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'top_{n_features}_feature_importance.png', dpi=300)
    plt.close()

def plot_features_by_importance_threshold(xgb_regressor, importance_threshold=0.95):
    """
    Plot the smallest set of features that together account for a specified percentage of
    the total importance, ensuring the plot is readable and well-formatted.

    Parameters:
    xgb_regressor: The trained XGBRegressor model.
    importance_threshold: The percentage of the total importance that the feature set should account for.
    """

    # Get feature importance dictionary from the xgb_regressor
    f_importance = xgb_regressor.get_booster().get_score(importance_type='weight')
    
    # Sort the feature importances in descending order
    sorted_f_importance = sorted(f_importance.items(), key=lambda item: item[1], reverse=True)
    
    # Normalize the feature importances to sum to 1
    total_importance = sum(f_importance.values())
    sorted_f_importance = [(feature, importance / total_importance) for feature, importance in sorted_f_importance]
    
    # Find the smallest set of features that together account for the importance_threshold
    cumulative_importance = 0.0
    selected_features = []
    for feature, importance in sorted_f_importance:
        cumulative_importance += importance
        selected_features.append((feature, importance))
        if cumulative_importance >= importance_threshold:
            break
    
    # Create a DataFrame
    df_importance = pd.DataFrame(selected_features, columns=['Feature', 'Normalized Importance']).sort_values(by='Normalized Importance', ascending=True)
    
    # Determine figure height dynamically: each feature should have 0.3 units of height
    fig_height = len(selected_features) * 0.3
    fig_height = max(min(fig_height, 10), 3)  # Set a reasonable range for figure height
    
    # Plot
    plt.figure(figsize=(10, fig_height))
    bars = plt.barh(df_importance['Feature'], df_importance['Normalized Importance'], color='skyblue')
    
    # Annotate with the normalized importance percentage
    for bar in bars:
        plt.text(bar.get_width() + bar.get_width()*0.05, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.2%}', va='center')
    
    # Add the total and cumulative importance to the title
    plt.title(f'Set of Features Accounting for {importance_threshold:.0%} of Total Importance')
    plt.xlabel('Normalized Importance')
    
    # Ensure the annotations fit within the plot
    plt.xlim(0, max(df_importance['Normalized Importance']) + 0.15)
    
    plt.tight_layout()
    plt.savefig(f'features_accounting_for_{importance_threshold:.0%}_importance.png', dpi=300)
    plt.close()

def get_top_features_excluding_docking_scores(df, n, score_columns, target='experimental', test_size=None, random_state=42):
    """
    Get the top n features from an XGBoost model, excluding features that are related to docking scores.

    Parameters:
    df (pandas.DataFrame): The dataframe with the input data.
    n (int): Number of top features to select.
    docking_score_prefixes (list of str): List of prefixes used in docking score feature names.
    target (str): The name of the target variable column.
    test_size (float): The proportion of the dataset to include in the test split. If None, the entire dataset is used.
    random_state (int): The seed used by the random number generator. If None, a random seed is used.

    Returns:
    list: Top n feature names excluding docking scores.
    """

    # Get feature columns excluding docking scores and other non-feature columns
    features_to_exclude = ['name', 'receptor', 'ligand', 'type', 'db', target] + score_columns
    feature_cols = df.drop(columns=features_to_exclude)
    
    # Split the data into features and target
    X = feature_cols
    y = df[target]

    # Split the data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the XGBoost regressor
    xgb_regressor = XGBRegressor(objective='reg:squarederror')

    # Fit the model on the training data
    xgb_regressor.fit(X_train, y_train)

    # Get feature importance
    importance = xgb_regressor.feature_importances_

    # Create a Series with feature importances
    feature_importance = pd.Series(importance, index=X.columns)

    # Sort features by importance
    top_features = feature_importance.nlargest(n).index.tolist()

    return top_features

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

def run_xgboost(X_train, y_train, X_test, labels, params, use_gpu, random_state = 42) -> tuple:
    # Extend the parameters dict with the device parameter
    if use_gpu:
        params['device'] = 'cuda'

    model = XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        tree_method='hist',
        eval_metric = 'auc',
        random_state=random_state,
        **params
    )

    # Fit the model on the training data        
    model.fit(X_train, y_train, eval_set=[(X_test, labels)], verbose=False)

    # Get the result
    evals_result  = model.evals_result()

    # Get the last AUC value from the validation set
    roc_auc = evals_result['validation_0']['auc'][-1]

    return model, roc_auc

def evaluate_feature_removal(X_train, y_train, X_test, labels, random_state=42, feature_importances = [], use_gpu = False, params = {}):
    """
    Evaluate the performance of an XGBoost model as features are removed one by one, 
    keeping docking score features in the training set.

    Parameters:
    pdbbind_df (pandas.DataFrame): The dataframe with the PDBbind input data.
    dudez_df (pandas.DataFrame): The dataframe with the DUDEz input data.
    target (str): The name of the target variable column.
    score_columns (list of str): List of prefixes used in docking score feature names.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    feature_importances (list): List of feature importances. If empty, the function will calculate the feature importances.

    Returns:
    pandas.DataFrame: A dataframe containing the MSE and R2 for each number of features used.
    """

    # All features for training, including docking scores
    all_features = X_train.columns.to_list()

    # If feature_importance is not provided, load the removable features
    if not feature_importances:
        # Features to consider for removal, excluding docking scores
        removable_features = [col for col in all_features if col not in score_columns]
    else:
        # The removable features will be the feature_importance
        removable_features = feature_importances

    results = []

    for i in tqdm(range(len(removable_features), 0, -1), desc='Feature removal'):
        # Select the current features for training
        current_features = removable_features[:i] + score_columns
        X_train_aux = X_train[current_features]
        X_test_aux = X_test[current_features]
        
        # Train and evaluate the model
        model, roc_auc = run_xgboost(X_train_aux, y_train, X_test_aux, labels, params, use_gpu, random_state = random_state)
        
        # Update removable_features to exclude the least important non-docking score feature
        feature_importances = pd.Series(model.feature_importances_, index=current_features)

        removable_feature_importances = feature_importances[removable_features]
        least_important_feature = removable_feature_importances.idxmin()
        removable_features = [f for f in removable_features if f != least_important_feature]

        # Store the results
        results.append({
            'num_features': i,
            'roc_auc': roc_auc,
            'removed_feature': least_important_feature,
            'model': model
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def plot_performance(results, filename = 'feature_removal_performance.png'):
    """
    Plot the performance of a model as features are removed, showing both MSE and R2 metrics.
    Annotate the plot with the minimum MSE and maximum R2.

    Parameters:
    results (pandas.DataFrame): DataFrame containing 'num_features', 'mse', and 'r2' columns.
    filename (str): The name of the file to save the plot. Default is 'feature_removal_performance.png'.
    """

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot MSE, R2, and roc_auc on the same y-axis
    #plt.plot(results['num_features'], results['mse'], label='MSE', color='tab:red')
    #plt.plot(results['num_features'], results['r2'], label='R2', color='tab:blue')
    plt.plot(results['num_features'], results['roc_auc'], label='roc_auc', color='tab:green')

    # Invert x-axis so fewer features are on the right
    plt.gca().invert_xaxis()

    # Set y-axis limits
    plt.ylim(0, 1)

    '''
    # Find the position of smallest MSE and annotate from below
    min_mse_position = results['mse'].idxmin()
    min_mse_value = results['mse'].min()
    min_mse_num_features = results['num_features'][min_mse_position]
    plt.annotate(f'Min MSE: {min_mse_value:.2f}\nFeatures: {min_mse_num_features}',
                 xy=(min_mse_num_features, min_mse_value),
                 xytext=(min_mse_num_features, min_mse_value - 0.1),  # Adjust text position for arrow from below
                 arrowprops=dict(facecolor='red', arrowstyle="->", connectionstyle="arc3"),
                 horizontalalignment='center', verticalalignment='top')

    # Find the position of highest R2 and annotate from above
    max_r2_position = results['r2'].idxmax()
    max_r2_value = results['r2'].max()
    max_r2_num_features = results['num_features'][max_r2_position]
    plt.annotate(f'Max R2: {max_r2_value:.2f}\nFeatures: {max_r2_num_features}',
                 xy=(max_r2_num_features, max_r2_value),
                 xytext=(max_r2_num_features, max_r2_value + 0.1),  # Adjust text position for arrow from above
                 arrowprops=dict(facecolor='blue', arrowstyle="->", connectionstyle="arc3"),
                 horizontalalignment='center', verticalalignment='bottom')
    '''

    # Find the position of highest roc_auc and annotate from above
    max_roc_auc_position = results['roc_auc'].idxmax()
    max_roc_auc_value = results['roc_auc'].max()
    max_roc_auc_num_features = results['num_features'][max_roc_auc_position]
    plt.annotate(f'Max roc_auc: {max_roc_auc_value:.2f}\nFeatures: {max_roc_auc_num_features}',
                    xy=(max_roc_auc_num_features, max_roc_auc_value),
                    xytext=(max_roc_auc_num_features, max_roc_auc_value + 0.1),  # Adjust text position for arrow from above
                    arrowprops=dict(facecolor='green', arrowstyle="->", connectionstyle="arc3"),
                    horizontalalignment='center', verticalalignment='bottom')

    # Add labels and title
    plt.xlabel('Number of used features (excluding docking scores)')
    plt.ylabel('Performance')
    plt.title('Performance of XGBoost model with different numbers of features')
    plt.legend(loc='best')
    # Add grid
    plt.grid(True)
    # Increase the x-axis tick values for each 50 features
    plt.xticks(np.arange(0, results['num_features'].max() + 1, 50))
    plt.tight_layout()

    # Save and show plot
    plt.savefig(filename, dpi=300)
    plt.close()

def run_grid_search(X_train, y_train, X_test, labels, param_grid, top_features, score_columns, use_gpu=True, random_state=42):
    """
    Runs a grid search over specified parameters for an XGBRegressor, evaluates using AUC, 
    and saves the best model. Optionally uses GPU by setting the device parameter.
    
    Parameters:
    - X_train, y_train: Training data
    - X_test: Test data for predictions
    - labels: Actual labels for evaluating predictions
    - param_grid: Hyperparameters to iterate over in the grid search
    - top_features: List of feature names to be used from X_test
    - score_columns: List of score column names to be included from X_test
    - use_gpu: Flag to enable or disable GPU usage
    """
    
    best_auc = 0
    best_params = None
    best_model = None

    print("Running grid search...")
    
    for params in tqdm(ParameterGrid(param_grid), desc='Grid search'):
        # Train and evaluate the model
        model, roc_auc = run_xgboost(X_train[top_features + score_columns], y_train, X_test[top_features + score_columns], labels, params, use_gpu, random_state = random_state)
        
        # Update the best model if the current model is better
        if roc_auc > best_auc:
            print("New best AUC:", roc_auc)
            best_auc = roc_auc
            best_params = params
            best_model = model
    
    # Save the best model
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("Best AUC: {:.4f}".format(best_auc))
    print("Best Parameters:", best_params)

def evaluate_and_plot(X_train, y_train, X_test, labels, results, iterations = 10, sort_order=(False), save_path='.', random_state=42, use_gpu=False, params={}):
    """
    Evaluates feature removal and plots performance, supporting both sorting approaches.

    Parameters:
    - pdbbind_df: DataFrame for PDBBind dataset (normalized).
    - dudez_df: DataFrame for DUD-EZ dataset (normalized).
    - score_columns: Columns to use for scoring.
    - results: Initial results to start with.
    - iterations: Number of iterations to perform.
    - sort_order: Tuple indicating the sorting order for 'roc_auc'.
    - save_path: Path to save the results and plots.
    - random_state: Random state for reproducibility.
    - use_gpu: Flag to enable or disable GPU usage.
    - params: Additional parameters for the XGBoost model.
    """

    lresults = results['removed_feature'].to_list()

    for i in tqdm(range(iterations), total=iterations, desc='Evaluating feature removal'):
        # If there is already a result for the current feature set, load it and skip
        if os.path.exists(f'{save_path}/result_{i + 1}.pkl'):
            # Load the result
            results = load_object(f'{save_path}/result_{i + 1}.pkl')
        else:
            # Evaluate feature removal
            results = evaluate_feature_removal(X_train, y_train, X_test, labels, random_state = random_state, feature_importances = lresults, use_gpu = use_gpu, params = params)
            save_object(results.drop(columns = 'model'), f'{save_path}/result_{i + 1}.pkl')

        plot_performance(results, filename=f'{save_path}/feature_removal_performance_{i + 1}.png')
        
        # Dynamic sorting based on the provided sort_order
        #results = results.sort_values(by=['roc_auc', 'mse', 'r2'], ascending=sort_order).reset_index(drop=True)
        results = results.sort_values(by=['roc_auc'], ascending=sort_order).reset_index(drop=True)
        lresults = results['removed_feature'].to_list()

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

use_pdb_train = True

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
from NNOptimizer import NNOptimizer
from optuna.samplers import TPESampler

def AOworker(pid,
              id,
              X_train, 
              X_test, 
              X_val,
              models_folder,
              storage,
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

num_processes = 4
storage_id = 9
storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
models_folder = f"/data/hd4tb/OCDocker/data/ocdb/models/autoencoder_{storage_id}"
run_autoencoder_optimization = False

# If models folder does not exist, create it
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

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
            storage,
            models_folder,
            42,               # random_seed
            True,             # use_gpu
            False,            # verbose
            "minimize",       # direction
            250,              # n_trials
            True,             # load_if_exists
            4,                # n_jobs
            "AO_Optimization" # study_name
            ) for pid in range(num_processes)
        ])

# Load the study
ao_study = optuna.load_study(study_name = f"AO_Optimization_{storage_id}_TPE", storage = storage)
ao_df = ao_study.trials_dataframe()
ao_df['combined_metric'] = abs(ao_df['value'] - ao_df['user_attrs_val_rmse'])

best_ao_df = ao_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])

# Recreate the autoencoder object for the best trial based on the best_ao_df
best_ao_trial = best_ao_df.iloc[0]

# Select the trial by the best_ao_trial number
best_ao_trial = ao_study.trials[best_ao_trial.number]

# Pick the params from the best_ao_trial
best_ao_params = best_ao_trial.params

with Pool(num_processes) as pool:
    # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
    pool.starmap(NNworker, [(
        pid,
        storage_id, 
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        best_ao_params,   # encoder
        1,                # output_size
        42,               # random_seed
        True,             # use_gpu
        False,            # verbose
        "minimize",       # direction
        40,               # n_trials
        True,             # load_if_exists
        4,                # n_jobs
        "NN_Optimization" # study_name
        ) for pid in range(num_processes)
    ])

# Load the study
nn_study = optuna.load_study(study_name = f"NN_Optimization_{storage_id}_TPE", storage = storage)
nn_df = nn_study.trials_dataframe()

nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']

best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])

# Define the number of models to select
n_models = 5

# Get the best n models in the best_nn_df
best_nn_df.head(n_models)

# Build the models
models = []

for i in range(n_models):
    # Get the best trial
    best_trial = nn_study.trials[best_nn_df.iloc[i].number]

    # Pick the params from the best_trial
    best_params = best_trial.params

    # Initialize the trainer
    trainer = NNOptimizer(
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        best_params,
        output_size = 1, 
        random_seed = 42,
        use_gpu = True, 
        verbose=False
    )

    # Train the model
    model = trainer.train_model()

    # Append the model to the list
    models.append(model)