
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
from xgboost import XGBRegressor

from PreXGBoostOptimizer import PreXGBoostOptimizer
from EvolutionaryFeatureSelector import EvolutionaryFeatureSelector
from EvolutionaryFeatureSelectorCustom import EvolutionaryFeatureSelectorCustom


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
    X_val = dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'], errors = 'ignore')
    y_val = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})
else:
    # Set the test size to 0.0 to use the entire dataset for training
    X_train = pdbbind_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db', 'experimental'], errors = 'ignore')
    y_train = pdbbind_standard_norm_df['experimental']

    X_test = dudez_standard_norm_df.drop(columns = ['receptor', 'ligand', 'name', 'type', 'db'], errors = 'ignore')
    y_test = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})

    # Set X and y for validation to None
    X_val = None
    y_val = None

'''
# Plot correlation matrix to show that the correlation is preserved after normalization, even though the values are different
plot_corr_matrix(dudez_standard_norm_df, columns = [col for col in df.columns if col.startswith(('VINA', 'SMINA', 'PLANTS', 'ODDT'))], scaler = 'standard')
plot_corr_matrix(dudez_minmax_norm_df, columns = [col for col in df.columns if col.startswith(('VINA', 'SMINA', 'PLANTS', 'ODDT'))], scaler = 'minmax')
plot_corr_matrix(pdbbind_standard_norm_df, columns = [col for col in df.columns if col.startswith(('VINA', 'SMINA', 'PLANTS', 'ODDT'))] + ['experimental'], scaler = 'standard')
plot_corr_matrix(pdbbind_minmax_norm_df, columns = [col for col in df.columns if col.startswith(('VINA', 'SMINA', 'PLANTS', 'ODDT'))] + ['experimental'], scaler = 'minmax')

# Since it is the same, no matter the scaler, we can use any of the normalized dataframes to calculate the correlation similarity
plot_correlation_similarity(dudez_standard_norm_df, dudez_minmax_norm_df, columns = [col for col in df.columns if col.startswith(('VINA', 'SMINA', 'PLANTS', 'ODDT'))])

###################
## DUDEz analysis
###################

# Prepare ROC data
labels = dudez_data['type'].map({'ligand': 1, 'decoy': 0})

# Plot ROC curves
plot_roc_curves(dudez_data, score_columns, labels, title = "ROC_DUDEz_clean")

# Calculate additional metrics
dudez_data_metrics, additional_metrics_cols = calculate_metrics(dudez_data, score_columns)

# Plot the ROC curve again using the additional metrics
plot_roc_curves(dudez_data_metrics, score_columns.tolist() + additional_metrics_cols, labels, title = "ROC_DUDEz_metrics")
'''

#########################
## Feature engineering
#########################

# Skipping outlier detection for now...

# Compute the z-score for the score columns
#dudez_zscore_df = compute_zscore(dudez_data[["receptor", "ligand", "type"] + score_columns], score_columns)

# Identify the outliers
#outliers = dudez_zscore_df[(dudez_zscore_df[["z_" + s for s in score_columns]] > 3).any(axis = 1)]

###############################################
## Exploratory analysis for feature selection
###############################################

# Create the feature_engineering directory if it does not exist
if not os.path.exists('feature_engineering'):
    os.makedirs('feature_engineering')

# Create the most_important and least_important directories if they do not exist
if not os.path.exists('feature_engineering/most_important'):
    os.makedirs('feature_engineering/most_important')

if not os.path.exists('feature_engineering/least_important'):
    os.makedirs('feature_engineering/least_important')

###############################################################################################################################################
## These steps are to determine that the importance of the features is not related to the performance of the model where ROC_AUC is maximized
###############################################################################################################################################

'''
# Check if the result.pkl file exists
if os.path.isfile('feature_engineering/result.pkl'):
    # Load the results from the pickle
    results = pd.read_pickle('feature_engineering/result.pkl')
else:
    # Train and evaluate multiple XGBoost models with different feature sets removing descriptors one by one
    results = evaluate_feature_removal(X_train, y_train, X_test, labels, random_state=42, feature_importances = [], use_gpu = False)

# Plot the results and save the pickle
plot_performance(results, filename='feature_engineering/feature_removal_performance.png')
# Save the results to a pickle file except for the model column
save_object(results.drop(columns = 'model'), 'feature_engineering/result.pkl')

# Evaluate feature removal and plot performance for the most and least important features
evaluate_and_plot(X_train, y_train, X_test, labels, results, iterations = 10, sort_order=(False), save_path='./feature_engineering/most_important') # Most important features
evaluate_and_plot(X_train, y_train, X_test, labels, results, iterations = 10, sort_order=(True), save_path='./feature_engineering/least_important') # Least important features
'''

print("Running XGBoost pre-optimization...")
# Create the PreXGBoostOptimizer object
pxgb = PreXGBoostOptimizer(X_train, y_train, X_test, y_test, X_val, y_val, params = {}, use_gpu = True, early_stopping_rounds = 50, random_state = 42, verbose = False)

n_jobs = 2

# If the X_val is None, the direction is set to maximize
if X_val is None:
    # Run the optimization
    study_pre, best_params_pre, best_score_pre = pxgb.optimize(study_name = "XGBoost pre-optimization", direction = "maximize", n_trials = 1000, n_jobs = n_jobs)
else:
    # Run the optimization
    study_pre, best_params_pre, best_score_pre = pxgb.optimize(study_name = "XGBoost pre-optimization", direction = "minimize", n_trials = 1000, n_jobs = n_jobs)

#best_params_pre = {}

def optimize_feature_selection(X_train, y_train, X_test, y_test, X_validation = None, y_validation = None, best_params = {}, algorithm = "ga", n_trials = 100, study_name: str = "Feature selection", random_state = 42, use_gpu = True, verbose = False, instance_id: int = -1):
    """
    Function to be executed by each process.
    
    :param instance_id: An identifier for the instance, could be used to modify the behavior per instance.
    """

    # Only print the instance_id if it is greater than -1
    if instance_id > -1:
        # Setup unique to this instance, potentially using instance_id to differentiate setups
        print(f"Running instance {instance_id}")
    
    if algorithm.lower() == "custom-ga":
        # Create the EvolutionaryFeatureSelectorCustom object
        evo = EvolutionaryFeatureSelectorCustom(X_train, y_train, X_test, y_test, X_validation = X_validation, y_validation = y_validation, xgboost_params = best_params, use_gpu = use_gpu, random_state = random_state, verbose = verbose) # type: ignore
    elif algorithm.lower() in ["cmaes", "ga"]:
        # Create the EvolutionaryFeatureSelector object
        evo = EvolutionaryFeatureSelector(X_train, y_train, X_test, y_test, X_validation = X_validation, y_validation = y_validation, xgboost_params = best_params, algorithm = algorithm, use_gpu = use_gpu, random_state = random_state, verbose = verbose) # type: ignore
    
    # Run the optimization
    study, best_features, best_score = evo.optimize(study_name = study_name, direction = "minimize", n_trials = n_trials, n_jobs = n_jobs)

    return study, best_features, best_score

print("Running feature selection...")

algorithm = "custom-ga"
random_state = 42
n_trials = 100
n_jobs = 30
use_gpu = True
verbose = False

study_fs, best_features_fs, best_score_fs = optimize_feature_selection(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    X_val, 
    y_val, 
    best_params_pre, 
    algorithm = algorithm, 
    n_trials = n_trials, 
    study_name = "Feature selection Custom GA",
    random_state = random_state, 
    use_gpu = use_gpu, 
    verbose = verbose, 
    instance_id = 0
)
