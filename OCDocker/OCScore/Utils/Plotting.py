#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage plotting operations in OCDocker in the context of
scoring functions.

They are imported as:

import OCDocker.OCScore.Utils.Plotting as ocscoreplot
'''

# Imports
###############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import auc, roc_curve
from typing import Union, Optional

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

# Methods
###############################################################################

def plot_correlation_similarity(df1 : pd.DataFrame, df2 : pd.DataFrame, columns : list = [], annot : bool = True, fontsize : Optional[float] = None, normalize : bool = True) -> None:
    ''' Plots the similarity of correlation matrices from two DataFrames.

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
    '''

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
            text.set_fontsize(fontsize)

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
            text.set_fontsize(fontsize)

    plt.tight_layout()
    plt.savefig('correlation_similarity_sorted.png')
    plt.close()


def plot_roc_curves(df : pd.DataFrame, feature_cols : list, labels : pd.Series, title : str = "ROC") -> None:
    ''' Plots ROC curves for a DataFrame.

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
    '''

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
