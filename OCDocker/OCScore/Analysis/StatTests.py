#!/usr/bin/env python3

# Description
###############################################################################
''' This module provides functions to perform statistical tests on scoring data.

It is imported as:

import OCDocker.OCScore.Analysis.StatTests as ocstat
'''

# Imports
###############################################################################

import math
import os
import pickle

from typing import Optional
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg
import scipy.stats as stats

import OCDocker.OCScore.Analysis.Plotting as ocstatplot

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


def run_statistical_tests(df: pd.DataFrame, n_trials: int, colour_mapping: dict[str, tuple[float, float, float]], output_dir: str) -> None:
    '''
    Perform Welch's ANOVA and Games-Howell post-hoc tests for RMSE and AUC,
    followed by visualization of statistical results.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered dataframe containing 'RMSE', 'AUC', and 'Methodology' columns.
    n_trials : int
        Number of top trials used for naming output files.
    colour_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping methodology names to RGB tuples for plotting.
    output_dir : str
        Directory to save output plots and CSV files.
    '''

    print("\nRunning Welch's ANOVA and Games-Howell post-hoc tests...")

    # Welch's ANOVA
    welch_auc = pg.welch_anova(dv = 'AUC', between = 'Methodology', data = df)
    welch_rmse = pg.welch_anova(dv = 'RMSE', between = 'Methodology', data = df)

    # Save Welch results
    welch_auc.to_csv(f"csvs/welch_anova_auc_{n_trials}.csv", index=False)
    welch_rmse.to_csv(f"csvs/welch_anova_rmse_{n_trials}.csv", index=False)

    # Games-Howell post-hoc tests
    gh_auc = pg.pairwise_gameshowell(dv = 'AUC', between = 'Methodology', data = df)
    gh_rmse = pg.pairwise_gameshowell(dv = 'RMSE', between = 'Methodology', data = df)

    # Save post-hoc results
    gh_auc.to_csv(f"csvs/games_howell_posthoc_AUC_{n_trials}.csv", index = False)
    gh_rmse.to_csv(f"csvs/games_howell_posthoc_RMSE_{n_trials}.csv", index = False)

    # Plot barplots with significant differences
    ocstatplot.plot_bar_with_significance(gh_auc, metric = "AUC", y_col = "diff", colour_mapping = colour_mapping, output_dir = output_dir)
    ocstatplot.plot_bar_with_significance(gh_rmse, metric = "RMSE", y_col = "diff", colour_mapping = colour_mapping, output_dir = output_dir)

    # Plot heatmaps of p-values
    ocstatplot.plot_heatmap(gh_auc, title = "Games-Howell p-values for AUC", metric = "AUC", output_dir = output_dir)
    ocstatplot.plot_heatmap(gh_rmse, title = "Games-Howell p-values for RMSE", metric = "RMSE", output_dir = output_dir)

    # Plot normality and variance diagnostics
    print("\nRunning normality and variance tests...")
    ocstatplot.plot_normality_and_variance_diagnostics(df, metric = 'AUC', n_trials = n_trials, output_dir = output_dir)
    ocstatplot.plot_normality_and_variance_diagnostics(df, metric = 'RMSE', n_trials = n_trials, output_dir = output_dir)

    print("\nStatistical analysis complete. Results saved to 'csvs/' and 'plots/'.")
    plt.close('all')


def load_pca_model(pickle_file: str) -> PCA:
    ''' Load PCA model from disk.
    
    Parameters
    ----------
    pickle_file : str
        Path to the PCA model pickle file.
    Returns
    -------
    PCA
        Fitted PCA model loaded from the pickle file.
    '''

    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def compute_pca_feature_importance(pca_model: PCA, feature_names: list[str]) -> pd.DataFrame:
    '''
    Compute feature importance from PCA model using squared loadings weighted by explained variance.

    Parameters
    ----------
    pca_model : PCA
        Fitted PCA model from sklearn.
    feature_names : list[str]
        List of feature names corresponding to PCA components.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and their computed importance scores.
    '''

    loadings = pca_model.components_
    explained_variance = pca_model.explained_variance_ratio_
    weighted_importance = np.square(loadings) * explained_variance[:, np.newaxis]
    total_importance = np.sum(weighted_importance, axis=0)

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': total_importance
    }).sort_values(by='Importance', ascending=False)

    return importance_df


def run_pca_analysis(
    data_matrix: pd.DataFrame,
    models_dir: str,
    output_dir: str,
    n_trials: int,
    n_features: int = 20
) -> None:
    '''
    Run PCA feature importance analysis for all methods in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Methodology' column (defines which PCA model to load).
    data_matrix : pd.DataFrame
        Original input data used to fit the PCA model (columns = features).
    models_dir : str
        Directory where PCA models (.pkl) are stored.
    output_dir : str
        Where to save barplots.
    n_trials : int
        Trial count label for output naming.
    n_features : int
        Number of top features to visualize.
    colour_mapping : dict
        Optional mapping of methodology names to colors.
    '''

    for pca_type in ['80', '85', '90', '95']:
        model_path = os.path.join(models_dir, f"pca{pca_type}.pkl")
        if not os.path.exists(model_path):
            print(f"[WARNING] PCA model not found: {model_path}")
            continue

        try:
            pca_model = load_pca_model(model_path)
            importance_df = compute_pca_feature_importance(pca_model=pca_model, feature_names=data_matrix.columns.tolist())

            ocstatplot.plot_pca_importance_barplot(importance_df, pca_type, n_features, n_trials, output_dir)
            ocstatplot.plot_pca_importance_histogram(importance_df, pca_type, n_trials, output_dir)
            ocstatplot.save_pca_importance_groups(importance_df, pca_type, n_trials, output_dir)
            ocstatplot.save_pca_importance_bins(importance_df, pca_type, n_trials, output_dir, n_bins=10)

            print(f"Saved PCA feature importance for {pca_type}.")
        except Exception as e:
            print(f"[ERROR] Failed to process {pca_type}: {e}")
