#!/usr/bin/env python3

# Description
###############################################################################
''' This module provides correlation analysis utilities for RMSE and AUC metrics.
It computes and visualizes Pearson correlations across methodologies and optionally
includes raw and consensus scores.

Usage:

import OCDocker.OCScore.Analysis.Correlation as occorrana
'''

# Imports
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional

# Functions
###############################################################################


def correlation_analysis(
    results_df: pd.DataFrame,
    final_metrics: pd.DataFrame,
    n_trials: int,
    error_threshold: float = 1.5,
    save_path: str = "plots",
    colour_mapping: Optional[dict] = None
) -> None:
    '''
    Compute and visualize correlation between RMSE and AUC per methodology,
    including raw and consensus scores.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame from parsed Optuna studies with RMSE, AUC and Methodology.
    final_metrics : pd.DataFrame
        DataFrame with raw scoring and consensus scores.
    n_trials : int
        Trial count label for output files.
    error_threshold : float
        Maximum RMSE to include in the filtered subset.
    save_path : str
        Directory to save output plot.
    colour_mapping : dict, optional
        Dictionary mapping methodologies to specific colors.
    '''

    print("Computing correlation metrics...")

    # Extract relevant columns and harmonize naming
    core_data = results_df[['study_name', 'study_type', 'best_combined_value', 'best_combined_auc', 'best_combined_metric']].copy()
    core_data.rename(columns={
        'study_type': 'Methodology',
        'best_combined_value': 'RMSE',
        'best_combined_auc': 'AUC',
        'best_combined_metric': 'combined_metric'
    }, inplace=True)

    # Merge with raw/consensus scores
    corr_data = pd.concat([core_data, final_metrics[['study_name', 'Methodology', 'RMSE', 'AUC', 'combined_metric']]], axis=0)

    # Filter out high-error experiments
    filtered = corr_data[corr_data['RMSE'] <= error_threshold]

    # Compute correlations
    correlation_df = pd.DataFrame(columns=['Methodology', 'Correlation'])
    correlation_df.loc[0] = ['All', filtered['RMSE'].corr(filtered['AUC'])]

    for method in filtered['Methodology'].unique():
        subset = filtered[filtered['Methodology'] == method]
        corr_val = subset['RMSE'].corr(subset['AUC'])
        correlation_df.loc[len(correlation_df)] = [method, corr_val]

    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    # Plot
    plt.figure(figsize=(20, 8))
    palette = colour_mapping if colour_mapping else "viridis"
    sns.barplot(
        data=correlation_df,
        x='Methodology',
        y='Correlation',
        hue='Methodology',
        palette=palette,
        legend=False
    )

    plt.title(f'Correlation between RMSE and AUC ({n_trials} Trials)')
    plt.xticks(rotation=90)
    plt.ylabel('Pearson Correlation')
    plt.grid(True, which='both', linestyle=':', linewidth=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/Experiments_Correlation_barplot_{n_trials}.png", dpi=300)
    plt.close()

    print("Correlation plot saved.")
