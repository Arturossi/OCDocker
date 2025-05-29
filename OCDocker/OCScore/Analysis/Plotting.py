#!/usr/bin/env python3

# Description
###############################################################################
""" This module provides functions to plot statistical results and performance metrics

It is imported as:

import OCDocker.OCScore.Analysis.Plotting as ocstatplot
"""

# Imports
###############################################################################

import math
import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import gridspec
from matplotlib.lines import Line2D
from scipy import stats

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################

def set_color_mapping(df: pd.DataFrame, palette_colour: str = "glasbey") -> dict[str, tuple[float, float, float]]:
    '''
    Set the color palette for plotting based on the unique methodologies in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'Methodology' column with unique methodologies.
    palette_colour : str
        Name of the color palette to use. Options include:
        - "glasbey"
        - "Set2"
        - "Set3"
        - "tab10"
        - "tab20"
        - "colorblind"
        - "pastel"
        - "bright"
        - "dark"
        - "deep"
        - "muted"
        - "viridis"
    
    Returns
    -------
    color_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping each methodology to a color in RGB format.

    Raises
    ------
    ValueError
        If an unsupported palette is provided.
    '''

    print("Setting the pallette, alpha, and error threshold for the plots.")

    if palette_colour == "glasbey":
        palette_colour = sns.color_palette(cc.glasbey, n_colors = df['Methodology'].nunique()) # type: ignore
    elif palette_colour in ["Set2", "Set3", "tab10", "tab20", "colorblind", "pastel", "bright", "dark", "deep", "muted", "viridis"]:
        # Use seaborn's built-in palettes
        palette_colour = sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique()) # type: ignore
    else:
        raise ValueError(f"Unsupported palette: {palette_colour}. Choose from 'glasbey', 'Set2', 'Set3', 'tab10', 'tab20', 'colorblind', 'pastel', 'bright', 'dark', 'deep', 'muted', or 'viridis'.")

    # Create a color mapping for methodologies
    color_mapping = {
        method: color for method, color in zip(
            df['Methodology'].unique(), 
            sns.color_palette(palette_colour, n_colors = df['Methodology'].nunique())
        )
    }

    return color_mapping

def plot_bar_with_significance(
    data: pd.DataFrame,
    metric: str,
    colour_mapping: dict[str, tuple[float, float, float]],
    output_dir: str,
    y_col: str = 'diff'
) -> None:
    '''
    Create a barplot of mean performance per methodology and annotate bars
    with counts of statistically significant differences.

    Parameters
    ----------
    data : pd.DataFrame
        Games-Howell post-hoc test results.
    metric : str
        Name of the metric analyzed ('AUC' or 'RMSE').
    colour_mapping : dict
        Dictionary mapping methodology names to RGB tuples or color codes.
    output_dir : str
        Directory to save the barplot image.
    y_col : str, optional
        Column used as bar height (usually 'diff'). Defaults to 'diff'.
    '''

    # Merge means from both columns A and B
    means = pd.concat([
        data[['A', 'mean(A)']].rename(columns={'A': 'Methodology', 'mean(A)': y_col}),
        data[['B', 'mean(B)']].rename(columns={'B': 'Methodology', 'mean(B)': y_col})
    ], ignore_index = True)
    means = means.groupby('Methodology').mean().reset_index()

    # Count significant differences
    significant_counts = data[data['pval'] < 0.05].groupby('A').size().reindex(means['Methodology'], fill_value = 0)

    # Extract colors in the correct order
    bar_colors = [colour_mapping.get(m, (0.5, 0.5, 0.5)) for m in means['Methodology']]

    # Plot
    plt.figure(figsize = (12, 8))
    sns.barplot(
        x = 'Methodology',
        y = y_col,
        data = means,
        hue = 'Methodology',
        palette = bar_colors,
        errorbar = 'sd',
        capsize = 0.2,
        legend = False
    )
    plt.xticks(rotation = 90)
    plt.title(f"{metric} Means with Significant Pairwise Differences")

    # Annotate with significance counts
    for i, count in enumerate(significant_counts):
        if count > 0:
            plt.text(i, means.iloc[i][y_col] + 0.01, f"*{count}", ha='center', color='red')

    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/games_howell_barplot_{metric}.png", dpi = 300)
    plt.close()

def plot_heatmap(df: pd.DataFrame, title: str, metric: str, output_dir: str) -> None:
    '''
    Create a symmetric heatmap of p-values between methodologies from Games-Howell test.

    Parameters
    ----------
    df : pd.DataFrame
        Games-Howell post-hoc results.
    title : str
        Plot title.
    metric : str
        Metric name for saving the figure.
    output_dir : str
        Directory to save the heatmap plot.
    '''

    methods = sorted(set(df['A']).union(df['B']))
    n = len(methods)
    p_matrix = np.ones((n, n))

    for _, row in df.iterrows():
        i, j = methods.index(row['A']), methods.index(row['B'])
        p_matrix[i, j] = row['pval']
        p_matrix[j, i] = row['pval']

    def fmt_full(x: float) -> str:
        return "0.0" if x == 0 else ("1.00" if x == 1.0 else (f"{x:.1e}" if x < 0.01 else f"{x:.2f}"))

    annot = np.vectorize(fmt_full)(p_matrix)
    fontsize = max(8, int(280 / n))

    width = max(14, 0.85 * n)
    height = max(9, 0.65 * n)

    # Use gridspec with constrained layout from the beginning
    fig = plt.figure(figsize = (width, height), constrained_layout = True)
    gs = gridspec.GridSpec(1, 2, width_ratios = [20, 1], figure = fig)

    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    sns.heatmap(
        p_matrix,
        annot = annot,
        xticklabels = methods,
        yticklabels = methods,
        cmap = "coolwarm",
        cbar_ax = cax,
        cbar_kws = {'label': 'p-value'},
        fmt = "",
        annot_kws = {"size": fontsize},
        ax = ax
    )

    cax.set_ylabel('p-value', fontsize = 14)

    for text in ax.texts:
        text.set_rotation(45)

    ax.set_xticklabels(methods, rotation = 45, ha = 'right')
    ax.set_yticklabels(methods, rotation = 0)
    ax.set_title(title, pad = 20)
    ax.tick_params(axis = 'x', pad = 6)
    ax.tick_params(axis = 'y', pad = 4)

    plt.savefig(f"{output_dir}/games_howell_heatmap_{metric}.png", dpi = 300)
    plt.close()

def plot_combined_metric_scatter(df: pd.DataFrame, n_trials: int, colour_mapping: dict[str, tuple[float, float, float]], output_dir: str, alpha: float = 0.9) -> None:
    '''
    Generate a detailed scatter plot showing RMSE vs AUC across methods with shading and symbol cues.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with RMSE, AUC, and Methodology columns.
    n_trials : int
        Number of top trials considered.
    colour_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping methodologies to colors.
    alpha : float
        Transparency for the markers.
    '''

    df = df.copy()
    df['AUC_adj'] = df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)
    df['AUC_category'] = df['AUC'].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')
    df.loc[df['AUC_category'] == '< 0.5', 'AUC'] = df['AUC_adj']

    plt.figure(figsize = (10, 8))

    # Scatter for AUC ≥ 0.5
    sns.scatterplot(
        data = df[df['AUC_category'] == '>= 0.5'],
        x = 'RMSE',
        y = 'AUC',
        hue = 'Methodology',
        palette = colour_mapping,
        alpha = alpha,
        marker = 'o',
        s = 100,
        legend = False
    )

    # Scatter for AUC < 0.5
    sns.scatterplot(
        data = df[df['AUC_category'] == '< 0.5'],
        x = 'RMSE',
        y = 'AUC',
        hue = 'Methodology',
        palette = colour_mapping,
        alpha = alpha,
        marker = '*',
        s = 130,
        legend = False
    )

    plt.xlabel('RMSE')
    plt.ylabel('AUC (adjusted)')
    plt.title(f'Combined Metric Comparison ({n_trials} Trials)')
    plt.grid(True)
    plt.minorticks_on()
    plt.grid(which = 'minor', linestyle = ':', linewidth = 0.3)

    # Legends
    method_labels = df['Methodology'].unique().tolist()
    method_handles = [Line2D([0], [0], color = colour_mapping[m], lw = 4.1) for m in method_labels]
    shape_handles = [
        Line2D([0], [0], marker = 'o', color = 'w', label = 'AUC ≥ 0.5', markerfacecolor = 'gray', markersize = 10),
        Line2D([0], [0], marker = '*', color = 'w', label = 'AUC < 0.5 (adjusted)', markerfacecolor = 'gray', markersize = 12)
    ]

    plt.figlegend(method_handles, method_labels, title = 'Methodology',
                  loc = 'lower center', bbox_to_anchor = (0.5, 0.07), ncol = 5)
    plt.figlegend(shape_handles, ['AUC ≥ 0.5', 'AUC < 0.5 (adjusted)'], title = 'Marker Type',
                  loc = 'lower center', bbox_to_anchor = (0.5, 0.01), ncol = 2)

    plt.tight_layout(rect = (0, 0.22, 1, 1))
    plt.savefig(f'{output_dir}/scatter_combined_metric_{n_trials}.png', bbox_inches = 'tight', dpi = 300)
    plt.close()

def plot_boxplots(df: pd.DataFrame, n_trials: int, colour_mapping: dict[str, tuple[float, float, float]], output_dir: str, show_simple_consensus: bool = False) -> None:
    '''
    Generate enhanced boxplots of RMSE and AUC across methodologies, with group shading and mean lines.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'RMSE', 'AUC', and 'Methodology'.
    n_trials : int
        Number of trials used for title and filenames.
    colour_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping methodologies to colors.
    output_dir : str
        Directory to save the boxplot images.
    show_simple_consensus : bool
        Whether to include the 'Simple consensus' box in the plots.
    '''

    plot_df = df.copy()
    if not show_simple_consensus:
        plot_df = plot_df[plot_df['Methodology'] != 'Simple consensus']

    plt.figure(figsize = (16, 12))
    mean_line_rmse, mean_line_auc = None, None

    for i, metric in enumerate(['RMSE', 'AUC']):
        plt.subplot(2, 1, i + 1)
        ax = sns.boxplot(
            data = plot_df,
            x = 'Methodology',
            y = metric,
            hue = 'Methodology',
            palette = colour_mapping,
            showfliers = False,
            legend = False
        )

        # Distinct line color for each metric
        mean_val = plot_df[metric].mean()
        line_color = 'red' if metric == 'RMSE' else 'blue'
        line = ax.axhline(mean_val, color = line_color, linestyle = '--', label = f'Mean {metric}')
        if i == 0:
            mean_line_rmse = line
        else:
            mean_line_auc = line

        plt.xticks(rotation = 90)
        plt.title(f'{metric} Distribution ({n_trials} Trials)')
        plt.grid(True, linestyle = ':', linewidth = 0.5)
        plt.minorticks_on()

        # Highlight NN, XGB, Transformer groups
        for prefix, color in [('NN', 'lightblue'), ('XGB', 'lightgreen'), ('Transformer', 'lightcoral')]:
            for method in plot_df['Methodology'].unique():
                if method.startswith(prefix):
                    idx = list(plot_df['Methodology'].unique()).index(method)
                    plt.axvspan(idx - 0.5, idx + 0.5, color = color, alpha = 0.2)

    # Add figure-level legend at the bottom
    plt.figlegend(
        handles = [mean_line_rmse, mean_line_auc],
        labels = ['Mean RMSE', 'Mean AUC'],
        loc = 'lower center',
        bbox_to_anchor = (0.5, 0.02),
        ncol = 2,
        frameon = False
    )

    # Adjust layout to avoid overlap
    plt.tight_layout(rect = (0, 0.08, 1, 1))
    plt.savefig(f'{output_dir}/boxplots_rmse_auc_{n_trials}.png', dpi = 300)
    plt.close()

def plot_barplots(df: pd.DataFrame, n_trials: int, colour_mapping: dict[str, tuple[float, float, float]], output_dir: str) -> None:
    '''
    Generate sorted barplots of mean RMSE and AUC across methodologies with annotations.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'RMSE', 'AUC', and 'Methodology'.
    n_trials : int
        Trial number for title and output naming.
    colour_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping methodologies to colors.
    output_dir : str
        Directory to save the barplot images.
    '''

    df_means = df.groupby('Methodology')[['RMSE', 'AUC']].mean().reset_index()

    plt.figure(figsize = (16, 6))
    for i, metric in enumerate(['RMSE', 'AUC']):
        plt.subplot(1, 2, i + 1)
        df_sorted = df_means.sort_values(by = metric)
        method_order = df_sorted['Methodology'].tolist()
        palette_sorted = {k: colour_mapping[k] for k in method_order}

        ax = sns.barplot(
            data = df_sorted,
            x = 'Methodology',
            y = metric,
            hue = 'Methodology',
            palette = palette_sorted,
            legend = False
        )
        for j, val in enumerate(df_sorted[metric]):
            plt.text(j, val + 0.01, f"{val:.2f}", ha = 'center', va = 'bottom', fontsize = 9)

        plt.xticks(rotation = 90)
        plt.title(f'{metric} Mean per Method ({n_trials} Trials)')
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which = 'minor', linestyle = ':', linewidth = 0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/barplot_rmse_auc_{n_trials}.png')
    plt.close()

import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Optional

def plot_normality_and_variance_diagnostics(
    df: pd.DataFrame,
    metric: str,
    n_trials: int,
    output_dir: str = "plots",
    highlight_significance: bool = True
) -> None:
    '''
    Generate QQ plots and run variance homogeneity tests (Levene and Bartlett) for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metric and Methodology.
    metric : str
        Metric column to evaluate (e.g., 'RMSE', 'AUC').
    n_trials : int
        Used for naming output plots.
    output_dir : str
        Directory to save the QQ plot.
    highlight_significance : bool
        If True, highlight QQ titles in red when p < 0.05.
    '''

    methods = df['Methodology'].unique()
    num_methods = len(methods)
    cols = 4
    rows = math.ceil(num_methods / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    group_data = []
    shapiro_results = []

    for i, method in enumerate(methods):
        data = df[df['Methodology'] == method][metric].dropna()
        group_data.append(data)

        stats.probplot(data, dist="norm", plot=axes[i])
        axes[i].tick_params(labelsize=8)

        if len(data) >= 3:
            shapiro_p = stats.shapiro(data).pvalue
        else:
            shapiro_p = float("nan")

        text_color = "red" if (highlight_significance and shapiro_p < 0.05) else "black"
        axes[i].set_title(f"{method}", fontsize=10, color=text_color)

        shapiro_results.append({
            "Methodology": method,
            f"Shapiro_{metric}_p": round(shapiro_p, 4),
            "Significant": "Yes" if shapiro_p < 0.05 else "No"
        })

        print(f"Shapiro p-value for {method} ({metric}): {shapiro_p:.4f}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"QQ Plots per Method ({metric}, {n_trials} Trials)", fontsize=14)

    if highlight_significance:
        # Add figure-level legend
        red_patch = Line2D([0], [0], marker='o', color='w', label='p < 0.05 (non-normal)',
                               markerfacecolor='red', markersize=10)
        black_patch = Line2D([0], [0], marker='o', color='w', label='p ≥ 0.05 (normal)',
                                 markerfacecolor='black', markersize=10)
        fig.legend(handles=[black_patch, red_patch],
                   loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=10, title='Shapiro-Wilk')

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    plt.savefig(f"{output_dir}/qq_{metric}_{n_trials}.png", dpi=300)
    plt.close()

    # Variance tests
    levene_stat, levene_p = stats.levene(*group_data)
    bartlett_stat, bartlett_p = stats.bartlett(*group_data)

    print(f"\nLevene test for {metric}: stat = {levene_stat:.4f}, p = {levene_p:.4f}")
    print(f"Bartlett test for {metric}: stat = {bartlett_stat:.4f}, p = {bartlett_p:.4f}")

    pd.DataFrame(shapiro_results).to_csv(f"{output_dir}/shapiro_{metric}_{n_trials}.csv", index=False)
    pd.DataFrame({
        "Test": ["Levene", "Bartlett"],
        "Statistic": [round(levene_stat, 4), round(bartlett_stat, 4)],
        "p-value": [round(levene_p, 4), round(bartlett_p, 4)]
    }).to_csv(f"{output_dir}/variance_tests_{metric}_{n_trials}.csv", index=False)

def plot_pca_importance_barplot(
    importance_df: pd.DataFrame,
    method: str,
    n_features: int,
    n_trials: int,
    output_dir: str
) -> None:
    '''
    Plot top PCA features with optional color mapping.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame containing PCA feature importances with columns 'Feature' and 'Importance'.
    method : str
        Methodology name to include in the plot title and filename.
    n_features : int
        Number of top features to visualize.
    n_trials : int
        Number of trials used for the analysis, included in the filename.
    output_dir : str
        Directory where the barplot will be saved.
    '''

    top_features = importance_df.head(n_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_features, x='Importance', y='Feature', color='teal', legend=False)
    plt.title(f"Top PCA Features - {method} ({n_trials} Trials)")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/pca_feature_importance_{method}_{n_trials}.png", dpi=300)
    plt.close()

def plot_pca_importance_histogram(
    importance_df: pd.DataFrame,
    method: str,
    n_trials: int,
    output_dir: str
) -> None:
    '''
    Plot a histogram showing the distribution of PCA feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    method : str
        Name of the methodology.
    n_trials : int
        Trial count for labeling.
    output_dir : str
        Directory to save the plot.
    '''

    plt.figure(figsize=(10, 6))
    sns.histplot(data=importance_df, x='Importance', bins=50, kde=True, color='teal')
    plt.title(f'Distribution of PCA Feature Importances – {method} ({n_trials} Trials)')
    plt.xlabel('Importance')
    plt.ylabel('Feature Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_importance_hist_{method}_{n_trials}.png", dpi=300)
    plt.close()

def save_pca_importance_groups(
    importance_df: pd.DataFrame,
    method: str,
    n_trials: int,
    output_dir: str
) -> None:
    '''
    Group features by identical importance values and export as CSV.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    method : str
        Name of the methodology.
    n_trials : int
        Trial count for labeling.
    output_dir : str
        Directory to save the CSV.
    '''
    os.makedirs(output_dir, exist_ok=True)

    grouped = importance_df.groupby('Importance')['Feature'].apply(list).reset_index()
    grouped = grouped.sort_values(by='Importance', ascending=False)

    grouped.to_csv(f"{output_dir}/pca_feature_importance_groups_{method}_{n_trials}.csv", index=False)

def save_pca_importance_bins(
    importance_df: pd.DataFrame,
    method: str,
    n_trials: int,
    output_dir: str,
    n_bins: int = 10
) -> None:
    '''
    Count and export how many features fall within each importance bin.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    method : str
        Name of the methodology.
    n_trials : int
        Trial count for labeling.
    output_dir : str
        Directory to save the CSV.
    n_bins : int
        Number of bins to use for importance ranges.
    '''
    os.makedirs(output_dir, exist_ok=True)

    bins = pd.cut(importance_df['Importance'], bins=n_bins)
    bin_counts = bins.value_counts().sort_index()
    bin_df = pd.DataFrame({'Importance Bin': bin_counts.index.astype(str), 'Feature Count': bin_counts.values})

    bin_df.to_csv(f"{output_dir}/pca_importance_bins_{method}_{n_trials}.csv", index=False)
