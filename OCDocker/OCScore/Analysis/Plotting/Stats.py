#!/usr/bin/env python3

# Description
###############################################################################
'''
Plotting helpers for statistical summaries (scatter/box/bar, diagnostics, PCA
importance). These utilities are used by Analysis workflows and StatTests.

They are imported as:

import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot
'''

# Imports
###############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import scipy.stats as sstats

from typing import Optional

# No config needed - OCScore modules

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
    output_dir : str
        Directory to save the scatter plot image.
    alpha : float, optional
        Transparency for the markers. Default is 0.9.
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
    method_handles = [mlines.Line2D([0], [0], color = colour_mapping[m], lw = 4.1) for m in method_labels]
    shape_handles = [
        mlines.Line2D([0], [0], marker = 'o', color = 'w', label = 'AUC ≥ 0.5', markerfacecolor = 'gray', markersize = 10),
        mlines.Line2D([0], [0], marker = '*', color = 'w', label = 'AUC < 0.5 (adjusted)', markerfacecolor = 'gray', markersize = 12)
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
        plot_df = plot_df[plot_df['Methodology'] != 'Mean consensus']

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


def plot_scatterplot(
        df_rmse: pd.DataFrame,
        df_auc: pd.DataFrame,
        df_all: pd.DataFrame,
        n_trials: int,
        colour_mapping: dict[str, tuple[float, float, float]],
        output_dir: str,
        orientation: str = 'horizontal',
        alpha: float = 0.9
    ) -> None:
    '''Create scatter plots of RMSE vs AUC for all methods and filtered subsets.

    Create a 1x3 panel of scatter plots (RMSE vs AUC):
    - All filtered points
    - RMSE-filtered subset
    - AUC-filtered subset

    Parameters
    ----------
    df_all : pd.DataFrame
        DataFrame with all filtered points.
    df_rmse : pd.DataFrame
        DataFrame filtered by RMSE threshold.
    df_auc : pd.DataFrame
        DataFrame filtered by AUC threshold.
    n_trials : int
        Number of top trials considered.
    colour_mapping : dict[str, tuple[float, float, float]]
        Dictionary mapping methodologies to colors.
    output_dir : str
        Directory to save the scatter plot image.
    orientation : str, optional
        Orientation of the scatter plot. Default is 'horizontal'. Options: 'horizontal', 'vertical'.
    alpha : float, optional
        Transparency for the markers. Default is 0.9.
    
    Raises
    ------
    ValueError
        If the orientation parameter is not 'horizontal' or 'vertical'.
    '''

    # Make orientation case-insensitive
    orientation = orientation.lower()

    if orientation == 'vertical':
        plt.figure(figsize=(8, 14))
    elif orientation == 'horizontal':
        plt.figure(figsize=(18, 8))
    else:
        # User-facing error: invalid orientation
        ocerror.Error.value_error(f"Invalid orientation: '{orientation}'. Must be 'horizontal' or 'vertical'.") # type: ignore
        raise ValueError(f"Orientation must be 'horizontal' or 'vertical', got {orientation}.")

    panels = [
        (df_rmse, 'Error vs. AUC (Smallest Error)'), 
        (df_auc, 'Error vs. AUC (Biggest AUC)'), 
        (df_all, 'Error vs. AUC (Smallest Error - AUC)')
    ]

    for i, (df, title) in enumerate(panels, start=1):

        df = df.copy()
        df['AUC_adj'] = df['AUC'].apply(lambda x: 1 - x if x < 0.5 else x)
        df['AUC_category'] = df['AUC'].apply(lambda x: '>= 0.5' if x >= 0.5 else '< 0.5')
        df.loc[df['AUC_category'] == '< 0.5', 'AUC'] = df['AUC_adj']

        if orientation == 'vertical':
            plt.subplot(3, 1, i)
        else:
            plt.subplot(1, 3, i)

        # Scatter for AUC ≥ 0.5
        sns.scatterplot(
            data = df[df['AUC_category'] == '>= 0.5'],
            x = 'RMSE',
            y = 'AUC',
            hue = 'Methodology',
            palette = colour_mapping,
            alpha = alpha,
            s = 30,
            legend = False,
        )

        # Scatter for AUC < 0.5
        sns.scatterplot(
            data = df[df['AUC_category'] == '< 0.5'],
            x ='RMSE',
            y ='AUC',
            hue = 'Methodology',
            palette = colour_mapping,
            alpha = alpha,
            s = 50,
            legend = False,
        )

        plt.title(title)
        plt.grid(True, linestyle=':', linewidth=0.5)
        plt.xlabel('RMSE')
        plt.ylabel('AUC')

    # Legends - define before use
    method_labels = df_all['Methodology'].unique().tolist()
    method_handles = [mlines.Line2D([0], [0], color = colour_mapping[m], lw = 4.1) for m in method_labels]
    shape_handles = [
        mlines.Line2D([0], [0], marker = 'o', color = 'w', label = 'AUC ≥ 0.5', markerfacecolor = 'gray', markersize = 10),
        mlines.Line2D([0], [0], marker = '*', color = 'w', label = 'AUC < 0.5 (adjusted)', markerfacecolor = 'gray', markersize = 12)
    ]

    if orientation == 'vertical':
        # Methodology legend
        plt.figlegend(method_handles, method_labels, title = 'Methodology',
                    loc = 'lower center', bbox_to_anchor = (0.5, 0.09), ncol = 5)

        # Shape legend
        plt.figlegend(shape_handles, ['AUC ≥ 0.5', 'AUC < 0.5 (adjusted)'], title = 'Marker Type',
                    loc = 'lower center', bbox_to_anchor = (0.5, 0.03), ncol = 2)
        plt.tight_layout(rect = (0, 0.18, 1, 1))
    else:
        # Methodology legend
        plt.figlegend(method_handles, method_labels, title = 'Methodology',
                    loc = 'lower center', bbox_to_anchor = (0.5, 0.09), ncol = 5)

        # Shape legend
        plt.figlegend(shape_handles, ['AUC ≥ 0.5', 'AUC < 0.5 (adjusted)'], title = 'Marker Type',
                    loc = 'lower center', bbox_to_anchor = (0.5, 0.02), ncol = 2)

    # Methodology legend
    plt.figlegend(method_handles, method_labels, title = 'Methodology',
                  loc = 'lower center', bbox_to_anchor = (0.5, 0.09), ncol = 5)

    # Shape legend
    plt.figlegend(shape_handles, ['AUC ≥ 0.5', 'AUC < 0.5 (adjusted)'], title = 'Marker Type',
                  loc = 'lower center', bbox_to_anchor = (0.5, 0.02), ncol = 2)

    if orientation == 'vertical':
        plt.subplots_adjust(bottom=0.28)
        plt.tight_layout(rect = (0, 0.25, 1, 1))
        
    plt.savefig(f'{output_dir}/scatter_rmse_auc_panels_{n_trials}.png', dpi=300)
    plt.close()


def plot_bar_with_significance(
        gh_df: pd.DataFrame,
        metric: str,
        y_col: str = 'diff',
        colour_mapping: Optional[dict[str, tuple[float, float, float]]] = None,
        output_dir: str = 'plots',
        top_n: Optional[int] = 30
    ) -> None:
    '''
    Plot Games-Howell pairwise differences as a horizontal bar chart.

    Parameters
    ----------
    gh_df : pd.DataFrame
        Output of pingouin.pairwise_gameshowell (expects columns 'A','B','diff','pval').
    metric : str
        Metric label for titling ('AUC' or 'RMSE').
    y_col : str
        Which column from gh_df to plot as bar length (default 'diff').
    colour_mapping : dict | None, optional
        Unused here, accepted for API compatibility. Default: None.
    output_dir : str
        Where to save the plot image. Default: 'plots'.
    top_n : int | None, optional
        If given, keep the top-N pairs by smallest p-value. Default: 30.
    '''

    df = gh_df.copy()
    if 'pval' not in df.columns:
        # pingouin sometimes returns 'pval'/'pval_corr'; tolerate variants
        pcol = next((c for c in df.columns if c.startswith('pval')), None)
        if pcol is None:
            # User-facing error: missing required data in DataFrame
            ocerror.Error.data_not_found("Games-Howell dataframe must contain a p-value column (pval, pval_corr, etc.)") # type: ignore
            raise ValueError('Games-Howell dataframe must contain a p-value column.')
        df['pval'] = df[pcol]

    df['pair'] = df['A'].astype(str) + ' vs ' + df['B'].astype(str)
    df.sort_values(by=['pval', y_col], ascending=[True, False], inplace=True)
    if top_n is not None:
        df = df.head(top_n)

    # Color positive diffs blue, negative red for quick read
    colors = df[y_col].map(lambda v: 'tab:blue' if v >= 0 else 'tab:red')

    plt.figure(figsize=(max(8, 0.25 * len(df)), max(6, 0.35 * len(df))))
    ax = sns.barplot(data=df, x=y_col, y='pair', palette=colors, orient='h')

    # Annotate p-values and significance stars
    def stars(p: float) -> str:
        '''Convert p-value to significance stars.
        
        Parameters
        ----------
        p : float
            The p-value to convert.
        
        Returns
        -------
        str
            Significance stars: '***' for p < 0.001, '**' for p < 0.01, '*' for p < 0.05, '' otherwise.
        '''
        
        return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

    for i, r in df.reset_index(drop=True).iterrows():
        ax.text(r[y_col] + (0.01 if r[y_col] >= 0 else -0.01), i,
                f"{r[y_col]:.3f}  (p={r['pval']:.2e}) {stars(r['pval'])}",
                ha='left' if r[y_col] >= 0 else 'right', va='center', fontsize=8)

    ax.set_title(f'Games-Howell pairwise differences — {metric}')
    ax.set_xlabel(f'Difference in {metric}')
    ax.set_ylabel('Pair (A vs B)')
    plt.grid(True, axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/games_howell_bar_{metric}.png", dpi=300)
    plt.close()


def plot_heatmap(
        gh_df: pd.DataFrame,
        title: str,
        metric: str,
        output_dir: str = 'plots'
    ) -> None:
    '''Heatmap of Games-Howell p-values across methodology pairs.
    
    Parameters
    ----------
    gh_df : pd.DataFrame
        Output of pingouin.pairwise_gameshowell (expects columns 'A','B
        'diff','pval').
    title : str
        Title for the heatmap.
    metric : str
        Metric label for titling ('AUC' or 'RMSE').
    output_dir : str
        Where to save the plot image. Default: 'plots'.
    '''

    df = gh_df.copy()
    pcol = 'pval' if 'pval' in df.columns else next((c for c in df.columns if c.startswith('pval')), None)
    if pcol is None:
        # User-facing error: missing required data in DataFrame
        ocerror.Error.data_not_found("Games-Howell dataframe must contain a p-value column (pval, pval_corr, etc.)") # type: ignore
        raise ValueError('Games-Howell dataframe must contain a p-value column.')
    mat = df.pivot(index='A', columns='B', values=pcol)
    # Mirror to make a symmetric matrix, leaving diagonal as NaN
    mat_full = mat.combine_first(mat.T)
    np.fill_diagonal(mat_full.values, np.nan)

    plt.figure(figsize=(max(8, 0.6 * mat_full.shape[1]), max(6, 0.35 * mat_full.shape[0])))
    ax = sns.heatmap(-np.log10(mat_full), cmap='mako', annot=False, cbar_kws={'label': '-log10(p)'})
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/games_howell_heatmap_{metric}.png", dpi=300)
    plt.close()


def plot_normality_and_variance_diagnostics(
        df: pd.DataFrame,
        metric: str,
        n_trials: int,
        output_dir: str = 'plots'
    ) -> None:
    ''' Perform and plot normality and variance diagnostics across methodologies.

    Quick diagnostics across groups:
    - Shapiro-Wilk p-values per methodology (bar of -log10 p)
    - Group variances (bar) and Levene's p-value annotated

    Parameters
    ----------
    df : pd.DataFrame
        Data containing 'Methodology' and the specified metric.
    metric : str
        Metric column to analyze (e.g., 'AUC' or 'RMSE').
    n_trials : int
        Number of trials for title and output naming.
    output_dir : str
        Directory to save the diagnostics plot. Default: 'plots'.
    '''

    # Compute Shapiro p-values and variances per group
    rows = []
    groups = []

    for method, sub in df.groupby('Methodology'):
        x = pd.to_numeric(sub[metric], errors='coerce').dropna().to_numpy()
        if x.size >= 3:
            try:
                p_shap = sstats.shapiro(x).pvalue
            except (ValueError, TypeError, AttributeError):
                # Fallback to NaN if statistical test fails
                p_shap = np.nan
        else:
            p_shap = np.nan
        var = float(np.var(x, ddof=1)) if x.size >= 2 else np.nan
        rows.append({'Methodology': method, 'p_shapiro': p_shap, 'variance': var})
        groups.append(x)

    diag = pd.DataFrame(rows).sort_values(by='p_shapiro', ascending=True)

    # Levene across all groups
    try:
        groups_nonempty = [g for g in groups if g.size >= 2]
        p_levene = sstats.levene(*groups_nonempty).pvalue if len(groups_nonempty) >= 2 else np.nan
    except (ValueError, TypeError, AttributeError):
        # Fallback to NaN if statistical test fails
        p_levene = np.nan

    # Plot two panels
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(data=diag, x='Methodology', y=-np.log10(diag['p_shapiro']), color='steelblue')
    plt.xticks(rotation=90)
    plt.ylabel('-log10 Shapiro p-value')
    plt.title(f'Normality (Shapiro) — {metric}')
    plt.grid(True, axis='y', linestyle=':', linewidth=0.5)

    plt.subplot(1, 2, 2)
    sns.barplot(data=diag, x='Methodology', y='variance', color='tab:orange')
    plt.xticks(rotation=90)
    plt.ylabel('Group variance')
    lev_txt = f"Levene p={p_levene:.2e}" if isinstance(p_levene, float) and np.isfinite(p_levene) else "Levene p=N/A"
    plt.title(f'Variance across groups — {metric} ({lev_txt})')
    plt.grid(True, axis='y', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/diagnostics_{metric}_{n_trials}.png", dpi=300)
    plt.close()


def plot_pca_importance_barplot(
        importance_df: pd.DataFrame,
        pca_type: str,
        n_features: int,
        n_trials: int,
        output_dir: str = 'plots'
    ) -> None:
    '''Barplot of top-N PCA feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    pca_type : str
        PCA type label for titling (e.g., '1', '2').
    n_features : int
        Number of top features to display.
    n_trials : int
        Number of trials for title and output naming.
    output_dir : str
        Directory to save the barplot image. Default: 'plots'.
    '''

    top = importance_df.head(n_features)

    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    sns.barplot(data=top, x='Importance', y='Feature', orient='h', color='steelblue')
    plt.title(f'PCA{pca_type}: Top {len(top)} feature importances')
    plt.xlabel('Importance (variance-weighted loadings)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca{pca_type}_importance_top{len(top)}_{n_trials}.png", dpi=300)
    plt.close()


def plot_pca_importance_histogram(
        importance_df: pd.DataFrame,
        pca_type: str,
        n_trials: int,
        output_dir: str = 'plots'
    ) -> None:
    '''Histogram of PCA feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    pca_type : str
        PCA type label for titling (e.g., '1', '2').
    n_trials : int
        Number of trials for title and output naming.
    output_dir : str
        Directory to save the histogram image. Default: 'plots'.
    '''

    plt.figure(figsize=(8, 5))
    sns.histplot(importance_df['Importance'], bins=30, color='tab:purple')
    plt.title(f'PCA{pca_type}: Distribution of feature importances')
    plt.xlabel('Importance')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca{pca_type}_importance_hist_{n_trials}.png", dpi=300)
    plt.close()


def save_pca_importance_groups(
        importance_df: pd.DataFrame,
        pca_type: str,
        n_trials: int,
        output_dir: str = 'plots'
    ) -> None:
    '''Assign coarse groups by quantiles and save as CSV.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    pca_type : str
        PCA type label for titling (e.g., '1', '2').
    n_trials : int
        Number of trials for title and output naming.
    output_dir : str
        Directory to save the plot image. Default: 'plots'.
    '''

    q = importance_df['Importance'].quantile
    bins = [0.0, q(0.2), q(0.4), q(0.6), q(0.8), q(1.0)]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    df = importance_df.copy()
    df['Group'] = pd.cut(df['Importance'], bins=bins, labels=labels, include_lowest=True, duplicates='drop')
    df.to_csv(f"{output_dir}/pca{pca_type}_importance_groups_{n_trials}.csv", index=False)


def save_pca_importance_bins(
        importance_df: pd.DataFrame,
        pca_type: str,
        n_trials: int,
        output_dir: str = 'plots',
        n_bins: int = 10
    ) -> None:
    '''Assign quantile bins (qcut) and save as CSV.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'Feature' and 'Importance' columns.
    pca_type : str
        PCA type label for titling (e.g., '1', '2').
    n_trials : int
        Number of trials for title and output naming.
    output_dir : str
        Directory to save the plot image. Default: 'plots'.
    n_bins : int
        Number of quantile bins to create. Default: 10.
    '''

    df = importance_df.copy()
    try:
        df['bin'] = pd.qcut(df['Importance'], q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # Not enough unique values; fallback to rank-based bins
        ranks = df['Importance'].rank(method='average', pct=True)
        df['bin'] = (ranks * (n_bins - 1)).astype(int)
    df.to_csv(f"{output_dir}/pca{pca_type}_importance_bins_{n_trials}.csv", index=False)
