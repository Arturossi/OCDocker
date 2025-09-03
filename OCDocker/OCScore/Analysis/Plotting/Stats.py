import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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