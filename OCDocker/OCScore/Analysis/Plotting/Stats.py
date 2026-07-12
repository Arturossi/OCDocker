#!/usr/bin/env python3

# Description
###############################################################################
'''
Plotting helpers for statistical summaries (scatter/box/bar, diagnostics, PCA
importance). These utilities are used by Analysis workflows and StatTests.

Usage:

import OCDocker.OCScore.Analysis.Plotting.Stats as ocstatplot
'''

# Imports
###############################################################################
import warnings

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sstats
import seaborn as sns
import OCDocker.Error as ocerror

from typing import Mapping, Optional, Sequence

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Palette for the shortcut-risk scatter (validated for CVD separation).
COLOR_REFERENCE = "#2a78d6"   # blue   -> the reference model
COLOR_RETAINED = "#0ca30c"    # green  -> beat the reference with a distributed explanation
COLOR_DISCARDED = "#d03b3b"   # red    -> beat the reference, but on a dominant feature
COLOR_OTHER = "#b6b4ab"       # gray   -> did not beat the reference; the rule does not apply
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_MUTED = "#52514e"
COLOR_GRID = "#e1e0d9"

# Functions
###############################################################################
## Private ##

## Public ##

def plot_ablation_bedroc_significance_bars(
        significance_df: pd.DataFrame,
        *,
        reference_policy: str = 'full_ocscore',
        metric_label: str = 'BEDROC',
        output_dir: str = 'plots',
        alpha: float = 0.05
    ) -> None:
    '''
    Plot per-policy BEDROC means vs a reference policy, colored by paired significance.

    Parameters
    ----------
    significance_df : pd.DataFrame
        Output of ``OCScore.Analysis.AblationSignificance.compute_ablation_significance``
        (expects columns 'policy', 'reference_mean', 'policy_mean', 'mean_diff',
        'pvalue_corrected', 'reject_null').
    reference_policy : str
        Name of the reference policy, used for the axis label and reference line. Default: 'full_ocscore'.
    metric_label : str
        Metric label for titling. Default: 'BEDROC'.
    output_dir : str
        Where to save the plot image. Default: 'plots'.
    alpha : float
        Family-wise significance threshold used only for the subtitle text. Default: 0.05.
    '''

    df = significance_df.sort_values(by='policy_mean', ascending=True).reset_index(drop=True)
    reference_mean = float(df['reference_mean'].iloc[0]) if not df.empty else float('nan')

    def stars(p: float) -> str:
        '''Convert a corrected p-value to significance stars.

        Parameters
        ----------
        p : float
            The corrected p-value to convert.

        Returns
        -------
        str
            Significance stars: '***' for p < 0.001, '**' for p < 0.01, '*' for p < 0.05, '' otherwise.
        '''

        if pd.isna(p):
            return ''
        return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))

    def bar_colour(row: pd.Series) -> str:
        if not bool(row['reject_null']):
            return 'tab:gray'
        return 'tab:blue' if row['mean_diff'] > 0 else 'tab:red'

    palette = dict(zip(df['policy'], df.apply(bar_colour, axis=1)))

    plt.figure(figsize=(10, max(6, 0.35 * len(df))))
    ax = sns.barplot(data=df, x='policy_mean', y='policy', hue='policy', palette=palette, legend=False, orient='h')
    ax.axvline(reference_mean, color='black', linestyle='--', linewidth=1, label=f'{reference_policy} (reference)')

    for i, (mean_val, p_corr) in enumerate(zip(df['policy_mean'], df['pvalue_corrected'])):
        ax.text(
            mean_val + 0.005,
            i,
            f"{mean_val:.3f} {stars(p_corr)}",
            ha='left',
            va='center',
            fontsize=8,
        )

    legend_handles = [
        mlines.Line2D([0], [0], color='black', linestyle='--', label=f'{reference_policy} (reference)'),
        mlines.Line2D([0], [0], color='tab:red', lw=6, label='Significantly worse'),
        mlines.Line2D([0], [0], color='tab:blue', lw=6, label='Significantly better'),
        mlines.Line2D([0], [0], color='tab:gray', lw=6, label=f'Not significant (Holm, alpha={alpha:g})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)

    ax.set_title(f'{metric_label} per feature-ablation policy vs {reference_policy}')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Feature policy')
    plt.grid(True, axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_{metric_label.lower()}_significance_bars.png", dpi=300, bbox_inches='tight')
    plt.close()


def classify_policies_by_shortcut_rule(
        plot_df: pd.DataFrame,
        *,
        reference_policy: str = 'full_ocscore',
        risk_threshold: float = 20.0,
        bedroc_column: str = 'bedroc_mean',
        risk_column: str = 'shortcut_risk_max_pct',
    ) -> tuple[list[str], list[str]]:
    '''
    Split policies into retained and discarded by the shortcut-risk rule.

    A policy is discarded when it beats the reference policy's mean metric *and*
    concentrates more than ``risk_threshold`` percent of its total SHAP importance
    in a single feature: the gain is real but rides on one dominant feature.
    Policies that do not beat the reference are not candidates, so the rule does
    not apply to them and they belong to neither group.

    Parameters
    ----------
    plot_df : pd.DataFrame
        One row per policy, with a ``'policy'`` column plus ``bedroc_column`` and
        ``risk_column``.
    reference_policy : str
        Policy whose mean metric defines the candidacy cutoff. Default: 'full_ocscore'.
    risk_threshold : float
        Maximum single-feature SHAP share, in percent, tolerated in a candidate. Default: 20.0.
    bedroc_column : str
        Column holding the per-policy mean metric. Default: 'bedroc_mean'.
    risk_column : str
        Column holding the per-policy shortcut risk, in percent. Default: 'shortcut_risk_max_pct'.

    Returns
    -------
    tuple[list[str], list[str]]
        Retained (low-risk) and discarded (high-risk) candidate policy names.

    Raises
    ------
    ValueError
        If ``reference_policy`` is absent from ``plot_df``.
    '''

    reference_rows = plot_df[plot_df['policy'] == reference_policy]
    if reference_rows.empty:
        raise ValueError(f"reference policy {reference_policy!r} is not present in plot_df.")

    reference_metric = float(reference_rows[bedroc_column].iloc[0])
    candidates = plot_df[
        (plot_df[bedroc_column] > reference_metric) & (plot_df['policy'] != reference_policy)
    ]
    retained = candidates[candidates[risk_column] <= risk_threshold]['policy'].tolist()
    discarded = candidates[candidates[risk_column] > risk_threshold]['policy'].tolist()
    return retained, discarded


def _detect_x_break(
        values: Sequence[float],
        min_gap_share: float = 0.35,
        pad_share: float = 0.12,
    ) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    '''
    Find an empty x-region wide enough to justify a broken axis.

    A single far-out policy (a sanity-check control, typically) otherwise squeezes
    every other point into a fraction of the axis. When the widest gap between two
    consecutive values spans at least ``min_gap_share`` of the full range, the axis
    is worth splitting there.

    Parameters
    ----------
    values : sequence[float]
        The x values to be plotted.
    min_gap_share : float, optional
        Minimum share of the full range that the largest gap must span for a break
        to be worthwhile, by default 0.35.
    pad_share : float, optional
        Padding added around each panel's points, as a share of that panel's own
        span, by default 0.12.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]] | None
        ``(left_xlim, right_xlim)`` when a break is warranted, else None.
    '''

    ordered = sorted(float(v) for v in values)
    if len(ordered) < 3:
        return None

    full_range = ordered[-1] - ordered[0]
    if full_range <= 0:
        return None

    gaps = [(b - a, i) for i, (a, b) in enumerate(zip(ordered, ordered[1:]))]
    widest, index = max(gaps)
    if widest / full_range < min_gap_share:
        return None

    left_values = ordered[: index + 1]
    right_values = ordered[index + 1:]

    def limits(subset: list[float]) -> tuple[float, float]:
        span = subset[-1] - subset[0]
        pad = max(span * pad_share, widest * 0.05)
        return subset[0] - pad, subset[-1] + pad

    return limits(left_values), limits(right_values)


def plot_bedroc_vs_shortcut_risk_scatter(
        plot_df: pd.DataFrame,
        *,
        reference_policy: str = 'full_ocscore',
        good_policies: Optional[Sequence[str]] = None,
        bad_policies: Optional[Sequence[str]] = None,
        risk_threshold: float = 20.0,
        bedroc_column: str = 'bedroc_mean',
        risk_column: str = 'shortcut_risk_max_pct',
        label_column: Optional[str] = None,
        highlight_policy: Optional[str] = None,
        metric_label: str = 'BEDROC',
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = 'Shortcut risk\n(max. % of SHAP importance in a single feature)',
        legend_labels: Optional[Mapping[str, str]] = None,
        threshold_note: Optional[str] = None,
        zone_note: Optional[str] = None,
        reference_note: Optional[str] = None,
        highlight_note: Optional[str] = None,
        label_offsets: Optional[Mapping[str, tuple[float, float]]] = None,
        break_x_axis: bool = True,
        figsize: tuple[float, float] = (6.9, 4.6),
        dpi: int = 300,
        output_dir: str = 'plots',
    ) -> None:
    '''
    Scatter per-policy mean BEDROC against SHAP shortcut risk, under the shortcut rule.

    Point colors are *derived from the rule*, not from a curated list: a policy is
    discarded when it beats the reference policy and still concentrates more than
    ``risk_threshold`` percent of its SHAP importance in one feature. The two guide
    lines drawn on the axes are exactly the two conditions of that rule, and the
    shaded region is the quadrant they delimit, so the marking a reader sees and the
    geometry they read it from cannot disagree. Pass ``good_policies`` /
    ``bad_policies`` only to override the derived grouping.

    When one policy sits far from every other on the x axis (a low-signal control,
    typically), it compresses the interesting cluster into a fraction of the width.
    ``break_x_axis`` splits the axis across that empty region instead, keeping every
    point visible.

    Parameters
    ----------
    plot_df : pd.DataFrame
        One row per policy, with a ``'policy'`` column plus ``bedroc_column`` and
        ``risk_column`` (e.g. the output of ``SHAP.Dominance.aggregate_dominant_feature_risk``
        merged with per-policy mean BEDROC).
    reference_policy : str
        Policy plotted as the reference marker, with a dotted vertical guide at its
        metric value. Default: 'full_ocscore'.
    good_policies : sequence[str] | None, optional
        Overrides the retained group. Derived from the rule when None. Default: None.
    bad_policies : sequence[str] | None, optional
        Overrides the discarded group. Derived from the rule when None. Default: None.
    risk_threshold : float
        Shortcut-risk cutoff, in percent; also the horizontal guide line. Default: 20.0.
    bedroc_column : str
        Column with the per-policy mean metric value. Default: 'bedroc_mean'.
    risk_column : str
        Column with the per-policy shortcut-risk value. Default: 'shortcut_risk_max_pct'.
    label_column : str | None, optional
        Column used to annotate each point (falls back to ``'policy'``). Default: None.
    highlight_policy : str | None, optional
        Policy to call out with an arrow and a bold label (the final recommendation,
        typically). Default: None.
    metric_label : str
        Metric name, used in the default x label and in the output filename. Default: 'BEDROC'.
    title, xlabel, ylabel : str | None, optional
        Plot text. Override to render the figure in another language.
    legend_labels : mapping[str, str] | None, optional
        Legend text, keyed by ``'reference'``, ``'retained'``, ``'discarded'`` and
        ``'other'``. Override to render the figure in another language.
    threshold_note, zone_note, reference_note, highlight_note : str | None, optional
        In-plot annotations for the risk guide line, the discard quadrant, the
        reference guide line and the highlighted policy. Override to render the
        figure in another language.
    label_offsets : mapping[str, tuple[float, float]] | None, optional
        Per-policy ``(dx, dy)`` label offset override, in points, for policies whose
        default offset collides with a nearby marker or label. Default: None.
    break_x_axis : bool, optional
        Split the x axis across a wide empty region when one is present. Default: True.
    figsize : tuple[float, float]
        Figure size in inches, sized to be embedded at roughly 1:1. Default: (6.9, 4.6).
    dpi : int
        Figure DPI. Default: 300.
    output_dir : str
        Where to save the plot image. Default: 'plots'.
    '''

    df = plot_df.reset_index(drop=True)
    label_col = label_column or 'policy'

    derived_good, derived_bad = classify_policies_by_shortcut_rule(
        df,
        reference_policy = reference_policy,
        risk_threshold = risk_threshold,
        bedroc_column = bedroc_column,
        risk_column = risk_column,
    )
    good = set(good_policies if good_policies is not None else derived_good)
    bad = set(bad_policies if bad_policies is not None else derived_bad)

    reference_metric = float(df.loc[df['policy'] == reference_policy, bedroc_column].iloc[0])

    styles = {
        'reference': (COLOR_REFERENCE, 'D', 115, COLOR_TEXT),
        'retained': (COLOR_RETAINED, '^', 95, 'none'),
        'discarded': (COLOR_DISCARDED, 'v', 95, 'none'),
        'other': (COLOR_OTHER, 'o', 62, 'none'),
    }

    def group_of(policy: str) -> str:
        if policy == reference_policy:
            return 'reference'
        if policy in good:
            return 'retained'
        if policy in bad:
            return 'discarded'
        return 'other'

    x_break = _detect_x_break(df[bedroc_column].tolist()) if break_x_axis else None

    if x_break is None:
        fig, ax_right = plt.subplots(figsize = figsize, dpi = dpi)
        axes = [ax_right]
        ax_left = None
    else:
        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, sharey = True, figsize = figsize, dpi = dpi,
            gridspec_kw = {'width_ratios': [1, 9], 'wspace': 0.035},
        )
        axes = [ax_left, ax_right]

    for ax in axes:
        ax.axhline(risk_threshold, color = COLOR_TEXT_MUTED, lw = 1.0, ls = (0, (4, 2)), zorder = 1)
        ax.grid(True, color = COLOR_GRID, lw = 0.7, ls = ':', zorder = 0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize = 9, colors = COLOR_TEXT_MUTED, length = 0)

    # the discard quadrant: beats the reference AND sits above the risk threshold
    right_limits = x_break[1] if x_break is not None else ax_right.get_xlim()
    ax_right.add_patch(mpatches.Rectangle(
        (reference_metric, risk_threshold),
        right_limits[1] - reference_metric, 100.0 - risk_threshold,
        facecolor = COLOR_DISCARDED, alpha = 0.055, edgecolor = 'none', zorder = 0.5,
    ))
    ax_right.axvline(reference_metric, color = COLOR_TEXT, lw = 0.9, ls = (0, (2, 2)), alpha = 0.55, zorder = 1)

    split_at = x_break[0][1] if x_break is not None else None
    for _, row in df.iterrows():
        color, marker, size, edge = styles[group_of(row['policy'])]
        ax = ax_left if (split_at is not None and row[bedroc_column] <= split_at) else ax_right
        ax.scatter(
            row[bedroc_column], row[risk_column], color = color, marker = marker, s = size,
            edgecolor = edge, linewidth = 1.1, zorder = 3,
        )
        offset = (label_offsets or {}).get(row['policy'], (7, 5))
        ax.annotate(
            str(row[label_col]), (row[bedroc_column], row[risk_column]),
            textcoords = 'offset points', xytext = offset, fontsize = 8.5, color = COLOR_TEXT,
            fontweight = 'bold' if row['policy'] == highlight_policy else 'normal', zorder = 4,
        )

    if highlight_policy is not None and highlight_note:
        target = df[df['policy'] == highlight_policy]
        if not target.empty:
            row = target.iloc[0]
            ax_right.annotate(
                highlight_note, (row[bedroc_column], row[risk_column]),
                textcoords = 'offset points', xytext = (16, 10), fontsize = 7.8,
                color = COLOR_RETAINED, fontweight = 'bold', zorder = 4,
                arrowprops = dict(arrowstyle = '-', color = COLOR_RETAINED, lw = 0.9,
                                  shrinkA = 0, shrinkB = 4, alpha = 0.8),
            )

    # inverted: the higher the point, the more distributed the explanation
    axes[0].set_ylim(100, 0)

    if x_break is not None:
        ax_left.set_xlim(*x_break[0])
        ax_right.set_xlim(*x_break[1])
        ax_left.set_xticks([round(df[bedroc_column].min(), 3)])
        ax_left.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)
        ax_right.tick_params(left = False)
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_color(COLOR_GRID)
        ax_left.spines['left'].set_color(COLOR_GRID)
        ax_right.spines['right'].set_visible(False)

        # diagonal break marks straddling the two panels
        mark = dict(
            marker = [(-1, -0.9), (1, 0.9)], markersize = 7, linestyle = 'none',
            color = COLOR_TEXT_MUTED, mec = COLOR_TEXT_MUTED, mew = 1.1, clip_on = False,
        )
        ax_left.plot([1, 1], [0, 1], transform = ax_left.transAxes, **mark)
        ax_right.plot([0, 0], [0, 1], transform = ax_right.transAxes, **mark)
    else:
        for spine in ('top', 'right'):
            ax_right.spines[spine].set_visible(False)
        for spine in ('bottom', 'left'):
            ax_right.spines[spine].set_color(COLOR_GRID)

    if threshold_note:
        ax_right.text(
            ax_right.get_xlim()[1], risk_threshold - 1.5, threshold_note,
            fontsize = 7.6, color = COLOR_TEXT_MUTED, ha = 'right', va = 'bottom', style = 'italic',
        )
    if zone_note:
        ax_right.text(
            ax_right.get_xlim()[1], 97.5, zone_note, fontsize = 7.8,
            color = COLOR_DISCARDED, ha = 'right', va = 'bottom', fontweight = 'bold', alpha = 0.85,
        )
    ax_right.text(
        reference_metric, -2.5,
        reference_note if reference_note is not None else f'{metric_label} of the reference model',
        fontsize = 7.6, color = COLOR_TEXT_MUTED, ha = 'center', va = 'bottom',
        style = 'italic', clip_on = False,
    )

    ax_right.set_xlabel(xlabel or f'{metric_label} (test)', fontsize = 10.5, color = COLOR_TEXT)
    ax_right.xaxis.set_label_coords(0.44, -0.11)
    axes[0].set_ylabel(ylabel, fontsize = 9.5, color = COLOR_TEXT)
    fig.suptitle(
        title or f'Ranking performance vs shortcut risk ({len(df)} policies)',
        fontsize = 11.0, color = COLOR_TEXT, fontweight = 'bold', x = 0.135, ha = 'left', y = 0.975,
    )

    text = {
        'reference': f'Full model ({reference_policy}, reference)',
        'retained': f'Retained: beat the reference, risk <= {risk_threshold:g}% (n={len(good)})',
        'discarded': f'Discarded: beat the reference, risk > {risk_threshold:g}% (n={len(bad)})',
        'other': 'Do not beat the reference (rule does not apply)',
        **dict(legend_labels or {}),
    }
    legend_handles = [
        mlines.Line2D([], [], color = COLOR_REFERENCE, marker = 'D', ls = 'none', ms = 8,
                      mec = COLOR_TEXT, label = text['reference']),
        mlines.Line2D([], [], color = COLOR_DISCARDED, marker = 'v', ls = 'none', ms = 8.5,
                      label = text['discarded']),
        mlines.Line2D([], [], color = COLOR_RETAINED, marker = '^', ls = 'none', ms = 8.5,
                      label = text['retained']),
        mlines.Line2D([], [], color = COLOR_OTHER, marker = 'o', ls = 'none', ms = 7,
                      label = text['other']),
    ]
    fig.subplots_adjust(left = 0.135, right = 0.985, top = 0.875, bottom = 0.265)
    fig.legend(
        handles = legend_handles, loc = 'lower center', frameon = False, fontsize = 7.8,
        labelcolor = COLOR_TEXT_MUTED, ncol = 2, bbox_to_anchor = (0.5, -0.005),
        handlelength = 1.2, columnspacing = 1.2, labelspacing = 0.4,
    )

    fig.savefig(f"{output_dir}/ablation_{metric_label.lower()}_vs_shortcut_risk_scatter.png", dpi = dpi)
    plt.close(fig)


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
            ocerror.Error.data_not_found("Games-Howell dataframe must contain a p-value column (pval, pval_corr, etc.)")
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

    df_plot = df.reset_index(drop = True)
    y_values = df_plot[y_col].to_numpy()
    p_values = df_plot['pval'].to_numpy()
    for i, (y_val, p_val) in enumerate(zip(y_values, p_values)):
        ax.text(
            y_val + (0.01 if y_val >= 0 else -0.01),
            i,
            f"{y_val:.3f}  (p={p_val:.2e}) {stars(p_val)}",
            ha='left' if y_val >= 0 else 'right',
            va='center',
            fontsize=8,
        )

    ax.set_title(f'Games-Howell pairwise differences — {metric}')
    ax.set_xlabel(f'Difference in {metric}')
    ax.set_ylabel('Pair (A vs B)')
    plt.grid(True, axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/games_howell_bar_{metric}.png", dpi=300)
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
        Whether to include consensus methodologies (any label ending with "consensus").
    '''

    plot_df = df.copy()
    if not show_simple_consensus:
        plot_df = plot_df[~plot_df['Methodology'].str.endswith('consensus', na = False)]

    plt.figure(figsize = (16, 12))
    mean_line_rmse, mean_line_auc = None, None

    for i, metric in enumerate(['RMSE', 'AUC']):
        plt.subplot(2, 1, i + 1)
        with warnings.catch_warnings():
            # Seaborn currently forwards a deprecated Matplotlib `vert` kwarg
            # internally in some versions; silence this third-party warning.
            warnings.filterwarnings(
                "ignore",
                message = "vert: bool will be deprecated in a future version.*",
                category = PendingDeprecationWarning,
            )
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
        ocerror.Error.data_not_found("Games-Howell dataframe must contain a p-value column (pval, pval_corr, etc.)")
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
        ocerror.Error.value_error(f"Invalid orientation: '{orientation}'. Must be 'horizontal' or 'vertical'.")
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
        df_auc_ge = df[df['AUC_category'] == '>= 0.5']
        if not df_auc_ge.empty:
            sns.scatterplot(
                data = df_auc_ge,
                x = 'RMSE',
                y = 'AUC',
                hue = 'Methodology',
                palette = colour_mapping,
                alpha = alpha,
                s = 30,
                legend = False,
            )

        # Scatter for AUC < 0.5
        df_auc_lt = df[df['AUC_category'] == '< 0.5']
        if not df_auc_lt.empty:
            sns.scatterplot(
                data = df_auc_lt,
                x ='RMSE',
                y ='AUC',
                hue = 'Methodology',
                palette = colour_mapping,
                alpha = alpha,
                s = 50,
                marker = '*',
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
