#!/usr/bin/env python3

# Description
###############################################################################
'''
Plotting utilities for feature impact analysis (2xK contingencies, residuals, chi-square share).

They are imported as:

import OCDocker.OCScore.Analysis.Plotting.ImpactPlots as ocimpactplots

This module complements `OCDocker.OCScore.Analysis.Impact` with visual summaries:
- Proportion deltas across categories
- Residuals lollipop for presence row only
- Per-category chi-square contribution bars
- Composite report (2x2 panel)
- Residuals matrix heatmap across features
'''

from __future__ import annotations
from typing import Iterable, Optional, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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


def prop_delta_2xk(contingency: pd.DataFrame) -> pd.DataFrame:
    '''For a 2xK contingency table, return per-category proportion deltas.

    delta = prop(feature==1) - prop(feature==0)

    Parameters
    ----------
    contingency : pd.DataFrame
        2xK contingency table with rows representing feature absence/presence (0/1)
        and columns representing metric categories.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with ['MetricCategory', 'prop_delta'].
    '''

    # Validate shape: function expects exactly 2 rows (absence/presence)
    if contingency.shape[0] != 2:
        # User-facing error: invalid contingency table shape
        ocerror.Error.value_error(f"Invalid contingency table shape: expected 2 rows, got {contingency.shape[0]}. Contingency must be 2xK.") # type: ignore
        raise ValueError("contingency must be 2xK.")
    props = contingency.div(contingency.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    
    # Identify presence/absence row keys; tolerate string/int indices
    if 1 in contingency.index:
        delta = props.loc[1] - props.loc[0]
    elif '1' in contingency.index:
        delta = props.loc['1'] - props.loc['0']
    else:
        # Fallback: use last minus first row when explicit indices are unknown
        delta = props.iloc[1] - props.iloc[0]
    
    # Return table with explicit column names
    return delta.to_frame("prop_delta").reset_index(names="MetricCategory") # type: ignore


def plot_prop_delta(contingency: pd.DataFrame, title: str = 'Proportion delta (1 - 0)', outpath: Optional[str] = None) -> None:
    '''Diverging bar chart of proportion deltas across metric categories.

    Parameters
    ----------
    contingency : pd.DataFrame
        2xK contingency table.
    title : str, optional
        Plot title. Default is 'Proportion delta (1 - 0)'.
    outpath : str | None, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    '''

    # Compute per-category Δ proportion
    df = prop_delta_2xk(contingency)
    # Create the diverging bar plot
    # Draw lollipop plot (horizontal stem + point)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="prop_delta", y="MetricCategory", orient="h",
                palette=df["prop_delta"].map(lambda v: "tab:red" if v>0 else "tab:blue")) # type: ignore
    # Draw reference at zero and annotate values
    plt.axvline(0, ls="--", c="k", lw=1)
    for _, r in df.iterrows():
        plt.text(r["prop_delta"] + (0.01 if r["prop_delta"]>=0 else -0.01),
                 r["MetricCategory"], f"{r['prop_delta']:.2f}",
                 va="center", ha="left" if r["prop_delta"]>=0 else "right")
    # Labeling and layout
    plt.title(title)
    plt.xlabel("Proportion delta (feature=1 minus feature=0)")
    plt.ylabel("")
    plt.tight_layout()
    # Optionally save to file
    # Optionally save to file
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()


def plot_residuals_lollipop(residuals_df: pd.DataFrame,
                            feature_name: str,
                            presence_level: Union[int, str] = 1,
                            title_suffix: str = 'Standardized residuals (feature=1)',
                            outpath: Optional[str] = None) -> None:
    '''Lollipop plot of standardized residuals for the 'presence' row only.

    Draw reference lines at ±2 and ±3.

    Parameters
    ----------
    residuals_df : pd.DataFrame
        Contingency residuals table (rows correspond to presence/absence).
    feature_name : str
        Feature name for the plot title.
    presence_level : int | str, optional
        Row key for presence (default 1). Falls back to last row if not found.
    title_suffix : str, optional
        Suffix to include in the plot title.
    outpath : str | None, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    '''

    # Select the 'presence' row using provided key (tolerate int/str),
    # fallback to last row when keys are not found
    row = None
    for key in (presence_level, str(presence_level)):
        if key in residuals_df.index:
            row = residuals_df.loc[key]
            break
    if row is None:
        row = residuals_df.iloc[-1]  # fallback: last row

    # Sort categories for visual stability and extract arrays
    s = row.sort_index()
    x = s.values
    cats = s.index.tolist()

    # Create the diverging bar plot
    plt.figure(figsize=(7, 4))
    y = np.arange(len(cats))
    plt.hlines(y, 0, x, lw=2) # type: ignore
    plt.plot(x, y, "o") # type: ignore
    for xi, yi, cat in zip(x, y, cats):
        plt.text(xi + (0.1 if xi>=0 else -0.1), yi, f"{xi:.2f}",
                 va="center", ha="left" if xi>=0 else "right")
    # Reference thresholds at ±2 and ±3 (common standardized residual heuristics)
    for thr, ls in [(2, "--"), (3, ":")]:
        plt.axvline(+thr, ls=ls, c="red", lw=1)
        plt.axvline(-thr, ls=ls, c="red", lw=1)
    # Labeling and layout
    plt.yticks(y, cats)
    plt.xlabel("Standardized residual")
    plt.title(f"{feature_name} — {title_suffix}")
    plt.tight_layout()
    # Optionally save to file
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()


def plot_chi2_contrib(contingency: pd.DataFrame,
                      feature_name: str,
                      presence_level: Union[int, str] = 1,
                      title: Optional[str] = None,
                      outpath: Optional[str] = None) -> None:
    '''Bar plot of per-category chi-square contributions for the presence row only.

    Contribution = (O-E)^2 / E ; normalized to percentage.

    Parameters
    ----------
    contingency : pd.DataFrame
        2xK contingency table.
    feature_name : str
        Feature name for the plot title.
    presence_level : int | str, optional
        Row key for presence (default 1). Falls back to last row if not found.
    title : str | None, optional
        Title override. If None, an informative default is used.
    outpath : str | None, optional
        If provided, saves the figure to this path.

    Returns
    -------
    None
    '''

    # Compute row/column totals and the expected frequencies under independence
    total = contingency.values.sum()
    row_sum = contingency.sum(axis=1).values[:, None] # type: ignore
    col_sum = contingency.sum(axis=0).values[None, :] # type: ignore
    expected = (row_sum @ col_sum) / total
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    # Select presence row robustly (int/str), fallback to last row
    row_key = presence_level if presence_level in contingency.index else \
              (str(presence_level) if str(presence_level) in contingency.index else contingency.index[-1])

    # Compute chi-square contribution per category and normalize to percentage share
    obs = contingency.loc[row_key].astype(float)
    exp = expected_df.loc[row_key].astype(float)
    contrib = ((obs - exp) ** 2) / exp
    share = 100 * contrib / contrib.sum()

    # Build a tidy dataframe for plotting
    df = share.reset_index()
    df.columns = ["MetricCategory", "Chi2SharePct"]

    # Render horizontal bar plot with annotations
    plt.figure(figsize=(7,4))
    sns.barplot(data=df, x="Chi2SharePct", y="MetricCategory", orient="h", color="steelblue")
    for _, r in df.iterrows():
        plt.text(r["Chi2SharePct"] + 0.5, r["MetricCategory"], f"{r['Chi2SharePct']:.1f}%", va="center")
    plt.xlabel("Share of Chi-square (%)")
    plt.ylabel("")
    plt.title(title or f"{feature_name} — per-category χ² contribution (feature=1)")
    plt.tight_layout()
    # Optionally save to file
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()


def feature_report_2xk(feature: str,
                       contingency: pd.DataFrame,
                       residuals_df: pd.DataFrame,
                       p_value: Optional[float] = None,
                       outpath: str = 'feature_report.png') -> None:
    '''Compose a 2x2 figure for a single feature.

    Layout:
      [0,0] proportion delta; [0,1] residual lollipop;
      [1,0] chi2 contribution; [1,1] legend/text box.

    Parameters
    ----------
    feature : str
        Feature name for titles.
    contingency : pd.DataFrame
        2xK contingency table.
    residuals_df : pd.DataFrame
        Residuals table compatible with the contingency categories.
    p_value : float | None, optional
        Optional p-value to include in the text box.
    outpath : str, optional
        Output image path (default 'feature_report.png').

    Returns
    -------
    None
    '''

    # Create a 2x2 figure grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) Proportion delta
    # Compute per-category Δ proportion
    df = prop_delta_2xk(contingency)
    sns.barplot(data=df, x="prop_delta", y="MetricCategory", orient="h",
                palette=df["prop_delta"].map(lambda v: "tab:red" if v>0 else "tab:blue"), ax=axes[0,0]) # type: ignore
    axes[0,0].axvline(0, ls="--", c="k", lw=1)
    axes[0,0].set_title("Proportion delta (1 - 0)")
    axes[0,0].set_xlabel("Δ proportion")
    axes[0,0].set_ylabel("")

    # (0,1) Residuals lollipop (feature=1)
    s = None
    for key in (1, '1'):
        if key in residuals_df.index:
            s = residuals_df.loc[key]
            break
    if s is None:
        s = residuals_df.iloc[-1]
    xv = s.values
    yv = np.arange(len(s))
    axes[0,1].hlines(yv, 0, xv, lw=2)
    axes[0,1].plot(xv, yv, "o")
    for xi, yi, cat in zip(xv, yv, s.index.tolist()):
        axes[0,1].text(xi + (0.1 if xi>=0 else -0.1), yi, f"{xi:.2f}",
                       va="center", ha="left" if xi>=0 else "right")
    # Reference thresholds at ±2 and ±3 (common standardized residual heuristics)
    for thr, ls in [(2, "--"), (3, ":")]:
        axes[0,1].axvline(+thr, ls=ls, c="red", lw=1)
        axes[0,1].axvline(-thr, ls=ls, c="red", lw=1)
    axes[0,1].set_yticks(yv)
    axes[0,1].set_yticklabels(s.index.tolist())
    axes[0,1].set_xlabel("Standardized residual")
    axes[0,1].set_title("Residuals (feature=1)")

    # (1,0) Chi-square contribution per category
    total = contingency.values.sum()
    row_sum = contingency.sum(axis=1).values[:, None] # type: ignore
    col_sum = contingency.sum(axis=0).values[None, :] # type: ignore
    expected = (row_sum @ col_sum) / total
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    row_key = 1 if 1 in contingency.index else ('1' if '1' in contingency.index else contingency.index[-1])
    # Compute chi-square contribution per category and normalize to percentage share
    obs = contingency.loc[row_key].astype(float)
    exp = expected_df.loc[row_key].astype(float)
    contrib = ((obs - exp) ** 2) / exp
    share = 100 * contrib / contrib.sum()
    sns.barplot(x=share.values, y=share.index, color="steelblue", orient="h", ax=axes[1,0])
    for yi, (cat, val) in enumerate(share.items()):
        axes[1,0].text(val + 0.5, yi, f"{val:.1f}%", va="center")
    axes[1,0].set_title("Per-category χ² contribution (feature=1)")
    axes[1,0].set_xlabel("Share (%)")
    axes[1,0].set_ylabel("")

    # (1,1) Text panel: how to read and optional p-value
    axes[1,1].axis('off')
    txt = "How to read:\n" \
          "• Right bars in Δ proportion ⇒ category over-represented when feature=1.\n" \
          "• Residuals > +2 or < -2 are notable.\n" \
          "• χ² share pinpoints which bins drive the association."
    if p_value is not None:
        txt += f"\n\np-value: {p_value:.2e}"
    axes[1,1].text(0, 1, txt, va="top")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def residuals_matrix_from_dict(residuals_dict: Dict[str, pd.DataFrame],
                               presence_level: Union[int, str] = 1) -> pd.DataFrame:
    '''Build a matrix (features x categories) with residuals for presence row only.

    Parameters
    ----------
    residuals_dict : dict[str, pd.DataFrame]
        Mapping feature -> residuals DataFrame (rows=presence/absence; cols=categories).
    presence_level : int | str, optional
        Row key for presence (default 1). Falls back to last row if not found.

    Returns
    -------
    pd.DataFrame
        Matrix with features as rows and categories as columns.
    '''
    # Build a dict of rows by selecting the 'presence' row from each residuals DF
    rows = {}
    for feat, resdf in residuals_dict.items():
        row = None
        for key in (presence_level, str(presence_level)):
            if key in resdf.index:
                row = resdf.loc[key]
                break
        if row is None:
            row = resdf.iloc[-1]
        rows[feat] = row
    # Assemble into a features x categories matrix
    return pd.DataFrame(rows).T


def plot_residuals_matrix(residuals_dict: Dict[str, pd.DataFrame],
                          presence_level: Union[int, str] = 1,
                          order_by: str = 'maxabs',
                          outpath: str = 'residuals_matrix.png') -> None:
    '''Heatmap of features (rows) vs metric categories (columns), values = residuals (feature=1).

    Parameters
    ----------
    residuals_dict : dict[str, pd.DataFrame]
        Mapping feature -> residuals DataFrame.
    presence_level : int | str, optional
        Row key for presence (default 1).
    order_by : str, optional
        How to order features. Options: 'maxabs' (default) or 'chi2' (reserved for future use).
    outpath : str, optional
        Output image path (default 'residuals_matrix.png').

    Returns
    -------
    None
    '''
    
    # Build matrix and order rows by the chosen criterion
    mat = residuals_matrix_from_dict(residuals_dict, presence_level=presence_level)
    idx = mat.abs().max(axis=1).sort_values(ascending=False).index if order_by == 'maxabs' else mat.index
    mat = mat.loc[idx]
    plt.figure(figsize=(max(6, mat.shape[1]*1.6), max(6, mat.shape[0]*0.3)))
    ax = sns.heatmap(mat, annot=False, cmap="coolwarm", center=0, cbar_kws={'label': 'Std residual'})
    # optional significance markers (|res|>=2)
    sig = (mat.abs() >= 2)
    ys, xs = np.where(sig.values)
    for y, x in zip(ys, xs):
        ax.text(x+0.5, y+0.5, "•", ha="center", va="center", color="k", fontsize=9)
    plt.title("Residuals (feature=1) across features")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
