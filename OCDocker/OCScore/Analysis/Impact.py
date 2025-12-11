#!/usr/bin/env python3

# Description
###############################################################################
''' This module provides utilities to summarize and visualize feature impact
using Net Benefit Score (NBS) and related statistics.

It exposes high-level functions:

- build_impact_overview: build a table with NBS, direction, strength and stats
- plot_impact_arrows_inline_labels: render arrow plot with inline labels
- get_neutral_features: list neutral features by |NBS| < tau or Direction == 'neutral'

Import as:

import OCDocker.OCScore.Analysis.Impact as ocimpact
'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# Helpers
###############################################################################


def _strength_from_v(v: float) -> str:
    '''
    Map Cramér's V value to a qualitative strength label.

    Parameters
    ----------
    v : float
        Cramér's V statistic value.

    Returns
    -------
    str
        One of: 'unknown', 'none', 'weak', 'moderate', 'strong', 'very strong'.
    '''

    if pd.isna(v):
        return "unknown"
    if v < 0.10:
        return "none"
    if v < 0.20:
        return "weak"
    if v < 0.30:
        return "moderate"
    if v < 0.50:
        return "strong"
    return "very strong"


def _strength_from_nbs_norm(nbs_norm: float, thresholds: Sequence[float] = (0.10, 0.20, 0.35)) -> str:
    '''
    Bucketize |NBS_norm| into qualitative strength classes.

    Parameters
    ----------
    nbs_norm : float
        Net Benefit Score normalized to [-1, 1].
    thresholds : Sequence[float]
        Cutoffs applied to |NBS_norm| to define 'weak', 'moderate', 'strong'.

    Returns
    -------
    str
        One of: 'none', 'weak', 'moderate', 'strong', 'very strong'.
    '''

    a = abs(float(nbs_norm))
    if a == 0:
        return "none"
    if a < thresholds[0]:
        return "weak"
    if a < thresholds[1]:
        return "moderate"
    if a < thresholds[2]:
        return "strong"
    return "very strong"


def _beneficial_categories(metric: str, categories: Iterable[str], custom: Optional[Iterable[str]] = None) -> set[str]:
    '''
    Decide which ordered categories are beneficial for the given metric.

    Parameters
    ----------
    metric : str
        Metric name ('AUC', 'RMSE', or other).
    categories : Iterable[str]
        Ordered category labels from the contingency table columns.
    custom : Optional[Iterable[str]]
        Explicit set of beneficial categories. If provided, overrides defaults.

    Returns
    -------
    set[str]
        Set of category labels considered beneficial.
    '''

    cats = [str(c) for c in categories]
    if custom is not None:
        return set(map(str, custom))

    m = metric.strip().upper()
    if m == "AUC":
        good = {c for c in cats if "high" in c.lower()}
    elif m == "RMSE":
        good = {c for c in cats if "low" in c.lower()}
    else:
        k = len(cats)
        good = set(cats[k // 2 :])
    return good if good else set(cats[-max(1, len(cats) // 2) :])


def _proportion_delta(contingency: pd.DataFrame, presence_level: Union[int, str] = 1) -> pd.Series:
    '''
    Compute Δp(c) = p(c | feature=1) - p(c | feature=0) for each category c.

    Parameters
    ----------
    contingency : pd.DataFrame
        Contingency table with rows as feature presence (0/1) and columns as categories.
    presence_level : Union[int, str]
        Row key representing presence (defaults to 1). Falls back if not present.

    Returns
    -------
    pd.Series
        Series of deltas indexed by category label.
    '''

    cont = contingency.copy()
    props = cont.div(cont.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    if presence_level in cont.index:
        k1 = presence_level
    elif str(presence_level) in cont.index:
        k1 = str(presence_level)
    else:
        k1 = cont.index[-1]

    if 0 in cont.index:
        k0 = 0
    elif "0" in cont.index:
        k0 = "0"
    else:
        k0 = cont.index[0]

    return (props.loc[k1] - props.loc[k0]).astype(float)


def _net_benefit(delta: pd.Series, beneficial: set[str]) -> float:
    '''
    Compute Net Benefit Score in [-1, 1] from Δp and beneficial categories.

    Parameters
    ----------
    delta : pd.Series
        Δp per category (output of _proportion_delta).
    beneficial : set[str]
        Set of categories deemed beneficial.

    Returns
    -------
    float
        Net Benefit Score (NBS).
    '''

    idx = delta.index.astype(str)
    good = [c for c in idx if c in beneficial]
    bad = [c for c in idx if c not in beneficial]
    return float(delta[good].sum() - delta[bad].sum())






# Public API
###############################################################################


def build_impact_overview(
    chi_df: pd.DataFrame,
    contingency_dict: dict[str, pd.DataFrame],
    metric: str,
    presence_level: Union[int, str] = 1,
    beneficial_custom: Optional[Iterable[str]] = None,
    tau: float = 0.05,
) -> pd.DataFrame:
    '''
    Build a clear impact table with NBS, direction, strength and stats.

    Parameters
    ----------
    chi_df : pd.DataFrame
        DataFrame with chi-square outcomes, requires at least columns
        ['Feature', "Cramér's V", 'Chi2 Statistic', 'p-value'].
    contingency_dict : dict[str, pd.DataFrame]
        Mapping from feature -> contingency table with rows as presence (0/1)
        and columns as ordered categories (strings).
    metric : str
        Metric name used to identify beneficial categories ('AUC' or 'RMSE').
    presence_level : Union[int, str], optional
        Row key considered as presence (default: 1). If not found, falls back.
    beneficial_custom : Optional[Iterable[str]], optional
        Explicit set of beneficial categories to use instead of defaults.
    tau : float, optional
        Tolerance to classify neutral direction by |NBS| < tau (default: 0.05).

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with columns: 'Feature', 'NBS', 'Direction', 'Strength',
        'Chi2', 'p-value', 'CramersV', 'FavoredCategory', 'HurtCategory',
        '|NBS|', 'NegLog10P'.
    '''

    any_cont = next(iter(contingency_dict.values()))
    categories = any_cont.columns.astype(str).tolist()
    beneficial = _beneficial_categories(metric, categories, custom=beneficial_custom)

    rows = []
    for _, r in chi_df.iterrows():
        feat = r['Feature']
        cont = contingency_dict.get(feat)
        if cont is None or cont.empty:
            rows.append({
                'Feature': feat,
                'NBS': np.nan,
                'Direction': 'neutral',
                'Strength': 'unknown',
                'Chi2': r.get('Chi2 Statistic', np.nan),
                'p-value': r.get('p-value', np.nan),
                'CramersV': r.get("Cramér's V", np.nan),
                'FavoredCategory': None,
                'HurtCategory': None,
            })
            continue

        delta = _proportion_delta(cont, presence_level=presence_level)
        nbs = _net_benefit(delta, beneficial)

        if abs(nbs) < tau:
            direction = 'neutral'
        else:
            direction = 'positive' if nbs > 0 else 'negative'

        favored = delta.idxmax()
        hurt = delta.idxmin()

        rows.append({
            'Feature': feat,
            'NBS': nbs,
            'Direction': direction,
            'Strength': _strength_from_v(r.get("Cramér's V", np.nan)),
            'Chi2': r.get('Chi2 Statistic', np.nan),
            'p-value': r.get('p-value', np.nan),
            'CramersV': r.get("Cramér's V", np.nan),
            'FavoredCategory': favored,
            'HurtCategory': hurt,
        })

    out = pd.DataFrame(rows)
    out['|NBS|'] = out['NBS'].abs()
    out['NegLog10P'] = -np.log10(np.clip(out['p-value'].astype(float), 1e-300, 1.0))
    return out.sort_values(['Direction', '|NBS|', 'NegLog10P'], ascending=[True, False, False])


def get_neutral_features(impact_df: pd.DataFrame, tau: float = 0.05) -> list[str]:
    '''
    Return a sorted list of neutral features by Direction or |NBS| < tau.

    Parameters
    ----------
    impact_df : pd.DataFrame
        DataFrame returned by build_impact_overview, with 'NBS' and 'Direction'.
    tau : float, optional
        Neutrality threshold on original NBS scale (default: 0.05).

    Returns
    -------
    list[str]
        Sorted list of neutral feature names.
    '''

    if 'Direction' in impact_df.columns:
        return (
            impact_df.loc[impact_df['Direction'] == 'neutral', 'Feature']
            .astype(str)
            .sort_values()
            .tolist()
        )
    return (
        impact_df.loc[impact_df['NBS'].abs() < tau, 'Feature']
        .astype(str)


        .sort_values()
        .tolist()
    )


def plot_impact_arrows_inline_labels(
    impact_df: pd.DataFrame,
    title: str,
    outpath: Optional[str] = None,
    tau: float = 0.05,
    thresholds: Sequence[float] = (0.10, 0.20, 0.35),
    xpad: float = 0.025,
    height_per_feature: float = 0.42,
    max_height: float = 28.0,
    font_size: int = 10,
) -> None:
    '''
    Render an arrow plot with inline feature labels based on NBS.

    Parameters
    ----------
    impact_df : pd.DataFrame
        DataFrame with columns ['Feature','NBS'] and optionally 'Direction'.
    title : str
        Plot title.
    outpath : Optional[str], optional
        Output image path. If None, the figure is not saved to disk.
    tau : float, optional
        Neutrality threshold on original NBS scale (default: 0.05).
    thresholds : Sequence[float], optional
        Thresholds for marker strength derived from |NBS_norm| (default: 0.10, 0.20, 0.35).
    xpad : float, optional
        Horizontal text offset relative to marker (default: 0.025).
    height_per_feature : float, optional
        Figure height contribution per feature (default: 0.42).
    max_height : float, optional
        Maximum figure height (default: 28.0).
    font_size : int, optional
        Font size for labels (default: 10).
    '''

    df = impact_df.copy()

    # Normalize to [-1, 1] for visualization
    df['NBS_norm'] = (df['NBS'] / 2.0).clip(-1.0, 1.0)
    tau_norm = tau / 2.0

    if 'Direction' not in df.columns:
        df['Direction'] = np.where(
            df['NBS'] > +tau, 'positive', np.where(df['NBS'] < -tau, 'negative', 'neutral')
        )

    df['Strength'] = df['NBS_norm'].apply(lambda v: _strength_from_nbs_norm(v, thresholds))

    color_map = {'positive': '#2ca02c', 'negative': '#d62728', 'neutral': '#7f7f7f'}
    marker_map = {'none': 'o', 'weak': 'o', 'moderate': 's', 'strong': 'D', 'very strong': 'P'}
    size_map = {'none': 40, 'weak': 60, 'moderate': 90, 'strong': 120, 'very strong': 150}
    alpha_map = {'none': 0.35, 'weak': 0.5, 'moderate': 0.7, 'strong': 0.9, 'very strong': 0.95}

    df['Color'] = df['Direction'].map(color_map).fillna('#7f7f7f')
    df['Marker'] = df['Strength'].map(marker_map).fillna('o')
    df['Size'] = df['Strength'].map(size_map).fillna(60)
    df['Alpha'] = df['Strength'].map(alpha_map).fillna(0.6)

    df = df.sort_values('NBS_norm').reset_index(drop=True)

    fig_h = min(max(6.0, height_per_feature * len(df)), max_height)
    plt.figure(figsize=(12, fig_h))

    for i, r in df.iterrows():
        plt.plot([0, r['NBS_norm']], [i, i], color=r['Color'], linewidth=2, alpha=r['Alpha'])
        plt.scatter(
            r['NBS_norm'],
            i,
            s=r['Size'],
            c=r['Color'],
            marker=r['Marker'],
            edgecolor='k',
            linewidth=0.4,
            zorder=3,
        )

    for i, r in df.iterrows():
        x = r['NBS_norm'] + (xpad if r['NBS_norm'] >= 0 else -xpad)
        ha = 'left' if r['NBS_norm'] >= 0 else 'right'
        plt.text(x, i, str(r['Feature']), va='center', ha=ha, fontsize=font_size)

    plt.yticks([], [])
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(+tau_norm, color='k', linestyle=':', linewidth=1)
    plt.axvline(-tau_norm, color='k', linestyle=':', linewidth=1)

    plt.xlabel("Net Benefit Score (normalized: −1 worse ← 0 → better +1)")
    plt.xlim(-1.05, 1.05)
    plt.title(title)

    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    leg_dir = [
        mpatches.Patch(color=color, label=lbl)
        for lbl, color in [('positive', '#2ca02c'), ('negative', '#d62728'), ('neutral', '#7f7f7f')]
    ]
    leg_str = [
        mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=8, label='none/weak'),
        mlines.Line2D([], [], color='k', marker='s', linestyle='None', markersize=8, label='moderate'),
        mlines.Line2D([], [], color='k', marker='D', linestyle='None', markersize=8, label='strong'),
        mlines.Line2D([], [], color='k', marker='P', linestyle='None', markersize=8, label='very strong'),
    ]
    lg1 = plt.legend(handles=leg_dir, title='direction', loc='upper left', frameon=False, fontsize=9)
    plt.gca().add_artist(lg1)
    plt.legend(handles=leg_str, title='strength (symbols)', loc='upper right', frameon=False, fontsize=9)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
        plt.close()
