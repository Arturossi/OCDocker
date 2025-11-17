from __future__ import annotations

# Description
###############################################################################
'''
Ranking metrics and tables for Test2-style analyses (no CLI, no I/O).

They are imported as:

import OCDocker.OCScore.Analysis.RankingMetrics as ocrank

This module consolidates ROC/PR/EF-ROC with bootstrap CIs and provides tabular
outputs consistent with your existing analysis style.

Public API (metrics/tables):
- roc_auc_per_target
- pr_auc_per_target
- efroc_per_target
- roc_auc_pooled
- pr_auc_pooled
- efroc_pooled
- build_test2_tables
- build_summary_table
'''

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


__all__ = [
    "BootstrapCI",
    "roc_auc_per_target",
    "pr_auc_per_target",
    "efroc_per_target",
    "roc_auc_pooled",
    "pr_auc_pooled",
    "efroc_pooled",
    "build_test2_tables",
    "build_summary_table",
]


# Classes
###############################################################################

# Functions
###############################################################################
@dataclass
class BootstrapCI:
    '''Bootstrap confidence interval dataclass.
    
    Attributes
    ----------
    point : float
        The point estimate (mean/median) of the metric.
    low : float
        The lower bound of the confidence interval (e.g., 2.5th percentile).
    high : float
        The upper bound of the confidence interval (e.g., 97.5th percentile).
    '''

    point: float
    low: float
    high: float


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _to_binary(y: pd.Series, positive_label: Optional[Union[str, int]]) -> np.ndarray:
    '''Map labels to 0/1 while tolerating strings/numbers/booleans.

    Parameters
    ----------
    y : pd.Series
        Labels to convert to binary.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.

    Returns
    -------
    np.ndarray
        Binary array (0/1).
    '''

    if y.dtype == bool:
        return y.astype(int).to_numpy()

    if positive_label is not None:
        return (y == positive_label).astype(int).to_numpy()

    if pd.api.types.is_numeric_dtype(y):
        vals = pd.to_numeric(y, errors="coerce").to_numpy()
        uniques = np.unique(vals[~np.isnan(vals)])
        if set(uniques).issubset({0, 1}):
            return vals.astype(int)
        return (vals > 0).astype(int)

    y_str = y.astype(str).str.lower()
    positives = {"1", "true", "yes", "y", "pos", "positive", "active", "ligand"}
    return y_str.isin(positives).astype(int).to_numpy()


def _safe_metric(metric_fn: Callable, y_true: np.ndarray, y_score: np.ndarray) -> float:
    '''Compute a metric defensively: handle NaNs, degenerate classes, exceptions.

    Parameters
    ----------
    metric_fn : Callable
        Metric function to compute.
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.

    Returns
    -------
    float
        Metric value, or NaN if computation fails.
    '''
    
    try:
        mask = np.isfinite(y_score)
        y = y_true[mask]
        s = y_score[mask]
        if y.size == 0 or len(np.unique(y)) < 2:
            return float("nan")
        return float(metric_fn(y, s))
    except (ValueError, TypeError, AttributeError):
        # Fallback to NaN if metric calculation fails
        return float("nan")


def _bootstrap_ci_on_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable,
    n_boot: int,
    seed: int,
) -> BootstrapCI:
    '''Percentile bootstrap [2.5%, 97.5%] on a score-based metric.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_score : np.ndarray
        Predicted scores.
    metric_fn : Callable
        Metric function to compute.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapCI
        Bootstrap confidence interval (low, mean, high).
    '''
    
    rng = np.random.default_rng(seed)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n = y_true.shape[0]
    if n == 0 or len(np.unique(y_true)) < 2:
        return BootstrapCI(float("nan"), float("nan"), float("nan"))

    vals: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = _safe_metric(metric_fn, y_true[idx], y_score[idx])
        if not np.isnan(v):
            vals.append(v)

    if not vals:
        return BootstrapCI(float("nan"), float("nan"), float("nan"))

    arr = np.array(vals)
    point = _safe_metric(metric_fn, y_true, y_score)
    low, high = np.quantile(arr, [0.025, 0.975])
    return BootstrapCI(point=point, low=float(low), high=float(high))


def _decide_flip(y_all: np.ndarray, s_all: np.ndarray) -> bool:
    '''Decide once per model if scores should be flipped (pooled ROC AUC < 0.5).

    Parameters
    ----------
    y_all : np.ndarray
        All true labels.
    s_all : np.ndarray
        All predicted scores.

    Returns
    -------
    bool
        True if scores should be flipped (ROC AUC < 0.5).
    '''
    
    auc = _safe_metric(roc_auc_score, y_all, s_all)
    return not np.isnan(auc) and auc < 0.5


def _apply_flip(s: np.ndarray, do_flip: bool) -> np.ndarray:
    return -s if do_flip else s


# --------------------------------------------------------------------------------------
# EF-ROC helpers
# --------------------------------------------------------------------------------------
def _efroc(y_true: np.ndarray, y_score: np.ndarray, epsilons: Iterable[float]) -> pd.DataFrame:
    '''Compute EF-ROC (Enrichment Factor at ROC) for multiple epsilon values.

    EF_ROC(eps) = TPR_at_FPR<=eps / eps. Random baseline = 1.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores.
    epsilons : Iterable[float]
        List of epsilon (FPR) values to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["epsilon", "ef_roc", "tpr_at_epsilon"].
    '''
    
    if len(np.unique(y_true)) < 2:
        rows = [(float(eps), float("nan"), float("nan")) for eps in epsilons]
        return pd.DataFrame(rows, columns=["epsilon", "ef_roc", "tpr_at_epsilon"])

    fpr, tpr, _ = roc_curve(y_true, y_score)
    rows = []
    for eps in epsilons:
        eps = float(eps)
        mask = fpr <= eps
        tpr_eps = float(np.max(tpr[mask])) if np.any(mask) else 0.0
        ef = tpr_eps / eps if eps > 0 else float("nan")
        rows.append((eps, ef, tpr_eps))
    return pd.DataFrame(rows, columns=["epsilon", "ef_roc", "tpr_at_epsilon"])


def _efroc_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    epsilons: Iterable[float],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    '''Bootstrap percentile CIs for EF-ROC across multiple epsilons.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_score : np.ndarray
        Predicted scores.
    epsilons : Iterable[float]
        List of epsilon (FPR) values to evaluate.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["epsilon", "ef_roc", "ci_low", "ci_high", "tpr_at_epsilon"].
    '''

    rng = np.random.default_rng(seed)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n = y_true.shape[0]

    ef_samples: Dict[float, list] = {float(e): [] for e in epsilons}

    if n == 0 or len(np.unique(y_true)) < 2:
        base = _efroc(y_true, y_score, epsilons)
        base["ci_low"] = np.nan
        base["ci_high"] = np.nan
        return base

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        df_b = _efroc(y_true[idx], y_score[idx], epsilons)
        for e, ef in zip(df_b["epsilon"], df_b["ef_roc"]):
            if not np.isnan(ef):
                ef_samples[float(e)].append(float(ef))

    base = _efroc(y_true, y_score, epsilons)
    lows, highs = [], []
    for e in base["epsilon"]:
        samples = np.array(ef_samples[float(e)]) if ef_samples[float(e)] else np.array([np.nan])
        samples = samples[~np.isnan(samples)]
        if samples.size == 0:
            low = high = float("nan")
        else:
            low, high = np.quantile(samples, [0.025, 0.975])
        lows.append(float(low))
        highs.append(float(high))
    base["ci_low"] = lows
    base["ci_high"] = highs
    return base


# --------------------------------------------------------------------------------------
# Public: per-target metrics
# --------------------------------------------------------------------------------------
def roc_auc_per_target(
    df: pd.DataFrame,
    target_col: str,
    label_col: str,
    score_cols: Sequence[str],
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute ROC AUC with 95% CI per target for each score model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    target_col : str
        Column name for the target.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the ROC AUC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["target", "model", "roc_auc", "ci_low", "ci_high", "n_pos", "n_neg"].
    '''

    rows = []
    for m in score_cols:
        y_all = _to_binary(df[label_col], positive_label)
        s_all = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y_all, s_all) if auto_flip else False

        for target, g in df.groupby(target_col, dropna=False):
            y = _to_binary(g[label_col], positive_label)
            s = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            s = _apply_flip(s, do_flip)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            ci = _bootstrap_ci_on_scores(y, s, roc_auc_score, n_boot, seed)
            rows.append(
                {
                    "target": target,
                    "model": m,
                    "roc_auc": ci.point,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "model"]).reset_index(drop=True)


def pr_auc_per_target(
    df: pd.DataFrame,
    target_col: str,
    label_col: str,
    score_cols: Sequence[str],
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute PR AUC (Average Precision) with 95% CI per target for each score model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    target_col : str
        Column name for the target.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the PR AUC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["target", "model", "pr_auc", "ci_low", "ci_high", "n_pos", "n_neg"].
    '''
    
    rows = []
    for m in score_cols:
        y_all = _to_binary(df[label_col], positive_label)
        s_all = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y_all, s_all) if auto_flip else False

        for target, g in df.groupby(target_col, dropna=False):
            y = _to_binary(g[label_col], positive_label)
            s = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            s = _apply_flip(s, do_flip)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            ci = _bootstrap_ci_on_scores(y, s, average_precision_score, n_boot, seed)
            rows.append(
                {
                    "target": target,
                    "model": m,
                    "pr_auc": ci.point,
                    "ci_low": ci.low,
                    "ci_high": ci.high,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "model"]).reset_index(drop=True)


def efroc_per_target(
    df: pd.DataFrame,
    target_col: str,
    label_col: str,
    score_cols: Sequence[str],
    epsilons: Sequence[float] = (0.01, 0.05, 0.10),
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute EF-ROC per target for each score model, with bootstrap CIs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    target_col : str
        Column name for the target.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    epsilons : Sequence[float]
        List of epsilon (FPR) values to evaluate.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the EF-ROC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["target", "model", "epsilon", "ef_roc", "ci_low", "ci_high", "tpr_at_epsilon", "n_pos", "n_neg"].
    '''
    
    all_rows = []
    for m in score_cols:
        y_all = _to_binary(df[label_col], positive_label)
        s_all = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y_all, s_all) if auto_flip else False

        for target, g in df.groupby(target_col, dropna=False):
            y = _to_binary(g[label_col], positive_label)
            s = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype=float)
            s = _apply_flip(s, do_flip)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())

            ef_df = _efroc_bootstrap_ci(y, s, epsilons, n_boot, seed)
            ef_df.insert(0, "target", target)
            ef_df.insert(1, "model", m)
            ef_df["n_pos"] = n_pos
            ef_df["n_neg"] = n_neg
            all_rows.append(ef_df)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["target","model","epsilon","ef_roc","ci_low","ci_high","tpr_at_epsilon","n_pos","n_neg"]
    )
    return out.sort_values(["target", "model", "epsilon"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Public: pooled metrics
# --------------------------------------------------------------------------------------
def roc_auc_pooled(
    df: pd.DataFrame,
    label_col: str,
    score_cols: Sequence[str],
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute pooled ROC AUC with 95% CI for each score model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the ROC AUC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["model", "roc_auc", "ci_low", "ci_high", "n_pos", "n_neg"].
    '''
    
    rows = []
    for m in score_cols:
        y = _to_binary(df[label_col], positive_label)
        s = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y, s) if auto_flip else False
        s = _apply_flip(s, do_flip)

        ci = _bootstrap_ci_on_scores(y, s, roc_auc_score, n_boot, seed)
        rows.append(
            {
                "model": m,
                "roc_auc": ci.point,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def pr_auc_pooled(
    df: pd.DataFrame,
    label_col: str,
    score_cols: Sequence[str],
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute pooled PR AUC (Average Precision) with 95% CI for each score model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the PR AUC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["model", "pr_auc", "ci_low", "ci_high", "n_pos", "n_neg"].
    '''

    rows = []
    for m in score_cols:
        y = _to_binary(df[label_col], positive_label)
        s = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y, s) if auto_flip else False
        s = _apply_flip(s, do_flip)

        ci = _bootstrap_ci_on_scores(y, s, average_precision_score, n_boot, seed)
        rows.append(
            {
                "model": m,
                "pr_auc": ci.point,
                "ci_low": ci.low,
                "ci_high": ci.high,
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("pr_auc", ascending=False).reset_index(drop=True)


def efroc_pooled(
    df: pd.DataFrame,
    label_col: str,
    score_cols: Sequence[str],
    epsilons: Sequence[float] = (0.01, 0.05, 0.10),
    n_boot: int = 500,
    seed: int = 0,
    positive_label: Optional[Union[str, int]] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    '''
    Compute pooled EF-ROC for each score model with bootstrap CIs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    label_col : str
        Column name for the labels.
    score_cols : Sequence[str]
        Column names for the score models.
    epsilons : Sequence[float]
        List of epsilon (FPR) values to evaluate.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    auto_flip : bool
        Whether to flip the scores if the EF-ROC is less than 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["model", "epsilon", "ef_roc", "ci_low", "ci_high", "tpr_at_epsilon", "n_pos", "n_neg"].
    '''
    
    all_rows = []
    for m in score_cols:
        y = _to_binary(df[label_col], positive_label)
        s = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype=float)
        do_flip = _decide_flip(y, s) if auto_flip else False
        s = _apply_flip(s, do_flip)

        ef_df = _efroc_bootstrap_ci(y, s, epsilons, n_boot, seed)
        ef_df.insert(0, "model", m)
        ef_df["n_pos"] = int((y == 1).sum())
        ef_df["n_neg"] = int((y == 0).sum())
        all_rows.append(ef_df)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["model","epsilon","ef_roc","ci_low","ci_high","tpr_at_epsilon","n_pos","n_neg"]
    )
    return out.sort_values(["model", "epsilon"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Aggregator: one call to get everything in your usual table format
# --------------------------------------------------------------------------------------
def build_test2_tables(
    df: pd.DataFrame,
    models: Sequence[str],
    target_col: str = "target",
    label_col: str = "active",
    positive_label: Optional[Union[str, int]] = None,
    n_boot: int = 500,
    seed: int = 0,
    epsilons: Sequence[float] = (0.01, 0.05, 0.10),
    auto_flip: bool = True,
) -> Dict[str, pd.DataFrame]:
    '''
    Convenience wrapper to compute all tables at once.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data.
    models : Sequence[str]
        Column names for the score models.
    target_col : str
        Column name for the target.
    label_col : str
        Column name for the labels.
    positive_label : Optional[Union[str, int]]
        Label to treat as positive (1). If None, infers from data.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    epsilons : Sequence[float]
        List of epsilon (FPR) values to evaluate.
    auto_flip : bool
        Whether to flip the scores if the ROC AUC is less than 0.5.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping with the following keys:
        - "roc_auc_per_target": ROC AUC per target with bootstrap CIs.
        - "pr_auc_per_target": PR AUC per target with bootstrap CIs.
        - "efroc_per_target": EF-ROC table per target across epsilons.
        - "roc_auc_pooled": Pooled ROC AUC across all targets with CIs.
        - "pr_auc_pooled": Pooled PR AUC across all targets with CIs.
        - "efroc_pooled": Pooled EF-ROC across epsilons with CIs.
        - "summary": Compact table combining pooled ROC/PR with counts.
    '''
    
    tables: Dict[str, pd.DataFrame] = {}

    tables["roc_auc_per_target"] = roc_auc_per_target(
        df=df,
        target_col=target_col,
        label_col=label_col,
        score_cols=models,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )

    tables["pr_auc_per_target"] = pr_auc_per_target(
        df=df,
        target_col=target_col,
        label_col=label_col,
        score_cols=models,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )

    tables["efroc_per_target"] = efroc_per_target(
        df=df,
        target_col=target_col,
        label_col=label_col,
        score_cols=models,
        epsilons=epsilons,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )

    pooled_roc = roc_auc_pooled(
        df=df,
        label_col=label_col,
        score_cols=models,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )
    pooled_pr = pr_auc_pooled(
        df=df,
        label_col=label_col,
        score_cols=models,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )
    pooled_ef = efroc_pooled(
        df=df,
        label_col=label_col,
        score_cols=models,
        epsilons=epsilons,
        n_boot=n_boot,
        seed=seed,
        positive_label=positive_label,
        auto_flip=auto_flip,
    )

    tables["roc_auc_pooled"] = pooled_roc
    tables["pr_auc_pooled"] = pooled_pr
    tables["efroc_pooled"] = pooled_ef

    # Summary aligns with your usual pattern (ROC + PR pooled)
    summary = pooled_roc[["model", "roc_auc", "ci_low", "ci_high", "n_pos", "n_neg"]].merge(
        pooled_pr[["model", "pr_auc"]], on="model", how="left"
    )
    summary = summary[["model", "roc_auc", "pr_auc", "n_pos", "n_neg"]].copy()
    tables["summary"] = summary.sort_values(["roc_auc", "pr_auc"], ascending=False).reset_index(drop=True)

    return tables


# --------------------------------------------------------------------------------------
# Formatting helper (parity with legacy clean_analysis)
# --------------------------------------------------------------------------------------
def build_summary_table(
    summary_targets: pd.DataFrame,
    summary_pooled: pd.DataFrame,
    models: Sequence[str],
    eps: Sequence[int] = (1, 5, 10, 20, 30),
    include_pr_auc: bool = False,
    pr_summary_targets: Optional[pd.DataFrame] = None,
    pr_summary_pooled: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    '''
    Create a presentation table combining median EF-ROC across targets and pooled EF-ROC at given epsilons.

    Optionally include PR-AUC (median and pooled).

    Parameters
    ----------
    summary_targets : pd.DataFrame
        DataFrame with the targets.
    summary_pooled : pd.DataFrame
        DataFrame with the pooled targets.
    models : Sequence[str]
        Column names for the score models.
    eps : Sequence[int]
        List of epsilon (FPR) values to evaluate.
    include_pr_auc : bool
        Whether to include the PR-AUC.
    pr_summary_targets : Optional[pd.DataFrame]
        DataFrame with the PR-AUC targets.
    pr_summary_pooled : Optional[pd.DataFrame]
        DataFrame with the PR-AUC pooled.

    Returns
    -------
    pd.DataFrame
        DataFrame with the summary table.
    '''
    eps = tuple(int(e) for e in eps)
    st = summary_targets[summary_targets["metric"].isin([f"EF_ROC_{e}%" for e in eps])].copy()
    med = (
        st[st["model"].isin(models)]
        .assign(val=lambda d: d.apply(lambda r: f"{r['median_across_targets']:.2f} [{r['CI95_lo']:.2f}–{r['CI95_hi']:.2f}]", axis=1))
        .pivot(index="model", columns="metric", values="val")
        .reindex(models)
    )

    sp = summary_pooled[summary_pooled["metric"].isin([f"EF_ROC_{e}%" for e in eps])].copy()
    poo = (
        sp[sp["model"].isin(models)]
        .assign(val=lambda d: d.apply(lambda r: f"{r['pooled_value']:.2f} [{r['CI95_lo']:.2f}–{r['CI95_hi']:.2f}]", axis=1))
        .pivot(index="model", columns="metric", values="val")
        .reindex(models)
    )

    rename_med = {f"EF_ROC_{e}%": f"Median EF-ROC {e}%" for e in eps}
    rename_poo = {f"EF_ROC_{e}%": f"Pooled EF-ROC {e}%" for e in eps}

    out = med.rename(columns=rename_med).join(poo.rename(columns=rename_poo))

    if include_pr_auc and pr_summary_targets is not None and pr_summary_pooled is not None:
        stp = pr_summary_targets[
            (pr_summary_targets["metric"].eq("PR_AUC")) & (pr_summary_targets["model"].isin(models))
        ].copy()
        med_pr = (
            stp.assign(val=lambda d: d.apply(lambda r: f"{r['median_across_targets']:.3f} [{r['CI95_lo']:.3f}–{r['CI95_hi']:.3f}]", axis=1))
            .pivot(index="model", columns="metric", values="val")
            .reindex(models)
        )
        spp = pr_summary_pooled[
            (pr_summary_pooled["metric"].eq("PR_AUC")) & (pr_summary_pooled["model"].isin(models))
        ].copy()
        poo_pr = (
            spp.assign(val=lambda d: d.apply(lambda r: f"{r['pooled_value']:.3f} [{r['CI95_lo']:.3f}–{r['CI95_hi']:.3f}]", axis=1))
            .pivot(index="model", columns="metric", values="val")
            .reindex(models)
        )
        out = out.join(med_pr.rename(columns={"PR_AUC": "Median PR-AUC"})).join(
            poo_pr.rename(columns={"PR_AUC": "Pooled PR-AUC"})
        )

    return out
