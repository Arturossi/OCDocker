from __future__ import annotations
"""
Test2 analysis (library style, no CLI, no I/O).

Consolidates unique metric functions (ROC/PR/EF-ROC with bootstrap) and provides
tabular outputs consistent with your existing analysis style.

Public API (metrics/tables):
- roc_auc_per_target
- pr_auc_per_target
- efroc_per_target
- roc_auc_pooled
- pr_auc_pooled
- efroc_pooled
- build_test2_tables
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict

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
]


# --------------------------------------------------------------------------------------
# Basic structures
# --------------------------------------------------------------------------------------
@dataclass
class BootstrapCI:
    point: float
    low: float
    high: float


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _to_binary(y: pd.Series, positive_label: Optional[str | int]) -> np.ndarray:
    """Map labels to 0/1 while tolerating strings/numbers/booleans."""
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


def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute a metric defensively: handle NaNs, degenerate classes, exceptions."""
    try:
        mask = np.isfinite(y_score)
        y = y_true[mask]
        s = y_score[mask]
        if y.size == 0 or len(np.unique(y)) < 2:
            return float("nan")
        return float(metric_fn(y, s))
    except Exception:
        return float("nan")


def _bootstrap_ci_on_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> BootstrapCI:
    """Percentile bootstrap [2.5%, 97.5%] on a score-based metric."""
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
    """Decide once per model if scores should be flipped (pooled ROC AUC < 0.5)."""
    auc = _safe_metric(roc_auc_score, y_all, s_all)
    return not np.isnan(auc) and auc < 0.5


def _apply_flip(s: np.ndarray, do_flip: bool) -> np.ndarray:
    return -s if do_flip else s


# --------------------------------------------------------------------------------------
# EF-ROC helpers
# --------------------------------------------------------------------------------------
def _efroc(y_true: np.ndarray, y_score: np.ndarray, epsilons: Iterable[float]) -> pd.DataFrame:
    """
    EF_ROC(eps) = TPR_at_FPR<=eps / eps. Random baseline = 1.
    Returns columns: ["epsilon","ef_roc","tpr_at_epsilon"].
    """
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
    """Bootstrap percentile CIs for EF-ROC across multiple epsilons."""
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute ROC AUC with 95% CI per target for each score model.

    Returns columns:
      ["target","model","roc_auc","ci_low","ci_high","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute PR AUC (Average Precision) with 95% CI per target for each score model.

    Returns columns:
      ["target","model","pr_auc","ci_low","ci_high","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute EF-ROC per target for each score model, with bootstrap CIs.

    Returns columns:
      ["target","model","epsilon","ef_roc","ci_low","ci_high","tpr_at_epsilon","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute pooled ROC AUC with 95% CI for each score model.

    Returns columns:
      ["model","roc_auc","ci_low","ci_high","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute pooled PR AUC (Average Precision) with 95% CI for each score model.

    Returns columns:
      ["model","pr_auc","ci_low","ci_high","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    auto_flip: bool = True,
) -> pd.DataFrame:
    """
    Compute pooled EF-ROC for each score model with bootstrap CIs.

    Returns columns:
      ["model","epsilon","ef_roc","ci_low","ci_high","tpr_at_epsilon","n_pos","n_neg"]
    """
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
    positive_label: Optional[str | int] = None,
    n_boot: int = 500,
    seed: int = 0,
    epsilons: Sequence[float] = (0.01, 0.05, 0.10),
    auto_flip: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper to compute all tables at once.

    Returns a dict of DataFrames:
      {
        "roc_auc_per_target": ...,
        "pr_auc_per_target": ...,
        "efroc_per_target": ...,
        "roc_auc_pooled": ...,
        "pr_auc_pooled": ...,
        "efroc_pooled": ...,
        "summary": ...
      }
    """
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
