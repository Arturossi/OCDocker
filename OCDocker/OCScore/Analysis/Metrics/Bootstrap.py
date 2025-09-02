
from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    strata: Optional[Iterable] = None,
) -> Tuple[float, float, float]:
    """Compute metric and (1-alpha) bootstrap CI.
    If strata is provided, resample within each stratum to preserve distribution.
    Returns (metric, low, high).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    est = metric_fn(y_true, y_score)
    if n_boot <= 0:
        return est, np.nan, np.nan

    if strata is None:
        idx = np.arange(len(y_true))
        boots = []
        for _ in range(n_boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            boots.append(metric_fn(y_true[b], y_score[b]))
    else:
        df = pd.DataFrame(dict(y=y_true, s=y_score, g=list(strata)))
        groups = df.groupby('g', dropna=False)
        boots = []
        for _ in range(n_boot):
            parts = []
            for _, sub in groups:
                b = sub.sample(n=len(sub), replace=True, random_state=rng.integers(1<<30))
                parts.append(b)
            bdf = pd.concat(parts, axis=0)
            boots.append(metric_fn(bdf['y'].to_numpy(), bdf['s'].to_numpy()))
    low = float(np.nanpercentile(boots, 100*alpha/2))
    high = float(np.nanpercentile(boots, 100*(1-alpha/2)))
    return float(est), low, high
