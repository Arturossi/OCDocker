
#!/usr/bin/env python3

# Description
###############################################################################
'''
Bootstrap utilities for metric confidence intervals.

They are imported as:

from OCDocker.OCScore.Analysis.Metrics import Bootstrap as Boot
'''

# Imports
###############################################################################

from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple
import numpy as np
import pandas as pd

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


def bootstrap_ci(
        y_true: np.ndarray,
        y_score: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_boot: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None,
        strata: Optional[Iterable] = None
    ) -> Tuple[float, float, float]:
    '''Compute metric and (1-alpha) bootstrap CI.
    If strata is provided, resample within each stratum to preserve distribution.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0/1 or boolean).
    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        a classifier).
    metric_fn : callable
        Function with signature metric_fn(y_true, y_score) that returns a float.
    n_boot : int, optional
        Number of bootstrap samples. Default is 1000. If <= 0, no boot
        is performed and (metric, NaN, NaN) is returned.
    alpha : float, optional
        Significance level for the (1-alpha) confidence interval. Default is 0.05
        for a 95% CI.
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    strata : array-like of shape (n_samples,) or None, optional
        If provided, resample within each stratum to preserve distribution.
        Default is None (no stratification).

    Returns
    -------
    tuple[float, float, float]
        Returns (metric, low, high).
    '''

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
