
#!/usr/bin/env python3

# Description
###############################################################################
'''
Utilities to visualize SHAP outputs.

Public helpers
--------------
- feature_importance_barh: horizontal bar chart of relative importance
- beeswarm: wrapper around shap.summary_plot
- shap_correlation_heatmap: correlation heatmap of SHAP values
'''

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap
import seaborn as sns

# Functions
###############################################################################


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _relative_importance(shap_2d: np.ndarray) -> np.ndarray:
    mean_abs = np.abs(shap_2d).mean(axis=0)
    denom = mean_abs.sum()
    if denom <= 0:
        return np.zeros_like(mean_abs)
    return (mean_abs / denom) * 100.0


def feature_importance_barh(
    shap_2d: np.ndarray,
    feature_names: List[str],
    out_png: str,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    '''Horizontal bar chart of relative SHAP importance per feature.

    Parameters
    ----------
    shap_2d : np.ndarray
        SHAP values array with shape (n_samples, n_features).
    feature_names : list[str]
        Names of features in column order.
    out_png : str
        Where to save the plot.
    top_k : int, optional
        Number of top features to display. Default is 20.
    figsize : tuple[int, int], optional
        Figure size. Default is (10, 6).

    Returns
    -------
    str
        Output path of the saved plot.
    '''

    # Ensure output directory exists
    _ensure_dir(os.path.dirname(out_png))
    
    # Compute relative importance as normalized mean |SHAP|
    rel = _relative_importance(shap_2d)
    order = np.argsort(rel)[::-1]
    k = min(top_k, len(order))
    # Render barh plot
    plt.figure(figsize=figsize)
    plt.barh(y=np.array(feature_names)[order][:k], width=rel[order][:k])
    plt.xlabel('Relative Importance (%)')
    plt.title('Descriptor Importance (SHAP)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png


def beeswarm(
    shap_2d: np.ndarray,
    X_eval: pd.DataFrame,
    out_png: str,
    figsize: Tuple[int, int] = (10, 6),
) -> str:
    '''Wrapper around shap.summary_plot to save a beeswarm plot.

    Parameters
    ----------
    shap_2d : np.ndarray
        SHAP values with shape (n_samples, n_features).
    X_eval : pd.DataFrame
        Evaluation features (columns = names).
    out_png : str
        Where to save the plot.
    figsize : tuple[int, int], optional
        Plot size (width, height). Default is (10, 6).

    Returns
    -------
    str
        Output path of the saved plot.
    '''

    # Ensure output directory exists
    _ensure_dir(os.path.dirname(out_png))
    
    # Compute relative importance as normalized mean |SHAP|
    shap.summary_plot(shap_2d, X_eval.to_numpy(), feature_names=X_eval.columns, show=False, plot_size=figsize)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    return out_png


def shap_correlation_heatmap(
    shap_values: Union[np.ndarray, pd.DataFrame],
    out_png: str,
    feature_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> str:
    '''Plot a heatmap of SHAP value correlations across features.

    Parameters
    ----------
    shap_values : array-like | pd.DataFrame
        SHAP values with shape (n_samples, n_features).
    out_png : str
        Where to save the heatmap image.
    feature_names : list[str] | None
        Optional feature names. If `shap_values` is a DataFrame, its columns are used.
    figsize : tuple[int, int]
        Figure size.

    Returns
    -------
    str
        Output path of the saved heatmap.
    '''
    _ensure_dir(os.path.dirname(out_png))
    
    # Compute relative importance as normalized mean |SHAP|
    if isinstance(shap_values, pd.DataFrame):
        df = shap_values.copy()
    else:
        arr = np.asarray(shap_values)
        cols = feature_names if feature_names is not None else [f"f{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)

    corr = df.corr()
    # Render barh plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("SHAP value correlations")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png
