
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap

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
    _ensure_dir(os.path.dirname(out_png))
    rel = _relative_importance(shap_2d)
    order = np.argsort(rel)[::-1]
    k = min(top_k, len(order))
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
    _ensure_dir(os.path.dirname(out_png))
    shap.summary_plot(shap_2d, X_eval.to_numpy(), feature_names=X_eval.columns, show=False, plot_size=figsize)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    return out_png
