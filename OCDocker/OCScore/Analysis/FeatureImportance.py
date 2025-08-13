from __future__ import annotations
"""
Test2 SHAP utilities (library style, no I/O, no plots).

Focus on reproducible SHAP computation with stratified background selection and
uniform outputs (arrays and tables). Works with tree/NN/black-box models via
auto selection of the appropriate Explainer.

Public API:
- build_stratified_background
- make_explainer
- compute_shap_values
- shap_importance_table
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except Exception as e:  # pragma: no cover
    shap = None  # Deferred error: raised when functions are called


__all__ = [
    "build_stratified_background",
    "make_explainer",
    "compute_shap_values",
    "shap_importance_table",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _require_shap() -> None:
    if shap is None:
        raise ImportError("shap is not installed. Please install `shap` to use Test2SHAP utilities.")


def _ensure_2d(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Guarantee 2D float64 array without copying unnecessarily."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float, copy=False)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X.astype(float, copy=False)


# --------------------------------------------------------------------------------------
# Background selection
# --------------------------------------------------------------------------------------
def build_stratified_background(
    X: np.ndarray | pd.DataFrame,
    meta: pd.DataFrame,
    strata_cols: Sequence[str],
    per_stratum: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a stratified background set by sampling up to `per_stratum` rows from each
    combination in `strata_cols` (e.g., ["target","active"]). Preserves class/target
    balance in the background while bounding its size.

    Returns: background array with shape (n_bg, n_features).
    """
    rng = np.random.default_rng(seed)
    X_arr = _ensure_2d(X)
    assert len(meta) == X_arr.shape[0], "meta rows must align with X rows"

    idxs: List[int] = []
    for combo, g in meta.groupby(list(strata_cols), dropna=False):
        g_idx = g.index.to_numpy()
        if g_idx.size <= per_stratum:
            idxs.extend(g_idx.tolist())
        else:
            take = rng.choice(g_idx, size=per_stratum, replace=False)
            idxs.extend(take.tolist())

    idxs = np.array(sorted(set(idxs)), dtype=int)
    return X_arr[idxs]


# --------------------------------------------------------------------------------------
# Explainer selection
# --------------------------------------------------------------------------------------
def make_explainer(
    model,
    background: np.ndarray,
    method: str = "auto",
    link: str | None = None,
    predict_fn: Callable | None = None,
):
    """
    Create a SHAP Explainer for the given model and background.

    - method="auto": TreeExplainer if tree model; DeepExplainer if torch/TF; else KernelExplainer.
    - link: optional link function (e.g., "logit") for KernelExplainer.
    - predict_fn: override prediction function (expects shape (n, n_classes) or (n,) ).

    Returns: (explainer, predict_proba_index)
      predict_proba_index = 1 is commonly used for binary classification when the
      explainer returns per-class SHAP values (lists). You can change that later.
    """
    _require_shap()
    bg = _ensure_2d(background)

    # Allow user to override prediction function (useful for custom wrappers)
    if predict_fn is not None:
        fn = predict_fn
    else:
        # Default: try predict_proba -> class 1 prob; else predict
        if hasattr(model, "predict_proba"):
            fn = lambda X: model.predict_proba(X)
        else:
            fn = lambda X: model.predict(X)

    # Heuristics for method auto-selection
    is_tree = any(hasattr(model, attr) for attr in ("apply", "tree_", "feature_importances_"))
    is_torch = "torch" in type(model).__module__
    is_tf = any(k in type(model).__module__ for k in ("tensorflow", "keras"))

    if method == "auto":
        if is_tree and hasattr(shap, "TreeExplainer"):
            explainer = shap.TreeExplainer(model, data=bg)
            proba_idx = 1
        elif (is_torch or is_tf) and hasattr(shap, "DeepExplainer"):
            explainer = shap.DeepExplainer(model, bg)
            proba_idx = 1
        else:
            link_obj = link if link in (None, "identity", "logit") else None
            explainer = shap.KernelExplainer(fn, bg, link=link_obj)
            proba_idx = 1
    elif method.lower() == "tree":
        explainer = shap.TreeExplainer(model, data=bg)
        proba_idx = 1
    elif method.lower() == "deep":
        explainer = shap.DeepExplainer(model, bg)
        proba_idx = 1
    elif method.lower() == "kernel":
        link_obj = link if link in (None, "identity", "logit") else None
        explainer = shap.KernelExplainer(fn, bg, link=link_obj)
        proba_idx = 1
    else:
        raise ValueError(f"Unknown method: {method}")

    return explainer, proba_idx


# --------------------------------------------------------------------------------------
# SHAP computation
# --------------------------------------------------------------------------------------
def compute_shap_values(
    explainer,
    X_eval: np.ndarray | pd.DataFrame,
    task: str = "binary",
    nsamples: int | str | None = "auto",
    class_index: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute SHAP values for the evaluation set.

    - For binary classification with explainers returning per-class arrays (list),
      we select class_index (default = 1).
    - nsamples is passed to KernelExplainer; ignored by Tree/Deep explainers when not applicable.

    Returns dict:
      {
        "shap_values": (n_samples, n_features),
        "base_values": (n_samples,) or scalar
      }
    """
    _require_shap()
    X_eval_arr = _ensure_2d(X_eval)

    # Some explainers expose .shap_values (callable) with optional nsamples
    try:
        vals = explainer.shap_values(X_eval_arr, nsamples=nsamples)  # KernelExplainer accepts nsamples
    except TypeError:
        vals = explainer.shap_values(X_eval_arr)  # Tree/Deep

    # Align output shape
    if isinstance(vals, list):
        # Per-class outputs; choose the desired class (binary: index 1)
        vals_use = np.asarray(vals[class_index], dtype=float)
    else:
        vals_use = np.asarray(vals, dtype=float)

    base = getattr(explainer, "expected_value", 0.0)
    if isinstance(base, (list, tuple, np.ndarray)):
        # Per-class base values; align with class_index if present
        if len(np.atleast_1d(base)) > class_index:
            base_val = np.atleast_1d(base)[class_index]
        else:
            base_val = np.atleast_1d(base).ravel()[0]
    else:
        base_val = float(base)

    # Some explainers return per-sample base values; broadcast if needed
    if np.ndim(base_val) == 0:
        base_values = np.full(X_eval_arr.shape[0], float(base_val), dtype=float)
    else:
        base_values = np.asarray(base_val, dtype=float)

    return {"shap_values": vals_use, "base_values": base_values}


# --------------------------------------------------------------------------------------
# Importance tables
# --------------------------------------------------------------------------------------
def shap_importance_table(
    shap_values: np.ndarray,
    feature_names: Sequence[str] | None = None,
    k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per feature and return a ranked table.

    Columns:
      ["feature","mean_abs_shap","rank"]
    """
    S = np.asarray(shap_values, dtype=float)
    mean_abs = np.nanmean(np.abs(S), axis=0)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(S.shape[1])]

    df = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs})
    df.sort_values("mean_abs_shap", ascending=False, inplace=True)
    df["rank"] = np.arange(1, df.shape[0] + 1)

    if k is not None and k > 0:
        df = df.head(int(k)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df
