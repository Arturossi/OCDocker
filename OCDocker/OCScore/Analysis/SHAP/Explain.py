
#!/usr/bin/env python3

# Description
###############################################################################
'''
SHAP computation helpers (background selection, explainer setup, shape wrangling)
for Analysis workflows.
'''

# Imports
###############################################################################

from __future__ import annotations
from typing import List, Optional, Union
import numpy as np
import pandas as pd
import torch
import shap

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

def _cuda_device() -> torch.device:
    """Return a CUDA device if available; otherwise CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _squeeze_shap(values: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Normalize SHAP outputs to a 2D array (n_samples, n_features)."""
    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        else:
            values = np.sum(np.stack(values, axis=0), axis=0)
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")
    return arr

def _stratified_indices(df: pd.DataFrame, n: int, by: Optional[List[str]], seed: int) -> np.ndarray:
    """Draw up to n indices, stratified by the values of `by` columns."""
    if by is None or len(by) == 0:
        rng = np.random.default_rng(seed)
        n = min(n, len(df))
        return rng.choice(len(df), size=n, replace=False)
    groups = df.groupby(by, dropna=False)
    sizes = groups.size()
    total = float(sizes.sum())
    rng = np.random.default_rng(seed)
    picks = []
    for key, idx in groups.indices.items():
        frac = sizes[key] / total
        k = max(1, int(round(frac * n)))
        local_choices = rng.choice(idx, size=min(k, len(idx)), replace=False)
        picks.extend(local_choices.tolist())
    if len(picks) > n:
        picks = rng.choice(picks, size=n, replace=False).tolist()
    return np.array(sorted(picks))

def compute_shap_values(
    neural,
    X_background: pd.DataFrame,
    X_eval: pd.DataFrame,
    explainer: str = "deep",
    background_size: Optional[int] = None,
    eval_size: Optional[int] = None,
    stratify_by: Optional[List[str]] = None,
    rng_seed: int = 0,
) -> np.ndarray:
    device = _cuda_device()
    if background_size is None:
        background_size = len(X_background)
    if eval_size is None:
        eval_size = len(X_eval)
    bkg_df = X_background.copy()
    eval_df = X_eval.copy()
    bkg_idx = _stratified_indices(bkg_df, background_size, stratify_by, rng_seed)
    eval_idx = _stratified_indices(eval_df, eval_size, stratify_by, rng_seed + 1)
    bkg_np = bkg_df.iloc[bkg_idx].to_numpy(dtype=np.float32)
    eval_np = eval_df.iloc[eval_idx].to_numpy(dtype=np.float32)
    if explainer.lower() == "deep":
        background_tensor = torch.tensor(bkg_np, dtype=torch.float32, device=device)
        deep_explainer = shap.DeepExplainer(neural.NN, background_tensor)
        shap_values = deep_explainer.shap_values(torch.tensor(eval_np, dtype=torch.float32, device=device))
    elif explainer.lower() == "kernel":
        def model_predict(x_numpy: np.ndarray) -> np.ndarray:
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = neural.NN(x_tensor).detach().cpu().numpy().squeeze()
            return out
        kernel_explainer = shap.KernelExplainer(model_predict, bkg_np)
        shap_values = kernel_explainer.shap_values(eval_np)
    else:
        raise ValueError("explainer must be 'deep' or 'kernel'")
    shap_2d = _squeeze_shap(shap_values)
    return shap_2d
