
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
import OCDocker.Error as ocerror
from typing import Any, List, Optional, Union
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
    '''Return a CUDA device if available; otherwise CPU.'''

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _squeeze_shap(values: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    '''Normalize SHAP outputs to a 2D array (n_samples, n_features).

    Parameters
    ----------
    values : array-like or list of array-like
        SHAP values as returned by shap explainers. Can be a single array or a
        list of arrays (e.g., for multi-class models).

    Returns
    -------
    np.ndarray
        SHAP values as a 2D array of shape (n_samples, n_features). 
    '''

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
    '''Draw up to n indices, stratified by the values of `by` columns.
    
    If by is None or empty, draw n random indices from the whole DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sample from.
    n : int
        Number of indices to draw.
    by : list of str or None
        Column names to stratify by. If None or empty, no stratification is done.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of drawn indices.
    '''

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
        frac = sizes[key] / total # type: ignore
        k = max(1, int(round(frac * n)))
        local_choices = rng.choice(idx, size=min(k, len(idx)), replace=False)
        picks.extend(local_choices.tolist())

    if len(picks) > n:
        picks = rng.choice(picks, size=n, replace=False).tolist()

    return np.array(sorted(picks))


def compute_shap_values(
        neural: Any,
        X_background: pd.DataFrame,
        X_eval: pd.DataFrame,
        explainer: str = "deep",
        background_size: Optional[int] = None,
        eval_size: Optional[int] = None,
        stratify_by: Optional[List[str]] = None,
        rng_seed: int = 0,
    ) -> np.ndarray:
    '''Compute SHAP values for a neural network model using Deep or Kernel explainer.

    Parameters
    ----------
    neural : object
        Object with a .NN attribute that is a PyTorch neural network model.
    X_background : pd.DataFrame
        Background dataset for SHAP. Should contain the same features as X_eval.
    X_eval : pd.DataFrame
        Evaluation dataset for which to compute SHAP values.
    explainer : str
        Type of SHAP explainer to use: "deep" or "kernel". Default is
        "deep".
    background_size : int, optional
        Number of samples to draw from X_background. If None, use all. Default is None.
    eval_size : int, optional
        Number of samples to draw from X_eval. If None, use all. Default is None.
    stratify_by : list of str, optional
        Column names to stratify sampling by. If None or empty, no
        stratification is done. Default is None.
    rng_seed : int
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    np.ndarray
        SHAP values as a 2D array of shape (n_samples, n_features).
    '''

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
            '''Predict using the neural network model for SHAP KernelExplainer.
            
            Parameters
            ----------
            x_numpy : np.ndarray
                Input feature array.
            
            Returns
            -------
            np.ndarray
                Model predictions as a numpy array.
            '''
            
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = neural.NN(x_tensor).detach().cpu().numpy().squeeze()
            return out
        kernel_explainer = shap.KernelExplainer(model_predict, bkg_np)
        shap_values = kernel_explainer.shap_values(eval_np)
    else:
        # User-facing error: invalid explainer type
        ocerror.Error.value_error(f"Invalid explainer type: '{explainer}'. Must be 'deep' or 'kernel'.") # type: ignore
        raise ValueError("explainer must be 'deep' or 'kernel'")
    
    shap_2d = _squeeze_shap(shap_values) # type: ignore

    return shap_2d
