#!/usr/bin/env python3

# Description
###############################################################################
'''
SHAP analysis for exported ``best_model/`` bundles from the staged pipeline.

Usage:

from OCDocker.OCScore.Analysis.SHAP.ExportRunner import run_export_shap_analysis
'''

# Imports
###############################################################################
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

import OCDocker.OCScore.Optimization.ModelExport as ocexport
from OCDocker.OCScore.Analysis.SHAP import Plots as shap_plots
from OCDocker.OCScore.Analysis.SHAP.Explain import compute_shap_values
from OCDocker.OCScore.Analysis.SHAP.Paths import OutputPaths
from OCDocker.OCScore.Utils.ContentHash import hash_feature_list

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################
class _ShapForwardModule(nn.Module):
    """Expose a single scalar output for DeepExplainer / KernelExplainer."""

    def __init__(self, model: nn.Module, task: str) -> None:
        super().__init__()
        self._model = model
        self._task = str(task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass without ``no_grad`` so DeepExplainer can backpropagate.'''

        self._model.eval()
        if self._task == "pdbbind_regression":
            outputs = self._model(x)
            prediction = outputs["prediction"] if isinstance(outputs, dict) else outputs
        else:
            prediction = self._model(x)
        return prediction.reshape(-1, 1)


class _ShapNeuralWrapper:
    """Adapter matching archived SHAP ``neural.NN`` contract."""

    def __init__(self, model: nn.Module, task: str) -> None:
        self.NN = _ShapForwardModule(model, task)


# Functions
###############################################################################
## Private ##

def _transform_features(X_df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    values = X_df.to_numpy(dtype=np.float32)
    if scaler is not None:
        values = scaler.transform(values)
    return pd.DataFrame(values, columns=list(X_df.columns))


def _require_split_indices(split_indices: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not split_indices:
        raise ValueError(
            "Export bundle is missing split_indices.npz. "
            "Re-export the model or provide a bundle with validation/test indices."
        )
    val_idx = split_indices.get("validation_indices")
    test_idx = split_indices.get("test_indices")
    if val_idx is None or test_idx is None:
        raise ValueError(
            "split_indices.npz must contain validation_indices and test_indices "
            "for export SHAP (validation background, test evaluation)."
        )
    return np.asarray(val_idx), np.asarray(test_idx)


def _prepare_feature_frames(
        dataframe: pd.DataFrame,
        selected_features: List[str],
        split_indices: dict[str, np.ndarray],
        scaler: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    missing = [column for column in selected_features if column not in dataframe.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Dataframe is missing selected export features: {preview}{suffix}")

    val_idx, test_idx = _require_split_indices(split_indices)
    feature_frame = dataframe[selected_features]
    X_background = _transform_features(feature_frame.iloc[val_idx].reset_index(drop=True), scaler)
    X_eval = _transform_features(feature_frame.iloc[test_idx].reset_index(drop=True), scaler)
    return X_background, X_eval, list(selected_features)


## Public ##

def run_export_shap_analysis(
        export_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        out_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        background_size: Optional[int] = None,
        eval_size: Optional[int] = None,
        explainer: str = "gradient",
        stratify_by: Optional[List[str]] = None,
        seed: int = 0,
        save_csv: bool = True,
    ) -> OutputPaths:
    '''Run SHAP on an exported staged-model bundle.

    Uses validation rows for the SHAP background and test rows for evaluation,
    matching the saved ``split_indices.npz`` from export.

    Parameters
    ----------
    export_dir : str | Path
        Exported ``best_model/`` directory.
    dataframe : pd.DataFrame
        Task-filtered reduced dataframe containing ``selected_features`` columns.
    out_dir : str | Path
        Directory for SHAP artifacts.
    device : str | torch.device | None, optional
        Torch device for model loading and SHAP, by default export CPU.
    background_size : int | None, optional
        Subsample size for validation background.
    eval_size : int | None, optional
        Subsample size for test evaluation.
    explainer : str, optional
        ``gradient``, ``deep``, ``kernel``, or ``permutation`` SHAP explainer, by default ``gradient``.
    stratify_by : list[str] | None, optional
        Optional dataframe columns for stratified subsampling.
    seed : int, optional
        Random seed for subsampling, by default 0.
    save_csv : bool, optional
        Write ``shap_values.csv`` when True, by default True.

    Returns
    -------
    OutputPaths
        Paths to generated SHAP artifacts.
    '''

    export_path = Path(export_dir)
    output_path = Path(out_dir)
    os.makedirs(output_path, exist_ok=True)

    bundle = ocexport.load_exported_model(export_path, device=device or "cpu")
    task = str(bundle["retrain_config"]["task"])
    selected_features = list(bundle["selected_features"])
    X_background, X_eval, feature_names = _prepare_feature_frames(
        dataframe,
        selected_features,
        bundle["split_indices"],
        bundle.get("scaler"),
    )

    model = bundle["model"]
    model.eval()
    neural = _ShapNeuralWrapper(model, task)
    shap_2d = compute_shap_values(
        neural=neural,
        X_background=X_background,
        X_eval=X_eval,
        explainer=explainer,
        background_size=background_size,
        eval_size=eval_size,
        stratify_by=stratify_by,
        rng_seed=seed,
    )

    shap_npy = str(output_path / "shap_values.npy")
    np.save(shap_npy, shap_2d)
    shap_csv = None
    if save_csv:
        shap_csv = str(output_path / "shap_values.csv")
        pd.DataFrame(shap_2d, columns=feature_names).to_csv(shap_csv, index=False)

    imp_png = str(output_path / "shap_feature_importance.png")
    bee_png = str(output_path / "shap_beeswarm_plot.png")
    shap_plots.feature_importance_barh(shap_2d, feature_names, out_png=imp_png, top_k=20)
    shap_plots.beeswarm(shap_2d, X_eval.iloc[: shap_2d.shape[0]], out_png=bee_png, rng_seed=seed)

    shap_report = {
        "export_dir": str(export_path.resolve()),
        "task": task,
        "explainer": explainer,
        "selected_features": selected_features,
        "selected_features_hash": hash_feature_list(selected_features),
        "n_selected_features": len(selected_features),
        "feature_policy": (bundle.get("feature_metadata") or {}).get("feature_policy"),
        "note": "SHAP values are comparable within this exported feature space; compare across policies only with the policy-specific feature spaces in mind.",
    }
    (output_path / "shap_report.json").write_text(
        json.dumps(shap_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return OutputPaths(
        out_dir=str(output_path.resolve()),
        feature_importance_png=imp_png,
        beeswarm_png=bee_png,
        shap_values_npy=shap_npy,
        shap_values_csv=shap_csv,
    )


__all__ = ["run_export_shap_analysis", "OutputPaths"]
