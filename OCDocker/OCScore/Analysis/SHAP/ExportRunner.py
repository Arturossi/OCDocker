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
from typing import Any, List, Mapping, Optional, Sequence, Union, cast

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
        return cast(torch.Tensor, prediction.reshape(-1, 1))


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


def _require_split_indices(
        split_indices: dict[str, np.ndarray],
        eval_split: str,
    ) -> tuple[np.ndarray, np.ndarray]:
    '''Return validation background indices and requested evaluation indices.

    Parameters
    ----------
    split_indices : dict[str, np.ndarray]
        Exported split-index mapping.
    eval_split : str
        Evaluation split name. Supported values are ``validation`` and ``test``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validation background indices and evaluation indices.
    '''

    if not split_indices:
        raise ValueError(
            "Export bundle is missing split_indices.npz. "
            "Re-export the model or provide a bundle with validation/test indices."
        )
    val_idx = split_indices.get("validation_indices")
    if val_idx is None:
        raise ValueError(
            "split_indices.npz must contain validation_indices "
            "for export SHAP validation background."
        )
    if eval_split == "validation":
        return np.asarray(val_idx), np.asarray(val_idx)
    if eval_split == "test":
        test_idx = split_indices.get("test_indices")
        if test_idx is None:
            raise ValueError(
                "split_indices.npz must contain test_indices for export SHAP test evaluation."
            )
        return np.asarray(val_idx), np.asarray(test_idx)
    raise ValueError("eval_split must be 'validation' or 'test'")


def _prepare_feature_frames(
        dataframe: pd.DataFrame,
        selected_features: List[str],
        split_indices: dict[str, np.ndarray],
        scaler: Any,
        eval_split: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    '''Prepare validation background and requested evaluation feature frames.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Task-specific dataframe.
    selected_features : list[str]
        Exported selected features.
    split_indices : dict[str, np.ndarray]
        Exported split indices.
    scaler : Any
        Optional fitted scaler.
    eval_split : str
        Evaluation split name.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]
        Background frame, evaluation frame, feature names, and evaluation
        metadata rows.
    '''

    missing = [column for column in selected_features if column not in dataframe.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Dataframe is missing selected export features: {preview}{suffix}")

    val_idx, eval_idx = _require_split_indices(split_indices, eval_split)
    feature_frame = dataframe[selected_features]
    X_background = _transform_features(feature_frame.iloc[val_idx].reset_index(drop=True), scaler)
    X_eval = _transform_features(feature_frame.iloc[eval_idx].reset_index(drop=True), scaler)
    eval_metadata = dataframe.iloc[eval_idx].reset_index(drop=True)
    return X_background, X_eval, list(selected_features), eval_metadata


def _align_table_to_eval_rows(
        table: Union[str, Path, Sequence[Any], pd.Series, pd.DataFrame],
        eval_indices: np.ndarray,
    ) -> pd.DataFrame:
    '''Align a full dataset table to SHAP evaluation rows.

    Parameters
    ----------
    table : str | Path | sequence | pd.Series | pd.DataFrame
        Full dataset metadata table.
    eval_indices : np.ndarray
        Evaluation indices from the exported split.

    Returns
    -------
    pd.DataFrame
        Metadata table aligned to SHAP rows.
    '''

    if isinstance(table, (str, Path)):
        frame = pd.read_csv(table)
    elif isinstance(table, pd.DataFrame):
        frame = table.copy()
    elif isinstance(table, pd.Series):
        frame = table.to_frame(name=table.name or "label")
    else:
        frame = pd.DataFrame({"label": list(table)})
    if len(frame) >= int(np.max(eval_indices)) + 1:
        return frame.iloc[eval_indices].reset_index(drop=True)
    if len(frame) == len(eval_indices):
        return frame.reset_index(drop=True)
    raise ValueError(
        "Metadata/label table must be either the full modeling dataframe "
        "or already aligned to the selected SHAP evaluation split."
    )


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
        policy: str = "policy",
        top_n: int = 20,
        family_spec: Optional[Union[str, Path, Mapping[str, Any]]] = None,
        dependence_features: Optional[Sequence[str]] = None,
        sample_metadata: Optional[Union[str, Path, pd.DataFrame]] = None,
        target_column: Optional[str] = None,
        labels: Optional[Union[str, Path, Sequence[Any], pd.Series, pd.DataFrame]] = None,
        label_column: Optional[str] = None,
        eval_split: str = "validation",
        include_log_importance_plots: bool = True,
        filter_zero_rows_log: bool = True,
    ) -> OutputPaths:
    '''Run SHAP on an exported staged-model bundle.

    Uses validation rows for the SHAP background and validation rows for evaluation,
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
        Subsample size for the selected SHAP evaluation split.
    explainer : str, optional
        ``gradient``, ``deep``, ``kernel``, or ``permutation`` SHAP explainer, by default ``gradient``.
    stratify_by : list[str] | None, optional
        Optional dataframe columns for stratified subsampling.
    seed : int, optional
        Random seed for subsampling, by default 0.
    save_csv : bool, optional
        Write ``shap_values.csv`` when True, by default True.
    policy : str, optional
        File-name policy prefix for reusable SHAP plots, by default "policy".
    top_n : int, optional
        Number of visible features in global plots, by default 20.
    family_spec : str | Path | mapping | None, optional
        Feature-family specification.
    dependence_features : sequence[str] | None, optional
        Features for dependence plots.
    sample_metadata : str | Path | pd.DataFrame | None, optional
        Sample metadata for target-family heatmap.
    target_column : str | None, optional
        Target column in sample metadata.
    labels : str | Path | sequence | pd.Series | pd.DataFrame | None, optional
        Labels for active-vs-decoy family distribution.
    label_column : str | None, optional
        Label column when labels are provided as a table.
    eval_split : str, optional
        Split to explain with SHAP. Supported values are ``validation`` and
        ``test``. Default is ``validation``.
    include_log_importance_plots : bool, optional
        Save log-scale feature and family importance companion plots.
    filter_zero_rows_log : bool, optional
        Remove zero rows from log-scale plots when True. When False, zero rows
        are plotted with a small positive floor.

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
    X_background, X_eval, feature_names, eval_metadata = _prepare_feature_frames(
        dataframe,
        selected_features,
        bundle["split_indices"],
        bundle.get("scaler"),
        eval_split,
    )
    _, eval_indices = _require_split_indices(bundle["split_indices"], eval_split)
    aligned_sample_metadata = None
    if sample_metadata is not None:
        aligned_sample_metadata = _align_table_to_eval_rows(sample_metadata, eval_indices)
    elif target_column is not None:
        aligned_sample_metadata = eval_metadata

    aligned_labels = None
    if labels is not None:
        aligned_labels = _align_table_to_eval_rows(labels, eval_indices)
    elif label_column is not None:
        aligned_labels = eval_metadata

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

    plot_artifacts = shap_plots.save_shap_plot_suite(
        shap_2d,
        feature_names,
        output_path,
        policy=policy,
        feature_matrix=X_eval.iloc[: shap_2d.shape[0]],
        dependence_features=dependence_features,
        family_spec=family_spec,
        sample_metadata=aligned_sample_metadata,
        target_column=target_column,
        labels=aligned_labels,
        label_column=label_column,
        top_n=top_n,
        rng_seed=seed,
        include_log_importance_plots=include_log_importance_plots,
        filter_zero_rows_log=filter_zero_rows_log,
    )

    shap_report = {
        "export_dir": str(export_path.resolve()),
        "task": task,
        "explainer": explainer,
        "eval_split": eval_split,
        "include_log_importance_plots": bool(include_log_importance_plots),
        "filter_zero_rows_log": bool(filter_zero_rows_log),
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
        feature_importance_png=str(plot_artifacts.get("feature_importance_png", "")),
        beeswarm_png=str(plot_artifacts.get("beeswarm_png", "")),
        shap_values_npy=shap_npy,
        shap_values_csv=shap_csv,
        artifacts=plot_artifacts,
    )


__all__ = ["run_export_shap_analysis", "OutputPaths"]
