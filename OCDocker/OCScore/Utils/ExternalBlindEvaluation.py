#!/usr/bin/env python3

# Description
###############################################################################
'''
External blind evaluation for exported OCScore models.

Loads frozen preprocessing and model artifacts, applies them once to an external
dataset, and writes predictions plus a provenance-rich evaluation report.
No feature reduction, Optuna, or fitting occurs on blind rows.
'''

from __future__ import annotations

# Imports
###############################################################################
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, cast

import numpy as np
import pandas as pd

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocranking
import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged
import OCDocker.Toolbox.Reproducibility as ocrepro

from OCDocker.OCScore.Utils.ContentHash import hash_dataframe_partition
from OCDocker.OCScore.Utils.ContentHash import hash_feature_list
from OCDocker.OCScore.Utils.ContentHash import hash_file
from OCDocker.OCScore.Utils.ContentHash import hash_json_dict

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

EXTERNAL_BLIND_EVAL_JSON = "external_blind_evaluation.json"
EXTERNAL_BLIND_PREDICTIONS_CSV = "external_blind_predictions.csv"


@dataclass
class ExternalBlindConfig:
    """Configuration for external blind evaluation."""

    export_dir: str | Path
    blind_csv: str | Path
    output_dir: str | Path
    label_column: Optional[str] = None
    group_column: Optional[str] = "receptor"
    kind_column: Optional[str] = "kind"
    device: str = "cpu"
    pdbbind_export_dir: Optional[str | Path] = None
    forbidden_dataset_hashes: Optional[Sequence[str]] = None
    command: Optional[list[str]] = None


def _optional_torch_version() -> Optional[str]:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return None


def _optional_cuda_info() -> Optional[dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
        }
    except ImportError:
        return None


def _resolve_label_column(task: str, dataframe: pd.DataFrame, label_column: Optional[str]) -> Optional[str]:
    if label_column and label_column in dataframe.columns:
        return label_column
    if task == "pdbbind_regression":
        for candidate in ("experimental", "pKd", "label", "target"):
            if candidate in dataframe.columns:
                return candidate
    if task == "dudez_screening":
        for candidate in ("label", "kind"):
            if candidate in dataframe.columns:
                return candidate
    return None


def _build_dudez_labels(dataframe: pd.DataFrame, label_column: str, kind_column: str) -> np.ndarray:
    if label_column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[label_column]):
        return cast(np.ndarray, dataframe[label_column].to_numpy(dtype=float))
    if kind_column not in dataframe.columns:
        raise ValueError(f"DUDEz blind evaluation requires {kind_column!r} or numeric labels.")
    normalized = dataframe[kind_column].astype(str).str.strip().str.lower()
    return np.where(normalized.isin({"ligands", "active", "actives"}), 1.0, 0.0)


def _compute_metrics(
        task: str,
        predictions: pd.DataFrame,
        label_column: Optional[str],
        group_column: Optional[str],
        kind_column: Optional[str],
        bedroc_alpha: float = 20.0,
    ) -> tuple[Optional[dict[str, Any]], str]:
    if label_column is None or label_column not in predictions.columns:
        return None, "labels_unavailable"

    if task == "pdbbind_regression":
        y_true = predictions[label_column].to_numpy(dtype=float)
        y_pred = predictions["ocscore_prediction"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not mask.any():
            return None, "labels_nonfinite"
        metrics = ocstaged.evaluate_regression_metrics(y_true[mask], y_pred[mask])
        return metrics, "pdbbind_regression_holdout"

    if task == "dudez_screening":
        y_true = _build_dudez_labels(predictions, label_column, kind_column or "kind")
        y_score = predictions["ocscore_probability"].to_numpy(dtype=float)
        groups = predictions[group_column].to_numpy() if group_column and group_column in predictions.columns else None
        metrics = ocranking.evaluate_screening_metrics(
            y_true.astype(int),
            y_score,
            groups=groups,
            higher_is_better=True,
            bedroc_alpha=bedroc_alpha,
        )
        metric_scope = "dudez_grouped_receptor" if groups is not None else "dudez_global"
        return metrics, metric_scope

    raise ValueError(f"Unsupported export task for external blind evaluation: {task}")


def run_external_blind_evaluation(config: ExternalBlindConfig) -> dict[str, Any]:
    '''Run one-shot external blind evaluation using a frozen export bundle.'''

    export_path = Path(config.export_dir).resolve()
    blind_path = Path(config.blind_csv).resolve()
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = ocexport.load_exported_model(export_path, device=config.device)
    retrain_config = bundle["retrain_config"]
    task = str(retrain_config["task"])
    bedroc_alpha = float((retrain_config.get("stage_config") or {}).get("bedroc_alpha", 20.0))
    feature_metadata = bundle["feature_metadata"]
    selected_features = list(bundle["selected_features"])
    selected_hash = hash_feature_list(selected_features)
    metadata_hash = feature_metadata.get("selected_features_hash")
    if metadata_hash and metadata_hash != selected_hash:
        raise ValueError(
            "Model checkpoint selected_features hash does not match feature_metadata.json."
        )

    blind_hash = hash_file(blind_path)
    forbidden = {str(value) for value in (config.forbidden_dataset_hashes or []) if value}
    auto_forbidden = feature_metadata.get("forbidden_evaluation_dataset_hashes") or []
    retrain_extra = (bundle.get("retrain_config") or {}).get("extra") or {}
    auto_forbidden = list(auto_forbidden) + list(retrain_extra.get("forbidden_evaluation_dataset_hashes") or [])
    forbidden.update(str(value) for value in auto_forbidden if value)

    blind_df = pd.read_csv(blind_path, low_memory=False)
    blind_content_hash = hash_dataframe_partition(blind_df, range(len(blind_df)))
    if blind_hash in forbidden or blind_content_hash in forbidden:
        raise ValueError(
            "Blind dataset hash matches a forbidden training/validation/test dataset hash."
        )

    extra_columns = sorted(set(blind_df.columns) - set(selected_features))
    missing_features = [column for column in selected_features if column not in blind_df.columns]
    if missing_features:
        raise ValueError(f"Blind dataset is missing required selected features: {missing_features}")

    scaler = bundle.get("scaler")
    retrain_scaler_required = task == "pdbbind_regression"
    if retrain_scaler_required and scaler is None:
        raise ValueError("PDBbind export requires a frozen scaler artifact for external blind evaluation.")

    predictions = ocexport.predict_from_export(
        export_path,
        blind_df,
        device=config.device,
        pdbbind_export_dir=config.pdbbind_export_dir,
    )

    label_column = _resolve_label_column(task, predictions, config.label_column)
    metrics, metric_scope = _compute_metrics(
        task,
        predictions,
        label_column,
        config.group_column,
        config.kind_column,
        bedroc_alpha=bedroc_alpha,
    )

    predictions_path = output_dir / EXTERNAL_BLIND_PREDICTIONS_CSV
    predictions.to_csv(predictions_path, index=False)

    model_checkpoint_path = export_path / ocexport.BEST_MODEL_FILENAME
    model_config_path = export_path / ocexport.RETRAIN_CONFIG_FILENAME
    scaler_path = export_path / ocexport.SCALER_FILENAME

    report: dict[str, Any] = {
        "task": task,
        "validation_mode": "external-blind",
        "model_export_dir": str(export_path),
        "model_checkpoint_hash": hash_file(model_checkpoint_path) if model_checkpoint_path.is_file() else None,
        "model_config_hash": hash_file(model_config_path) if model_config_path.is_file() else None,
        "selected_features_hash": selected_hash,
        "removed_features_hash": feature_metadata.get("removed_features_hash"),
        "blind_dataset_path": str(blind_path),
        "blind_dataset_hash": blind_hash,
        "blind_dataset_content_hash": blind_content_hash,
        "n_rows": int(len(predictions)),
        "n_features_required": len(selected_features),
        "n_features_missing": len(missing_features),
        "missing_features": missing_features,
        "extra_features_ignored": extra_columns,
        "preprocessing_artifacts_used": {
            "selected_features_json": feature_metadata.get("feature_metadata_path"),
            "scaler_path": str(scaler_path) if scaler_path.is_file() else None,
            "feature_order": selected_features,
        },
        "scaler_strategy": retrain_config.get("preprocessing", {}).get("scaler"),
        "scaler_fit_scope": "train_only",
        "metrics": metrics,
        "metric_scope": metric_scope,
        "group_column": config.group_column,
        "label_column": label_column,
        "n_groups_total": int(predictions[config.group_column].nunique())
        if config.group_column and config.group_column in predictions.columns
        else None,
        "n_groups_used": int(predictions[config.group_column].nunique())
        if config.group_column and config.group_column in predictions.columns and metrics is not None
        else None,
        "command": list(config.command or []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_version": ocrepro.generate_reproducibility_manifest(include_python_packages=False).get("ocdocker_version"),
        "python_version": sys.version,
        "torch_version": _optional_torch_version(),
        "cuda_info_if_available": _optional_cuda_info(),
        "artifact_hashes": {
            "predictions_csv": hash_file(predictions_path),
        },
    }
    if scaler_path.is_file():
        report["artifact_hashes"]["scaler"] = hash_file(scaler_path)
    report["evaluation_json_hash"] = hash_json_dict(report)

    eval_path = output_dir / EXTERNAL_BLIND_EVAL_JSON
    eval_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["external_blind_evaluation_json"] = str(eval_path)
    report["external_blind_predictions_csv"] = str(predictions_path)
    return report


__all__ = [
    "EXTERNAL_BLIND_EVAL_JSON",
    "EXTERNAL_BLIND_PREDICTIONS_CSV",
    "ExternalBlindConfig",
    "run_external_blind_evaluation",
]
