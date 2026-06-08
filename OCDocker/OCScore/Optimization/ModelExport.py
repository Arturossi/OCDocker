#!/usr/bin/env python3

# Description
###############################################################################
'''
Export and reload best OCScore Optuna models for inference and retraining.

After a PDBbind or DUDEz study completes, :func:`export_best_model_bundle` writes a
``best_model/`` directory containing weights, architecture, retraining config,
feature metadata, and a compact trial summary.

Usage:

from OCDocker.OCScore.Optimization.ModelExport import load_exported_model
'''

# Imports
###############################################################################
from __future__ import annotations

import copy
import json
import subprocess

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

import torch.nn as nn

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Optimization.StagedOptuna import DUDEzScreeningModel
from OCDocker.OCScore.Optimization.StagedOptuna import FeatureExtractor
from OCDocker.OCScore.Optimization.StagedOptuna import PDBbindRegressionModel
from OCDocker.OCScore.Optimization.StagedOptuna import _linear_out_features
from OCDocker.OCScore.Optimization.StagedOptuna import apply_fine_tuning_mode
from OCDocker.OCScore.Optimization.StagedOptuna import build_dudez_model
from OCDocker.OCScore.Optimization.StagedOptuna import build_pdbbind_model
from OCDocker.OCScore.Utils.ContentHash import hash_dataframe_partition
from OCDocker.OCScore.Utils.ContentHash import hash_feature_list

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

LOGGER = oclogging.get_logger("ocscore.optimization.model_export")

BEST_MODEL_FILENAME = "best_model.pt"
ARCHITECTURE_FILENAME = "architecture.json"
RETRAIN_CONFIG_FILENAME = "retrain_config.json"
SUMMARY_FILENAME = "best_trial_summary.json"
FEATURE_METADATA_FILENAME = "feature_metadata.json"
SCALER_FILENAME = "scaler.joblib"
CALIBRATOR_FILENAME = "probability_calibrator.joblib"
SPLIT_INDICES_FILENAME = "split_indices.npz"


# Functions
###############################################################################
## Private ##

def _json_ready(value: Any) -> Any:
    '''Recursively convert values to JSON-serializable forms.'''

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _describe_feature_extractor(extractor: FeatureExtractor) -> dict[str, Any]:
    hidden_sizes = _linear_out_features(extractor.encoder)[:-1]
    return {
        "hidden_sizes": [int(size) for size in hidden_sizes],
        "latent_dim": int(extractor.latent_dim),
        "projection_dim": int(extractor.output_dim if extractor.projection is not None else 0),
        "output_dim": int(extractor.output_dim),
    }


def _build_architecture_document(
        task: str,
        model_config: dict[str, Any],
        input_size: int,
        model: nn.Module,
    ) -> dict[str, Any]:
    document: dict[str, Any] = {
        "task": task,
        "input_size": int(input_size),
        "model_config": _json_ready(model_config),
    }
    if isinstance(model, PDBbindRegressionModel):
        extractor = model.feature_extractor
        document["encoder"] = {
            "hidden_sizes": list(model_config.get("encoder_hidden_sizes", [])),
            "latent_dim": int(model_config.get("encoder_latent_dim", 0)),
            "depth": int(model_config.get("encoder_depth", 0)),
            "is_monotonic": bool(model_config.get("encoder_is_monotonic", True)),
            "activation": str(model_config.get("encoder_activation", "GELU")),
            "dropout": float(model_config.get("encoder_dropout", 0.0)),
            "resolved": _describe_feature_extractor(extractor),
        }
        document["projection"] = {
            "projection_dim": int(model_config.get("projection_dim", 0)),
            "enabled": int(model_config.get("projection_dim", 0)) > 0,
        }
        decoder_sizes = model_config.get("decoder_hidden_sizes")
        document["decoder"] = {
            "enabled": bool(decoder_sizes),
            "hidden_sizes": list(decoder_sizes) if decoder_sizes else [],
            "depth": int(model_config.get("decoder_depth", 0)),
            "lambda_rec": float(model_config.get("decoder_lambda_rec", 0.0)),
        }
        document["dae"] = {
            "enabled": float(model_config.get("decoder_lambda_rec", 0.0)) > 0.0,
            "noise_type": str(model_config.get("dae_noise_type", "none")),
            "mask_prob": float(model_config.get("dae_mask_prob", 0.0)),
            "gaussian_std": float(model_config.get("dae_gaussian_std", 0.0)),
        }
        document["regression_head"] = {
            "loss": str(model_config.get("pdbbind_regression_loss", "mse")),
            "huber_delta": float(model_config.get("pdbbind_huber_delta", 1.0)),
        }
        return document

    if isinstance(model, DUDEzScreeningModel):
        document["feature_extractor"] = _describe_feature_extractor(model.feature_extractor)
        document["classifier"] = {
            "hidden_size": int(model_config.get("dudez_classifier_hidden_size", 128)),
            "dropout": float(model_config.get("dudez_classifier_dropout", 0.0)),
            "activation": str(model_config.get("dudez_classifier_activation", "GELU")),
        }
        document["transfer"] = {
            "use_transfer": bool(model_config.get("dudez_use_transfer", True)),
            "fine_tuning_mode": str(model_config.get("dudez_fine_tuning_mode", "partial")),
            "num_unfrozen_layers": int(model_config.get("dudez_num_unfrozen_layers", 1)),
        }
        document["training"] = {
            "use_class_weighting": bool(model_config.get("dudez_use_class_weighting", True)),
        }
        return document

    raise ValueError(f"Unsupported model type for architecture export: {type(model)}")


def _build_retrain_config(
        task: str,
        model_config: dict[str, Any],
        selected_features: Sequence[str],
        stage_config: Mapping[str, Any],
        splits: Mapping[str, Any],
        best_trial_number: int,
        best_params: Mapping[str, Any],
        objective_metric: str,
        direction: str,
        random_seed: Optional[int],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
    config: dict[str, Any] = {
        "task": task,
        "objective_metric": objective_metric,
        "direction": direction,
        "best_trial_number": int(best_trial_number),
        "best_params": _json_ready(dict(best_params)),
        "resolved_model_config": _json_ready(model_config),
        "selected_features": list(selected_features),
        "input_size": int(len(selected_features)),
        "stage_config": _json_ready(dict(stage_config)),
        "split_config": _json_ready(splits.get("split_config", {})),
        "split_diagnostics": _json_ready(splits.get("split_diagnostics", {})),
        "random_seed": random_seed,
        "optimizer": {
            "learning_rate": float(model_config.get("optimizer_learning_rate", 0.0)),
            "weight_decay": float(model_config.get("optimizer_weight_decay", 0.0)),
            "batch_size": int(model_config.get("optimizer_batch_size", 0)),
        },
        "preprocessing": {
            "scaler": "standard" if task == "pdbbind_regression" else "none",
            "scaler_path": SCALER_FILENAME if task == "pdbbind_regression" else None,
        },
        "split_indices_path": SPLIT_INDICES_FILENAME,
        "feature_metadata_path": FEATURE_METADATA_FILENAME,
        "architecture_path": ARCHITECTURE_FILENAME,
        "weights_path": BEST_MODEL_FILENAME,
    }
    if extra:
        config["extra"] = _json_ready(dict(extra))
    return config


def _save_split_indices(export_dir: Path, splits: Mapping[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key in ("train_indices", "validation_indices", "test_indices"):
        if key in splits and splits[key] is not None:
            arrays[key] = np.asarray(splits[key], dtype=np.int64)
    if arrays:
        np.savez_compressed(export_dir / SPLIT_INDICES_FILENAME, **arrays)


def _load_split_indices(export_dir: Path) -> dict[str, np.ndarray]:
    path = export_dir / SPLIT_INDICES_FILENAME
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _build_model_from_export(
        export_dir: Path,
        device: torch.device,
        transferred_extractor: Optional[FeatureExtractor] = None,
    ) -> nn.Module:
    retrain_config = _read_json(export_dir / RETRAIN_CONFIG_FILENAME)
    architecture = _read_json(export_dir / ARCHITECTURE_FILENAME)
    model_config = dict(retrain_config["resolved_model_config"])
    input_size = int(retrain_config["input_size"])
    task = str(retrain_config["task"])

    if task == "pdbbind_regression":
        model = build_pdbbind_model(input_size=input_size, params=model_config)
    elif task == "dudez_screening":
        if bool(model_config.get("dudez_use_transfer", True)):
            if transferred_extractor is None:
                pdbbind_export = (retrain_config.get("extra") or {}).get("pdbbind_best_model_export_dir")
                if not pdbbind_export:
                    raise ValueError(
                        "DUDEz transfer export requires pdbbind_best_model_export_dir in retrain_config.extra "
                        "or an explicit transferred_extractor."
                    )
                pdbbind_bundle = load_exported_model(Path(pdbbind_export), device=device)
                transferred_extractor = pdbbind_bundle["model"].feature_extractor
            model = build_dudez_model(
                input_size=input_size,
                params=model_config,
                transferred_extractor=transferred_extractor,
            )
        else:
            extractor_arch = architecture.get("feature_extractor", {})
            model = build_dudez_model(
                input_size=input_size,
                params=model_config,
                feature_extractor_architecture=extractor_arch,
            )
    else:
        raise ValueError(f"Unsupported export task: {task}")

    weights_path = export_dir / BEST_MODEL_FILENAME
    payload = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _resolve_transferred_extractor(
        retrain_config: Mapping[str, Any],
        device: torch.device,
        pdbbind_export_dir: str | Path | None = None,
    ) -> Optional[FeatureExtractor]:
    task = str(retrain_config.get("task", ""))
    if task != "dudez_screening":
        return None

    model_config = dict(retrain_config.get("resolved_model_config") or {})
    if not bool(model_config.get("dudez_use_transfer", True)):
        return None

    export_path = pdbbind_export_dir
    if export_path is None:
        export_path = (retrain_config.get("extra") or {}).get("pdbbind_best_model_export_dir")
    if not export_path:
        raise ValueError(
            "DUDEz transfer export requires pdbbind_best_model_export_dir in retrain_config.extra "
            "or an explicit pdbbind_export_dir."
        )

    pdbbind_path = Path(export_path)
    if not pdbbind_path.is_dir():
        raise ValueError(
            f"DUDEz transfer export could not load linked PDBbind export directory: {pdbbind_path}"
        )

    pdbbind_bundle = load_exported_model(pdbbind_path, device=device)
    return pdbbind_bundle["model"].feature_extractor


def _forward_export_predictions(
        model: nn.Module,
        features: np.ndarray,
        task: str,
        device: torch.device,
    ) -> np.ndarray:
    model = model.to(device)
    model.eval()
    tensor = torch.tensor(np.asarray(features, dtype=np.float32), device=device)
    with torch.no_grad():
        if task == "pdbbind_regression":
            if not isinstance(model, PDBbindRegressionModel):
                raise TypeError("Expected PDBbindRegressionModel for pdbbind_regression export.")
            outputs = model(tensor)["prediction"]
        elif task == "dudez_screening":
            if not isinstance(model, DUDEzScreeningModel):
                raise TypeError("Expected DUDEzScreeningModel for dudez_screening export.")
            outputs = model(tensor)
        else:
            raise ValueError(f"Unsupported export task: {task}")
    return outputs.detach().cpu().numpy().reshape(-1)


## Public ##

def validate_export_features(dataframe: pd.DataFrame, selected_features: Sequence[str]) -> None:
    '''Ensure a dataframe contains all exported selected feature columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input feature table.
    selected_features : Sequence[str]
        Feature names required by the export bundle.

    Raises
    ------
    ValueError
        If any selected feature column is missing.
    '''

    missing = [column for column in selected_features if column not in dataframe.columns]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(f"Dataframe is missing selected export features: {preview}{suffix}")


def transform_export_features(
        dataframe: pd.DataFrame,
        selected_features: Sequence[str],
        scaler: Any | None,
    ) -> np.ndarray:
    '''Extract and optionally scale exported feature columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input rows containing ``selected_features``.
    selected_features : Sequence[str]
        Feature names in model input order.
    scaler : Any or None
        Optional fitted scaler (PDBbind exports).

    Returns
    -------
    np.ndarray
        Feature matrix ready for model forward pass.
    '''

    values = dataframe[list(selected_features)].to_numpy(dtype=np.float32)
    if scaler is not None:
        values = scaler.transform(values)
    return values


def predict_from_export(
        export_dir: str | Path,
        dataframe: pd.DataFrame,
        *,
        device: torch.device | str | None = "cpu",
        pdbbind_export_dir: str | Path | None = None,
    ) -> pd.DataFrame:
    '''Score rows from a wide feature table using an exported best-model bundle.

    Parameters
    ----------
    export_dir : str or Path
        Exported ``best_model/`` directory.
    dataframe : pd.DataFrame
        Wide pipeline feature table containing export ``selected_features``.
    device : torch.device or str, optional
        Torch device for inference, by default CPU.
    pdbbind_export_dir : str or Path, optional
        Override path to the linked PDBbind export for DUDEz transfer models.

    Returns
    -------
    pd.DataFrame
        Input metadata with ``ocscore_prediction`` and, for DUDEz exports,
        ``ocscore_probability``.
    '''

    export_path = Path(export_dir)
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    retrain_config = _read_json(export_path / RETRAIN_CONFIG_FILENAME)
    task = str(retrain_config["task"])
    transferred_extractor = _resolve_transferred_extractor(
        retrain_config,
        resolved_device,
        pdbbind_export_dir=pdbbind_export_dir,
    )
    bundle = load_exported_model(
        export_path,
        device=resolved_device,
        transferred_extractor=transferred_extractor,
    )
    selected_features = list(bundle["selected_features"])
    validate_export_features(dataframe, selected_features)

    feature_values = dataframe[selected_features].to_numpy(dtype=float)
    finite_mask = np.isfinite(feature_values).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped:
        LOGGER.info("Dropped %s rows with non-finite feature values.", dropped)
    if not finite_mask.any():
        raise ValueError("No rows remain after dropping non-finite feature values.")

    clean_df = dataframe.loc[finite_mask].reset_index(drop=True)
    features = transform_export_features(clean_df, selected_features, bundle.get("scaler"))
    predictions = _forward_export_predictions(
        bundle["model"],
        features,
        task=task,
        device=bundle["device"],
    )

    output = clean_df.copy()
    output["ocscore_prediction"] = predictions
    if task == "dudez_screening":
        output["ocscore_probability"] = torch.sigmoid(torch.tensor(predictions)).numpy()
        calibrator = bundle.get("calibrator")
        if calibrator is not None:
            output["ocscore_probability_calibrated"] = calibrator.predict(
                np.asarray(predictions, dtype=float)
            )
    return output


def _build_forbidden_evaluation_dataset_hashes(
        source_dataframe: Optional[pd.DataFrame],
        splits: Mapping[str, Any],
    ) -> list[str]:
    '''Compute partition hashes that external blind evaluation must reject.'''

    if source_dataframe is None:
        return []
    hashes: list[str] = []
    for key in ("train_indices", "validation_indices", "test_indices"):
        indices = splits.get(key)
        if indices is None:
            continue
        hashes.append(hash_dataframe_partition(source_dataframe, indices))
    return sorted(set(hashes))


def export_best_model_bundle(
        export_dir: str | Path,
        task: str,
        model: nn.Module,
        model_config: Mapping[str, Any],
        selected_features: Sequence[str],
        best_trial_number: int,
        best_objective_value: float,
        validation_metrics: Mapping[str, Any],
        test_metrics: Mapping[str, Any],
        stage_config: Mapping[str, Any],
        splits: Mapping[str, Any],
        objective_metric: str,
        direction: str,
        best_params: Mapping[str, Any],
        random_seed: Optional[int] = None,
        source_checkpoint_path: Optional[str] = None,
        training_metrics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        calibrator: Optional[Any] = None,
        validate: bool = True,
        source_dataframe: Optional[Any] = None,
    ) -> dict[str, str]:
    '''Export the best completed trial model into a reloadable bundle.

    Parameters
    ----------
    export_dir : str | Path
        Directory that will contain ``best_model.pt`` and companion metadata.
    task : str
        ``"pdbbind_regression"`` or ``"dudez_screening"``.
    model : nn.Module
        Best trained model instance.
    model_config : Mapping[str, Any]
        Resolved model configuration after conditional search-space logic.
    selected_features : Sequence[str]
        Feature names in training column order.
    best_trial_number : int
        Optuna trial number for the exported model.
    best_objective_value : float
        Final objective value on validation data.
    validation_metrics : Mapping[str, Any]
        Validation metrics for the best model.
    test_metrics : Mapping[str, Any]
        Test metrics for the best model.
    stage_config : Mapping[str, Any]
        Serialized stage configuration (Optuna settings, splits, pruning, etc.).
    splits : Mapping[str, Any]
        Prepared split payload including indices and optional scaler.
    objective_metric : str
        Effective objective metric name.
    direction : str
        Optuna optimization direction.
    best_params : Mapping[str, Any]
        Raw Optuna trial parameters.
    random_seed : int | None, optional
        Random seed used by the stage, by default None.
    source_checkpoint_path : str | None, optional
        Path to the source ``*_best.pt`` checkpoint, by default None.
    training_metrics : Mapping[str, Any] | None, optional
        Optional training diagnostics, by default None.
    extra : Mapping[str, Any] | None, optional
        Additional export metadata, by default None.
    calibrator : Any | None, optional
        Fitted :class:`~OCDocker.OCScore.Analysis.Metrics.Calibration.ProbabilityCalibrator`
        for DUDEz exports (saved as ``probability_calibrator.joblib``).
    validate : bool, optional
        Run :func:`validate_export_bundle` before returning, by default True.
    source_dataframe : pd.DataFrame | None, optional
        Source reduced dataframe used to compute forbidden blind-evaluation hashes.

    Returns
    -------
    dict[str, str]
        Absolute paths for exported artifacts.
    '''

    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    model_config_dict = dict(model_config)
    input_size = int(len(selected_features))

    extra_payload = dict(extra or {})
    feature_policy_metadata = extra_payload.get("feature_policy")
    feature_metadata = {
        "selected_features": list(selected_features),
        "feature_order": list(selected_features),
        "input_size": input_size,
        "task": task,
        "selected_features_hash": hash_feature_list(selected_features),
    }
    if feature_policy_metadata:
        feature_metadata["feature_policy"] = _json_ready(feature_policy_metadata)
    forbidden_hashes = _build_forbidden_evaluation_dataset_hashes(source_dataframe, splits)
    if forbidden_hashes:
        feature_metadata["forbidden_evaluation_dataset_hashes"] = forbidden_hashes
    _write_json(export_path / FEATURE_METADATA_FILENAME, feature_metadata)

    scaler = splits.get("scaler")
    if scaler is not None:
        joblib.dump(scaler, export_path / SCALER_FILENAME)
    if calibrator is not None and task == "dudez_screening":
        joblib.dump(calibrator, export_path / CALIBRATOR_FILENAME)

    splits_payload = dict(splits)
    if "split_config" not in splits_payload and "split_config" in stage_config:
        splits_payload["split_config"] = stage_config.get("split_config")
    _save_split_indices(export_path, splits_payload)

    architecture_path = export_path / ARCHITECTURE_FILENAME
    retrain_config_path = export_path / RETRAIN_CONFIG_FILENAME
    weights_path = export_path / BEST_MODEL_FILENAME
    summary_path = export_path / SUMMARY_FILENAME

    architecture = _build_architecture_document(
        task=task,
        model_config=model_config_dict,
        input_size=input_size,
        model=model,
    )
    _write_json(architecture_path, architecture)

    retrain_config = _build_retrain_config(
        task=task,
        model_config=model_config_dict,
        selected_features=selected_features,
        stage_config=stage_config,
        splits=splits_payload,
        best_trial_number=best_trial_number,
        best_params=best_params,
        objective_metric=objective_metric,
        direction=direction,
        random_seed=random_seed,
        extra={
            **extra_payload,
            **({"forbidden_evaluation_dataset_hashes": forbidden_hashes} if forbidden_hashes else {}),
        },
    )
    _write_json(retrain_config_path, retrain_config)

    torch.save(
        {
            "task": task,
            "input_size": input_size,
            "model_config": model_config_dict,
            "model_state_dict": model.state_dict(),
        },
        weights_path,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    summary = {
        "trial_number": int(best_trial_number),
        "final_objective_value": float(best_objective_value),
        "objective_metric": objective_metric,
        "direction": direction,
        "task": task,
        "validation_metrics": _json_ready(validation_metrics),
        "test_metrics": _json_ready(test_metrics),
        "training_metrics": _json_ready(training_metrics or {}),
        "checkpoint_path": str(weights_path.resolve()),
        "source_checkpoint_path": source_checkpoint_path,
        "architecture_path": str(architecture_path.resolve()),
        "retrain_config_path": str(retrain_config_path.resolve()),
        "feature_metadata_path": str((export_path / FEATURE_METADATA_FILENAME).resolve()),
        "export_dir": str(export_path.resolve()),
        "timestamp_utc": timestamp,
        "git_commit": _git_commit_hash(),
    }
    if feature_policy_metadata:
        summary["feature_policy"] = _json_ready(feature_policy_metadata)
    _write_json(summary_path, summary)

    paths = {
        "export_dir": str(export_path.resolve()),
        "best_model_path": str(weights_path.resolve()),
        "architecture_path": str(architecture_path.resolve()),
        "retrain_config_path": str(retrain_config_path.resolve()),
        "best_trial_summary_path": str(summary_path.resolve()),
        "feature_metadata_path": str((export_path / FEATURE_METADATA_FILENAME).resolve()),
    }
    if scaler is not None:
        paths["scaler_path"] = str((export_path / SCALER_FILENAME).resolve())
    if (export_path / CALIBRATOR_FILENAME).exists():
        paths["calibrator_path"] = str((export_path / CALIBRATOR_FILENAME).resolve())
    if (export_path / SPLIT_INDICES_FILENAME).exists():
        paths["split_indices_path"] = str((export_path / SPLIT_INDICES_FILENAME).resolve())

    if validate:
        validate_export_bundle(export_path)

    LOGGER.info("Exported best model bundle to %s", export_path)
    return paths


def validate_export_bundle(export_dir: str | Path, device: Optional[torch.device] = None) -> dict[str, Any]:
    '''Rebuild the exported model and verify weights load successfully.

    Parameters
    ----------
    export_dir : str | Path
        Exported ``best_model/`` directory.
    device : torch.device | None, optional
        Device used for reconstruction smoke test, by default CPU.

    Returns
    -------
    dict[str, Any]
        Validation metadata including parameter counts.
    '''

    export_path = Path(export_dir)
    required = [
        export_path / BEST_MODEL_FILENAME,
        export_path / ARCHITECTURE_FILENAME,
        export_path / RETRAIN_CONFIG_FILENAME,
        export_path / FEATURE_METADATA_FILENAME,
        export_path / SUMMARY_FILENAME,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Export bundle is incomplete. Missing: {missing}")

    resolved_device = device or torch.device("cpu")
    retrain_config = _read_json(export_path / RETRAIN_CONFIG_FILENAME)
    transferred_extractor = None
    if str(retrain_config["task"]) == "dudez_screening":
        extra = retrain_config.get("extra") or {}
        pdbbind_export = extra.get("pdbbind_best_model_export_dir")
        if pdbbind_export and Path(pdbbind_export).exists():
            transferred_extractor = load_exported_model(Path(pdbbind_export), device=resolved_device)["model"].feature_extractor

    model = _build_model_from_export(
        export_path,
        device=resolved_device,
        transferred_extractor=transferred_extractor,
    )
    parameter_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "export_dir": str(export_path.resolve()),
        "task": str(retrain_config["task"]),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "validated": True,
    }


def load_exported_model(
        export_dir: str | Path,
        device: Optional[torch.device | str] = "cpu",
        transferred_extractor: Optional[FeatureExtractor] = None,
    ) -> dict[str, Any]:
    '''Load an exported best-model bundle for inference or evaluation.

    Parameters
    ----------
    export_dir : str | Path
        Exported ``best_model/`` directory.
    device : torch.device | str | None, optional
        Target device, by default CPU.
    transferred_extractor : FeatureExtractor | None, optional
        Optional transferred extractor for DUDEz transfer exports.

    Returns
    -------
    dict[str, Any]
        Loaded model and metadata keys including ``model``, ``scaler``,
        ``selected_features``, ``architecture``, ``retrain_config``, and ``summary``.
    '''

    export_path = Path(export_dir)
    resolved_device = torch.device(device) if device is not None else torch.device("cpu")
    model = _build_model_from_export(
        export_path,
        device=resolved_device,
        transferred_extractor=transferred_extractor,
    )

    scaler = None
    scaler_path = export_path / SCALER_FILENAME
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    calibrator = None
    calibrator_path = export_path / CALIBRATOR_FILENAME
    if calibrator_path.exists():
        calibrator = joblib.load(calibrator_path)

    feature_metadata = _read_json(export_path / FEATURE_METADATA_FILENAME)
    selected_features = list(feature_metadata["selected_features"])
    return {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "selected_features": selected_features,
        "feature_metadata": feature_metadata,
        "architecture": _read_json(export_path / ARCHITECTURE_FILENAME),
        "retrain_config": _read_json(export_path / RETRAIN_CONFIG_FILENAME),
        "summary": _read_json(export_path / SUMMARY_FILENAME),
        "split_indices": _load_split_indices(export_path),
        "device": resolved_device,
        "export_dir": str(export_path.resolve()),
    }


def retrain_from_export(
        export_dir: str | Path,
        pdbbind_df: Optional[Any] = None,
        dudez_df: Optional[Any] = None,
        device: Optional[torch.device | str] = None,
        use_saved_split_indices: bool = True,
    ) -> dict[str, Any]:
    '''Prepare data splits and a fresh model for retraining from an export bundle.

    This does not run training; it returns the model, optimizer-related settings,
    prepared splits, and metadata needed to launch a training loop.

    Parameters
    ----------
    export_dir : str | Path
        Exported ``best_model/`` directory.
    pdbbind_df : pd.DataFrame | None, optional
        Reduced PDBbind dataframe for regression retraining.
    dudez_df : pd.DataFrame | None, optional
        Reduced DUDEz dataframe for screening retraining.
    device : torch.device | str | None, optional
        Target device, by default CPU.
    use_saved_split_indices : bool, optional
        Reuse exported split indices when available, by default True.

    Returns
    -------
    dict[str, Any]
        Retraining payload with ``model``, ``splits``, ``model_config``, and
        ``stage_config``.
    '''

    bundle = load_exported_model(export_dir, device=device or "cpu")
    retrain_config = bundle["retrain_config"]
    task = str(retrain_config["task"])
    selected_features = bundle["selected_features"]
    model_config = dict(retrain_config["resolved_model_config"])
    stage_config = dict(retrain_config.get("stage_config", {}))
    split_config_dict = dict(retrain_config.get("split_config", {}))

    if task == "pdbbind_regression":
        if pdbbind_df is None:
            raise ValueError("pdbbind_df is required to retrain a PDBbind export.")
        from OCDocker.OCScore.Optimization.StagedOptuna import PDBbindSplitConfig
        from OCDocker.OCScore.Optimization.StagedOptuna import prepare_pdbbind_regression_data

        split_config = PDBbindSplitConfig(**split_config_dict) if split_config_dict else PDBbindSplitConfig()
        splits = prepare_pdbbind_regression_data(pdbbind_df, selected_features, split_config=split_config)
        if use_saved_split_indices and bundle["split_indices"]:
            from sklearn.preprocessing import StandardScaler

            indices = bundle["split_indices"]
            train_idx = indices.get("train_indices")
            val_idx = indices.get("validation_indices")
            test_idx = indices.get("test_indices")
            if train_idx is not None and val_idx is not None and test_idx is not None:
                X = pdbbind_df[selected_features].to_numpy(dtype=np.float32)
                y = pdbbind_df[split_config.target_column].to_numpy(dtype=np.float32)
                scaler = StandardScaler()
                splits = {
                    "X_train": scaler.fit_transform(X[train_idx]).astype(np.float32),
                    "y_train": y[train_idx],
                    "X_val": scaler.transform(X[val_idx]).astype(np.float32),
                    "y_val": y[val_idx],
                    "X_test": scaler.transform(X[test_idx]).astype(np.float32),
                    "y_test": y[test_idx],
                    "train_indices": train_idx,
                    "validation_indices": val_idx,
                    "test_indices": test_idx,
                    "scaler": scaler,
                    "split_diagnostics": splits.get("split_diagnostics", {}),
                }
        model = build_pdbbind_model(input_size=len(selected_features), params=model_config)
    elif task == "dudez_screening":
        if dudez_df is None:
            raise ValueError("dudez_df is required to retrain a DUDEz export.")
        from OCDocker.OCScore.Optimization.StagedOptuna import DUDEzSplitConfig
        from OCDocker.OCScore.Optimization.StagedOptuna import derive_dudez_labels
        from OCDocker.OCScore.Optimization.StagedOptuna import prepare_dudez_screening_data

        kind_column = str(stage_config.get("kind_column", "kind"))
        target_group_column = str(stage_config.get("target_group_column", "receptor"))
        labels = derive_dudez_labels(dudez_df, kind_column=kind_column)
        groups = (
            dudez_df[target_group_column].to_numpy()
            if target_group_column in dudez_df.columns
            else None
        )
        split_config = DUDEzSplitConfig(**split_config_dict) if split_config_dict else DUDEzSplitConfig()
        splits = prepare_dudez_screening_data(
            dudez_df,
            selected_features,
            labels,
            groups=groups,
            split_config=split_config,
            target_group_column=target_group_column if groups is not None else None,
        )
        transferred_extractor = None
        if bool(model_config.get("dudez_use_transfer", True)):
            extra = retrain_config.get("extra") or {}
            pdbbind_export = extra.get("pdbbind_best_model_export_dir")
            if pdbbind_export:
                transferred_extractor = load_exported_model(pdbbind_export, device=bundle["device"])["model"].feature_extractor
        model = build_dudez_model(
            input_size=len(selected_features),
            params=model_config,
            transferred_extractor=transferred_extractor,
            feature_extractor_architecture=bundle["architecture"].get("feature_extractor"),
        )
    else:
        raise ValueError(f"Unsupported export task: {task}")

    resolved_device = bundle["device"]
    model.to(resolved_device)
    return {
        "model": model,
        "splits": splits,
        "model_config": model_config,
        "stage_config": stage_config,
        "retrain_config": retrain_config,
        "selected_features": selected_features,
        "device": resolved_device,
    }


__all__ = [
    "ARCHITECTURE_FILENAME",
    "BEST_MODEL_FILENAME",
    "FEATURE_METADATA_FILENAME",
    "RETRAIN_CONFIG_FILENAME",
    "SCALER_FILENAME",
    "SUMMARY_FILENAME",
    "export_best_model_bundle",
    "load_exported_model",
    "predict_from_export",
    "retrain_from_export",
    "transform_export_features",
    "validate_export_bundle",
    "validate_export_features",
]
