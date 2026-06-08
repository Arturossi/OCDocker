#!/usr/bin/env python3

# Description
###############################################################################
'''
Current staged Optuna protocol for OCScore PDBbind/DUDEz modeling.

It is imported as:

from OCDocker.OCScore.Optimization.StagedOptuna import StagedProtocol
'''

# Imports
###############################################################################
from __future__ import annotations

import copy
import json
import math
import os
import random
import threading
import time

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim

from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocrank
from OCDocker.OCScore.Analysis.Metrics.Calibration import enrich_dudez_export_metrics
from OCDocker.OCScore.Analysis.Metrics.Ranking import evaluate_screening_metrics
import OCDocker.Toolbox.Logging as oclogging

from OCDocker.OCScore.Utils.FixedOuterSplit import FixedOuterSplitAssignment
from OCDocker.OCScore.Utils.FixedOuterSplit import build_replica_split_alignment_metadata
from OCDocker.OCScore.Utils.FixedOuterSplit import validate_replica_split_alignment
from OCDocker.OCScore.Optimization.OptunaSearchSpace import DUDEzSearchSpaceConfig
from OCDocker.OCScore.Optimization.OptunaSearchSpace import EncoderSearchSpace
from OCDocker.OCScore.Optimization.OptunaSearchSpace import PDBBIND_SEARCH_PHASE_ENCODER_REGRESSION
from OCDocker.OCScore.Optimization.OptunaSearchSpace import PDBBIND_SEARCH_PHASE_FULL
from OCDocker.OCScore.Optimization.OptunaSearchSpace import PDBbindSearchSpaceConfig
from OCDocker.OCScore.Optimization.OptunaSearchSpace import pdbbind_search_space_for_phase
from OCDocker.OCScore.Optimization.OptunaSearchSpace import validate_pdbbind_search_phase
from OCDocker.OCScore.Optimization.OptunaSearchSpace import build_activation_module
from OCDocker.OCScore.Optimization.OptunaSearchSpace import search_space_to_summary
from OCDocker.OCScore.Utils.DUDEzScaling import DUDEzScalingConfig
from OCDocker.OCScore.Utils.DUDEzScaling import scale_dudez_features
from OCDocker.OCScore.Utils.DUDEzSplit import DUDEzSplitConfig
from OCDocker.OCScore.Utils.DUDEzSplit import dudez_receptor_heldout_complete_config
from OCDocker.OCScore.Utils.DUDEzSplit import split_dudez_by_receptor_and_kind
from OCDocker.OCScore.Utils.PDBbindSplit import PDBbindSplitConfig
from OCDocker.OCScore.Utils.PDBbindSplit import split_pdbbind_regression
from OCDocker.OCScore.Optimization.OptunaStorage import resolve_optuna_storage
from OCDocker.OCScore.Optimization.Protocol import ProtocolContext, ProtocolStage, StagedProtocol

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

LOGGER = oclogging.get_logger("ocscore.optimization.staged_optuna")

OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS_ENV = "OCSCORE_OPTUNA_SQLITE_LOCK_RETRIES"
OPTUNA_SQLITE_LOCK_RETRY_SECONDS_ENV = "OCSCORE_OPTUNA_SQLITE_LOCK_RETRY_SECONDS"
DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS = 5
DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_SECONDS = 5.0


def _sqlite_lock_retry_attempts() -> int:
    raw = os.environ.get(OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS_ENV)
    if raw is None:
        return DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS


def _sqlite_lock_retry_base_seconds() -> float:
    raw = os.environ.get(OPTUNA_SQLITE_LOCK_RETRY_SECONDS_ENV)
    if raw is None:
        return DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_OPTUNA_SQLITE_LOCK_RETRY_SECONDS


def _is_sqlite_database_locked_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if "database is locked" in message or "database table is locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _sleep_after_sqlite_lock(stage_label: str, attempt: int, max_attempts: int, exc: BaseException) -> None:
    delay = min(60.0, _sqlite_lock_retry_base_seconds() * (2 ** max(0, attempt - 1)))
    LOGGER.warning(
        "Optuna SQLite storage locked during %s; retrying attempt %s/%s after %.1fs: %s",
        stage_label,
        attempt + 1,
        max_attempts,
        delay,
        exc,
    )
    if delay > 0:
        time.sleep(delay)


def _create_study_with_sqlite_lock_retry(stage_label: str, **kwargs) -> optuna.study.Study:
    max_attempts = _sqlite_lock_retry_attempts()
    for attempt in range(1, max_attempts + 1):
        try:
            return optuna.create_study(**kwargs)
        except Exception as exc:
            if not _is_sqlite_database_locked_error(exc) or attempt >= max_attempts:
                raise
            _sleep_after_sqlite_lock(stage_label, attempt, max_attempts, exc)
    raise RuntimeError("unreachable Optuna create_study retry state")


def _optimize_with_sqlite_lock_retry(
        study: optuna.study.Study,
        objective,
        *,
        n_trials: int,
        n_jobs: int,
        stage_label: str,
    ) -> None:
    target_trial_count = len(study.trials) + int(n_trials)
    max_attempts = _sqlite_lock_retry_attempts()
    attempt = 1
    while True:
        remaining_trials = max(0, target_trial_count - len(study.trials))
        if remaining_trials <= 0:
            return
        try:
            study.optimize(objective, n_trials=remaining_trials, n_jobs=n_jobs)
            return
        except Exception as exc:
            if not _is_sqlite_database_locked_error(exc) or attempt >= max_attempts:
                raise
            _sleep_after_sqlite_lock(stage_label, attempt, max_attempts, exc)
            attempt += 1


DEFAULT_PDBBIND_SEARCH_SPACE = PDBbindSearchSpaceConfig()
DEFAULT_DUDEZ_SEARCH_SPACE = DUDEzSearchSpaceConfig()

DUDEZ_PRIMARY_METRICS = ["BEDROC", "PR-AUC", "ROC-AUC", "EF1%", "EF5%", "NDCG@1%", "NDCG@5%"]
DUDEZ_EARLY_ENRICHMENT_RANKING_METRICS = frozenset({
    "BEDROC",
    "EF1%",
    "EF5%",
    "NDCG@1%",
    "NDCG@5%",
})
PDBBIND_REPORT_METRICS = ["RMSE", "MAE", "Pearson r", "Spearman rho", "R2"]
DUDEZ_REPORT_METRICS = [
    "ROC-AUC",
    "PR-AUC",
    "BEDROC",
    "EF1%",
    "EF5%",
    "NDCG@1%",
    "NDCG@5%",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "TP",
    "FP",
    "TN",
    "FN",
    "Brier",
    "Log-loss",
    "ECE",
    "Brier_calibrated",
    "Log-loss_calibrated",
    "ECE_calibrated",
]
OCSCORE_NON_FEATURE_COLUMNS = [
    "receptor",
    "name",
    "dataset",
    "kind",
    "label",
    "Protein",
    "resolution",
    "release_year",
    "-logKd/Ki",
    "Ki/Kd",
    "Ki/Kd_relation",
    "Ki/Kd_value",
    "Ki/Kd_order",
    "Ki/Kd_raw_value",
    "Ki/Kd_raw_unit",
    "dG",
    "dG_kcal_mol",
    "reference",
    "ligand_name",
    "index_comment",
    "experimental",
]
FORBIDDEN_MIXED_OBJECTIVE_PATTERNS = [
    "RMSE-AUC",
    "RMSE-BEDROC",
    "RMSE-PR-AUC",
    "RMSE+",
    "WEIGHTED",
    "COMBINED",
]


# Classes
###############################################################################

@dataclass
class PDBbindOptunaConfig:
    """Configuration for the PDBbind regression Optuna stage.

    Parameters
    ----------
    target_column : str, optional
        Regression target column, by default "experimental".
    n_trials : int, optional
        Number of Optuna trials, by default 10.
    epochs : int, optional
        Training epochs per trial, by default 100.
    storage : str | None, optional
        Optuna storage URL. Use "auto" to create ``optuna.db`` in the protocol
        output directory (shared across stages and replicas), by default "auto".
    study_name : str, optional
        Optuna study name, by default "PDBbind_Regression_Optimization".
    load_if_exists : bool, optional
        Reuse an existing Optuna study with the same name, by default False.
    validation_size : float, optional
        Fraction of non-test data held out for validation, by default 0.2.
    test_size : float, optional
        Fraction held out for test metrics, by default 0.2.
    objective_metric : str, optional
        Stage objective metric. Must be "RMSE", by default "RMSE".
    direction : str, optional
        Optuna direction. Must be "minimize", by default "minimize".
    sampler_seed : int | None, optional
        Seed for the Optuna sampler, by default None.
    random_seed : int | None, optional
        Stage-specific seed override, by default None.
    use_gpu : bool, optional
        Use CUDA when available, by default True.
    verbose : bool, optional
        Reserved for compatibility with OCScore verbosity patterns, by default False.
    n_jobs : int, optional
        Number of parallel Optuna jobs, by default 1.
    search_space : PDBbindSearchSpaceConfig | None, optional
        Centralized Optuna search-space definition. If None, defaults are used.
    search_phase : str, optional
        Staged search phase: ``full`` (default) or ``encoder_regression`` (Phase 1:
        encoder + regression only). Ignored when ``search_space`` is set explicitly.
    enable_pruning : bool, optional
        When False, Optuna uses ``NopPruner`` (recommended for Phase 1 experiments).
    pruner_n_startup_trials : int | None, optional
        MedianPruner startup trials. When None, ``max(10% of n_trials, 5)``.
    pruner_n_warmup_steps : int | None, optional
        MedianPruner warmup epochs before pruning. When None,
        ``max(10% of epochs, 10)``.
    split_config : PDBbindSplitConfig | None, optional
        Affinity-aware PDBbind split configuration. When None, defaults to
        quantile-bin stratified splitting using ``target_column``,
        ``validation_size``, and ``test_size``.
    """

    target_column: str = "experimental"
    n_trials: int = 10
    epochs: int = 100
    storage: Optional[str] = "auto"
    study_name: str = "PDBbind_Regression_Optimization"
    search_phase: str = PDBBIND_SEARCH_PHASE_FULL
    enable_pruning: bool = True
    pruner_n_startup_trials: Optional[int] = None
    pruner_n_warmup_steps: Optional[int] = None
    load_if_exists: bool = False
    validation_size: float = 0.2
    test_size: float = 0.2
    objective_metric: str = "RMSE"
    direction: str = "minimize"
    sampler_seed: Optional[int] = None
    random_seed: Optional[int] = None
    use_gpu: bool = True
    verbose: bool = False
    n_jobs: int = 1
    search_space: Optional[PDBbindSearchSpaceConfig] = None
    split_config: Optional[PDBbindSplitConfig] = None


@dataclass
class DUDEzOptunaConfig:
    """Configuration for the DUDEz screening Optuna stage.

    Parameters
    ----------
    kind_column : str, optional
        Column used to derive active/decoy labels, by default "kind".
    target_group_column : str, optional
        Column used for target-grouped splits, by default "receptor".
    primary_metric : str, optional
        DUDEz validation objective metric, by default "BEDROC".
    n_trials : int, optional
        Number of Optuna trials, by default 10.
    epochs : int, optional
        Training epochs per trial, by default 100.
    storage : str | None, optional
        Optuna storage URL. Use "auto" to create ``optuna.db`` in the protocol
        output directory (shared across stages and replicas), by default "auto".
    study_name : str, optional
        Optuna study name, by default "DUDEz_Screening_Optimization".
    load_if_exists : bool, optional
        Reuse an existing Optuna study with the same name, by default False.
    validation_size : float, optional
        Fraction of non-test data held out for validation, by default 0.2.
    test_size : float, optional
        Fraction held out for test metrics, by default 0.2.
    direction : str, optional
        Optuna direction. Must be "maximize", by default "maximize".
    sampler_seed : int | None, optional
        Seed for the Optuna sampler, by default None.
    random_seed : int | None, optional
        Stage-specific seed override, by default None.
    use_gpu : bool, optional
        Use CUDA when available, by default True.
    verbose : bool, optional
        Reserved for compatibility with OCScore verbosity patterns, by default False.
    n_jobs : int, optional
        Number of parallel Optuna jobs, by default 1.
    allow_scratch : bool, optional
        Allow from-scratch DUDEz feature extractors as a tunable option, by default True.
    use_class_weighting : bool, optional
        Apply positive-class weighting in BCEWithLogitsLoss, by default True.
    search_space : DUDEzSearchSpaceConfig | None, optional
        Centralized Optuna search-space definition. If None, defaults are used.
    split_config : DUDEzSplitConfig | None, optional
        Receptor/kind-aware train/validation/test split configuration. When None,
        defaults to receptor-wise stratified splitting using ``validation_size``,
        ``test_size``, ``target_group_column``, and ``kind_column``.
    dudez_scaling_config : DUDEzScalingConfig | None, optional
        DUDEz feature scaling policy. When None, the PDBbind-fitted scaler is reused
        strictly for transfer-compatible DUDEz inputs.
    calibration_report_mode : str, optional
        Controls diagnostic vs validated calibration reporting in export metrics,
        by default ``"ranking_only"``.
    """

    kind_column: str = "kind"
    target_group_column: str = "receptor"
    primary_metric: str = "BEDROC"
    n_trials: int = 10
    epochs: int = 100
    storage: Optional[str] = "auto"
    study_name: str = "DUDEz_Screening_Optimization"
    load_if_exists: bool = False
    validation_size: float = 0.2
    test_size: float = 0.2
    direction: str = "maximize"
    sampler_seed: Optional[int] = None
    random_seed: Optional[int] = None
    use_gpu: bool = True
    verbose: bool = False
    n_jobs: int = 1
    allow_scratch: bool = True
    use_class_weighting: bool = True
    search_space: Optional[DUDEzSearchSpaceConfig] = None
    split_config: Optional[DUDEzSplitConfig] = None
    dudez_scaling_config: Optional[DUDEzScalingConfig] = None
    calibration_report_mode: str = "ranking_only"


@dataclass
class StageArtifacts:
    """Paths and metrics produced by a stage.

    Parameters
    ----------
    checkpoint_path : str | None, optional
        Saved model checkpoint path, by default None.
    optuna_trials_path : str | None, optional
        CSV path with Optuna trials, by default None.
    optuna_summary_path : str | None, optional
        JSON path with Optuna summary metadata, by default None.
    optuna_storage_path : str | None, optional
        Optuna storage path or URL, by default None.
    metrics : dict[str, Any], optional
        Stage validation/test metrics, by default an empty dictionary.
    """

    checkpoint_path: Optional[str] = None
    optuna_trials_path: Optional[str] = None
    optuna_summary_path: Optional[str] = None
    optuna_storage_path: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)


class TabularDataset(Dataset):
    """Simple tabular dataset for PyTorch training.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    target : np.ndarray
        Regression targets or binary labels.
    """

    def __init__(self, features: np.ndarray, target: np.ndarray) -> None:
        '''Initialize the tabular dataset.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix.
        target : np.ndarray
            Regression targets or binary labels.
        '''

        self.features = torch.tensor(np.asarray(features), dtype=torch.float32)
        self.target = torch.tensor(np.asarray(target), dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Return one feature/target pair.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Feature tensor and target tensor for ``idx``.
        '''

        return self.features[idx], self.target[idx]

    def __len__(self) -> int:
        '''Return dataset length.

        Returns
        -------
        int
            Number of rows in the dataset.
        '''

        return len(self.features)


class FeatureExtractor(nn.Module):
    """Monotonic MLP feature extractor with optional projection block.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    hidden_sizes : Sequence[int]
        Monotonic hidden layer sizes.
    latent_dim : int
        Latent encoder dimension.
    activation : str, optional
        Activation name, by default "GELU".
    dropout : float, optional
        Dropout probability, by default 0.0.
    projection_dim : int, optional
        Optional projection output dimension. A value of 0 disables projection,
        by default 0.
    """

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            latent_dim: int,
            activation: str = "GELU",
            dropout: float = 0.0,
            projection_dim: int = 0,
        ) -> None:
        super(FeatureExtractor, self).__init__()
        layers: list[nn.Module] = []
        prev = int(input_size)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, int(hidden)))
            layers.append(_build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden)
        layers.append(nn.Linear(prev, int(latent_dim)))
        layers.append(_build_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = int(latent_dim)

        self.projection: Optional[nn.Sequential]
        if projection_dim and int(projection_dim) > 0:
            self.projection = nn.Sequential(
                nn.Linear(self.latent_dim, int(projection_dim)),
                _build_activation(activation),
            )
            self.output_dim = int(projection_dim)
        else:
            self.projection = None
            self.output_dim = self.latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Compute feature embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Latent or projected feature tensor.
        '''

        latent = self.encoder(x)
        if self.projection is not None:
            return self.projection(latent)
        return latent


class PDBbindRegressionModel(nn.Module):
    """PDBbind affinity regression model with optional weak decoder.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Encoder/projection module used for affinity prediction.
    input_size : int
        Original feature dimension used by the optional decoder.
    activation : str
        Activation name used by the optional decoder.
    decoder_sizes : Sequence[int] | None, optional
        Decoder hidden sizes. If None, reconstruction is disabled.
    """

    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            input_size: int,
            activation: str,
            decoder_sizes: Optional[Sequence[int]] = None,
        ) -> None:
        super(PDBbindRegressionModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.regression_head = nn.Linear(feature_extractor.output_dim, 1)
        if decoder_sizes:
            self.decoder = _build_decoder(feature_extractor.output_dim, input_size, decoder_sizes, activation)
        else:
            self.decoder = None

    def forward(self, x: torch.Tensor, return_reconstruction: bool = False) -> dict[str, torch.Tensor | None]:
        '''Run affinity prediction and optional reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        return_reconstruction : bool, optional
            Return decoder reconstruction when available, by default False.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Feature embedding, regression prediction, and optional reconstruction.
        '''

        features = self.feature_extractor(x)
        prediction = self.regression_head(features)
        reconstruction = None
        if return_reconstruction and self.decoder is not None:
            reconstruction = self.decoder(features)
        return {
            "features": features,
            "prediction": prediction,
            "reconstruction": reconstruction,
        }


class DUDEzScreeningModel(nn.Module):
    """DUDEz classifier/ranking model using a feature extractor and new head.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Transferred or newly initialized feature extractor.
    classifier_hidden_size : int, optional
        Hidden size for the classifier head, by default 128.
    dropout : float, optional
        Classifier dropout probability, by default 0.0.
    activation : str, optional
        Activation name, by default "GELU".
    """

    def __init__(
            self,
            feature_extractor: FeatureExtractor,
            classifier_hidden_size: int = 128,
            dropout: float = 0.0,
            activation: str = "GELU",
        ) -> None:
        super(DUDEzScreeningModel, self).__init__()
        self.feature_extractor = feature_extractor
        head_layers: list[nn.Module] = [
            nn.Linear(feature_extractor.output_dim, int(classifier_hidden_size)),
            _build_activation(activation),
        ]
        if dropout > 0.0:
            head_layers.append(nn.Dropout(float(dropout)))
        head_layers.append(nn.Linear(int(classifier_hidden_size), 1))
        self.classifier_head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Run DUDEz screening prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            One-dimensional classifier logits.
        '''

        features = self.feature_extractor(x)
        return self.classifier_head(features).view(-1)


def _require_fixed_outer_split(context: ProtocolContext) -> FixedOuterSplitAssignment:
    '''Load fixed outer split metadata from the protocol context.'''

    payload = context.metadata.get("fixed_outer_split")
    if isinstance(payload, bool):
        payload = context.metadata.get("fixed_outer_split_metadata")
    if payload is None:
        raise ValueError(
            "Staged train requires fixed_outer_split metadata. "
            "Rebuild the run with train-only feature reduction and a fixed outer split."
        )
    if isinstance(payload, FixedOuterSplitAssignment):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError(
            "fixed_outer_split metadata must be a fixed split assignment mapping, "
            f"not {type(payload).__name__}."
        )
    return FixedOuterSplitAssignment.from_dict(payload)


def _replica_name_from_context(context: ProtocolContext) -> str:
    return str(context.metadata.get("replica_name") or context.output_dir)


class PDBbindOptunaStage:
    """Optuna stage that optimizes PDBbind regression by validation RMSE only.

    Parameters
    ----------
    config : PDBbindOptunaConfig | None, optional
        Stage configuration. If None, default PDBbind settings are used.
    """

    name = "pdbbind_optuna"

    def __init__(self, config: Optional[PDBbindOptunaConfig] = None) -> None:
        '''Initialize the PDBbind Optuna stage.

        Parameters
        ----------
        config : PDBbindOptunaConfig | None, optional
            Stage configuration. If None, default PDBbind settings are used.
        '''

        self.config = config or PDBbindOptunaConfig()
        self._best_value = float("inf")
        self._best_payload: dict[str, Any] = {}
        self._best_lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_best_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._best_lock = threading.Lock()

    def run(self, context: ProtocolContext) -> ProtocolContext:
        '''Run PDBbind regression optimization.

        Parameters
        ----------
        context : ProtocolContext
            Protocol context with reduced PDBbind data and selected features.

        Returns
        -------
        ProtocolContext
            Updated context with PDBbind artifacts.
        '''

        _validate_pdbbind_objective(self.config.objective_metric)
        _validate_stage_direction(self.config.direction, "minimize", "PDBbind")
        training_seed = context.random_seed if self.config.random_seed is None else self.config.random_seed
        _set_random_seed(training_seed)
        stage_dir = context.ensure_output_dir() / "pdbbind"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _validate_selected_features(context.pdbbind_df, context.selected_features, "PDBbind")
        if self.config.target_column not in context.pdbbind_df.columns:
            raise ValueError(f"PDBbind target column not found: {self.config.target_column}")

        fixed_outer = _require_fixed_outer_split(context)
        split_config = _resolve_pdbbind_split_config(
            self.config,
            random_seed=int(fixed_outer.outer_split_seed),
        )
        splits = prepare_pdbbind_regression_data(
            context.pdbbind_df,
            context.selected_features,
            split_config=split_config,
            fixed_train_indices=fixed_outer.pdbbind_train_indices,
            fixed_validation_indices=fixed_outer.pdbbind_validation_indices,
            fixed_test_indices=fixed_outer.pdbbind_test_indices,
        )
        split_indices = serialize_split_indices(splits)
        validate_replica_split_alignment(
            fixed_outer,
            replica_name=_replica_name_from_context(context),
            pdbbind_split_indices=split_indices,
            selected_features=context.selected_features,
            strict=True,
        )
        split_alignment = build_replica_split_alignment_metadata(
            fixed_outer,
            replica_seed=int(training_seed),
            replica_name=_replica_name_from_context(context),
            pdbbind_split_indices=split_indices,
            selected_features=context.selected_features,
        )

        device = _resolve_device(self.config.use_gpu)
        storage, storage_path = resolve_optuna_storage(self.config.storage, context.ensure_output_dir())
        sampler = TPESampler(
            seed=self.config.sampler_seed if self.config.sampler_seed is not None else training_seed
        )
        pruner = _build_pdbbind_pruner(self.config)
        study = _create_study_with_sqlite_lock_retry(
            "pdbbind_optuna:create_study",
            direction=self.config.direction,
            study_name=self.config.study_name,
            storage=storage,
            load_if_exists=self.config.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial,
                splits,
                stage_dir,
                device,
                training_seed,
                context.selected_features,
            )

        _optimize_with_sqlite_lock_retry(
            study,
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            stage_label="pdbbind_optuna:optimize",
        )

        best_trial = study.best_trial
        best_model = self._best_payload["model"]
        best_model.eval()
        val_pred, val_true = _predict_regression(best_model, splits["X_val"], splits["y_val"], device)
        test_pred, test_true = _predict_regression(best_model, splits["X_test"], splits["y_test"], device)
        val_metrics = evaluate_regression_metrics(val_true, val_pred)
        test_metrics = evaluate_regression_metrics(test_true, test_pred)

        checkpoint_path = str(stage_dir / "pdbbind_best.pt")
        checkpoint_payload = {
            "task": "pdbbind_regression",
            "objective_metric": "RMSE",
            "direction": "minimize",
            "selected_features": list(context.selected_features),
            "target_column": self.config.target_column,
            "best_trial_number": best_trial.number,
            "best_params": dict(best_trial.params),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_config": self._best_payload["model_config"],
            "model_state_dict": best_model.state_dict(),
            "feature_extractor_state_dict": best_model.feature_extractor.state_dict(),
            "scaler": splits["scaler"],
        }
        torch.save(checkpoint_payload, checkpoint_path)

        if not self._best_payload:
            raise RuntimeError("PDBbind optimization finished without a valid best-model payload.")
        pdbbind_export_extra = {}
        feature_policy_metadata = context.metadata.get("feature_policy")
        if feature_policy_metadata:
            pdbbind_export_extra["feature_policy"] = feature_policy_metadata
        export_paths = _export_stage_best_model(
            export_dir=stage_dir / "best_model",
            task="pdbbind_regression",
            model=best_model,
            model_config=self._best_payload["model_config"],
            selected_features=context.selected_features,
            best_trial_number=best_trial.number,
            best_objective_value=float(best_trial.value),
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            stage_config=_pdbbind_stage_config_payload(self.config, split_config, training_seed),
            splits={**splits, "split_config": asdict(split_config)},
            objective_metric=self.config.objective_metric,
            direction=self.config.direction,
            best_params=dict(best_trial.params),
            random_seed=training_seed,
            source_checkpoint_path=checkpoint_path,
            extra=pdbbind_export_extra,
            source_dataframe=context.pdbbind_df,
        )

        artifacts = _save_optuna_artifacts(study, stage_dir, "pdbbind")
        artifacts.checkpoint_path = checkpoint_path
        artifacts.optuna_storage_path = storage_path
        artifacts.metrics = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        result = {
            "task": "affinity_regression",
            "objective_metric": "RMSE",
            "direction": "minimize",
            "report_only_metrics": ["MAE", "Pearson r", "Spearman rho", "R2"],
            "best_trial": best_trial.number,
            "best_value": float(best_trial.value),
            "best_params": dict(best_trial.params),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint_path": checkpoint_path,
            "optuna_trials_path": artifacts.optuna_trials_path,
            "optuna_summary_path": artifacts.optuna_summary_path,
            "optuna_storage_path": artifacts.optuna_storage_path,
            "split_config": asdict(split_config),
            "split_diagnostics": splits.get("split_diagnostics", {}),
            "split_indices": serialize_split_indices(splits),
            "split_alignment": split_alignment,
            "search_space": pdbbind_search_space_summary(_resolve_pdbbind_search_space(self.config)),
            "search_phase": validate_pdbbind_search_phase(self.config.search_phase),
            "enable_pruning": bool(self.config.enable_pruning),
            "pruner": _pdbbind_pruner_summary(self.config),
            "target_column": self.config.target_column,
            "best_model_export_dir": export_paths["export_dir"],
            "best_model_export": export_paths,
        }
        context.artifacts["pdbbind_model"] = copy.deepcopy(best_model).cpu()
        context.artifacts["pdbbind_best_model_export_dir"] = export_paths["export_dir"]
        context.artifacts["pdbbind_checkpoint_path"] = checkpoint_path
        context.artifacts["pdbbind_scaler"] = splits["scaler"]
        context.stage_results[self.name] = result
        context.protocol_log.setdefault("checkpoints", {})["pdbbind"] = checkpoint_path
        return context

    def _objective(
            self,
            trial: optuna.Trial,
            splits: dict[str, Any],
            stage_dir: Path,
            device: torch.device,
            random_seed: int,
            selected_features: Sequence[str],
        ) -> float:
        _set_random_seed(random_seed + trial.number)
        search_space = _resolve_pdbbind_search_space(self.config)
        params = suggest_pdbbind_trial_params(
            trial,
            input_dim=splits["X_train"].shape[1],
            search_space=search_space,
        )
        model = build_pdbbind_model(input_size=splits["X_train"].shape[1], params=params).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(params["optimizer_learning_rate"]),
            weight_decay=float(params["optimizer_weight_decay"]),
        )
        batch_size = int(params["optimizer_batch_size"])
        train_loader = DataLoader(TabularDataset(splits["X_train"], splits["y_train"]), batch_size=batch_size, shuffle=True)
        if params["pdbbind_regression_loss"] == "huber":
            regression_loss = nn.HuberLoss(delta=float(params["pdbbind_huber_delta"]))
        else:
            regression_loss = nn.MSELoss()
        reconstruction_loss = nn.MSELoss()

        best_state = None
        best_val_rmse = float("inf")
        for epoch in range(int(self.config.epochs)):
            model.train()
            for features, target in train_loader:
                features = features.to(device)
                target = target.to(device).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                dae_enabled = float(params["decoder_lambda_rec"]) > 0.0
                model_input = features
                if dae_enabled:
                    model_input = _apply_dae_noise(
                        features,
                        noise_type=str(params.get("dae_noise_type", "none")),
                        mask_prob=float(params.get("dae_mask_prob", 0.0)),
                        gaussian_std=float(params.get("dae_gaussian_std", 0.0)),
                    )
                outputs = model(model_input, return_reconstruction=dae_enabled)
                loss = compute_regression_reconstruction_loss(
                    prediction=outputs["prediction"],
                    target=target,
                    reconstruction=outputs["reconstruction"],
                    features=features,
                    regression_loss=regression_loss,
                    reconstruction_loss=reconstruction_loss,
                    lambda_rec=float(params["decoder_lambda_rec"]),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_pred, val_true = _predict_regression(model, splits["X_val"], splits["y_val"], device)
            val_rmse = evaluate_regression_metrics(val_true, val_pred)["RMSE"]
            trial.report(float(val_rmse), epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        val_pred, val_true = _predict_regression(model, splits["X_val"], splits["y_val"], device)
        val_metrics = evaluate_regression_metrics(val_true, val_pred)
        for key, value in val_metrics.items():
            trial.set_user_attr(f"validation_{key}", value)
        trial.set_user_attr("objective_metric", "RMSE")
        trial.set_user_attr("task", "pdbbind_regression")
        trial.set_user_attr(
            "pdbbind_search_phase",
            validate_pdbbind_search_phase(self.config.search_phase),
        )
        trial.set_user_attr("checkpoint_candidate_dir", str(stage_dir))

        with self._best_lock:
            if val_metrics["RMSE"] < self._best_value:
                self._best_value = val_metrics["RMSE"]
                self._best_payload = {
                    "model": copy.deepcopy(model).cpu(),
                    "model_config": params,
                }

        return float(val_metrics["RMSE"])


class TransferFeatureExtractorStage:
    """Transfer reusable PDBbind feature extractor layers to the DUDEz stage.

    Notes
    -----
    The final PDBbind regression head is not transferred to DUDEz. The decoder
    is also excluded because it is only a PDBbind reconstruction regularizer.
    """

    name = "transfer_feature_extractor"

    def run(self, context: ProtocolContext) -> ProtocolContext:
        '''Transfer the PDBbind feature extractor without the regression head.

        Parameters
        ----------
        context : ProtocolContext
            Protocol context containing a PDBbind model or checkpoint path.

        Returns
        -------
        ProtocolContext
            Updated context with ``transferred_feature_extractor``.
        '''

        pdbbind_model = context.artifacts.get("pdbbind_model")
        if pdbbind_model is None:
            checkpoint_path = context.artifacts.get("pdbbind_checkpoint_path")
            if not checkpoint_path:
                raise ValueError("PDBbind model or checkpoint is required before transfer.")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_config = payload["model_config"]
            pdbbind_model = build_pdbbind_model(
                input_size=len(payload["selected_features"]),
                params=model_config,
            )
            pdbbind_model.load_state_dict(payload["model_state_dict"])

        feature_extractor = copy.deepcopy(pdbbind_model.feature_extractor).cpu()
        context.artifacts["transferred_feature_extractor"] = feature_extractor
        context.stage_results[self.name] = {
            "transferred_components": ["feature_extractor", "projection_block_if_present"],
            "excluded_components": ["regression_head", "decoder"],
            "source_checkpoint_path": context.artifacts.get("pdbbind_checkpoint_path"),
        }
        return context


class DUDEzOptunaStage:
    """Optuna stage that optimizes DUDEz screening by one screening metric only.

    Parameters
    ----------
    config : DUDEzOptunaConfig | None, optional
        Stage configuration. If None, default DUDEz settings are used.
    """

    name = "dudez_optuna"

    def __init__(self, config: Optional[DUDEzOptunaConfig] = None) -> None:
        '''Initialize the DUDEz Optuna stage.

        Parameters
        ----------
        config : DUDEzOptunaConfig | None, optional
            Stage configuration. If None, default DUDEz settings are used.
        '''

        self.config = config or DUDEzOptunaConfig()
        self._best_value = -float("inf")
        self._best_payload: dict[str, Any] = {}
        self._best_lock = threading.Lock()
        self.effective_primary_metric = self.config.primary_metric

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_best_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._best_lock = threading.Lock()

    def run(self, context: ProtocolContext) -> ProtocolContext:
        '''Run DUDEz screening optimization.

        Parameters
        ----------
        context : ProtocolContext
            Protocol context containing DUDEz data and transferred extractor.

        Returns
        -------
        ProtocolContext
            Updated context with DUDEz artifacts.
        '''

        _validate_dudez_objective(self.config.primary_metric)
        _validate_stage_direction(self.config.direction, "maximize", "DUDEz")
        training_seed = context.random_seed if self.config.random_seed is None else self.config.random_seed
        _set_random_seed(training_seed)
        stage_dir = context.ensure_output_dir() / "dudez"
        stage_dir.mkdir(parents=True, exist_ok=True)
        _validate_selected_features(context.dudez_df, context.selected_features, "DUDEz")
        labels = derive_dudez_labels(context.dudez_df, kind_column=self.config.kind_column)
        groups = (
            context.dudez_df[self.config.target_group_column].to_numpy()
            if self.config.target_group_column in context.dudez_df.columns
            else None
        )
        fixed_outer = _require_fixed_outer_split(context)
        split_config = _resolve_dudez_split_config(
            self.config,
            random_seed=int(fixed_outer.outer_split_seed),
        )
        scaling_config = self.config.dudez_scaling_config or DUDEzScalingConfig(strategy="pdbbind_scaler", strict=True)
        splits = prepare_dudez_screening_data(
            context.dudez_df,
            context.selected_features,
            labels,
            groups=groups,
            split_config=split_config,
            target_group_column=self.config.target_group_column if groups is not None else None,
            scaling_config=scaling_config,
            pdbbind_scaler=context.artifacts.get("pdbbind_scaler"),
            fixed_train_indices=fixed_outer.dudez_train_indices,
            fixed_validation_indices=fixed_outer.dudez_validation_indices,
            fixed_test_indices=fixed_outer.dudez_test_indices,
        )
        split_indices = serialize_split_indices(splits)
        validate_replica_split_alignment(
            fixed_outer,
            replica_name=_replica_name_from_context(context),
            pdbbind_split_indices={
                "train": fixed_outer.pdbbind_train_indices,
                "validation": fixed_outer.pdbbind_validation_indices,
                "test": fixed_outer.pdbbind_test_indices,
            },
            selected_features=context.selected_features,
            dudez_split_indices=split_indices,
            strict=True,
        )
        split_alignment = build_replica_split_alignment_metadata(
            fixed_outer,
            replica_seed=int(training_seed),
            replica_name=_replica_name_from_context(context),
            pdbbind_split_indices={
                "train": fixed_outer.pdbbind_train_indices,
                "validation": fixed_outer.pdbbind_validation_indices,
                "test": fixed_outer.pdbbind_test_indices,
            },
            selected_features=context.selected_features,
            dudez_split_indices=split_indices,
        )

        transferred = context.artifacts.get("transferred_feature_extractor")
        if transferred is None:
            raise ValueError("DUDEzOptunaStage requires TransferFeatureExtractorStage output.")

        device = _resolve_device(self.config.use_gpu)
        storage, storage_path = resolve_optuna_storage(self.config.storage, context.ensure_output_dir())
        sampler = TPESampler(
            seed=self.config.sampler_seed if self.config.sampler_seed is not None else training_seed
        )
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=_median_pruner_n_startup_trials_default(self.config.n_trials),
            n_warmup_steps=_median_pruner_n_warmup_steps_default(self.config.epochs),
        )
        study = _create_study_with_sqlite_lock_retry(
            "dudez_optuna:create_study",
            direction=self.config.direction,
            study_name=self.config.study_name,
            storage=storage,
            load_if_exists=self.config.load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial,
                splits,
                transferred,
                stage_dir,
                device,
                training_seed,
                context.selected_features,
            )

        _optimize_with_sqlite_lock_retry(
            study,
            objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            stage_label="dudez_optuna:optimize",
        )

        best_trial = study.best_trial
        best_model = self._best_payload["model"]
        best_model.eval()
        val_score, val_true = _predict_screening(best_model, splits["X_val"], splits["y_val"], device)
        test_score, test_true = _predict_screening(best_model, splits["X_test"], splits["y_test"], device)
        val_metrics = evaluate_screening_metrics(
            val_true,
            val_score,
            groups=splits.get("val_groups"),
            higher_is_better=True,
        )
        test_metrics = evaluate_screening_metrics(
            test_true,
            test_score,
            groups=splits.get("test_groups"),
            higher_is_better=True,
        )
        calibrator = enrich_dudez_export_metrics(
            val_metrics,
            test_metrics,
            val_true=val_true,
            val_scores=val_score,
            test_true=test_true,
            test_scores=test_score,
            report_mode=self.config.calibration_report_mode,  # type: ignore[arg-type]
        )

        checkpoint_path = str(stage_dir / "dudez_best.pt")
        checkpoint_payload = {
            "task": "dudez_screening",
            "objective_metric": self.effective_primary_metric,
            "requested_primary_metric": self.config.primary_metric,
            "direction": "maximize",
            "selected_features": list(context.selected_features),
            "kind_column": self.config.kind_column,
            "target_group_column": self.config.target_group_column,
            "best_trial_number": best_trial.number,
            "best_params": dict(best_trial.params),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_config": self._best_payload["model_config"],
            "model_state_dict": best_model.state_dict(),
            "feature_extractor_state_dict": best_model.feature_extractor.state_dict(),
        }
        torch.save(checkpoint_payload, checkpoint_path)

        if not self._best_payload:
            raise RuntimeError("DUDEz optimization finished without a valid best-model payload.")
        dudez_export_extra = {}
        feature_policy_metadata = context.metadata.get("feature_policy")
        if feature_policy_metadata:
            dudez_export_extra["feature_policy"] = feature_policy_metadata
        pdbbind_export_dir = context.artifacts.get("pdbbind_best_model_export_dir")
        if pdbbind_export_dir:
            dudez_export_extra["pdbbind_best_model_export_dir"] = pdbbind_export_dir
        export_paths = _export_stage_best_model(
            export_dir=stage_dir / "best_model",
            task="dudez_screening",
            model=best_model,
            model_config=self._best_payload["model_config"],
            selected_features=context.selected_features,
            best_trial_number=best_trial.number,
            best_objective_value=float(best_trial.value),
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            stage_config=_dudez_stage_config_payload(self.config, split_config, training_seed),
            splits={**splits, "split_config": asdict(split_config)},
            objective_metric=self.effective_primary_metric,
            direction=self.config.direction,
            best_params=dict(best_trial.params),
            random_seed=training_seed,
            source_checkpoint_path=checkpoint_path,
            extra=dudez_export_extra,
            calibrator=calibrator,
            source_dataframe=context.dudez_df,
        )

        artifacts = _save_optuna_artifacts(study, stage_dir, "dudez")
        artifacts.checkpoint_path = checkpoint_path
        artifacts.optuna_storage_path = storage_path
        artifacts.metrics = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        result = {
            "task": "active_decoy_screening",
            "objective_metric": self.effective_primary_metric,
            "requested_primary_metric": self.config.primary_metric,
            "direction": "maximize",
            "report_only_metrics": [metric for metric in DUDEZ_REPORT_METRICS if metric != self.effective_primary_metric],
            "best_trial": best_trial.number,
            "best_value": float(best_trial.value),
            "best_params": dict(best_trial.params),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoint_path": checkpoint_path,
            "pdbbind_checkpoint_path": context.artifacts.get("pdbbind_checkpoint_path"),
            "optuna_trials_path": artifacts.optuna_trials_path,
            "optuna_summary_path": artifacts.optuna_summary_path,
            "optuna_storage_path": artifacts.optuna_storage_path,
            "split_config": asdict(split_config),
            "split_diagnostics": splits.get("split_diagnostics", {}),
            "split_indices": serialize_split_indices(splits),
            "split_alignment": split_alignment,
            "scaling_metadata": splits.get("scaling_metadata", {}),
            "metrics_scope": "grouped" if splits.get("val_groups") is not None else "global",
            "search_space": dudez_search_space_summary(_resolve_dudez_search_space(self.config)),
            "kind_column": self.config.kind_column,
            "best_model_export_dir": export_paths["export_dir"],
            "best_model_export": export_paths,
        }
        context.artifacts["dudez_model"] = copy.deepcopy(best_model).cpu()
        context.artifacts["dudez_best_model_export_dir"] = export_paths["export_dir"]
        context.artifacts["dudez_checkpoint_path"] = checkpoint_path
        context.stage_results[self.name] = result
        context.protocol_log.setdefault("checkpoints", {})["dudez"] = checkpoint_path
        return context

    def _objective(
            self,
            trial: optuna.Trial,
            splits: dict[str, Any],
            transferred_extractor: FeatureExtractor,
            stage_dir: Path,
            device: torch.device,
            random_seed: int,
            selected_features: Sequence[str],
        ) -> float:
        _set_random_seed(random_seed + trial.number)
        search_space = _resolve_dudez_search_space(self.config)
        params = suggest_dudez_trial_params(
            trial,
            allow_scratch=self.config.allow_scratch,
            search_space=search_space,
        )
        use_class_weighting = bool(params.get("dudez_use_class_weighting", self.config.use_class_weighting))
        model = build_dudez_model(
            input_size=splits["X_train"].shape[1],
            params=params,
            transferred_extractor=transferred_extractor,
            feature_extractor_architecture=_fresh_extractor_params_from_transfer(transferred_extractor),
        ).to(device)
        optimizer = optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(params["optimizer_learning_rate"]),
            weight_decay=float(params["optimizer_weight_decay"]),
        )
        batch_size = int(params["optimizer_batch_size"])
        train_loader = DataLoader(TabularDataset(splits["X_train"], splits["y_train"]), batch_size=batch_size, shuffle=True)
        pos_weight = _positive_class_weight(splits["y_train"], device) if use_class_weighting else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        metrics_scope = "grouped" if splits.get("val_groups") is not None else "global"
        best_state = None
        best_metric = -float("inf")
        last_train_loss = float("nan")
        for epoch in range(int(self.config.epochs)):
            model.train()
            epoch_losses: list[float] = []
            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device).view(-1)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            last_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

            val_score, val_true = _predict_screening(model, splits["X_val"], splits["y_val"], device)
            val_metrics = evaluate_screening_metrics(
                val_true,
                val_score,
                groups=splits.get("val_groups"),
                higher_is_better=True,
            )
            primary_metric, metric_value = _resolve_dudez_objective_value(
                self.config.primary_metric,
                val_metrics,
            )
            self.effective_primary_metric = primary_metric
            trial.report(metric_value, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if metric_value > best_metric:
                best_metric = metric_value
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        val_score, val_true = _predict_screening(model, splits["X_val"], splits["y_val"], device)
        val_metrics = evaluate_screening_metrics(
            val_true,
            val_score,
            groups=splits.get("val_groups"),
            higher_is_better=True,
        )
        primary_metric, objective_value = _resolve_dudez_objective_value(
            self.config.primary_metric,
            val_metrics,
        )
        self.effective_primary_metric = primary_metric
        _set_dudez_trial_metric_attrs(trial, val_metrics, primary_metric, metrics_scope)

        training_diagnostics = _collect_dudez_training_diagnostics(
            model,
            splits,
            device,
            criterion,
            train_loader,
        )
        trial.set_user_attr("train_bce_loss_last_epoch", last_train_loss)
        for key, value in training_diagnostics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("requested_primary_metric", self.config.primary_metric)
        trial.set_user_attr("task", "dudez_screening")
        trial.set_user_attr("checkpoint_candidate_dir", str(stage_dir))

        with self._best_lock:
            if objective_value > self._best_value:
                self._best_value = objective_value
                self._best_payload = {
                    "model": copy.deepcopy(model).cpu(),
                    "model_config": params,
                }

        return objective_value


# Functions
###############################################################################
## Private ##

def _build_activation(name: str) -> nn.Module:
    '''Build one activation module from a centralized search-space name.

    Parameters
    ----------
    name : str
        Activation name defined in :mod:`OCDocker.OCScore.Optimization.OptunaSearchSpace`.

    Returns
    -------
    torch.nn.Module
        Instantiated activation module.
    '''

    return build_activation_module(name)


def _resolve_pdbbind_search_space(config: PDBbindOptunaConfig) -> PDBbindSearchSpaceConfig:
    if config.search_space is not None:
        return config.search_space
    return pdbbind_search_space_for_phase(config.search_phase)


def _median_pruner_n_startup_trials_default(n_trials: int) -> int:
    '''Default MedianPruner startup count: ``max(10% of trials, 5)``.'''

    trials = max(1, int(n_trials))
    return max(int(trials * 0.1), 5)


def _median_pruner_n_warmup_steps_default(epochs: int) -> int:
    '''Default MedianPruner warmup steps: ``max(10% of epochs, 10)``.'''

    training_epochs = max(1, int(epochs))
    return max(int(training_epochs * 0.1), 10)


def _resolve_pdbbind_pruner_settings(config: PDBbindOptunaConfig) -> dict[str, Any]:
    '''Return effective PDBbind pruner settings for logging or study creation.'''

    if not config.enable_pruning:
        return {"type": "NopPruner"}

    n_trials = max(1, int(config.n_trials))
    epochs = max(1, int(config.epochs))

    if config.pruner_n_startup_trials is not None:
        n_startup = max(1, int(config.pruner_n_startup_trials))
    else:
        n_startup = _median_pruner_n_startup_trials_default(n_trials)

    if config.pruner_n_warmup_steps is not None:
        n_warmup = max(0, int(config.pruner_n_warmup_steps))
    else:
        n_warmup = _median_pruner_n_warmup_steps_default(epochs)

    return {
        "type": "MedianPruner",
        "n_startup_trials": n_startup,
        "n_warmup_steps": n_warmup,
    }


def _build_pdbbind_pruner(config: PDBbindOptunaConfig) -> optuna.pruners.BasePruner:
    '''Build the Optuna pruner for a PDBbind study from stage configuration.'''

    settings = _resolve_pdbbind_pruner_settings(config)
    if settings["type"] == "NopPruner":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(settings["n_startup_trials"]),
        n_warmup_steps=int(settings["n_warmup_steps"]),
    )


def _pdbbind_pruner_summary(config: PDBbindOptunaConfig) -> dict[str, Any]:
    return _resolve_pdbbind_pruner_settings(config)


def pdbbind_phase1_experiment_config(
        *,
        n_trials: int = 40,
        epochs: int = 100,
        study_name: str = "PDBbind_EncoderRegression_Phase1",
        enable_pruning: bool = False,
        **kwargs: Any,
    ) -> PDBbindOptunaConfig:
    '''Preset PDBbind Optuna config for Phase 1 encoder-regression experiments.

    Parameters
    ----------
    n_trials : int, optional
        Number of Optuna trials (default 40, within the 30–50 Phase 1 band).
    epochs : int, optional
        Training epochs per trial.
    study_name : str, optional
        Distinct study name so Phase 1 does not resume the full-search study.
    enable_pruning : bool, optional
        When False (default), pruning is disabled for the experiment.
    **kwargs
        Additional :class:`PDBbindOptunaConfig` fields.

    Returns
    -------
    PDBbindOptunaConfig
        Phase 1 experiment configuration.
    '''

    return PDBbindOptunaConfig(
        n_trials=int(n_trials),
        epochs=int(epochs),
        study_name=str(study_name),
        search_phase=PDBBIND_SEARCH_PHASE_ENCODER_REGRESSION,
        enable_pruning=bool(enable_pruning),
        **kwargs,
    )


def _resolve_dudez_search_space(config: DUDEzOptunaConfig) -> DUDEzSearchSpaceConfig:
    return config.search_space or DEFAULT_DUDEZ_SEARCH_SPACE


def _build_decoder(input_size: int, output_size: int, decoder_sizes: Sequence[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_size)
    for size in decoder_sizes:
        layers.append(nn.Linear(prev, int(size)))
        layers.append(_build_activation(activation))
        prev = int(size)
    layers.append(nn.Linear(prev, int(output_size)))
    return nn.Sequential(*layers)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _pdbbind_stage_config_payload(
        config: PDBbindOptunaConfig,
        split_config: PDBbindSplitConfig,
        random_seed: int,
    ) -> dict[str, Any]:
    return {
        "target_column": config.target_column,
        "n_trials": config.n_trials,
        "epochs": config.epochs,
        "study_name": config.study_name,
        "search_phase": validate_pdbbind_search_phase(config.search_phase),
        "enable_pruning": bool(config.enable_pruning),
        "pruner": _pdbbind_pruner_summary(config),
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "objective_metric": config.objective_metric,
        "direction": config.direction,
        "sampler_seed": config.sampler_seed,
        "random_seed": random_seed,
        "use_gpu": config.use_gpu,
        "n_jobs": config.n_jobs,
        "split_config": asdict(split_config),
        "search_space": pdbbind_search_space_summary(_resolve_pdbbind_search_space(config)),
    }


def _dudez_stage_config_payload(
        config: DUDEzOptunaConfig,
        split_config: DUDEzSplitConfig,
        random_seed: int,
    ) -> dict[str, Any]:
    return {
        "kind_column": config.kind_column,
        "target_group_column": config.target_group_column,
        "primary_metric": config.primary_metric,
        "n_trials": config.n_trials,
        "epochs": config.epochs,
        "study_name": config.study_name,
        "validation_size": config.validation_size,
        "test_size": config.test_size,
        "direction": config.direction,
        "sampler_seed": config.sampler_seed,
        "random_seed": random_seed,
        "use_gpu": config.use_gpu,
        "n_jobs": config.n_jobs,
        "allow_scratch": config.allow_scratch,
        "use_class_weighting": config.use_class_weighting,
        "split_config": asdict(split_config),
        "search_space": dudez_search_space_summary(_resolve_dudez_search_space(config)),
    }


def _export_stage_best_model(**kwargs: Any) -> dict[str, str]:
    from OCDocker.OCScore.Optimization.ModelExport import export_best_model_bundle

    return export_best_model_bundle(**kwargs)


def _fresh_extractor_params_from_transfer(extractor: FeatureExtractor) -> dict[str, Any]:
    hidden_sizes = _linear_out_features(extractor.encoder)[:-1]
    latent_dim = extractor.latent_dim
    projection_dim = extractor.output_dim if extractor.projection is not None else 0
    return {
        "hidden_sizes": hidden_sizes,
        "latent_dim": latent_dim,
        "activation": "GELU",
        "dropout": 0.0,
        "projection_dim": projection_dim,
    }


def _linear_out_features(module: nn.Module) -> list[int]:
    return [int(child.out_features) for child in module.modules() if isinstance(child, nn.Linear)]


def _positive_class_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    labels = np.asarray(y).astype(int)
    pos = max(1, int(labels.sum()))
    neg = max(1, int(len(labels) - pos))
    return torch.tensor([neg / pos], device=device, dtype=torch.float32)


def _predict_regression(
        model: PDBbindRegressionModel,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        pred = model(features)["prediction"]
    return pred.detach().cpu().numpy().reshape(-1), np.asarray(y, dtype=float).reshape(-1)


def _predict_screening(
        model: DUDEzScreeningModel,
        X: np.ndarray,
        y: np.ndarray,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        features = torch.tensor(np.asarray(X), dtype=torch.float32, device=device)
        logits = model(features)
    return logits.detach().cpu().numpy().reshape(-1), np.asarray(y, dtype=int).reshape(-1)


def _resolve_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    if method == "spearman":
        return float(pd.Series(x).rank(method="average").corr(pd.Series(y).rank(method="average")))
    return float(np.corrcoef(x, y)[0, 1])


def _safe_group_split(
        idx: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_seed: int,
        groups: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
    if len(idx) < 4:
        split = max(1, len(idx) - 1)
        return idx[:split], idx[split:]

    if groups is not None and len(np.unique(groups[idx])) > 2:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
            train_rel, test_rel = next(splitter.split(idx, y[idx], groups=groups[idx]))
            return idx[train_rel], idx[test_rel]
        except ValueError:
            pass

    stratify = y[idx] if _can_stratify(y[idx]) else None
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return len(counts) > 1 and np.min(counts) >= 2


def _safe_metric(metric_fn: Any, y_true: np.ndarray, y_score: np.ndarray, default: float = 0.0) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return default
        value = float(metric_fn(y_true, y_score))
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


def _save_optuna_artifacts(study: optuna.study.Study, stage_dir: Path, prefix: str) -> StageArtifacts:
    trials_path = stage_dir / f"{prefix}_optuna_trials.csv"
    summary_path = stage_dir / f"{prefix}_optuna_summary.json"
    try:
        study.trials_dataframe().to_csv(trials_path, index=False)
    except Exception:
        pd.DataFrame().to_csv(trials_path, index=False)
    payload = {
        "study_name": study.study_name,
        "direction": str(study.direction),
        "best_trial": getattr(study.best_trial, "number", None),
        "best_value": _finite_or_none(float(study.best_value)) if len(study.trials) else None,
        "best_params": dict(study.best_params) if len(study.trials) else {},
        "best_trial_user_attrs": dict(getattr(study.best_trial, "user_attrs", {})) if len(study.trials) else {},
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return StageArtifacts(
        optuna_trials_path=str(trials_path),
        optuna_summary_path=str(summary_path),
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _top_k_count(n: int, fraction: float) -> int:
    return max(1, min(n, int(math.ceil(float(fraction) * n))))


def _apply_dae_noise(
        features: torch.Tensor,
        *,
        noise_type: str,
        mask_prob: float,
        gaussian_std: float,
    ) -> torch.Tensor:
    if noise_type in {"none", "", "off"}:
        return features

    x = features
    if noise_type in {"mask", "mask+gaussian", "gaussian+mask"} and float(mask_prob) > 0.0:
        prob = float(mask_prob)
        keep = torch.rand_like(x) >= prob
        x = x * keep.to(dtype=x.dtype)

    if noise_type in {"gaussian", "mask+gaussian", "gaussian+mask"} and float(gaussian_std) > 0.0:
        std = float(gaussian_std)
        x = x + torch.randn_like(x) * std
    return x

def _validate_stage_direction(direction: str, expected: str, label: str) -> None:
    if direction != expected:
        raise ValueError(f"{label} Optuna direction must be {expected}.")


def _validate_dudez_objective(metric: str) -> None:
    normalized = _normalize_metric_name(metric)
    _validate_no_mixed_objective(metric)
    if normalized in {"RMSE", "MAE", "R2", "PEARSONR", "SPEARMANRHO"}:
        raise ValueError("DUDEz objective metric must be a screening/classification metric, not a regression metric.")
    if metric not in DUDEZ_PRIMARY_METRICS:
        raise ValueError(f"Unsupported DUDEz objective metric: {metric}")


def _validate_no_mixed_objective(metric: str) -> None:
    normalized = _normalize_metric_name(metric)
    if any(pattern in normalized for pattern in FORBIDDEN_MIXED_OBJECTIVE_PATTERNS):
        raise ValueError(f"Mixed regression/screening objective is not allowed: {metric}")
    if "RMSE" in normalized and any(token in normalized for token in ("AUC", "BEDROC", "NDCG", "EF")):
        raise ValueError(f"Mixed regression/screening objective is not allowed: {metric}")


def _validate_pdbbind_objective(metric: str) -> None:
    _validate_no_mixed_objective(metric)
    if metric != "RMSE":
        raise ValueError("PDBbind Optuna objective must be validation RMSE.")


def _validate_selected_features(
        df: pd.DataFrame,
        selected_features: Sequence[str],
        label: str,
        reserved_columns: Optional[Sequence[str]] = None,
    ) -> None:
    missing = [col for col in selected_features if col not in df.columns]
    if missing:
        raise ValueError(f"{label} dataframe is missing selected features: {missing}")
    if not selected_features:
        raise ValueError("selected_features must not be empty.")

    reserved = set(OCSCORE_NON_FEATURE_COLUMNS if reserved_columns is None else reserved_columns)
    reserved_selected = [col for col in selected_features if col in reserved]
    if reserved_selected:
        raise ValueError(
            f"{label} selected_features includes metadata/target columns that must not be model inputs: "
            f"{reserved_selected}"
        )


def _normalize_metric_name(metric: str) -> str:
    return str(metric).upper().replace(" ", "").replace("_", "-")


## Public ##

def apply_fine_tuning_mode(feature_extractor: nn.Module, mode: str, num_unfrozen_layers: int = 1) -> None:
    '''Apply frozen, partial, or full fine-tuning to a transferred extractor.

    Parameters
    ----------
    feature_extractor : nn.Module
        Feature extractor to update in-place.
    mode : str
        One of ``"frozen"``, ``"partial"``, or ``"full"``.
    num_unfrozen_layers : int, optional
        Number of final linear layers unfrozen when mode is ``"partial"``.
    '''

    if mode == "full":
        for param in feature_extractor.parameters():
            param.requires_grad = True
        return

    for param in feature_extractor.parameters():
        param.requires_grad = False

    if mode != "partial":
        return

    linears = [module for module in feature_extractor.modules() if isinstance(module, nn.Linear)]
    for module in linears[-max(1, int(num_unfrozen_layers)):]:
        for param in module.parameters():
            param.requires_grad = True


def build_dudez_model(
        input_size: int,
        params: dict[str, Any],
        transferred_extractor: Optional[FeatureExtractor] = None,
        feature_extractor_architecture: Optional[dict[str, Any]] = None,
    ) -> DUDEzScreeningModel:
    '''Build a DUDEz screening model from sampled parameters.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    params : dict[str, Any]
        Sampled DUDEz model and training parameters.
    transferred_extractor : FeatureExtractor | None, optional
        Transferred PDBbind feature extractor used when transfer is enabled.
    feature_extractor_architecture : dict[str, Any] | None, optional
        Resolved extractor architecture for scratch DUDEz models.

    Returns
    -------
    DUDEzScreeningModel
        Initialized DUDEz screening model.
    '''

    if bool(params.get("dudez_use_transfer", True)):
        if transferred_extractor is None:
            raise ValueError("transferred_extractor is required when dudez_use_transfer is True.")
        extractor = copy.deepcopy(transferred_extractor)
    else:
        architecture = feature_extractor_architecture or {}
        extractor = FeatureExtractor(
            input_size=input_size,
            hidden_sizes=architecture.get("hidden_sizes", []),
            latent_dim=int(architecture.get("latent_dim", 32)),
            activation=str(params.get("dudez_classifier_activation", "GELU")),
            dropout=float(params.get("encoder_dropout", 0.0)),
            projection_dim=int(architecture.get("projection_dim", 0)),
        )
    apply_fine_tuning_mode(
        extractor,
        mode=str(params.get("dudez_fine_tuning_mode", "partial")),
        num_unfrozen_layers=int(params.get("dudez_num_unfrozen_layers", 1)),
    )
    return DUDEzScreeningModel(
        feature_extractor=extractor,
        classifier_hidden_size=int(params.get("dudez_classifier_hidden_size", 128)),
        dropout=float(params.get("dudez_classifier_dropout", 0.0)),
        activation=str(params.get("dudez_classifier_activation", "GELU")),
    )


def build_pdbbind_model(input_size: int, params: dict[str, Any]) -> PDBbindRegressionModel:
    '''Build a PDBbind regression model from sampled parameters.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    params : dict[str, Any]
        Sampled PDBbind model and training parameters.

    Returns
    -------
    PDBbindRegressionModel
        Initialized PDBbind regression model.
    '''

    extractor = FeatureExtractor(
        input_size=input_size,
        hidden_sizes=params["encoder_hidden_sizes"],
        latent_dim=int(params["encoder_latent_dim"]),
        activation=str(params["encoder_activation"]),
        dropout=float(params["encoder_dropout"]),
        projection_dim=int(params["projection_dim"]),
    )
    decoder_sizes = params.get("decoder_hidden_sizes")
    if float(params["decoder_lambda_rec"]) > 0.0 and not decoder_sizes:
        raise ValueError(
            "decoder_lambda_rec > 0 requires sampled decoder_hidden_sizes from suggest_decoder_hidden_sizes()."
        )
    return PDBbindRegressionModel(
        feature_extractor=extractor,
        input_size=input_size,
        activation=str(params["encoder_activation"]),
        decoder_sizes=decoder_sizes,
    )


def compute_regression_reconstruction_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        reconstruction: Optional[torch.Tensor],
        features: torch.Tensor,
        regression_loss: nn.Module,
        reconstruction_loss: nn.Module,
        lambda_rec: float,
    ) -> torch.Tensor:
    '''Compute regression loss plus optional reconstruction regularization.

    Parameters
    ----------
    prediction : torch.Tensor
        Regression predictions.
    target : torch.Tensor
        Regression targets.
    reconstruction : torch.Tensor | None
        Reconstructed input features, if decoder regularization is enabled.
    features : torch.Tensor
        Original input features.
    regression_loss : nn.Module
        Primary regression loss.
    reconstruction_loss : nn.Module
        Reconstruction loss.
    lambda_rec : float
        Reconstruction loss weight. A value of 0 disables reconstruction.

    Returns
    -------
    torch.Tensor
        Total loss for backpropagation.
    '''

    reg_loss = regression_loss(prediction, target)
    if lambda_rec <= 0.0 or reconstruction is None:
        return reg_loss
    return reg_loss + float(lambda_rec) * reconstruction_loss(reconstruction, features)


def derive_dudez_labels(df: pd.DataFrame, kind_column: str = "kind") -> np.ndarray:
    '''Derive DUDEz labels from kind values.

    ``ligands``/``ligand`` map to 1 and ``decoys``/``decoy`` map to 0.

    Parameters
    ----------
    df : pd.DataFrame
        DUDEz dataframe containing active/decoy kind values.
    kind_column : str, optional
        Column used to derive labels, by default "kind".

    Returns
    -------
    np.ndarray
        Binary labels as float32 values.
    '''

    source_column = kind_column
    if source_column not in df.columns and "type" in df.columns:
        source_column = "type"
    if source_column not in df.columns:
        raise ValueError(f"DUDEz dataframe must contain {kind_column!r} to derive active/decoy labels.")

    normalized = df[source_column].astype("string").str.strip().str.lower()
    labels = normalized.map({"ligands": 1, "ligand": 1, "actives": 1, "active": 1, "decoys": 0, "decoy": 0})
    if labels.isna().any():
        unknown = sorted(normalized[labels.isna()].dropna().unique().tolist())
        raise ValueError(f"Unsupported DUDEz kind values: {unknown}")
    return labels.to_numpy(dtype=np.float32)


def dudez_search_space_summary(
        search_space: Optional[DUDEzSearchSpaceConfig] = None,
    ) -> dict[str, Any]:
    '''Return the DUDEz Optuna search-space summary.

    Parameters
    ----------
    search_space : DUDEzSearchSpaceConfig | None, optional
        Search-space configuration. Defaults to :data:`DEFAULT_DUDEZ_SEARCH_SPACE`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable DUDEz search-space description.
    '''

    space = search_space or DEFAULT_DUDEZ_SEARCH_SPACE
    summary = search_space_to_summary(space)
    summary["loss"] = "BCEWithLogitsLoss"
    return summary


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    '''Evaluate PDBbind regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Experimental affinity values.
    y_pred : np.ndarray
        Predicted affinity values.

    Returns
    -------
    dict[str, float]
        RMSE, MAE, Pearson r, Spearman rho, and R2 metrics.
    '''

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson r": _safe_corr(y_true, y_pred, method="pearson"),
        "Spearman rho": _safe_corr(y_true, y_pred, method="spearman"),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0,
    }


def _is_dudez_early_enrichment_metric(metric_name: str) -> bool:
    return str(metric_name) in DUDEZ_EARLY_ENRICHMENT_RANKING_METRICS


def _dudez_ranking_objective_is_valid(requested_metric: str, metrics: dict[str, float]) -> bool:
    '''Return whether early-enrichment metrics are usable for optimization.'''

    if not _is_dudez_early_enrichment_metric(requested_metric):
        return True
    if not bool(metrics.get("ranking_metrics_valid")):
        return False
    groups_used = metrics.get("n_groups_used")
    if groups_used is not None and not np.isnan(float(groups_used)) and float(groups_used) <= 0.0:
        return False
    value = metrics.get(requested_metric)
    if value is None:
        return False
    numeric = float(value)
    return not (np.isnan(numeric) or np.isinf(numeric))


def _invalid_dudez_ranking_reason(requested_metric: str, metrics: dict[str, float]) -> str:
    if not bool(metrics.get("ranking_metrics_valid")):
        return "constant_or_tied_scores"
    groups_used = metrics.get("n_groups_used")
    if groups_used is not None and not np.isnan(float(groups_used)) and float(groups_used) <= 0.0:
        return "no_valid_ranking_groups"
    value = metrics.get(requested_metric)
    if value is None or np.isnan(float(value)) or np.isinf(float(value)):
        return f"non_finite_{requested_metric}"
    return "unknown"


def _resolve_dudez_objective_value(
        requested_metric: str,
        metrics: dict[str, float],
    ) -> tuple[str, float]:
    '''Resolve the effective DUDEz objective metric and numeric value.'''

    if not _dudez_ranking_objective_is_valid(requested_metric, metrics):
        raise optuna.exceptions.TrialPruned(
            _invalid_dudez_ranking_reason(requested_metric, metrics)
        )

    primary_metric = resolve_dudez_primary_metric(requested_metric, metrics)
    if not _dudez_ranking_objective_is_valid(primary_metric, metrics):
        raise optuna.exceptions.TrialPruned(
            _invalid_dudez_ranking_reason(primary_metric, metrics)
        )

    raw_value = metrics.get(primary_metric)
    if raw_value is None:
        raise optuna.exceptions.TrialPruned(f"missing_{primary_metric}")
    objective_value = float(raw_value)
    if np.isnan(objective_value) or np.isinf(objective_value):
        raise optuna.exceptions.TrialPruned(f"non_finite_{primary_metric}")
    return primary_metric, objective_value


def pdbbind_search_space_summary(
        search_space: Optional[PDBbindSearchSpaceConfig] = None,
    ) -> dict[str, Any]:
    '''Return the PDBbind Optuna search-space summary.

    Parameters
    ----------
    search_space : PDBbindSearchSpaceConfig | None, optional
        Search-space configuration. Defaults to :data:`DEFAULT_PDBBIND_SEARCH_SPACE`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable PDBbind search-space description.
    '''

    return search_space_to_summary(search_space or DEFAULT_PDBBIND_SEARCH_SPACE)


def serialize_split_indices(splits: Mapping[str, Any]) -> dict[str, list[int]]:
    '''Return JSON-serializable train/validation/test row indices from split dicts.'''

    return {
        "train": [int(i) for i in np.asarray(splits["train_indices"]).reshape(-1)],
        "validation": [int(i) for i in np.asarray(splits["validation_indices"]).reshape(-1)],
        "test": [int(i) for i in np.asarray(splits["test_indices"]).reshape(-1)],
    }


def summarize_dudez_split_diagnostics(
        df: pd.DataFrame,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        groups: Optional[np.ndarray] = None,
        target_group_column: Optional[str] = None,
    ) -> dict[str, Any]:
    '''Summarize DUDEz split composition for logging and reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        Reduced DUDEz dataframe.
    labels : np.ndarray
        Binary active/decoy labels.
    train_idx : np.ndarray
        Training row indices.
    val_idx : np.ndarray
        Validation row indices.
    test_idx : np.ndarray
        Test row indices.
    groups : np.ndarray | None, optional
        Target/receptor group labels aligned with ``labels``.
    target_group_column : str | None, optional
        Source column name used for grouped metrics.

    Returns
    -------
    dict[str, Any]
        JSON-compatible split diagnostics.
    '''

    def _split_summary(indices: np.ndarray, split_name: str) -> dict[str, Any]:
        split_labels = np.asarray(labels, dtype=int)[indices]
        actives = int(np.sum(split_labels == 1))
        decoys = int(np.sum(split_labels == 0))
        total = int(len(split_labels))
        summary: dict[str, Any] = {
            "split": split_name,
            "n_rows": total,
            "n_actives": actives,
            "n_decoys": decoys,
            "active_fraction": float(actives / total) if total else float("nan"),
            "n_targets": None,
            "targets_with_zero_actives": [],
            "targets_with_zero_decoys": [],
            "per_target_counts": {},
        }
        if groups is None:
            return summary

        split_groups = np.asarray(groups)[indices]
        unique_targets = np.unique(split_groups)
        summary["n_targets"] = int(len(unique_targets))
        per_target: dict[str, dict[str, int]] = {}
        zero_actives: list[str] = []
        zero_decoys: list[str] = []
        for target in unique_targets:
            mask = split_groups == target
            target_labels = split_labels[mask]
            target_actives = int(np.sum(target_labels == 1))
            target_decoys = int(np.sum(target_labels == 0))
            target_name = str(target)
            per_target[target_name] = {
                "n_rows": int(mask.sum()),
                "n_actives": target_actives,
                "n_decoys": target_decoys,
            }
            if target_actives == 0:
                zero_actives.append(target_name)
            if target_decoys == 0:
                zero_decoys.append(target_name)
        summary["targets_with_zero_actives"] = zero_actives
        summary["targets_with_zero_decoys"] = zero_decoys
        summary["per_target_counts"] = per_target
        return summary

    diagnostics = {
        "target_group_column": target_group_column,
        "splits": {
            "train": _split_summary(np.asarray(train_idx), "train"),
            "validation": _split_summary(np.asarray(val_idx), "validation"),
            "test": _split_summary(np.asarray(test_idx), "test"),
        },
    }
    return diagnostics


def _log_dudez_split_diagnostics(diagnostics: dict[str, Any]) -> None:
    '''Log DUDEz split diagnostics at DEBUG level.

    Parameters
    ----------
    diagnostics : dict[str, Any]
        Split diagnostics from :func:`summarize_dudez_split_diagnostics` or
        :func:`split_dudez_by_receptor_and_kind`.
    '''

    if diagnostics.get("strategy"):
        LOGGER.debug(
            "DUDEz split strategy=%s seed=%s relaxed=%s n_groups_used_validation=%s",
            diagnostics.get("strategy"),
            diagnostics.get("random_seed"),
            diagnostics.get("relaxed_split"),
            diagnostics.get("n_groups_used_validation"),
        )

    for split_name, summary in diagnostics.get("splits", {}).items():
        LOGGER.debug(
            "DUDEz %s split: rows=%s actives=%s decoys=%s active_fraction=%.4f targets=%s "
            "valid_metric_groups=%s",
            split_name,
            summary.get("n_rows"),
            summary.get("n_actives", summary.get("n_ligands")),
            summary.get("n_decoys"),
            summary.get("active_fraction"),
            summary.get("n_targets", summary.get("n_receptors")),
            summary.get("n_valid_metric_groups"),
        )
        zero_actives = summary.get("targets_with_zero_actives") or []
        zero_decoys = summary.get("targets_with_zero_decoys") or []
        if zero_actives:
            LOGGER.warning(
                "DUDEz %s split has targets with zero actives: %s",
                split_name,
                zero_actives[:10],
            )
        if zero_decoys:
            LOGGER.warning(
                "DUDEz %s split has targets with zero decoys: %s",
                split_name,
                zero_decoys[:10],
            )
        invalid = summary.get("invalid_metric_groups") or []
        if invalid:
            LOGGER.warning(
                "DUDEz %s split invalid metric groups: %s",
                split_name,
                invalid[:10],
            )


def _set_dudez_trial_split_metric_attrs(
        trial: optuna.Trial,
        metrics: dict[str, float],
        metrics_scope: str,
        *,
        split_label: str,
    ) -> None:
    '''Store DUDEz screening metrics for one split on an Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial receiving user attributes.
    metrics : dict[str, float]
        Screening metrics for the split.
    metrics_scope : str
        ``"global"`` or ``"grouped"``.
    split_label : str
        Split prefix, for example ``"validation"`` or ``"test"``.
    '''

    prefix = f"{split_label}_"
    if split_label == "validation":
        trial.set_user_attr("metrics_scope", metrics_scope)
        trial.set_user_attr("ranking_metrics_valid", metrics.get("ranking_metrics_valid"))
    if split_label == "test":
        trial.set_user_attr("test_ranking_metrics_valid", metrics.get("ranking_metrics_valid"))

    trial.set_user_attr(f"{prefix}score_std", metrics.get("score_std"))
    trial.set_user_attr(f"{prefix}n_unique_scores", metrics.get("n_unique_scores"))
    trial.set_user_attr(f"{prefix}n_groups_total", metrics.get("n_groups_total"))
    trial.set_user_attr(f"{prefix}n_groups_used", metrics.get("n_groups_used"))
    trial.set_user_attr(
        f"{prefix}n_groups_invalid_constant_score",
        metrics.get("n_groups_invalid_constant_score"),
    )
    trial.set_user_attr(
        f"{prefix}n_groups_invalid_one_class",
        metrics.get("n_groups_invalid_one_class"),
    )
    trial.set_user_attr(
        f"{prefix}n_groups_invalid_nonfinite",
        metrics.get("n_groups_invalid_nonfinite"),
    )
    for metric_name in DUDEZ_REPORT_METRICS:
        if metric_name in metrics:
            trial.set_user_attr(f"{prefix}{metric_name}", metrics[metric_name])
        global_name = f"{metric_name}_global"
        if global_name in metrics:
            trial.set_user_attr(f"{prefix}{global_name}", metrics[global_name])
        std_name = f"{metric_name}_group_std"
        if std_name in metrics:
            trial.set_user_attr(f"{prefix}{std_name}", metrics[std_name])


def _set_dudez_trial_metric_attrs(
        trial: optuna.Trial,
        metrics: dict[str, float],
        primary_metric: str,
        metrics_scope: str,
    ) -> None:
    '''Store DUDEz validation metric diagnostics on an Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial receiving user attributes.
    metrics : dict[str, float]
        Validation screening metrics.
    primary_metric : str
        Effective objective metric name.
    metrics_scope : str
        ``"global"`` or ``"grouped"``.
    '''

    trial.set_user_attr("objective_metric", primary_metric)
    _set_dudez_trial_split_metric_attrs(
        trial,
        metrics,
        metrics_scope,
        split_label="validation",
    )


def _set_dudez_trial_test_metric_attrs(
        trial: optuna.Trial,
        metrics: dict[str, float],
        metrics_scope: str,
        primary_metric: str,
    ) -> None:
    '''Store held-out test screening metrics on an Optuna trial.

    Reserved for tooling that records a one-shot post-selection evaluation.
    Must not be called from Optuna objectives during hyperparameter search.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial receiving user attributes.
    metrics : dict[str, float]
        Test screening metrics.
    metrics_scope : str
        ``"global"`` or ``"grouped"``.
    primary_metric : str
        Effective validation objective metric name (for cross-reference).
    '''

    _set_dudez_trial_split_metric_attrs(
        trial,
        metrics,
        metrics_scope,
        split_label="test",
    )
    trial.set_user_attr("test_objective_metric_name", primary_metric)
    if primary_metric in metrics:
        trial.set_user_attr("test_objective_metric_value", metrics[primary_metric])


def _collect_dudez_training_diagnostics(
        model: DUDEzScreeningModel,
        splits: dict[str, Any],
        device: torch.device,
        criterion: nn.Module,
        train_loader: DataLoader,
    ) -> dict[str, float]:
    '''Collect training/validation diagnostics for one DUDEz trial.

    Parameters
    ----------
    model : DUDEzScreeningModel
        Trained screening model.
    splits : dict[str, Any]
        Prepared DUDEz train/validation arrays.
    device : torch.device
        Torch device.
    criterion : nn.Module
        BCE loss used during training.
    train_loader : DataLoader
        Training dataloader.

    Returns
    -------
    dict[str, float]
        Training and validation diagnostics for logging.
    '''

    model.train()
    train_losses: list[float] = []
    with torch.no_grad():
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).view(-1)
            logits = model(features)
            train_losses.append(float(criterion(logits, labels.float()).item()))

    val_score, val_true = _predict_screening(model, splits["X_val"], splits["y_val"], device)
    val_labels = np.asarray(val_true, dtype=int)
    val_scores = np.asarray(val_score, dtype=float)
    active_mask = val_labels == 1
    decoy_mask = val_labels == 0
    diagnostics = {
        "train_bce_loss_mean": float(np.mean(train_losses)) if train_losses else float("nan"),
        "validation_bce_loss": float(
            criterion(
                torch.tensor(val_scores, dtype=torch.float32, device=device),
                torch.tensor(val_labels, dtype=torch.float32, device=device),
            ).item()
        ),
        "validation_score_std": float(np.std(val_scores)) if val_scores.size else 0.0,
        "validation_mean_logit_active": float(np.mean(val_scores[active_mask])) if active_mask.any() else float("nan"),
        "validation_mean_logit_decoy": float(np.mean(val_scores[decoy_mask])) if decoy_mask.any() else float("nan"),
        "validation_active_fraction": float(active_mask.mean()) if val_labels.size else float("nan"),
    }
    return diagnostics


def _resolve_dudez_split_config(
        config: DUDEzOptunaConfig,
        random_seed: int,
    ) -> DUDEzSplitConfig:
    '''Build the effective DUDEz split configuration for a stage run.

    Parameters
    ----------
    config : DUDEzOptunaConfig
        DUDEz Optuna stage configuration.
    random_seed : int
        Effective random seed for the stage.

    Returns
    -------
    DUDEzSplitConfig
        Resolved split configuration.
    '''

    if config.split_config is not None:
        resolved = config.split_config
        if resolved.random_seed != random_seed:
            return DUDEzSplitConfig(
                **{**asdict(resolved), "random_seed": int(random_seed)}
            )
        return resolved

    return dudez_receptor_heldout_complete_config(
        random_seed=int(random_seed),
        receptor_column=config.target_group_column,
        kind_column=config.kind_column,
        relaxed_split=False,
    )


def _resolve_pdbbind_split_config(
        config: PDBbindOptunaConfig,
        random_seed: int,
    ) -> PDBbindSplitConfig:
    '''Build the effective PDBbind split configuration for a stage run.'''

    if config.split_config is not None:
        resolved = config.split_config
        if resolved.random_seed != random_seed:
            return PDBbindSplitConfig(
                **{**asdict(resolved), "random_seed": int(random_seed)}
            )
        return resolved

    train_size = 1.0 - float(config.validation_size) - float(config.test_size)
    return PDBbindSplitConfig(
        strategy="affinity_quantile_stratified",
        target_column=config.target_column,
        receptor_column="receptor",
        n_affinity_bins=5,
        train_size=train_size,
        validation_size=float(config.validation_size),
        test_size=float(config.test_size),
        random_seed=int(random_seed),
        relaxed_split=False,
    )


def prepare_dudez_screening_data(
        df: pd.DataFrame,
        selected_features: Sequence[str],
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        split_config: DUDEzSplitConfig,
        target_group_column: Optional[str] = None,
        scaling_config: Optional[DUDEzScalingConfig] = None,
        pdbbind_scaler: Optional[StandardScaler] = None,
        *,
        fixed_train_indices: Optional[Sequence[int]] = None,
        fixed_validation_indices: Optional[Sequence[int]] = None,
        fixed_test_indices: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
    '''Prepare train/validation/test arrays for DUDEz screening.

    Parameters
    ----------
    df : pd.DataFrame
        Reduced DUDEz dataframe.
    selected_features : Sequence[str]
        Descriptor columns selected by feature reduction.
    labels : np.ndarray
        Binary active/decoy labels.
    groups : np.ndarray | None
        Optional target groups aligned with ``df`` (used for grouped metrics).
    split_config : DUDEzSplitConfig
        Receptor/kind-aware split configuration.
    target_group_column : str | None, optional
        Target column name recorded in split diagnostics.

    Returns
    -------
    dict[str, Any]
        Train, validation, and test arrays with source indices.
    '''

    X = df[list(selected_features)].to_numpy(dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    group_array = None if groups is None else np.asarray(groups)
    if (
        fixed_train_indices is not None
        and fixed_validation_indices is not None
        and fixed_test_indices is not None
    ):
        train_idx = np.asarray(fixed_train_indices, dtype=int)
        val_idx = np.asarray(fixed_validation_indices, dtype=int)
        test_idx = np.asarray(fixed_test_indices, dtype=int)
        split_diagnostics = {
            "strategy": split_config.strategy,
            "fixed_outer_split": True,
            "random_seed": int(split_config.random_seed),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }
    else:
        split_result = split_dudez_by_receptor_and_kind(df, split_config)
        train_idx = split_result.train_idx
        val_idx = split_result.val_idx
        test_idx = split_result.test_idx
        split_diagnostics = split_result.diagnostics
    if target_group_column is not None:
        split_diagnostics["target_group_column"] = target_group_column
    _log_dudez_split_diagnostics(split_diagnostics)

    if scaling_config is None:
        default_strategy = "pdbbind_scaler" if pdbbind_scaler is not None else "dudez_train_scaler"
        scaling = DUDEzScalingConfig(strategy=default_strategy, strict=True)
    else:
        scaling = scaling_config
    X_train, X_val, X_test, scaling_metadata, dudez_scaler = scale_dudez_features(
        X,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        config=scaling,
        selected_features=selected_features,
        pdbbind_scaler=pdbbind_scaler,
    )
    return {
        "X_train": X_train,
        "y_train": y[train_idx],
        "X_val": X_val,
        "y_val": y[val_idx],
        "X_test": X_test,
        "y_test": y[test_idx],
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "test_indices": test_idx,
        "train_groups": None if group_array is None else group_array[train_idx],
        "val_groups": None if group_array is None else group_array[val_idx],
        "test_groups": None if group_array is None else group_array[test_idx],
        "split_diagnostics": split_diagnostics,
        "scaling_metadata": scaling_metadata,
        "dudez_scaler": dudez_scaler,
    }


def prepare_pdbbind_regression_data(
        df: pd.DataFrame,
        selected_features: Sequence[str],
        split_config: PDBbindSplitConfig,
        *,
        fixed_train_indices: Optional[Sequence[int]] = None,
        fixed_validation_indices: Optional[Sequence[int]] = None,
        fixed_test_indices: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
    '''Prepare scaled train/validation/test arrays for PDBbind regression.

    Parameters
    ----------
    df : pd.DataFrame
        Reduced PDBbind dataframe.
    selected_features : Sequence[str]
        Descriptor columns selected by feature reduction.
    split_config : PDBbindSplitConfig
        PDBbind split configuration (used for metadata when fixed indices are supplied).
    fixed_train_indices : Sequence[int] | None, optional
        When provided, reuse this fixed training partition instead of recomputing a split.
    fixed_validation_indices : Sequence[int] | None, optional
        Fixed validation partition indices.
    fixed_test_indices : Sequence[int] | None, optional
        Fixed test partition indices.

    Returns
    -------
    dict[str, Any]
        Scaled train, validation, and test arrays with source indices and scaler.
    '''

    X = df[list(selected_features)].to_numpy(dtype=np.float32)
    y = df[split_config.target_column].to_numpy(dtype=np.float32)
    if (
        fixed_train_indices is not None
        and fixed_validation_indices is not None
        and fixed_test_indices is not None
    ):
        train_idx = np.asarray(fixed_train_indices, dtype=int)
        val_idx = np.asarray(fixed_validation_indices, dtype=int)
        test_idx = np.asarray(fixed_test_indices, dtype=int)
        split_diagnostics = {
            "strategy": split_config.strategy,
            "fixed_outer_split": True,
            "random_seed": int(split_config.random_seed),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        }
    else:
        split_result = split_pdbbind_regression(df, split_config, target=y)
        train_idx = split_result.train_idx
        val_idx = split_result.val_idx
        test_idx = split_result.test_idx
        split_diagnostics = split_result.diagnostics

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_val = scaler.transform(X[val_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    return {
        "X_train": X_train,
        "y_train": y[train_idx],
        "X_val": X_val,
        "y_val": y[val_idx],
        "X_test": X_test,
        "y_test": y[test_idx],
        "train_indices": train_idx,
        "validation_indices": val_idx,
        "test_indices": test_idx,
        "scaler": scaler,
        "split_diagnostics": split_diagnostics,
    }


def resolve_dudez_primary_metric(requested_metric: str, metrics: dict[str, float]) -> str:
    '''Return the effective DUDEz objective metric.

    BEDROC is preferred by default. If it is missing or non-finite, PR-AUC is
    used as the primary objective.

    Parameters
    ----------
    requested_metric : str
        Requested screening objective metric.
    metrics : dict[str, float]
        Validation screening metrics.

    Returns
    -------
    str
        Effective metric name used for optimization.
    '''

    if requested_metric != "BEDROC":
        return requested_metric

    value = metrics.get(requested_metric)
    if value is None:
        return "PR-AUC"
    numeric = float(value)
    if np.isnan(numeric) or np.isinf(numeric):
        return "PR-AUC"

    if not bool(metrics.get("ranking_metrics_valid")):
        return requested_metric
    groups_used = metrics.get("n_groups_used")
    if groups_used is not None and not np.isnan(float(groups_used)) and float(groups_used) <= 0.0:
        return "PR-AUC"
    return requested_metric


def run_staged_ocscore_optuna_protocol(
        context: ProtocolContext,
        pdbbind_config: Optional[PDBbindOptunaConfig] = None,
        dudez_config: Optional[DUDEzOptunaConfig] = None,
    ) -> ProtocolContext:
    '''Run the current staged OCScore Optuna protocol.

    Parameters
    ----------
    context : ProtocolContext
        Initial protocol context.
    pdbbind_config : PDBbindOptunaConfig | None, optional
        PDBbind stage configuration, by default None.
    dudez_config : DUDEzOptunaConfig | None, optional
        DUDEz stage configuration, by default None.

    Returns
    -------
    ProtocolContext
        Updated protocol context after all stages complete.
    '''

    protocol = StagedProtocol([
        PDBbindOptunaStage(config=pdbbind_config),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(config=dudez_config),
    ])
    return protocol.run(context)


def suggest_dudez_trial_params(
        trial: optuna.Trial,
        allow_scratch: bool = True,
        search_space: Optional[DUDEzSearchSpaceConfig] = None,
    ) -> dict[str, Any]:
    '''Sample DUDEz screening hyperparameters for one Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    allow_scratch : bool, optional
        Allow sampling from-scratch feature extractors, by default True.
    search_space : DUDEzSearchSpaceConfig | None, optional
        Centralized search-space definition.

    Returns
    -------
    dict[str, Any]
        Sampled DUDEz hyperparameters.
    '''

    space = search_space or DEFAULT_DUDEZ_SEARCH_SPACE
    activation_options = list(space.activation_options)
    use_transfer = True
    if allow_scratch:
        use_transfer = bool(
            trial.suggest_categorical(
                "dudez_use_transfer",
                list(space.dudez_head.use_transfer_options),
            )
        )
    fine_tuning_mode = (
        trial.suggest_categorical(
            "dudez_fine_tuning_mode",
            list(space.dudez_head.fine_tuning_mode_options),
        )
        if use_transfer
        else "full"
    )
    return {
        "dudez_use_transfer": use_transfer,
        "dudez_fine_tuning_mode": fine_tuning_mode,
        "dudez_num_unfrozen_layers": int(
            trial.suggest_categorical(
                "dudez_num_unfrozen_layers",
                list(space.dudez_head.num_unfrozen_layers_options),
            )
        ),
        "dudez_classifier_hidden_size": int(
            trial.suggest_categorical(
                "dudez_classifier_hidden_size",
                list(space.dudez_head.classifier_hidden_size_options),
            )
        ),
        "dudez_classifier_dropout": float(
            trial.suggest_float(
                "dudez_classifier_dropout",
                space.dudez_head.classifier_dropout_min,
                space.dudez_head.classifier_dropout_max,
            )
        ),
        "dudez_classifier_activation": str(
            trial.suggest_categorical("dudez_classifier_activation", activation_options)
        ),
        "dudez_use_class_weighting": bool(
            trial.suggest_categorical(
                "dudez_use_class_weighting",
                list(space.dudez_head.use_class_weighting_options),
            )
        ),
        "optimizer_learning_rate": float(
            trial.suggest_float(
                "optimizer_learning_rate",
                space.optimizer.learning_rate_min,
                space.optimizer.learning_rate_max,
                log=True,
            )
        ),
        "optimizer_weight_decay": float(
            trial.suggest_float(
                "optimizer_weight_decay",
                space.optimizer.weight_decay_min,
                space.optimizer.weight_decay_max,
                log=True,
            )
        ),
        "optimizer_batch_size": int(
            trial.suggest_categorical(
                "optimizer_batch_size",
                list(space.optimizer.batch_size_options),
            )
        ),
    }


class InvalidEncoderArchitecture(ValueError):
    """Raised when sampled encoder params are structurally invalid."""


def _nearest_lower_power_of_two(value: int) -> int:
    v = int(value)
    if v <= 0:
        return 0
    return 1 << (v.bit_length() - 1)


def _max_consecutive_equal_run(values: Sequence[int]) -> int:
    """Return the longest run of identical consecutive values."""

    if not values:
        return 0
    if len(values) == 1:
        return 1

    best = 1
    current = 1
    for idx in range(1, len(values)):
        if int(values[idx]) == int(values[idx - 1]):
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def _max_allowed_equal_chain(depth: int, *, max_chain: int = 2) -> int:
    """Maximum allowed length of identical consecutive hidden sizes.

    For depth=2 this is 1 (no ``[x, x]``). For deeper stacks, at most
    ``max_chain`` identical layers may appear in a row (default 2).
    """

    d = int(depth)
    if d <= 1:
        return 1
    return min(int(max_chain), d - 1)


def _count_equal_plateau_blocks(values: Sequence[int]) -> int:
    """Count contiguous runs of identical values with length >= 2.

    Examples
    --------
    ``[256, 128, 128, 64]`` → 1 block; ``[256, 256, 128, 128]`` → 2 blocks.
    """

    if len(values) < 2:
        return 0

    blocks = 0
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and int(values[end]) == int(values[idx]):
            end += 1
        if end - idx >= 2:
            blocks += 1
        idx = end
    return int(blocks)


def _max_allowed_plateau_blocks(depth: int) -> int:
    """Maximum number of separate equal-width plateau blocks per architecture.

    Depth 2 allows none (no ``[x, x]``). Depth >= 3 allows at most one block
  (e.g. ``256, 128, 128, 64``) but not two (e.g. ``256, 256, 128, 128``).
    """

    d = int(depth)
    if d <= 2:
        return 0
    return 1


def _hidden_sizes_respect_equal_chain_limit(
    hidden_sizes: Sequence[int],
    *,
    depth: int,
    max_chain: int = 2,
) -> bool:
    """Return whether hidden sizes satisfy plateau rules for encoder/decoder stacks.

    Rules:
    - No run of identical consecutive layers longer than
      :func:`_max_allowed_equal_chain` (default 2; depth 2 forbids any repeat).
    - At most :func:`_max_allowed_plateau_blocks` separate repetition blocks
      (runs of length >= 2).
    """

    if _max_consecutive_equal_run(hidden_sizes) > _max_allowed_equal_chain(
        depth,
        max_chain=max_chain,
    ):
        return False
    return _count_equal_plateau_blocks(hidden_sizes) <= _max_allowed_plateau_blocks(depth)


def _encoder_allowed_hidden_sizes(
    input_dim: int,
    *,
    min_hidden: int,
) -> list[int]:
    """Return the deterministic power-of-two ladder for encoder hidden sizes."""

    min_h = int(min_hidden)
    if min_h <= 0:
        min_h = 32
    max_h = _nearest_lower_power_of_two(int(input_dim))
    if max_h <= 0:
        return []
    if max_h < min_h:
        return []

    out: list[int] = []
    size = int(max_h)
    while size >= int(min_h):
        out.append(int(size))
        size //= 2
    return out


def _encoder_small_input_hidden_sizes(
    input_dim: int,
    *,
    min_hidden: int,
    minimum_hidden: int = 2,
) -> list[int]:
    """Return a power-of-two compression ladder for small input dimensions."""

    max_h = _nearest_lower_power_of_two(int(input_dim))
    below_configured_min = _nearest_lower_power_of_two(max(int(min_hidden) - 1, 1))
    if below_configured_min > 0:
        max_h = min(max_h, below_configured_min)
    if max_h <= 0:
        return []
    floor = max(int(minimum_hidden), 1)
    out: list[int] = []
    size = int(max_h)
    while size >= floor:
        out.append(int(size))
        size //= 2
    return out


def _encoder_architecture_candidates_from_ladder(
    *,
    allowed_hidden: Sequence[int],
    depth_options: Sequence[int],
    latent_dim_options: Sequence[int],
    plateaus_allowed: bool = True,
    max_hidden_layers: int | None = None,
    max_equal_chain: int = 2,
) -> list[dict[str, Any]]:
    """Return deterministic valid monotonic encoder candidates from a ladder."""

    allowed_hidden = sorted({int(v) for v in allowed_hidden if int(v) > 0}, reverse=True)
    if not allowed_hidden:
        return []

    depths = sorted({int(d) for d in depth_options if int(d) > 0})
    if max_hidden_layers is not None:
        depths = [d for d in depths if int(d) <= int(max_hidden_layers)]
    if not depths:
        return []

    latent_ladder = sorted({int(v) for v in latent_dim_options if int(v) > 0})
    if not latent_ladder:
        return []

    candidates: list[dict[str, Any]] = []

    def _extend(prefix: list[int], *, depth: int) -> None:
        if len(prefix) >= int(depth):
            hidden_sizes = list(prefix)
            if not _hidden_sizes_respect_equal_chain_limit(
                hidden_sizes,
                depth=int(depth),
                max_chain=int(max_equal_chain),
            ):
                return
            last_hidden = int(hidden_sizes[-1])
            for latent in latent_ladder:
                if int(latent) <= int(last_hidden):
                    candidates.append({
                        "encoder_hidden_sizes": hidden_sizes,
                        "encoder_latent_dim": int(latent),
                        "encoder_depth": int(depth),
                        "encoder_is_monotonic": True,
                    })
            return

        last = int(prefix[-1]) if prefix else None
        for v in allowed_hidden:
            if last is not None:
                if plateaus_allowed:
                    if int(v) > last:
                        continue
                else:
                    if int(v) >= last:
                        continue
            _extend([*prefix, int(v)], depth=int(depth))

    for depth in depths:
        _extend([], depth=int(depth))

    candidates.sort(key=lambda c: (int(c["encoder_depth"]), list(c["encoder_hidden_sizes"]), int(c["encoder_latent_dim"])))
    return candidates


def _encoder_architecture_candidates(
    *,
    input_dim: int,
    depth_options: Sequence[int],
    latent_dim_options: Sequence[int],
    min_hidden: int,
    plateaus_allowed: bool = True,
    max_hidden_layers: int | None = None,
    max_equal_chain: int = 2,
) -> list[dict[str, Any]]:
    """Return deterministic valid monotonic encoder architecture candidates."""

    allowed_hidden = _encoder_allowed_hidden_sizes(int(input_dim), min_hidden=int(min_hidden))
    candidates = _encoder_architecture_candidates_from_ladder(
        allowed_hidden=allowed_hidden,
        depth_options=depth_options,
        latent_dim_options=latent_dim_options,
        plateaus_allowed=plateaus_allowed,
        max_hidden_layers=max_hidden_layers,
        max_equal_chain=max_equal_chain,
    )
    if candidates:
        return candidates

    # The current candidate list stays first. Adaptive small-input compressions
    # are appended only when no current architecture is valid for this input.
    small_hidden = _encoder_small_input_hidden_sizes(
        int(input_dim),
        min_hidden=int(min_hidden),
    )
    if not small_hidden:
        return []
    small_depths = sorted({1, *[int(d) for d in depth_options if int(d) > 0]})
    small_latents = sorted(
        {
            *[int(v) for v in latent_dim_options if int(v) > 0],
            *[int(v) for v in small_hidden if int(v) > 0],
        },
    )
    return _encoder_architecture_candidates_from_ladder(
        allowed_hidden=small_hidden,
        depth_options=small_depths,
        latent_dim_options=small_latents,
        plateaus_allowed=False,
        max_hidden_layers=max_hidden_layers,
        max_equal_chain=max_equal_chain,
    )


def _decoder_architecture_candidates(
    *,
    decoder_start_dim: int,
    input_dim: int,
    depth_options: Sequence[int],
    hidden_size_options: Sequence[int],
    plateaus_allowed: bool = True,
    max_equal_chain: int = 2,
) -> list[dict[str, Any]]:
    """Return deterministic monotonic non-decreasing decoder candidates."""

    start = int(decoder_start_dim)
    end = int(input_dim)
    if start <= 0 or end <= 0:
        return []
    if start > end:
        return []

    ladder = sorted(
        {int(v) for v in hidden_size_options if int(v) >= start and int(v) <= end},
    )
    if not ladder:
        return []

    depths = sorted({int(d) for d in depth_options if int(d) > 0})
    if not depths:
        return []

    candidates: list[dict[str, Any]] = []

    def _extend(prefix: list[int], *, depth: int) -> None:
        if len(prefix) >= depth:
            sizes = list(prefix)
            if not _hidden_sizes_respect_equal_chain_limit(
                sizes,
                depth=int(depth),
                max_chain=int(max_equal_chain),
            ):
                return
            candidates.append({
                "decoder_hidden_sizes": sizes,
                "decoder_depth": int(depth),
                "decoder_is_monotonic_increasing": True,
                "decoder_is_monotonic_decreasing": False,
            })
            return
        last = int(prefix[-1]) if prefix else int(start)
        for v in ladder:
            if plateaus_allowed:
                if int(v) < last:
                    continue
            else:
                if int(v) <= last:
                    continue
            _extend([*prefix, int(v)], depth=int(depth))

    for depth in depths:
        _extend([], depth=int(depth))

    candidates.sort(key=lambda c: (int(c["decoder_depth"]), list(c["decoder_hidden_sizes"])))
    return candidates


def _resolve_projection_dim(
        trial: optuna.Trial,
        sampled_projection_dim: int,
        latent_dim: int,
    ) -> int:
    """Return the effective projection size without expanding encoder output."""

    sampled = int(sampled_projection_dim)
    latent = int(latent_dim)
    if sampled <= 0:
        return 0
    if sampled <= latent:
        return sampled
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("projection_dim_sampled", int(sampled))
        trial.set_user_attr("projection_dim_effective", 0)
        trial.set_user_attr("projection_disabled_reason", "projection_would_expand_encoder")
    return 0


def suggest_encoder_architecture(
        trial: optuna.Trial,
        input_dim: int,
        search_space: Optional[EncoderSearchSpace] = None,
    ) -> tuple[int, list[int], int]:
    '''Sample monotonic encoder hidden sizes and latent dimension.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    input_dim : int
        Input feature dimension.
    search_space : EncoderSearchSpace | None, optional
        Encoder search-space definition.

    Returns
    -------
    tuple[int, list[int], int]
        Encoder architecture index, hidden layer sizes, and latent dimension.
    '''

    encoder_space = search_space or DEFAULT_PDBBIND_SEARCH_SPACE.encoder

    min_hidden = (
        min(int(v) for v in encoder_space.hidden_size_options)
        if encoder_space.hidden_size_options
        else 32
    )
    candidates = _encoder_architecture_candidates(
        input_dim=int(input_dim),
        depth_options=list(encoder_space.depth_options),
        latent_dim_options=list(encoder_space.latent_dim_options),
        min_hidden=int(min_hidden),
        plateaus_allowed=True,
        max_hidden_layers=int(encoder_space.max_hidden_layers),
    )
    if not candidates:
        raise InvalidEncoderArchitecture(
            f"No valid encoder architectures for input_dim={input_dim} (min_hidden={min_hidden})."
        )

    encoder_architecture_index = int(
        trial.suggest_int("encoder_architecture_index", 0, len(candidates) - 1)
    )
    chosen = candidates[int(encoder_architecture_index)]
    hidden_sizes = list(chosen["encoder_hidden_sizes"])
    latent_dim = int(chosen["encoder_latent_dim"])
    depth = int(chosen["encoder_depth"])

    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("encoder_architecture_index", int(encoder_architecture_index))
        trial.set_user_attr("encoder_hidden_sizes", list(hidden_sizes))
        trial.set_user_attr("encoder_latent_dim", int(latent_dim))
        trial.set_user_attr("encoder_depth", int(depth))
        trial.set_user_attr("encoder_is_monotonic", True)

    # Defensive validation (belt-and-suspenders).
    sizes = [*hidden_sizes, latent_dim]
    if any(left < right for left, right in zip(sizes, sizes[1:])):
        raise InvalidEncoderArchitecture(f"Invalid encoder architecture: {sizes}")
    if latent_dim > int(hidden_sizes[-1]):
        raise InvalidEncoderArchitecture(f"Invalid encoder architecture: {sizes}")
    return int(encoder_architecture_index), hidden_sizes, int(latent_dim)


def suggest_decoder_hidden_sizes(
        trial: optuna.Trial,
        search_space: DecoderSearchSpace,
        latent_dim: int,
        projection_dim: int,
        input_dim: int,
    ) -> tuple[Optional[list[int]], float]:
    '''Sample optional decoder hidden sizes for PDBbind reconstruction.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    search_space : DecoderSearchSpace
        Decoder search-space definition.
    latent_dim : int
        Encoder latent dimension.
    input_dim : int
        Original input feature dimension.

    Returns
    -------
    tuple[list[int] | None, float]
        Decoder hidden sizes and reconstruction-loss weight.
    '''

    lambda_rec = float(
        trial.suggest_categorical("decoder_lambda_rec", list(search_space.lambda_rec_options))
    )
    if lambda_rec <= 0.0:
        return None, lambda_rec

    decoder_start_dim = int(projection_dim) if int(projection_dim) > 0 else int(latent_dim)
    candidates = _decoder_architecture_candidates(
        decoder_start_dim=decoder_start_dim,
        input_dim=int(input_dim),
        depth_options=list(search_space.depth_options),
        hidden_size_options=list(search_space.hidden_size_options),
        plateaus_allowed=True,
    )
    if not candidates:
        if any(float(v) <= 0.0 for v in search_space.lambda_rec_options):
            if hasattr(trial, "set_user_attr"):
                trial.set_user_attr("decoder_start_dim", int(decoder_start_dim))
                trial.set_user_attr("decoder_lambda_rec_sampled", float(lambda_rec))
                trial.set_user_attr("decoder_disabled_reason", "no_valid_decoder_architecture")
            return None, 0.0
        raise InvalidEncoderArchitecture(
            f"No valid decoder architectures for start_dim={decoder_start_dim} input_dim={input_dim}."
        )
    idx = int(trial.suggest_int("decoder_architecture_index", 0, len(candidates) - 1))
    chosen = candidates[int(idx)]
    decoder_hidden_sizes = list(chosen["decoder_hidden_sizes"])
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr("decoder_start_dim", int(decoder_start_dim))
        trial.set_user_attr("decoder_architecture_index", int(idx))
        trial.set_user_attr("decoder_hidden_sizes", list(decoder_hidden_sizes))
        trial.set_user_attr("decoder_depth", int(chosen["decoder_depth"]))
        trial.set_user_attr("decoder_is_monotonic_increasing", bool(chosen.get("decoder_is_monotonic_increasing", True)))
        trial.set_user_attr("decoder_is_monotonic_decreasing", bool(chosen.get("decoder_is_monotonic_decreasing", False)))
    return decoder_hidden_sizes, lambda_rec


def suggest_pdbbind_trial_params(
        trial: optuna.Trial,
        input_dim: int,
        search_space: Optional[PDBbindSearchSpaceConfig] = None,
    ) -> dict[str, Any]:
    '''Sample PDBbind regression hyperparameters for one Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    input_dim : int
        Input feature dimension.
    search_space : PDBbindSearchSpaceConfig | None, optional
        Centralized search-space definition.

    Returns
    -------
    dict[str, Any]
        Sampled PDBbind hyperparameters.
    '''

    space = search_space or DEFAULT_PDBBIND_SEARCH_SPACE
    activation_options = list(space.activation_options)
    encoder_architecture_index, hidden_sizes, latent_dim = suggest_encoder_architecture(
        trial,
        input_dim=input_dim,
        search_space=space.encoder,
    )
    projection_dim_sampled = int(
        trial.suggest_categorical("projection_dim", list(space.projection.projection_dim_options))
    )
    projection_dim = _resolve_projection_dim(
        trial,
        sampled_projection_dim=projection_dim_sampled,
        latent_dim=latent_dim,
    )
    decoder_hidden_sizes, lambda_rec = suggest_decoder_hidden_sizes(
        trial,
        search_space=space.decoder,
        latent_dim=latent_dim,
        projection_dim=projection_dim,
        input_dim=input_dim,
    )
    dae_noise_type = "none"
    dae_mask_prob = 0.0
    dae_gaussian_std = 0.0
    if float(lambda_rec) > 0.0:
        dae_noise_type = str(trial.suggest_categorical("dae_noise_type", ["none", "mask", "gaussian", "mask+gaussian"]))
        dae_mask_prob = float(trial.suggest_float("dae_mask_prob", 0.0, 0.2))
        dae_gaussian_std = float(trial.suggest_float("dae_gaussian_std", 0.0, 0.1))
        if hasattr(trial, "set_user_attr"):
            trial.set_user_attr("dae_enabled", True)
            trial.set_user_attr("dae_noise_type", dae_noise_type)
            trial.set_user_attr("dae_mask_prob", dae_mask_prob)
            trial.set_user_attr("dae_gaussian_std", dae_gaussian_std)
    return {
        "encoder_architecture_index": int(encoder_architecture_index),
        "encoder_hidden_sizes": hidden_sizes,
        "encoder_latent_dim": latent_dim,
        "encoder_depth": len(hidden_sizes),
        "encoder_is_monotonic": True,
        "projection_dim": projection_dim,
        "encoder_activation": str(trial.suggest_categorical("encoder_activation", activation_options)),
        "encoder_dropout": float(
            trial.suggest_float(
                "encoder_dropout",
                space.encoder.dropout_min,
                space.encoder.dropout_max,
            )
        ),
        "optimizer_learning_rate": float(
            trial.suggest_float(
                "optimizer_learning_rate",
                space.optimizer.learning_rate_min,
                space.optimizer.learning_rate_max,
                log=True,
            )
        ),
        "optimizer_weight_decay": float(
            trial.suggest_float(
                "optimizer_weight_decay",
                space.optimizer.weight_decay_min,
                space.optimizer.weight_decay_max,
                log=True,
            )
        ),
        "optimizer_batch_size": int(
            trial.suggest_categorical(
                "optimizer_batch_size",
                list(space.optimizer.batch_size_options),
            )
        ),
        "decoder_hidden_sizes": decoder_hidden_sizes,
        "decoder_depth": len(decoder_hidden_sizes) if decoder_hidden_sizes else 0,
        "decoder_lambda_rec": lambda_rec,
        "dae_noise_type": dae_noise_type,
        "dae_mask_prob": dae_mask_prob,
        "dae_gaussian_std": dae_gaussian_std,
        "pdbbind_regression_loss": str(
            trial.suggest_categorical(
                "pdbbind_regression_loss",
                list(space.pdbbind_head.regression_loss_options),
            )
        ),
        "pdbbind_huber_delta": float(
            trial.suggest_float(
                "pdbbind_huber_delta",
                space.pdbbind_head.huber_delta_min,
                space.pdbbind_head.huber_delta_max,
            )
        ),
    }


__all__ = [
    "DUDEzSplitConfig",
    "PDBbindSplitConfig",
    "DUDEzOptunaConfig",
    "DUDEzOptunaStage",
    "DUDEzScreeningModel",
    "FeatureExtractor",
    "OCSCORE_NON_FEATURE_COLUMNS",
    "PDBbindOptunaConfig",
    "PDBbindOptunaStage",
    "PDBbindRegressionModel",
    "ProtocolContext",
    "ProtocolStage",
    "StagedProtocol",
    "TransferFeatureExtractorStage",
    "apply_fine_tuning_mode",
    "build_dudez_model",
    "build_pdbbind_model",
    "compute_regression_reconstruction_loss",
    "derive_dudez_labels",
    "dudez_search_space_summary",
    "evaluate_regression_metrics",
    "evaluate_screening_metrics",
    "PDBBIND_SEARCH_PHASE_ENCODER_REGRESSION",
    "PDBBIND_SEARCH_PHASE_FULL",
    "pdbbind_phase1_experiment_config",
    "pdbbind_search_space_summary",
    "prepare_dudez_screening_data",
    "dudez_receptor_heldout_complete_config",
    "split_dudez_by_receptor_and_kind",
    "summarize_dudez_split_diagnostics",
    "prepare_pdbbind_regression_data",
    "resolve_dudez_primary_metric",
    "run_staged_ocscore_optuna_protocol",
    "DEFAULT_DUDEZ_SEARCH_SPACE",
    "DEFAULT_PDBBIND_SEARCH_SPACE",
    "suggest_decoder_hidden_sizes",
    "suggest_dudez_trial_params",
    "suggest_encoder_architecture",
    "suggest_pdbbind_trial_params",
]
