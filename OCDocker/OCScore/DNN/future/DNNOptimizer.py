#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the future Neural Network pipeline.

It is imported as:

from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer
'''

# Imports
###############################################################################

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import GroupShuffleSplit, train_test_split

from torch.utils.data import DataLoader

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.OCScore.DNN.future.datasets import EnergyDataset, TargetRankingDataset, TargetBatchSampler
from OCDocker.OCScore.DNN.future.losses import (
    UncertaintyWeighting,
    focal_binary_loss,
    lambda_rank_ndcg_loss,
    supervised_contrastive_loss
)
from OCDocker.OCScore.DNN.future.metrics import compute_classification_metrics
from OCDocker.OCScore.DNN.future.models import MultiTaskModel, parse_encoder_params

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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


class DNNOptimizer:
    """Future DNN optimizer with multi-stage, multi-task training.

    Parameters
    ----------
    X_train : np.ndarray | pd.DataFrame
        Primary regression training features.
    y_train : np.ndarray | pd.Series
        Primary regression training targets.
    X_test : np.ndarray | pd.DataFrame
        Primary regression testing features.
    y_test : np.ndarray | pd.Series
        Primary regression testing targets.
    X_validation : np.ndarray | pd.DataFrame | None, optional
        Ranking/classification features (if provided). Default is None.
    y_validation : np.ndarray | pd.Series | None, optional
        Ranking/classification labels (if provided). Default is None.
    mask : list[int|bool] | np.ndarray, optional
        Feature mask for ablation/feature selection.
    storage : str, optional
        Optuna storage string.
    encoder_params : dict | None, optional
        Encoder params (old-style).
    output_size : int, optional
        Output size (kept for compatibility). Default is 1.
    random_seed : int, optional
        Random seed. Default is 42.
    use_gpu : bool, optional
        Use GPU if available. Default is True.
    verbose : bool, optional
        Verbose mode. Default is False.
    future_config : dict | None, optional
        Configuration overrides for future pipeline.

    Notes
    -----
    The training flow is split into two stages:
    - stage1: regression + optional reconstruction (pretraining on continuous targets).
    - stage2: ranking/classification with optional energy/reconstruction regularization.

    Data Flow
    ---------
    - Regression data: (X_train, y_train) are used for stage1 energy regression.
      (X_test, y_test) are used as the stage1 validation split.
    - Ranking data: (X_validation, y_validation) are used to build stage2 train/val
      loaders (grouped by target ids). If these are missing, stage2 cannot run.
    - Target grouping: if X_validation is a DataFrame and contains the column
      given by data.ranking_target_column (default "receptor"), that column is used
      as target ids. Otherwise, future_config["ranking_targets"] can supply them.
      If still unavailable, all samples are treated as one target.
    - Custom splits: future_config may provide pre-split dictionaries
      "ranking_train_data" and "ranking_val_data" with keys {X, y, targets}.

    Clarification
    ------------
    The reconstruction head in stage1 is only a regularizer. It does not replace
    the standalone Autoencoder pipeline and is not a dimensionality reduction
    step by itself. If you want explicit dimensionality reduction, run it
    upstream (e.g., PCA/AE) and pass the resulting embeddings as X, setting
    lambda_recon=0 to disable reconstruction.

    Configuration
    -------------
    The future_config dict is merged into the defaults using keys below:

    model
        - shared_sizes : list[int]
            Hidden sizes for the shared encoder (used when encoder_params is None).
        - shared_activation : str
            Activation for shared encoder layers.
        - decoder_sizes : list[int] | None
            Reconstruction decoder sizes; if None and recon loss enabled,
            a mirrored decoder is built automatically.
        - head_sizes : list[int]
            Hidden sizes for energy/activity heads.
        - embedding_dim : int | None
            Output size for embedding head (None disables).
        - dropout : float
            Dropout probability for encoder/heads.
        - batch_norm : bool
            Whether to use BatchNorm1d.

    stage1
        - enabled : bool
            Whether to run stage1 training.
        - epochs : int
            Number of training epochs.
        - batch_size : int
            Batch size for regression data.
        - lr, weight_decay : float
            Optimizer hyperparameters.
        - lambda_recon, lambda_energy : float
            Weights for reconstruction and energy losses.
        - energy_loss : str
            Energy loss type ("mse" or "huber").
        - noise_type : str
            Input noise type ("mask", "gaussian", or "none").
        - mask_prob, gaussian_std : float
            Noise parameters.
        - clip_grad : float
            Gradient clipping max-norm (0 disables).
        - early_stopping_patience : int
            Stop after this many epochs without improvement.

    stage2
        - enabled : bool
            Whether to run stage2 ranking/classification.
        - epochs : int
            Number of training epochs.
        - batch_size_per_target : int | None
            Optional per-target batch size for ranking.
        - split_target_batches : bool
            Whether to split large targets into multiple batches.
        - lr, weight_decay : float
            Optimizer hyperparameters.
        - lambda_rank, lambda_cls, lambda_con : float
            Weights for ranking, classification, and contrastive losses.
        - lambda_energy, lambda_recon : float
            Optional regularizers from regression data.
        - rank_k_fractions : tuple[float, float]
            Top-k fractions for LambdaRank weighting.
        - rank_weights : tuple[float, float]
            Weights per k fraction.
        - temperature : float
            Temperature for contrastive loss.
        - clip_grad : float
            Gradient clipping max-norm (0 disables).
        - use_focal : bool
            Use focal loss instead of BCE for classification.
        - focal_alpha, focal_gamma : float
            Focal loss parameters.
        - bce_pos_weight : float | None
            Optional positive class weight for BCE.
        - early_stopping_patience : int
            Stop after this many epochs without improvement.
        - energy_batch_ratio : float
            Ratio of regression batches used in stage2 regularization.

    optimization
        - loss_balancing : str
            "fixed" or "uncertainty" (learns task weights).
        - metric_for_best : str
            Metric key used to track best validation model.
        - multi_objective : bool
            Whether to return a multi-objective tuple to Optuna.
        - objective_metric : str
            Metric key to optimize when not multi-objective.

    data
        - ranking_validation_fraction : float
            Fraction of ranking data held out for validation.
        - ranking_split_by_target : bool
            If True, split by target ids (GroupShuffleSplit).
        - ranking_target_column : str
            Column name in X_validation (DataFrame) containing target ids.

    Example
    -------
    >>> trainer = DNNOptimizer(X_train, y_train, X_test, y_test, X_validation, y_validation)
    >>> trainer.optimize(n_trials=5)
    >>> # AE -> DNN pipeline with precomputed embeddings
    >>> import torch
    >>> from OCDocker.OCScore.Dimensionality.future.Autoencoder import Autoencoder
    >>> ae = Autoencoder(input_size=20, encoder_hidden_sizes=[32, 16], latent_dim=8, energy_head_sizes=None)
    >>> with torch.no_grad():
    ...     Z_train = ae.encode(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()
    ...     Z_test = ae.encode(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
    >>> dnn = DNNOptimizer.from_embeddings(
    ...     Z_train, y_train, Z_test, y_test,
    ...     future_config={"stage1": {"lambda_recon": 0.0, "noise_type": "none"}, "stage2": {"enabled": False}}
    ... )
    >>> dnn.optimize(n_trials=1)
    """

    def __init__(
            self,
            X_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame],
            y_test: Union[np.ndarray, pd.Series],
            X_validation: Union[np.ndarray, pd.DataFrame, None] = None,
            y_validation: Union[np.ndarray, pd.Series, None] = None,
            mask: Union[list[Union[int, bool]], np.ndarray] = [],
            storage: str = "sqlite:///NNoptimization.db",
            encoder_params: Union[None, dict] = None,
            output_size: int = 1,
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False,
            future_config: Optional[dict] = None
        ) -> None:
        '''Initialize the future DNN optimizer.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame
            Primary regression training features.
        y_train : np.ndarray | pd.Series
            Primary regression training targets.
        X_test : np.ndarray | pd.DataFrame
            Primary regression testing features.
        y_test : np.ndarray | pd.Series
            Primary regression testing targets.
        X_validation : np.ndarray | pd.DataFrame | None, optional
            Ranking/classification features, by default None.
        y_validation : np.ndarray | pd.Series | None, optional
            Ranking/classification labels, by default None.
        mask : list[int | bool] | np.ndarray, optional
            Feature mask, by default [].
        storage : str, optional
            Optuna storage string, by default "sqlite:///NNoptimization.db".
        encoder_params : dict | None, optional
            Legacy encoder parameters, by default None.
        output_size : int, optional
            Output size (compat), by default 1.
        random_seed : int, optional
            Random seed, by default 42.
        use_gpu : bool, optional
            Use GPU if available, by default True.
        verbose : bool, optional
            Verbose mode, by default False.
        future_config : dict | None, optional
            Configuration overrides, by default None.
        '''

        self.random_seed = random_seed
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.storage = storage
        self.encoder_params = encoder_params
        self.output_size = output_size

        self.set_random_seed()

        # Prepare mask
        if isinstance(mask, list):
            mask = np.asarray(mask)
        if mask is not None and len(mask) == 0:
            mask = None
        self.mask = torch.tensor(mask, dtype=torch.float32).to(self.device) if mask is not None else None

        # Merge configuration
        self.config = self._merge_config(future_config)

        # Prepare primary regression data
        self.X_reg_train = self._to_numpy(X_train)
        self.y_reg_train = np.asarray(y_train).reshape(-1, 1)
        self.X_reg_test = self._to_numpy(X_test)
        self.y_reg_test = np.asarray(y_test).reshape(-1, 1)

        # Determine input size
        self.input_size = self._infer_input_size(self.X_reg_train)

        # Prepare ranking data (train/val)
        self.rank_train = None
        self.rank_val = None

        self._prepare_ranking_data(X_validation, y_validation, future_config)


    @classmethod
    def from_embeddings(
            cls,
            X_embeddings_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            X_embeddings_test: Union[np.ndarray, pd.DataFrame],
            y_test: Union[np.ndarray, pd.Series],
            X_embeddings_validation: Union[np.ndarray, pd.DataFrame, None] = None,
            y_validation: Union[np.ndarray, pd.Series, None] = None,
            **kwargs: Any
        ) -> "DNNOptimizer":
        '''Construct a DNNOptimizer from precomputed embeddings.

        Parameters
        ----------
        X_embeddings_train : np.ndarray | pd.DataFrame
            Training embeddings (output of a dimensionality reducer).
        y_train : np.ndarray | pd.Series
            Training regression targets.
        X_embeddings_test : np.ndarray | pd.DataFrame
            Test embeddings.
        y_test : np.ndarray | pd.Series
            Test regression targets.
        X_embeddings_validation : np.ndarray | pd.DataFrame | None, optional
            Optional ranking/classification embeddings, by default None.
        y_validation : np.ndarray | pd.Series | None, optional
            Optional ranking/classification labels, by default None.
        **kwargs : Any
            Additional keyword arguments forwarded to DNNOptimizer.

        Returns
        -------
        DNNOptimizer
            Configured optimizer instance.

        Notes
        -----
        When using embeddings, set stage1.lambda_recon=0 and noise_type="none"
        to avoid reconstructing already-reduced features.
        '''

        return cls(
            X_train=X_embeddings_train,
            y_train=y_train,
            X_test=X_embeddings_test,
            y_test=y_test,
            X_validation=X_embeddings_validation,
            y_validation=y_validation,
            **kwargs
        )


    def set_random_seed(self) -> None:
        '''Set the random seed for reproducibility.'''

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.random_seed)
        else:
            self.device = torch.device('cpu')

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    def _merge_config(self, future_config: Optional[dict]) -> dict:
        '''Merge default configuration with overrides.

        Parameters
        ----------
        future_config : dict | None
            User-provided configuration overrides.

        Returns
        -------
        dict
            Merged configuration dictionary.
        '''

        default_config = {
            "model": {
                "shared_sizes": [512, 256, 128],
                "shared_activation": "GELU",
                "decoder_sizes": None,
                "head_sizes": [128, 64],
                "embedding_dim": 64,
                "dropout": 0.1,
                "batch_norm": True
            },
            "stage1": {
                "enabled": True,
                "epochs": 200,
                "batch_size": 256,
                "lr": 1e-3,
                "weight_decay": 1e-6,
                "lambda_recon": 1.0,
                "lambda_energy": 1.0,
                "energy_loss": "huber",
                "noise_type": "mask",
                "mask_prob": 0.1,
                "gaussian_std": 0.01,
                "clip_grad": 1.0,
                "early_stopping_patience": 20
            },
            "stage2": {
                "enabled": True,
                "epochs": 200,
                "batch_size_per_target": None,
                "split_target_batches": False,
                "lr": 1e-4,
                "weight_decay": 1e-6,
                "lambda_rank": 1.0,
                "lambda_cls": 1.0,
                "lambda_con": 0.2,
                "lambda_energy": 0.1,
                "lambda_recon": 0.0,
                "rank_k_fractions": (0.01, 0.05),
                "rank_weights": (0.5, 0.5),
                "temperature": 0.1,
                "clip_grad": 1.0,
                "use_focal": False,
                "focal_alpha": 0.25,
                "focal_gamma": 2.0,
                "bce_pos_weight": None,
                "early_stopping_patience": 30,
                "energy_batch_ratio": 1.0
            },
            "optimization": {
                "loss_balancing": "fixed",  # fixed | uncertainty
                "metric_for_best": "AUC",
                "multi_objective": False,
                "objective_metric": "AUC"
            },
            "data": {
                "ranking_validation_fraction": 0.2,
                "ranking_split_by_target": True,
                "ranking_target_column": "receptor"
            }
        }

        if not future_config:
            return default_config

        # Deep merge with overrides
        # Only one level deep to keep config predictable and avoid accidental deletions.
        merged = copy.deepcopy(default_config)
        for key, sub in future_config.items():
            if isinstance(sub, dict) and key in merged:
                merged[key].update(sub)
            else:
                merged[key] = sub

        return merged


    def _to_numpy(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        '''Convert input data to numpy array.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Numpy array representation.
        '''

        if isinstance(data, pd.DataFrame):
            return data.values
        return np.asarray(data)


    def _infer_input_size(self, data: np.ndarray) -> int:
        '''Infer input feature dimension from data.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix.

        Returns
        -------
        int
            Input feature dimension.
        '''

        return int(data.shape[1])


    def _prepare_ranking_data(
            self,
            X_validation: Union[np.ndarray, pd.DataFrame, None],
            y_validation: Union[np.ndarray, pd.Series, None],
            future_config: Optional[dict]
        ) -> None:
        '''Prepare ranking/classification data splits for stage2.

        Parameters
        ----------
        X_validation : np.ndarray | pd.DataFrame | None
            Feature matrix for ranking/classification dataset.
        y_validation : np.ndarray | pd.Series | None
            Labels for ranking/classification dataset.
        future_config : dict | None
            Optional overrides containing pre-split data or target ids.

        '''

        if future_config and "ranking_train_data" in future_config:
            rank_train = future_config.get("ranking_train_data")
            rank_val = future_config.get("ranking_val_data")

            self.rank_train = rank_train
            self.rank_val = rank_val
            return

        if X_validation is None or y_validation is None:
            return

        # Convert ranking data
        X_rank = self._to_numpy(X_validation)
        y_rank = np.asarray(y_validation).astype(int)

        # Extract target IDs if available
        target_ids = None
        if isinstance(X_validation, pd.DataFrame):
            target_col = self.config["data"].get("ranking_target_column", "receptor")
            if target_col in X_validation.columns:
                target_ids = X_validation[target_col].values

        if future_config and "ranking_targets" in future_config:
            target_ids = future_config["ranking_targets"]

        if target_ids is None:
            # Fallback: use a single target for all samples
            target_ids = np.array(["TARGET_0"] * len(y_rank))

        target_ids = np.asarray(target_ids)

        # Split ranking data into train/val
        val_fraction = float(self.config["data"].get("ranking_validation_fraction", 0.2))
        if val_fraction <= 0.0 or val_fraction >= 1.0:
            self.rank_train = {"X": X_rank, "y": y_rank, "targets": target_ids}
            self.rank_val = None
            return

        if self.config["data"].get("ranking_split_by_target", True):
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=self.random_seed)
            idx_train, idx_val = next(splitter.split(X_rank, y_rank, groups=target_ids))
        else:
            idx_train, idx_val = train_test_split(
                np.arange(len(y_rank)),
                test_size=val_fraction,
                random_state=self.random_seed,
                stratify=y_rank if len(np.unique(y_rank)) > 1 else None
            )

        self.rank_train = {
            "X": self._select_rows(X_rank, idx_train),
            "y": y_rank[idx_train],
            "targets": target_ids[idx_train]
        }
        self.rank_val = {
            "X": self._select_rows(X_rank, idx_val),
            "y": y_rank[idx_val],
            "targets": target_ids[idx_val]
        }


    def _select_rows(self, data: np.ndarray, idx: np.ndarray) -> np.ndarray:
        '''Select rows from an array by indices.

        Parameters
        ----------
        data : np.ndarray
            Input array.
        idx : np.ndarray
            Indices to select.

        Returns
        -------
        np.ndarray
            Selected rows.
        '''

        return data[idx]


    def _build_model(self, model_config: dict) -> nn.Module:
        '''Build the multi-task model for the future pipeline.

        Parameters
        ----------
        model_config : dict
            Model configuration dictionary.

        Returns
        -------
        nn.Module
            Initialized model.
        '''

        mask = self.mask
        model_cfg = copy.deepcopy(model_config)

        recon_enabled = self.config["stage1"].get("lambda_recon", 0.0) > 0 or self.config["stage2"].get("lambda_recon", 0.0) > 0
        if recon_enabled and model_cfg.get("decoder_sizes") is None:
            # Mirror encoder sizes to build a light decoder when reconstruction is enabled.
            if isinstance(self.encoder_params, dict):
                layer_sizes, _ = parse_encoder_params(self.encoder_params)
                decoder_sizes = list(reversed(layer_sizes[:-1])) + [int(self.input_size)]
            else:
                decoder_sizes = list(reversed(model_cfg["shared_sizes"][:-1])) + [int(self.input_size)]
            model_cfg["decoder_sizes"] = decoder_sizes

        model = MultiTaskModel(
            input_size=self.input_size,
            encoder_params=self.encoder_params,
            shared_sizes=model_cfg["shared_sizes"],
            shared_activation=model_cfg["shared_activation"],
            decoder_sizes=model_cfg.get("decoder_sizes"),
            head_sizes=model_cfg["head_sizes"],
            embedding_dim=model_cfg.get("embedding_dim"),
            dropout=model_cfg.get("dropout", 0.0),
            batch_norm=model_cfg.get("batch_norm", True),
            mask=mask
        )

        return model.to(self.device)


    def _apply_noise(self, x: torch.Tensor, stage_cfg: dict) -> torch.Tensor:
        '''Apply input noise during stage1 training.

        Parameters
        ----------
        x : torch.Tensor
            Input batch tensor.
        stage_cfg : dict
            Stage configuration.

        Returns
        -------
        torch.Tensor
            Noised tensor.
        '''

        noise_type = stage_cfg.get("noise_type", "mask")
        if noise_type == "none":
            return x

        if noise_type == "mask":
            mask_prob = float(stage_cfg.get("mask_prob", 0.1))
            if mask_prob <= 0.0:
                return x
            # Bernoulli mask zeros out a fraction of features.
            mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_prob))
            return x * mask

        if noise_type == "gaussian":
            std = float(stage_cfg.get("gaussian_std", 0.01))
            if std <= 0.0:
                return x
            # Additive Gaussian noise for denoising pretraining.
            noise = torch.randn_like(x) * std
            return x + noise

        return x


    def _make_reg_loader(self, split: str = "train") -> DataLoader:
        '''Create DataLoader for primary regression dataset.

        Parameters
        ----------
        split : str, optional
            Split name ("train" or "val"), by default "train".

        Returns
        -------
        DataLoader
            DataLoader for regression data.
        '''

        if split == "train":
            dataset = EnergyDataset(self.X_reg_train, self.y_reg_train, mask=self.mask.cpu().numpy() if self.mask is not None else None)
            batch_size = self.config["stage1"]["batch_size"]
            shuffle = True
        else:
            dataset = EnergyDataset(self.X_reg_test, self.y_reg_test, mask=self.mask.cpu().numpy() if self.mask is not None else None)
            batch_size = self.config["stage1"]["batch_size"]
            shuffle = False

        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


    def _make_ranking_loaders(self) -> tuple[DataLoader, Optional[DataLoader]]:
        '''Create DataLoaders for ranking/classification dataset.

        Returns
        -------
        tuple[DataLoader, Optional[DataLoader]]
            Training and validation loaders.
        '''

        if self.rank_train is None:
            raise ValueError("Ranking training data not provided")

        train_dataset = TargetRankingDataset(
            self.rank_train["X"],
            self.rank_train["y"],
            self.rank_train["targets"],
            mask=self.mask.cpu().numpy() if self.mask is not None else None
        )

        sampler = TargetBatchSampler(
            train_dataset.target_to_indices,
            batch_size=self.config["stage2"].get("batch_size_per_target"),
            shuffle=True,
            split_target_batches=self.config["stage2"].get("split_target_batches", False)
        )

        train_loader = DataLoader(dataset=train_dataset, batch_sampler=sampler)

        val_loader = None
        if self.rank_val is not None:
            val_dataset = TargetRankingDataset(
                self.rank_val["X"],
                self.rank_val["y"],
                self.rank_val["targets"],
                mask=self.mask.cpu().numpy() if self.mask is not None else None
            )
            val_sampler = TargetBatchSampler(val_dataset.target_to_indices, batch_size=None, shuffle=False)
            val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler)

        return train_loader, val_loader


    def _compute_energy_loss(self, preds: torch.Tensor, targets: torch.Tensor, stage_cfg: dict) -> torch.Tensor:
        '''Compute energy regression loss.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted energies.
        targets : torch.Tensor
            Target energies.
        stage_cfg : dict
            Stage configuration containing loss type.

        Returns
        -------
        torch.Tensor
            Loss value.
        '''

        if stage_cfg.get("energy_loss", "huber") == "mse":
            return nn.MSELoss()(preds, targets)
        return nn.HuberLoss()(preds, targets)


    def _train_stage1(self, model: nn.Module) -> None:
        '''Train stage1 on regression + reconstruction objectives.

        Parameters
        ----------
        model : nn.Module
            Model to train.
        '''

        stage_cfg = self.config["stage1"]
        if not stage_cfg.get("enabled", True):
            return

        model.train()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=stage_cfg["lr"],
            weight_decay=stage_cfg["weight_decay"]
        )

        loss_balancing = self.config["optimization"].get("loss_balancing", "fixed")
        balancer = None
        if loss_balancing == "uncertainty":
            # Learn task weights from data to avoid manual tuning.
            balancer = UncertaintyWeighting(["recon", "energy"]).to(self.device)
            optimizer.add_param_group({"params": balancer.parameters()})

        train_loader = self._make_reg_loader("train")
        val_loader = self._make_reg_loader("val")

        best_metric = float("inf")
        patience = int(stage_cfg.get("early_stopping_patience", 20))
        patience_counter = 0
        best_state = None

        for epoch in range(stage_cfg["epochs"]):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                features, energies = batch
                features = features.to(self.device)
                energies = energies.to(self.device)

                # Denoising pretraining: reconstruct clean features from noisy inputs.
                noisy = self._apply_noise(features, stage_cfg)
                outputs = model(noisy, return_reconstruction=stage_cfg["lambda_recon"] > 0)

                recon_loss = torch.tensor(0.0, device=self.device)
                if stage_cfg["lambda_recon"] > 0 and outputs["reconstruction"] is not None:
                    recon_loss = nn.MSELoss()(outputs["reconstruction"], features)
                energy_loss = self._compute_energy_loss(outputs["energy"], energies, stage_cfg)

                losses = {"recon": recon_loss, "energy": energy_loss}

                if balancer is not None:
                    loss, _ = balancer(losses)
                else:
                    loss = stage_cfg["lambda_recon"] * recon_loss + stage_cfg["lambda_energy"] * energy_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), stage_cfg["clip_grad"])
                optimizer.step()

                running_loss += loss.item()

            # Validation energy loss
            model.eval()
            val_energy = []
            with torch.no_grad():
                for batch in val_loader:
                    features, energies = batch
                    features = features.to(self.device)
                    energies = energies.to(self.device)
                    outputs = model(features)
                    loss_val = self._compute_energy_loss(outputs["energy"], energies, stage_cfg)
                    val_energy.append(loss_val.item())

            mean_val = float(np.mean(val_energy)) if val_energy else float("inf")

            if mean_val < best_metric:
                # Track the best state based on validation energy loss.
                best_metric = mean_val
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose:
                ocprint.printv(f"[Stage1] Epoch {epoch+1}/{stage_cfg['epochs']} loss={running_loss:.4f} val_energy={mean_val:.4f}")

            if patience_counter >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)


    def _train_stage2(self, model: nn.Module) -> Dict[str, float]:
        '''Train stage2 on ranking/classification objectives.

        Parameters
        ----------
        model : nn.Module
            Model to train.

        Returns
        -------
        Dict[str, float]
            Validation metrics if available.
        '''

        stage_cfg = self.config["stage2"]
        if not stage_cfg.get("enabled", True):
            return {}

        train_loader, val_loader = self._make_ranking_loaders()
        reg_loader = self._make_reg_loader("train") if stage_cfg.get("lambda_energy", 0.0) > 0 or stage_cfg.get("lambda_recon", 0.0) > 0 else None

        optimizer = optim.AdamW(
            model.parameters(),
            lr=stage_cfg["lr"],
            weight_decay=stage_cfg["weight_decay"]
        )

        loss_balancing = self.config["optimization"].get("loss_balancing", "fixed")
        balancer = None
        if loss_balancing == "uncertainty":
            # Balancer adapts weights across ranking/classification/contrastive terms.
            balancer = UncertaintyWeighting(["rank", "cls", "con", "energy", "recon"]).to(self.device)
            optimizer.add_param_group({"params": balancer.parameters()})

        # Compute pos_weight for BCE if needed (handles strong class imbalance).
        pos_weight = None
        if stage_cfg.get("bce_pos_weight") is None and self.rank_train is not None:
            labels = np.asarray(self.rank_train["y"]).astype(int)
            pos = max(1, int(labels.sum()))
            neg = max(1, int(len(labels) - pos))
            # Balance BCE for strong class imbalance.
            pos_weight = torch.tensor([neg / pos], device=self.device, dtype=torch.float32)
        elif stage_cfg.get("bce_pos_weight") is not None:
            pos_weight = torch.tensor([float(stage_cfg["bce_pos_weight"])], device=self.device, dtype=torch.float32)

        best_metric = -float("inf")
        patience = int(stage_cfg.get("early_stopping_patience", 30))
        patience_counter = 0
        best_state = None

        for epoch in range(stage_cfg["epochs"]):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()
                features, labels, _targets = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = model(features)
                scores = outputs["activity"].view(-1)

                # Ranking loss focuses on early recognition.
                rank_loss = lambda_rank_ndcg_loss(scores, labels, stage_cfg["rank_k_fractions"], stage_cfg["rank_weights"])

                if stage_cfg.get("use_focal", False):
                    cls_loss = focal_binary_loss(scores, labels, alpha=stage_cfg["focal_alpha"], gamma=stage_cfg["focal_gamma"])
                else:
                    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(scores, labels.float())

                con_loss = supervised_contrastive_loss(outputs["embedding"], labels, temperature=stage_cfg["temperature"])

                losses = {
                    "rank": rank_loss,
                    "cls": cls_loss,
                    "con": con_loss
                }

                if balancer is not None:
                    loss, _ = balancer(losses)
                else:
                    loss = (
                        stage_cfg["lambda_rank"] * rank_loss +
                        stage_cfg["lambda_cls"] * cls_loss +
                        stage_cfg["lambda_con"] * con_loss
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), stage_cfg["clip_grad"])
                optimizer.step()

                running_loss += loss.item()

            # Optional energy regularization using regression data
            if reg_loader is not None and (stage_cfg["lambda_energy"] > 0 or stage_cfg["lambda_recon"] > 0):
                # Energy/reconstruction regularize stage2 when requested.
                model.train()
                for batch in reg_loader:
                    optimizer.zero_grad()
                    features, energies = batch
                    features = features.to(self.device)
                    energies = energies.to(self.device)

                    outputs = model(features, return_reconstruction=stage_cfg["lambda_recon"] > 0)

                    energy_loss = self._compute_energy_loss(outputs["energy"], energies, stage_cfg)
                    recon_loss = torch.tensor(0.0, device=self.device)
                    if stage_cfg["lambda_recon"] > 0 and outputs["reconstruction"] is not None:
                        recon_loss = nn.MSELoss()(outputs["reconstruction"], features)

                    losses = {"energy": energy_loss, "recon": recon_loss}

                    if balancer is not None:
                        energy_total, _ = balancer(losses)
                    else:
                        energy_total = stage_cfg["lambda_energy"] * energy_loss + stage_cfg["lambda_recon"] * recon_loss

                    energy_total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), stage_cfg["clip_grad"])
                    optimizer.step()

            # Validation
            metrics = {}
            if val_loader is not None:
                metrics = self._evaluate_ranking(model, val_loader)
                metric_key = self.config["optimization"].get("metric_for_best", "AUC")
                metric_value = metrics.get(metric_key, 0.0)

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose:
                    ocprint.printv(f"[Stage2] Epoch {epoch+1}/{stage_cfg['epochs']} loss={running_loss:.4f} {metric_key}={metric_value:.4f}")

                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Return final validation metrics if available
        if val_loader is not None:
            return self._evaluate_ranking(model, val_loader)
        return {}


    def _evaluate_ranking(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        '''Evaluate ranking/classification metrics.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.
        loader : DataLoader
            DataLoader for ranking/classification data.

        Returns
        -------
        Dict[str, float]
            Metrics dictionary.
        '''

        model.eval()

        all_scores = []
        all_labels = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                features, labels, targets = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = model(features)
                scores = outputs["activity"].view(-1)

                all_scores.append(scores.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                all_targets.append(np.asarray(targets))

        y_score = np.concatenate(all_scores) if all_scores else np.array([])
        y_true = np.concatenate(all_labels) if all_labels else np.array([])
        target_ids = np.concatenate(all_targets) if all_targets else np.array([])

        return compute_classification_metrics(y_true, y_score, target_ids, self.config["stage2"]["rank_k_fractions"])


    def _evaluate_energy(self, model: nn.Module) -> Dict[str, float]:
        '''Evaluate energy regression performance on validation split.

        Parameters
        ----------
        model : nn.Module
            Model to evaluate.

        Returns
        -------
        Dict[str, float]
            Energy regression metrics.
        '''

        model.eval()

        loader = self._make_reg_loader("val")
        preds = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                features, energies = batch
                features = features.to(self.device)
                outputs = model(features)
                preds.append(outputs["energy"].detach().cpu().numpy())
                targets.append(energies.detach().cpu().numpy())

        if not preds:
            return {"energy_rmse": float("inf"), "energy_mae": float("inf")}

        y_pred = np.concatenate(preds).reshape(-1)
        y_true = np.concatenate(targets).reshape(-1)

        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))

        return {"energy_rmse": rmse, "energy_mae": mae}


    def objective(self, trial: optuna.Trial) -> float | tuple:
        '''Objective function for Optuna.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.

        Returns
        -------
        float | tuple
            Optimization objective value(s).
        '''

        # Suggest hyperparameters
        shared_width = trial.suggest_categorical("shared_width", [128, 256, 512])
        shared_depth = trial.suggest_int("shared_depth", 2, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        lr_stage1 = trial.suggest_float("lr_stage1", 1e-4, 5e-3, log=True)
        lr_stage2 = trial.suggest_float("lr_stage2", 1e-5, 5e-4, log=True)

        model_config = copy.deepcopy(self.config["model"])
        model_config["shared_sizes"] = [shared_width for _ in range(shared_depth)]
        model_config["dropout"] = dropout

        # Update training config with trial suggestions
        self.config["stage1"]["lr"] = lr_stage1
        self.config["stage2"]["lr"] = lr_stage2

        # Build model
        model = self._build_model(model_config)

        # Stage 1 pretraining
        self._train_stage1(model)

        # Stage 2 fine-tuning
        metrics = self._train_stage2(model)

        # Compute overall metrics for Optuna
        auc_value = metrics.get("AUC", 0.0)
        ndcg_value = metrics.get("NDCG@1%", 0.0)
        ef_value = metrics.get("EF@1%", 0.0)

        energy_metrics = self._evaluate_energy(model)

        trial.set_user_attr("AUC", auc_value)
        trial.set_user_attr("PR_AUC", metrics.get("PR_AUC", 0.0))
        trial.set_user_attr("log_loss", metrics.get("log_loss", float("inf")))
        trial.set_user_attr("EF@1%", ef_value)
        trial.set_user_attr("EF@5%", metrics.get("EF@5%", 0.0))
        trial.set_user_attr("NDCG@1%", ndcg_value)
        trial.set_user_attr("NDCG@5%", metrics.get("NDCG@5%", 0.0))
        trial.set_user_attr("pAUC@1%", metrics.get("pAUC@1%", 0.0))
        trial.set_user_attr("pAUC@5%", metrics.get("pAUC@5%", 0.0))
        trial.set_user_attr("energy_rmse", energy_metrics.get("energy_rmse", float("inf")))
        trial.set_user_attr("energy_mae", energy_metrics.get("energy_mae", float("inf")))

        if self.config["optimization"].get("multi_objective", False):
            # Minimize energy RMSE and maximize AUC and NDCG@1%
            return (
                energy_metrics.get("energy_rmse", float("inf")),
                auc_value,
                ndcg_value
            )

        objective_metric = self.config["optimization"].get("objective_metric", "AUC")
        return float(metrics.get(objective_metric, auc_value))


    def optimize(
            self,
            direction: str = "maximize",
            n_trials: int = 10,
            study_name: str = "NN_Future_Optimization",
            load_if_exists: bool = True,
            sampler: optuna.samplers.BaseSampler = TPESampler(),
            n_jobs: int = 1
        ) -> optuna.study.Study:
        '''Optimize the future pipeline using Optuna.

        Parameters
        ----------
        direction : str, optional
            Direction of optimization (ignored if multi-objective). Default is "maximize".
        n_trials : int, optional
            Number of trials. Default is 10.
        study_name : str, optional
            Study name. Default is "NN_Future_Optimization".
        load_if_exists : bool, optional
            Load existing study. Default True.
        sampler : optuna.samplers.BaseSampler, optional
            Optuna sampler. Default TPESampler().
        n_jobs : int, optional
            Number of parallel jobs. Default 1.

        Returns
        -------
        optuna.study.Study
            Optuna study object.
        '''

        if self.config["optimization"].get("multi_objective", False):
            study = optuna.create_study(
                directions=["minimize", "maximize", "maximize"],
                study_name=study_name,
                storage=self.storage,
                load_if_exists=load_if_exists,
                sampler=sampler
            )
        else:
            study = optuna.create_study(
                direction=direction,
                study_name=study_name,
                storage=self.storage,
                load_if_exists=load_if_exists,
                sampler=sampler
            )

        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

        if self.verbose:
            ocprint.printv(f"Best trial: {study.best_trial}")

        return study


# Methods
###############################################################################
