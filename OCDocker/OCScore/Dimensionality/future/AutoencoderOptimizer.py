#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the future Autoencoder pipeline.

It is imported as:

from OCDocker.OCScore.Dimensionality.future.AutoencoderOptimizer import AutoencoderOptimizer
'''

# Imports
###############################################################################

from __future__ import annotations

import copy
import random
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

import torch

import optuna
from optuna.samplers import TPESampler

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.OCScore.Dimensionality.future.Autoencoder import Autoencoder
from OCDocker.OCScore.Dimensionality.future.AETrainer import AETrainer

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


class AutoencoderOptimizer:
    """Future Autoencoder optimizer (denoising + multi-task).

    Parameters
    ----------
    X_train : np.ndarray | pd.DataFrame | pd.Series
        Training features.
    X_test : np.ndarray | pd.DataFrame | pd.Series
        Test features.
    X_validation : np.ndarray | pd.DataFrame | pd.Series | None, optional
        Validation features. Default None.
    encoding_dims : tuple, optional
        Min/max latent dimensions. Default (16, 256).
    storage : str, optional
        Optuna storage string. Default "sqlite:///autoencoder.db".
    models_folder : str, optional
        Folder to save models. Default "./models/Autoencoder/".
    random_seed : int, optional
        Random seed. Default 42.
    use_gpu : bool, optional
        Use GPU if available. Default True.
    verbose : bool, optional
        Verbose mode. Default False.
    y_train : np.ndarray | pd.Series | None, optional
        Energy labels for training. Default None.
    y_test : np.ndarray | pd.Series | None, optional
        Energy labels for testing. Default None.
    y_validation : np.ndarray | pd.Series | None, optional
        Energy labels for validation. Default None.
    X_unlabeled : np.ndarray | pd.DataFrame | None, optional
        Extra unlabeled data for reconstruction. Default None.
    future_config : dict | None, optional
        Configuration overrides for the future pipeline.

    Notes
    -----
    The configuration supports two training stages:
    - stage1: denoising reconstruction + optional energy supervision (default enabled).
    - stage2: optional fine-tuning stage with alternate weights/noise settings.

    Data Flow
    ---------
    - Features: X_train is used for training; X_validation (if provided) is used
      for validation, otherwise X_test is used as the evaluation split.
    - Energy labels: y_train/y_validation/y_test are optional. If labels are not
      provided, the energy head is disabled and only reconstruction is optimized.
    - Extra unlabeled data: X_unlabeled is concatenated to X_train for
      reconstruction-only learning (no energy labels are expected for it).

    Configuration
    -------------
    The future_config dict is merged into the defaults using keys below:

    model
        - encoder_hidden_sizes : list[int]
            Hidden sizes for the encoder (excluding latent).
        - latent_dim : int
            Latent embedding dimension.
        - decoder_sizes : list[int] | None
            Decoder sizes; if None, a mirrored decoder is built.
        - activation : str
            Activation for encoder/decoder hidden layers.
        - latent_activation : str
            Activation applied to latent embeddings.
        - decoder_output_activation : str
            Activation for the decoder output layer.
        - dropout, latent_dropout : float
            Dropout probabilities.
        - norm : str
            Normalization type ("batch", "layer", "none").
        - use_vae : bool
            Enable VAE reparameterization and KL term.
        - energy_head_sizes : list[int] | None
            Hidden sizes for energy head (None disables).

    stage1 / stage2
        - enabled : bool
            Whether to run the stage.
        - epochs, batch_size : int
            Training schedule and batch size.
        - lr, weight_decay : float
            Optimizer hyperparameters.
        - clip_grad : float
            Gradient clipping max-norm (0 disables).
        - recon_loss : str
            Reconstruction loss type ("mse", "rmse", "mae", "huber").
        - energy_loss : str
            Energy loss type ("mse", "rmse", "mae", "huber").
        - huber_delta : float
            Delta parameter for Huber/SmoothL1.
        - lambda_recon, lambda_energy : float
            Weights for reconstruction and energy losses.
        - lambda_l2 : float
            L2 penalty weight on latent embeddings.
        - lambda_contractive : float
            Contractive penalty weight (Jacobian norm).
        - beta_vae : float
            KL weight when use_vae is True.
        - noise_type : str
            Noise type ("mask", "gaussian", "swap", "mask+gaussian", "none").
        - mask_prob, gaussian_std, swap_prob : float
            Noise parameters.
        - ramp_epochs_energy, ramp_epochs_recon : int
            Epochs for ramping loss weights.
        - ramp_type : str
            Ramp schedule ("linear" or "sigmoid").
        - early_stopping_patience : int
            Stop after this many epochs without improvement.
        - mixed_precision : bool
            Enable AMP when running on CUDA.

    optimization
        - loss_balancing : str
            "fixed", "uncertainty", or "gradnorm".
        - gradnorm_alpha : float
            GradNorm alpha (only used when loss_balancing="gradnorm").
        - objective_metric : str
            Metric key to optimize (e.g., "val_combined_loss").
        - search_vae : bool
            If True, Optuna can toggle VAE usage and beta.

    checkpoint
        - save_best : bool
            Save best model checkpoint.
        - save_encoder : bool
            Save encoder-only checkpoint.

    data
        - use_energy_head : bool
            If False, energy head is disabled regardless of labels.

    Example
    -------
    >>> trainer = AutoencoderOptimizer(X_train, X_test, X_validation, verbose=True)
    >>> study = trainer.optimize(n_trials=10)
    """

    def __init__(
            self,
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, np.ndarray, pd.DataFrame, pd.Series] = None,
            encoding_dims: tuple = (16, 256),
            storage: str = "sqlite:///autoencoder.db",
            models_folder: str = "./models/Autoencoder/",
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False,
            y_train: Union[None, np.ndarray, pd.Series] = None,
            y_test: Union[None, np.ndarray, pd.Series] = None,
            y_validation: Union[None, np.ndarray, pd.Series] = None,
            X_unlabeled: Union[None, np.ndarray, pd.DataFrame] = None,
            future_config: Optional[dict] = None
        ) -> None:
        '''Initialize the future autoencoder optimizer.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame | pd.Series
            Training features.
        X_test : np.ndarray | pd.DataFrame | pd.Series
            Test features.
        X_validation : np.ndarray | pd.DataFrame | pd.Series | None, optional
            Validation features, by default None.
        encoding_dims : tuple, optional
            Latent dimension bounds, by default (16, 256).
        storage : str, optional
            Optuna storage string, by default "sqlite:///autoencoder.db".
        models_folder : str, optional
            Folder to save checkpoints, by default "./models/Autoencoder/".
        random_seed : int, optional
            Random seed, by default 42.
        use_gpu : bool, optional
            Use GPU if available, by default True.
        verbose : bool, optional
            Verbose mode, by default False.
        y_train : np.ndarray | pd.Series | None, optional
            Training energy labels, by default None.
        y_test : np.ndarray | pd.Series | None, optional
            Test energy labels, by default None.
        y_validation : np.ndarray | pd.Series | None, optional
            Validation energy labels, by default None.
        X_unlabeled : np.ndarray | pd.DataFrame | None, optional
            Extra unlabeled features, by default None.
        future_config : dict | None, optional
            Configuration overrides, by default None.
        '''

        self.random_seed = random_seed
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.storage = storage
        self.models_folder = models_folder

        self.set_random_seed()

        self.X_train = self._to_numpy(X_train)
        self.X_test = self._to_numpy(X_test)
        self.X_validation = self._to_numpy(X_validation) if X_validation is not None else None
        self.X_unlabeled = self._to_numpy(X_unlabeled) if X_unlabeled is not None else None

        self.y_train = self._to_numpy(y_train) if y_train is not None else None
        self.y_test = self._to_numpy(y_test) if y_test is not None else None
        self.y_validation = self._to_numpy(y_validation) if y_validation is not None else None


        self.input_size = int(self.X_train.shape[1])
        self.encoding_dims = encoding_dims

        self.config = self._merge_config(future_config)

        self.activation_functions = ["GELU", "LeakyReLU", "Mish", "ReLU", "SELU", "Identity"]


    def set_random_seed(self) -> None:
        '''Set random seeds for reproducibility.'''

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

        # Conservative latent size default based on input dimensionality.
        latent_default = min(128, max(8, self.input_size // 2))

        default_config = {
            "model": {
                "encoder_hidden_sizes": [512, 256],
                "latent_dim": latent_default,
                "decoder_sizes": None,
                "activation": "GELU",
                "latent_activation": "Identity",
                "decoder_output_activation": "Identity",
                "dropout": 0.1,
                "latent_dropout": 0.1,
                "norm": "batch",
                "use_vae": False,
                "energy_head_sizes": [128, 64]
            },
            "stage1": {
                "enabled": True,
                "epochs": 200,
                "batch_size": 256,
                "lr": 1e-3,
                "weight_decay": 1e-6,
                "clip_grad": 1.0,
                "recon_loss": "huber",
                "energy_loss": "huber",
                "huber_delta": 1.0,
                "lambda_recon": 1.0,
                "lambda_energy": 0.5,
                "lambda_l2": 0.0,
                "lambda_contractive": 0.0,
                "beta_vae": 1.0,
                "noise_type": "mask",
                "mask_prob": 0.1,
                "gaussian_std": 0.01,
                "swap_prob": 0.05,
                "ramp_epochs_energy": 20,
                "ramp_epochs_recon": 0,
                "ramp_type": "linear",
                "early_stopping_patience": 20,
                "mixed_precision": False
            },
            "stage2": {
                "enabled": False,
                "epochs": 100,
                "batch_size": 256,
                "lr": 5e-4,
                "weight_decay": 1e-6,
                "clip_grad": 1.0,
                "recon_loss": "huber",
                "energy_loss": "huber",
                "huber_delta": 1.0,
                "lambda_recon": 1.0,
                "lambda_energy": 0.2,
                "lambda_l2": 0.0,
                "lambda_contractive": 0.0,
                "beta_vae": 1.0,
                "noise_type": "none",
                "mask_prob": 0.0,
                "gaussian_std": 0.0,
                "swap_prob": 0.0,
                "ramp_epochs_energy": 0,
                "ramp_epochs_recon": 0,
                "ramp_type": "linear",
                "early_stopping_patience": 20,
                "mixed_precision": False
            },
            "optimization": {
                "loss_balancing": "fixed",
                "gradnorm_alpha": 0.5,
                "objective_metric": "val_combined_loss",
                "search_vae": False
            },
            "checkpoint": {
                "save_best": True,
                "save_encoder": True
            },
            "data": {
                "use_energy_head": True
            }
        }

        if not future_config:
            return default_config

        # One-level deep merge to keep overrides predictable.
        merged = copy.deepcopy(default_config)
        for key, sub in future_config.items():
            if isinstance(sub, dict) and key in merged:
                merged[key].update(sub)
            else:
                merged[key] = sub

        return merged


    def _to_numpy(self, data: Union[None, np.ndarray, pd.DataFrame, pd.Series]) -> Optional[np.ndarray]:
        '''Convert input data to numpy array.

        Parameters
        ----------
        data : np.ndarray | pd.DataFrame | pd.Series | None
            Input data.

        Returns
        -------
        np.ndarray | None
            Numpy array representation or None.
        '''

        if data is None:
            return None
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.values.astype(np.float32)
        return np.asarray(data, dtype=np.float32)


    def _build_model(self, model_cfg: dict) -> Autoencoder:
        '''Build Autoencoder model from configuration.

        Parameters
        ----------
        model_cfg : dict
            Model configuration.

        Returns
        -------
        Autoencoder
            Initialized autoencoder model.
        '''

        return Autoencoder(
            input_size=self.input_size,
            encoder_hidden_sizes=model_cfg.get("encoder_hidden_sizes", []),
            latent_dim=int(model_cfg.get("latent_dim", 64)),
            decoder_sizes=model_cfg.get("decoder_sizes", None),
            activation=model_cfg.get("activation", "GELU"),
            latent_activation=model_cfg.get("latent_activation", "Identity"),
            decoder_output_activation=model_cfg.get("decoder_output_activation", "Identity"),
            dropout=float(model_cfg.get("dropout", 0.0)),
            latent_dropout=float(model_cfg.get("latent_dropout", 0.0)),
            norm=model_cfg.get("norm", "batch"),
            use_vae=bool(model_cfg.get("use_vae", False)),
            energy_head_sizes=model_cfg.get("energy_head_sizes", None),
            device=self.device
        )


    def _prepare_trial_config(self, trial: optuna.Trial) -> dict:
        '''Prepare trial-specific configuration.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial instance.

        Returns
        -------
        dict
            Configuration dictionary updated with trial suggestions.
        '''
        
        cfg = copy.deepcopy(self.config)
        model_cfg = cfg.get("model", {})
        stage1_cfg = cfg.get("stage1", {})

        min_latent, max_latent = self.encoding_dims
        # Clamp latent dimension bounds to feature dimensionality.
        max_latent = min(int(max_latent), int(self.input_size))
        min_latent = min(int(min_latent), max_latent)

        latent_dim = trial.suggest_int("latent_dim", min_latent, max_latent)
        depth = trial.suggest_int("depth", 1, 4)
        width = trial.suggest_categorical("width", [128, 256, 512, 1024])

        model_cfg["latent_dim"] = latent_dim
        model_cfg["encoder_hidden_sizes"] = [width for _ in range(depth)]
        model_cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)
        model_cfg["latent_dropout"] = trial.suggest_float("latent_dropout", 0.0, 0.3)
        model_cfg["activation"] = trial.suggest_categorical("activation", self.activation_functions)
        model_cfg["norm"] = trial.suggest_categorical("norm", ["batch", "layer", "none"])

        if cfg.get("optimization", {}).get("search_vae", False):
            # Optional VAE search extends the objective to generative structure.
            model_cfg["use_vae"] = trial.suggest_categorical("use_vae", [False, True])
            stage1_cfg["beta_vae"] = trial.suggest_float("beta_vae", 0.5, 4.0)

        stage1_cfg["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        stage1_cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)

        stage1_cfg["noise_type"] = trial.suggest_categorical("noise_type", ["mask", "gaussian", "swap", "mask+gaussian", "none"])
        # Noise strengths are tuned independently for denoising robustness.
        stage1_cfg["mask_prob"] = trial.suggest_float("mask_prob", 0.0, 0.3)
        stage1_cfg["gaussian_std"] = trial.suggest_float("gaussian_std", 0.0, 0.05)
        stage1_cfg["swap_prob"] = trial.suggest_float("swap_prob", 0.0, 0.2)

        stage1_cfg["lambda_energy"] = trial.suggest_float("lambda_energy", 0.0, 1.0)

        cfg["model"] = model_cfg
        cfg["stage1"] = stage1_cfg

        return cfg


    def objective(self, trial: optuna.Trial) -> float:
        '''Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial instance.

        Returns
        -------
        float
            Objective value.
        '''

        config = self._prepare_trial_config(trial)

        if self.y_train is None or not config.get("data", {}).get("use_energy_head", True):
            # Disable energy head when labels are not provided.
            config["model"]["energy_head_sizes"] = None

        model = self._build_model(config.get("model", {}))

        run_name = f"autoencoder_future_{trial.number}"
        trainer = AETrainer(
            model=model,
            config=config,
            device=self.device,
            verbose=self.verbose,
            models_folder=self.models_folder,
            run_name=run_name
        )

        # Prefer validation if available; fallback to test split for evaluation metrics.
        eval_X = self.X_validation if self.X_validation is not None else self.X_test
        eval_y = self.y_validation if self.y_validation is not None else self.y_test

        train_metrics = trainer.fit(
            self.X_train,
            eval_X,
            self.y_train,
            eval_y,
            X_unlabeled=self.X_unlabeled
        )

        stage_name = "stage2" if config.get("stage2", {}).get("enabled", False) else "stage1"

        eval_train = trainer.evaluate(self.X_train, self.y_train, stage=stage_name)
        eval_val = trainer.evaluate(eval_X, eval_y, stage=stage_name) if eval_X is not None else {}

        trial.set_user_attr("recon_loss_train", eval_train.get("recon_loss", float("inf")))
        trial.set_user_attr("recon_loss_val", eval_val.get("recon_loss", float("inf")))
        trial.set_user_attr("energy_loss_train", eval_train.get("energy_loss", 0.0))
        trial.set_user_attr("energy_loss_val", eval_val.get("energy_loss", 0.0))

        trial.set_user_attr("recon_rmse_train", eval_train.get("recon_rmse", float("inf")))
        trial.set_user_attr("recon_rmse_val", eval_val.get("recon_rmse", float("inf")))
        trial.set_user_attr("recon_mae_train", eval_train.get("recon_mae", float("inf")))
        trial.set_user_attr("recon_mae_val", eval_val.get("recon_mae", float("inf")))
        trial.set_user_attr("recon_huber_train", eval_train.get("recon_huber", float("inf")))
        trial.set_user_attr("recon_huber_val", eval_val.get("recon_huber", float("inf")))

        trial.set_user_attr("energy_rmse_train", eval_train.get("energy_rmse", 0.0))
        trial.set_user_attr("energy_rmse_val", eval_val.get("energy_rmse", 0.0))
        trial.set_user_attr("energy_mae_train", eval_train.get("energy_mae", 0.0))
        trial.set_user_attr("energy_mae_val", eval_val.get("energy_mae", 0.0))
        trial.set_user_attr("energy_huber_train", eval_train.get("energy_huber", 0.0))
        trial.set_user_attr("energy_huber_val", eval_val.get("energy_huber", 0.0))

        trial.set_user_attr("embedding_variance", train_metrics.get("embedding_variance", []))
        trial.set_user_attr("embedding_collapse_rate", train_metrics.get("embedding_collapse_rate", 0.0))
        trial.set_user_attr("embedding_mean_norm", train_metrics.get("embedding_mean_norm", 0.0))
        trial.set_user_attr("embedding_energy_spearman", train_metrics.get("embedding_energy_spearman", 0.0))

        if self.verbose:
            ocprint.printv(f"[AutoencoderOptimizer] Trial {trial.number} completed")

        objective_metric = config.get("optimization", {}).get("objective_metric", "val_combined_loss")

        if objective_metric == "train_combined_loss":
            return float(eval_train.get("combined_loss", float("inf")))

        return float(eval_val.get("combined_loss", eval_train.get("combined_loss", float("inf"))))


    def optimize(
            self,
            direction: str = "minimize",
            n_trials: int = 10,
            study_name: str = "AE_Future_Optimization",
            load_if_exists: bool = True,
            sampler: optuna.samplers.BaseSampler = TPESampler(),
            n_jobs: int = 1
        ) -> optuna.study.Study:
        '''Optimize the autoencoder using Optuna.

        Parameters
        ----------
        direction : str, optional
            Optimization direction. Default "minimize".
        n_trials : int, optional
            Number of trials. Default 10.
        study_name : str, optional
            Study name. Default "AE_Future_Optimization".
        load_if_exists : bool, optional
            Load existing study if present. Default True.
        sampler : optuna.samplers.BaseSampler, optional
            Optuna sampler. Default TPESampler().
        n_jobs : int, optional
            Number of parallel jobs. Default 1.

        Returns
        -------
        optuna.study.Study
            Optuna study object.
        '''

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(1, n_trials // 10),
            n_warmup_steps=10
        )

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=self.storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner
        )

        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)

        if self.verbose:
            ocprint.printv("[AutoencoderOptimizer] Best trial:")
            trial = study.best_trial
            ocprint.printv(f"  Value: {trial.value}")
            ocprint.printv("  Params:")
            for key, value in trial.params.items():
                ocprint.printv(f"    {key}: {value}")

        return study
