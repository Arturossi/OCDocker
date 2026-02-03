#!/usr/bin/env python3

# Description
###############################################################################
'''Trainer for the future Autoencoder pipeline (denoising + multi-task).'''

# Imports
###############################################################################
from __future__ import annotations

import os

import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, List, Optional

import OCDocker.Toolbox.Printing as ocprint

from OCDocker.OCScore.Dimensionality.future.Autoencoder import Autoencoder
from OCDocker.OCScore.Dimensionality.future.datasets import AutoencoderDataset
from OCDocker.OCScore.Dimensionality.future.losses import contractive_penalty
from OCDocker.OCScore.Dimensionality.future.losses import energy_loss
from OCDocker.OCScore.Dimensionality.future.losses import kl_divergence
from OCDocker.OCScore.Dimensionality.future.losses import reconstruction_loss
from OCDocker.OCScore.Dimensionality.future.utils import apply_noise
from OCDocker.OCScore.Dimensionality.future.utils import embedding_stats
from OCDocker.OCScore.Dimensionality.future.utils import ramp_weight
from OCDocker.OCScore.Dimensionality.future.utils import spearman_corr
from OCDocker.OCScore.DNN.future.losses import UncertaintyWeighting

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


class EarlyStopping:
    """Simple early stopping helper.

    Parameters
    ----------
    patience : int, optional
        Number of epochs without improvement to wait, by default 20.
    min_delta : float, optional
        Minimum improvement to reset patience, by default 0.0.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0) -> None:
        '''Initialize early stopping state.

        Parameters
        ----------
        patience : int, optional
            Number of epochs without improvement to wait, by default 20.
        min_delta : float, optional
            Minimum improvement to reset patience, by default 0.0.
        '''

        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        '''Update early stopping state.

        Parameters
        ----------
        value : float
            Current monitored value.

        Returns
        -------
        bool
            True if training should stop.
        '''

        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class AETrainer:
    """Trainer for the future Autoencoder.

    Notes
    -----
    This trainer supports two stages with separate configs:
    - stage1: denoising reconstruction + optional energy supervision (default enabled).
    - stage2: optional fine-tuning stage with different weights/noise settings.

    Data Flow
    ---------
    - Each batch provides (features, energies, energy_mask).
    - energy_mask marks which samples have valid energy labels; energy loss is
      only computed on those samples.
    - Unlabeled samples can be added via X_unlabeled and are used only for
      reconstruction (energy_mask False).

    Example
    -------
    >>> trainer = AETrainer(model, config, device)
    >>> metrics = trainer.fit(X_train, X_val, y_train, y_val)
    """

    def __init__(
            self,
            model: Autoencoder,
            config: dict,
            device: torch.device,
            verbose: bool = False,
            models_folder: Optional[str] = None,
            run_name: str = "autoencoder_future"
        ) -> None:
        '''Initialize the trainer.

        Parameters
        ----------
        model : Autoencoder
            Autoencoder model.
        config : dict
            Training configuration.
        device : torch.device
            Execution device.
        verbose : bool, optional
            Verbose mode, by default False.
        models_folder : str | None, optional
            Folder to save checkpoints, by default None.
        run_name : str, optional
            Checkpoint base name, by default "autoencoder_future".
        '''

        self.model = model.to(device)
        self.config = config
        self.device = device
        self.verbose = verbose
        self.models_folder = models_folder
        self.run_name = run_name

        self.loss_balancing = self.config.get("optimization", {}).get("loss_balancing", "fixed")

        self.uncertainty = None
        if self.loss_balancing == "uncertainty":
            self.uncertainty = UncertaintyWeighting(["recon", "energy"]).to(self.device)

        self.gradnorm_alpha = float(self.config.get("optimization", {}).get("gradnorm_alpha", 0.5))
        self.gradnorm_weights = None
        self.gradnorm_initial = None

    def _build_dataset(
            self,
            X: Optional[np.ndarray],
            y: Optional[np.ndarray],
            feature_mask: Optional[np.ndarray]
        ) -> AutoencoderDataset:
        '''Build AutoencoderDataset from arrays.

        Parameters
        ----------
        X : np.ndarray | None
            Feature matrix.
        y : np.ndarray | None
            Energy targets.
        feature_mask : np.ndarray | None
            Feature mask.

        Returns
        -------
        AutoencoderDataset
            Prepared dataset instance.
        '''

        if X is None:
            raise ValueError("X cannot be None when building dataset")
        return AutoencoderDataset(X, energies=y, feature_mask=feature_mask)

    def _build_optimizer(self, stage_cfg: dict) -> optim.Optimizer:
        '''Build optimizer for a training stage.

        Parameters
        ----------
        stage_cfg : dict
            Stage configuration.

        Returns
        -------
        optim.Optimizer
            Configured optimizer.
        '''

        lr = float(stage_cfg.get("lr", 1e-3))
        weight_decay = float(stage_cfg.get("weight_decay", 1e-6))

        params = list(self.model.parameters())
        if self.uncertainty is not None:
            # Include uncertainty weights as learnable parameters.
            params += list(self.uncertainty.parameters())

        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def _combine_losses(
            self,
            rec_loss: torch.Tensor,
            energy_loss_val: Optional[torch.Tensor],
            lambda_rec: float,
            lambda_energy: float,
            kld: torch.Tensor,
            beta_vae: float,
            l2_penalty: torch.Tensor,
            lambda_l2: float,
            contractive: torch.Tensor,
            lambda_contractive: float
        ) -> torch.Tensor:
        '''Combine reconstruction, energy, and regularization losses.

        Parameters
        ----------
        rec_loss : torch.Tensor
            Reconstruction loss.
        energy_loss_val : torch.Tensor | None
            Energy regression loss.
        lambda_rec : float
            Reconstruction weight.
        lambda_energy : float
            Energy weight.
        kld : torch.Tensor
            KL divergence term.
        beta_vae : float
            KL weight for VAE.
        l2_penalty : torch.Tensor
            Latent L2 penalty term.
        lambda_l2 : float
            L2 penalty weight.
        contractive : torch.Tensor
            Contractive penalty term.
        lambda_contractive : float
            Contractive penalty weight.

        Returns
        -------
        torch.Tensor
            Combined loss.
        '''

        losses: Dict[str, torch.Tensor] = {}
        if lambda_rec > 0.0:
            losses["recon"] = rec_loss * lambda_rec
        if lambda_energy > 0.0 and energy_loss_val is not None:
            losses["energy"] = energy_loss_val * lambda_energy

        if self.loss_balancing == "uncertainty" and self.uncertainty is not None and len(losses) > 0:
            # Learn task weights dynamically.
            total, _ = self.uncertainty(losses)
        elif self.loss_balancing == "gradnorm" and len(losses) > 0:
            # Balance tasks by equalizing gradient norms.
            total = self._gradnorm_total(losses)
        else:
            total = sum(losses.values()) if losses else torch.tensor(0.0, device=self.device)

        if beta_vae > 0.0:
            total = total + beta_vae * kld
        if lambda_l2 > 0.0:
            total = total + lambda_l2 * l2_penalty
        if lambda_contractive > 0.0:
            total = total + lambda_contractive * contractive

        return total

    def _compute_embedding_metrics(self, dataset) -> Dict[str, object]:
        '''Compute embedding statistics and energy correlation.

        Parameters
        ----------
        dataset : Dataset
            Dataset to encode.

        Returns
        -------
        Dict[str, object]
            Embedding statistics and correlation metrics.
        '''

        loader = DataLoader(dataset, batch_size=512)

        embeddings = []
        energy_embeddings = []
        energies = []

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                features, energy_vals, energy_mask = batch
                features = features.to(self.device)
                energy_vals = energy_vals.to(self.device)
                energy_mask = energy_mask.to(self.device)

                z = self.model.encode(features, sample=False)
                embeddings.append(z.detach().cpu().numpy())

                if energy_mask.any():
                    # Correlate embedding norms with energy when labels exist.
                    energies.append(energy_vals[energy_mask].detach().cpu().numpy().reshape(-1))
                    energy_embeddings.append(z[energy_mask].detach().cpu().numpy())

        if not embeddings:
            return {
                "embedding_variance": [],
                "embedding_collapse_rate": 0.0,
                "embedding_mean_norm": 0.0,
                "embedding_energy_spearman": 0.0
            }

        emb = np.concatenate(embeddings, axis=0)
        stats = embedding_stats(emb)

        spearman_value = 0.0
        if energies:
            energy_vec = np.concatenate(energies, axis=0)
            emb_energy = np.concatenate(energy_embeddings, axis=0)
            scores = np.linalg.norm(emb_energy, axis=1)
            spearman_value = spearman_corr(-energy_vec, scores)

        return {
            "embedding_variance": stats.get("variance", []),
            "embedding_collapse_rate": stats.get("collapse_rate", 0.0),
            "embedding_mean_norm": stats.get("mean_norm", 0.0),
            "embedding_energy_spearman": spearman_value
        }

    def _evaluate(self, loader: DataLoader, stage_cfg: dict) -> Dict[str, float]:
        '''Evaluate reconstruction and energy losses.

        Parameters
        ----------
        loader : DataLoader
            DataLoader to evaluate.
        stage_cfg : dict
            Stage configuration.

        Returns
        -------
        Dict[str, float]
            Evaluation metrics.
        '''

        self.model.eval()

        recon_type = stage_cfg.get("recon_loss", "mse")
        energy_type = stage_cfg.get("energy_loss", "huber")
        huber_delta = float(stage_cfg.get("huber_delta", 1.0))

        total_mse = 0.0
        total_mae = 0.0
        total_huber = 0.0
        total_count = 0

        energy_mse = 0.0
        energy_mae = 0.0
        energy_huber = 0.0
        energy_count = 0

        with torch.no_grad():
            for batch in loader:
                features, energies, energy_mask = batch
                features = features.to(self.device)
                energies = energies.to(self.device)
                energy_mask = energy_mask.to(self.device)

                outputs = self.model(features, sample=False)
                recon = outputs["reconstruction"]

                diff = recon - features
                total_mse += float((diff ** 2).sum().item())
                total_mae += float(diff.abs().sum().item())

                huber = nn.SmoothL1Loss(beta=huber_delta, reduction="sum")(recon, features)
                total_huber += float(huber.item())
                total_count += int(diff.numel())

                if outputs["energy"] is not None and energy_mask.any():
                    energy_pred = outputs["energy"][energy_mask]
                    energy_true = energies[energy_mask]
                    e_diff = energy_pred - energy_true
                    energy_mse += float((e_diff ** 2).sum().item())
                    energy_mae += float(e_diff.abs().sum().item())
                    energy_huber += float(nn.SmoothL1Loss(beta=huber_delta, reduction="sum")(energy_pred, energy_true).item())
                    energy_count += int(energy_pred.numel())

        if total_count == 0:
            return {}

        mse = total_mse / total_count
        mae = total_mae / total_count
        huber = total_huber / total_count
        rmse = float(np.sqrt(mse))

        recon_loss_value = self._select_loss(mse, rmse, mae, huber, recon_type)

        energy_metrics = {
            "energy_mse": 0.0,
            "energy_rmse": 0.0,
            "energy_mae": 0.0,
            "energy_huber": 0.0,
            "energy_loss": 0.0
        }

        if energy_count > 0:
            e_mse = energy_mse / energy_count
            e_mae = energy_mae / energy_count
            e_huber = energy_huber / energy_count
            e_rmse = float(np.sqrt(e_mse))
            energy_loss_value = self._select_loss(e_mse, e_rmse, e_mae, e_huber, energy_type)

            energy_metrics = {
                "energy_mse": e_mse,
                "energy_rmse": e_rmse,
                "energy_mae": e_mae,
                "energy_huber": e_huber,
                "energy_loss": energy_loss_value
            }

        lambda_rec = float(stage_cfg.get("lambda_recon", 1.0))
        lambda_energy = float(stage_cfg.get("lambda_energy", 0.0))
        combined = lambda_rec * recon_loss_value + lambda_energy * energy_metrics.get("energy_loss", 0.0)

        metrics = {
            "recon_mse": mse,
            "recon_rmse": rmse,
            "recon_mae": mae,
            "recon_huber": huber,
            "recon_loss": recon_loss_value,
            "combined_loss": combined
        }
        metrics.update(energy_metrics)

        return metrics

    def _gradnorm_total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        '''Compute GradNorm-balanced loss.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Task losses.

        Returns
        -------
        torch.Tensor
            Balanced total loss.
        '''

        if self.gradnorm_weights is None or len(self.gradnorm_weights) != len(losses):
            self.gradnorm_weights = torch.ones(len(losses), device=self.device)
        if self.gradnorm_initial is None or len(self.gradnorm_initial) != len(losses):
            self.gradnorm_initial = torch.tensor([loss.detach().item() for loss in losses.values()], device=self.device)

        shared_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
        g_norms = []
        for loss in losses.values():
            # Per-task gradients computed on shared encoder parameters.
            grads = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=False)
            norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
            g_norms.append(norm)

        g_norms = torch.stack(g_norms)
        g_avg = torch.mean(g_norms).detach()

        losses_tensor = torch.stack([loss.detach() for loss in losses.values()])
        loss_ratios = losses_tensor / (self.gradnorm_initial + 1e-8)
        target = g_avg * (loss_ratios ** self.gradnorm_alpha)

        with torch.no_grad():
            self.gradnorm_weights = self.gradnorm_weights * (target / (g_norms + 1e-8))
            self.gradnorm_weights = self.gradnorm_weights * (len(losses) / self.gradnorm_weights.sum())

        weighted = [w * l for w, l in zip(self.gradnorm_weights, losses.values())]
        return sum(weighted)

    def _run_stage(
            self,
            stage_name: str,
            train_dataset,
            val_dataset,
            stage_cfg: dict
        ) -> Dict[str, object]:
        '''Run a training stage.

        Parameters
        ----------
        stage_name : str
            Stage identifier (e.g., "stage1" or "stage2").
        train_dataset : Dataset
            Training dataset.
        val_dataset : Dataset | None
            Validation dataset.
        stage_cfg : dict
            Stage configuration.

        Returns
        -------
        Dict[str, object]
            Stage metrics including history and best validation loss.
        '''

        batch_size = int(stage_cfg.get("batch_size", 256))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None

        optimizer = self._build_optimizer(stage_cfg)

        # Mixed precision is optional and only enabled on CUDA.
        use_amp = bool(stage_cfg.get("mixed_precision", False) and self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        early_stopping = EarlyStopping(stage_cfg.get("early_stopping_patience", 20))

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        best_epoch = 0

        for epoch in range(int(stage_cfg.get("epochs", 100))):
            train_metrics = self._train_epoch(train_loader, optimizer, scaler, stage_cfg, epoch)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader, stage_cfg)

            combined_val = float(val_metrics.get("combined_loss", train_metrics.get("combined_loss", 0.0)))
            if combined_val < best_val:
                best_val = combined_val
                best_epoch = epoch

            if self.verbose:
                msg = f"[AETrainer] {stage_name} epoch {epoch+1}"
                msg += f" train={train_metrics.get('combined_loss', 0.0):.4f}"
                if val_metrics:
                    msg += f" val={combined_val:.4f}"
                ocprint.printv(msg)

            history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

            if val_loader is not None and early_stopping.step(combined_val):
                if self.verbose:
                    ocprint.printv(f"[AETrainer] Early stopping on {stage_name} at epoch {epoch+1}")
                break

        return {
            "stage": stage_name,
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "history": history
        }

    def _save_checkpoints(self) -> None:
        '''Save model and encoder checkpoints.'''
        os.makedirs(self.models_folder, exist_ok=True)
        base = self.run_name

        model_path = os.path.join(self.models_folder, f"{base}_best.pt")
        encoder_path = os.path.join(self.models_folder, f"{base}_encoder_best.pt")

        torch.save(self.model.state_dict(), model_path)
        if self.config.get("checkpoint", {}).get("save_encoder", True):
            torch.save(self.model.encoder.state_dict(), encoder_path)

    def _select_loss(self, mse: float, rmse: float, mae: float, huber: float, loss_type: str) -> float:
        '''Select a loss value based on loss type.

        Parameters
        ----------
        mse : float
            Mean squared error.
        rmse : float
            Root mean squared error.
        mae : float
            Mean absolute error.
        huber : float
            Huber loss value.
        loss_type : str
            Loss type selector.

        Returns
        -------
        float
            Selected loss value.
        '''

        if loss_type == "rmse":
            return rmse
        if loss_type == "mae":
            return mae
        if loss_type == "huber":
            return huber
        return mse

    def _train_epoch(
            self,
            loader: DataLoader,
            optimizer: optim.Optimizer,
            scaler: torch.cuda.amp.GradScaler,
            stage_cfg: dict,
            epoch: int
        ) -> Dict[str, float]:
        '''Train a single epoch.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        optimizer : optim.Optimizer
            Optimizer instance.
        scaler : torch.cuda.amp.GradScaler
            AMP gradient scaler.
        stage_cfg : dict
            Stage configuration.
        epoch : int
            Epoch index.

        Returns
        -------
        Dict[str, float]
            Aggregated training losses.
        '''

        self.model.train()

        total_loss = 0.0
        total_rec = 0.0
        total_energy = 0.0
        num_batches = 0

        recon_type = stage_cfg.get("recon_loss", "mse")
        energy_type = stage_cfg.get("energy_loss", "huber")
        huber_delta = float(stage_cfg.get("huber_delta", 1.0))

        lambda_rec = ramp_weight(stage_cfg.get("lambda_recon", 1.0), epoch, stage_cfg.get("ramp_epochs_recon", 0), stage_cfg.get("ramp_type", "linear"))
        lambda_energy = ramp_weight(stage_cfg.get("lambda_energy", 0.0), epoch, stage_cfg.get("ramp_epochs_energy", 0), stage_cfg.get("ramp_type", "linear"))

        lambda_l2 = float(stage_cfg.get("lambda_l2", 0.0))
        lambda_contractive = float(stage_cfg.get("lambda_contractive", 0.0))
        beta_vae = float(stage_cfg.get("beta_vae", stage_cfg.get("beta", 1.0)))

        for batch in loader:
            features, energies, energy_mask = batch
            features = features.to(self.device)
            energies = energies.to(self.device)
            energy_mask = energy_mask.to(self.device)

            noisy = apply_noise(
                features,
                noise_type=stage_cfg.get("noise_type", "none"),
                mask_prob=stage_cfg.get("mask_prob", 0.0),
                gaussian_std=stage_cfg.get("gaussian_std", 0.0),
                swap_prob=stage_cfg.get("swap_prob", 0.0)
            )

            if lambda_contractive > 0.0:
                # Contractive penalty requires gradients w.r.t. input.
                noisy.requires_grad_(True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = self.model(noisy, sample=True)
                recon = outputs["reconstruction"]
                latent = outputs["latent"]

                rec_loss = reconstruction_loss(recon, features, loss_type=recon_type, huber_delta=huber_delta)
                energy_loss_val = None

                if outputs["energy"] is not None and energy_mask.any():
                    # Only compute energy loss where labels are available.
                    energy_pred = outputs["energy"][energy_mask]
                    energy_true = energies[energy_mask]
                    energy_loss_val = energy_loss(energy_pred, energy_true, loss_type=energy_type, huber_delta=huber_delta)

                kld = torch.tensor(0.0, device=self.device)
                if self.model.use_vae:
                    # KL term only applies when VAE is enabled.
                    kld = kl_divergence(outputs["mu"], outputs["logvar"])

                l2_penalty = latent.pow(2).mean() if lambda_l2 > 0.0 else torch.tensor(0.0, device=self.device)

                contractive = torch.tensor(0.0, device=self.device)
                if lambda_contractive > 0.0:
                    # Contractive penalty requires input gradients.
                    contractive = contractive_penalty(latent, noisy)

                total = self._combine_losses(
                    rec_loss,
                    energy_loss_val,
                    lambda_rec,
                    lambda_energy,
                    kld,
                    beta_vae,
                    l2_penalty,
                    lambda_l2,
                    contractive,
                    lambda_contractive
                )

            scaler.scale(total).backward()

            if stage_cfg.get("clip_grad", 0.0) > 0.0:
                # Unscale before clipping for correct gradient norms.
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(stage_cfg.get("clip_grad", 1.0)))

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(total.detach().cpu().item())
            total_rec += float(rec_loss.detach().cpu().item())
            if energy_loss_val is not None:
                total_energy += float(energy_loss_val.detach().cpu().item())
            num_batches += 1

        if num_batches == 0:
            return {}

        return {
            "combined_loss": total_loss / num_batches,
            "recon_loss": total_rec / num_batches,
            "energy_loss": total_energy / num_batches
        }

    def evaluate(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            feature_mask: Optional[np.ndarray] = None,
            stage: str = "stage1"
        ) -> Dict[str, float]:
        '''Evaluate reconstruction/energy metrics on a dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray | None, optional
            Energy targets, by default None.
        feature_mask : np.ndarray | None, optional
            Feature mask to apply, by default None.
        stage : str, optional
            Stage configuration name, by default "stage1".

        Returns
        -------
        Dict[str, float]
            Dictionary of reconstruction/energy metrics.
        '''

        dataset = self._build_dataset(X, y, feature_mask)
        loader = DataLoader(dataset, batch_size=512)
        stage_cfg = self.config.get(stage, self.config.get("stage1", {}))
        return self._evaluate(loader, stage_cfg)

    def fit(
            self,
            X_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_train: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_mask: Optional[np.ndarray] = None,
            X_unlabeled: Optional[np.ndarray] = None
        ) -> Dict[str, object]:
        '''Train the autoencoder on the provided data.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        X_val : np.ndarray | None, optional
            Validation feature matrix, by default None.
        y_train : np.ndarray | None, optional
            Training energy targets, by default None.
        y_val : np.ndarray | None, optional
            Validation energy targets, by default None.
        feature_mask : np.ndarray | None, optional
            Feature mask to apply, by default None.
        X_unlabeled : np.ndarray | None, optional
            Additional unlabeled data for reconstruction, by default None.

        Returns
        -------
        Dict[str, object]
            Training metrics and embedding statistics.

        Notes
        -----
        Stage semantics are defined by the configuration:
        - stage1 focuses on denoising reconstruction and (if available) energy supervision.
        - stage2 is optional and can reweight losses or change noise to refine the latent space.
        If y_train/y_val are None, the energy head is ignored and only reconstruction
        loss is optimized (energy_mask is all False).
        '''

        train_dataset = self._build_dataset(X_train, y_train, feature_mask)
        val_dataset = self._build_dataset(X_val, y_val, feature_mask) if X_val is not None else None

        if X_unlabeled is not None:
            # Concatenate unlabeled data for reconstruction-only regularization.
            unlabeled_dataset = self._build_dataset(X_unlabeled, None, feature_mask)
            train_dataset = ConcatDataset([train_dataset, unlabeled_dataset])

        best_state = None
        best_val = float("inf")
        best_metrics: Dict[str, object] = {}

        for stage_name in ["stage1", "stage2"]:
            stage_cfg = self.config.get(stage_name, {})
            if not stage_cfg.get("enabled", stage_name == "stage1"):
                continue

            if self.verbose:
                ocprint.printv(f"[AETrainer] Starting {stage_name}")

            metrics = self._run_stage(stage_name, train_dataset, val_dataset, stage_cfg)

            stage_best = float(metrics.get("best_val_loss", float("inf")))
            if stage_best < best_val:
                # Track best-performing state across stages.
                best_val = stage_best
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                best_metrics = metrics

        if best_state is not None:
            self.model.load_state_dict(best_state, strict=False)

        if self.models_folder and self.config.get("checkpoint", {}).get("save_best", True):
            self._save_checkpoints()

        embed_metrics = self._compute_embedding_metrics(val_dataset or train_dataset)
        best_metrics.update(embed_metrics)

        return best_metrics
# Functions
###############################################################################
## Private ##

## Public ##
