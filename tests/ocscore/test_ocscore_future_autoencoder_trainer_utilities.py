#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore future Autoencoder and AETrainer helper branches.
'''

# Imports
###############################################################################
from pathlib import Path

import numpy as np

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

import OCDocker.OCScore.Dimensionality.future.AETrainer as ocfaetrainer
import OCDocker.OCScore.Dimensionality.future.Autoencoder as ocfauto

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

def _build_autoencoder(use_vae: bool = False) -> ocfauto.Autoencoder:
    return ocfauto.Autoencoder(
        input_size=4,
        encoder_hidden_sizes=[3],
        latent_dim=2,
        decoder_sizes=[3, 4],
        activation="ReLU",
        norm="none",
        use_vae=use_vae,
        energy_head_sizes=[2],
    )


## Public ##

@pytest.mark.order(440)
def test_future_autoencoder_helper_builders_and_encode_paths(tmp_path):
    assert isinstance(ocfauto._build_activation("LeakyReLU", {"negative_slope": 0.2}), nn.LeakyReLU)
    assert isinstance(ocfauto._build_activation("GELU", {"approximate": "tanh"}), nn.GELU)
    assert isinstance(ocfauto._build_activation("Mish", {}), nn.Mish)
    assert isinstance(ocfauto._build_activation("SELU", {}), nn.SELU)
    assert isinstance(ocfauto._build_activation("Identity", {}), nn.Identity)
    assert isinstance(ocfauto._build_activation("Unknown", {}), nn.ReLU)

    assert isinstance(ocfauto._build_norm("batch", 4), nn.BatchNorm1d)
    assert isinstance(ocfauto._build_norm("layer", 4), nn.LayerNorm)
    assert ocfauto._build_norm("none", 4) is None

    model = _build_autoencoder(use_vae=False)
    x = torch.randn(3, 4)
    latent = model.encode(x, sample=False, return_stats=False)
    latent_stats = model.encode(x, sample=False, return_stats=True)
    out = model.forward(x, sample=False)
    recon = model.reconstruct(x, sample=False)

    assert latent.shape == (3, 2)
    assert len(latent_stats) == 3
    assert out["reconstruction"].shape == (3, 4)
    assert out["energy"] is not None
    assert recon.shape == (3, 4)
    assert model.get_decoder_topology() == ["Linear", "Norm", "Activation"]
    assert model.get_encoder_topology() == ["Linear", "Norm", "Activation"]

    save_path = tmp_path / "encoder_state.pt"
    model.save_encoder(str(save_path))
    assert save_path.exists()
    model.load_encoder(str(save_path), map_location="cpu")

    vae_model = _build_autoencoder(use_vae=True)
    z, mu, logvar = vae_model.encode(x, sample=True, return_stats=True)
    assert z.shape == mu.shape == logvar.shape


@pytest.mark.order(441)
def test_future_aetrainer_early_stopping_and_builders(monkeypatch, tmp_path):
    stopper = ocfaetrainer.EarlyStopping(patience=2, min_delta=0.0)
    assert stopper.step(1.0) is False
    assert stopper.step(1.1) is False
    assert stopper.step(1.2) is True

    model = _build_autoencoder(use_vae=False)
    trainer = ocfaetrainer.AETrainer(
        model=model,
        config={"optimization": {"loss_balancing": "fixed"}, "checkpoint": {"save_encoder": True}},
        device=torch.device("cpu"),
        verbose=False,
        models_folder=str(tmp_path),
        run_name="ae_unit",
    )

    with pytest.raises(ValueError, match="cannot be None"):
        trainer._build_dataset(None, None, None)

    dataset = trainer._build_dataset(
        X=np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float),
        y=np.array([0.5], dtype=float),
        feature_mask=np.array([1.0, 0.0, 1.0, 1.0], dtype=float),
    )
    assert len(dataset) == 1

    optimizer = trainer._build_optimizer({"lr": 1e-3, "weight_decay": 1e-6})
    assert optimizer is not None

    trainer._save_checkpoints()
    assert Path(tmp_path / "ae_unit_best.pt").exists()
    assert Path(tmp_path / "ae_unit_encoder_best.pt").exists()

    trainer.config["checkpoint"]["save_encoder"] = False
    trainer.run_name = "ae_unit_noenc"
    trainer._save_checkpoints()
    assert Path(tmp_path / "ae_unit_noenc_best.pt").exists()
    assert not Path(tmp_path / "ae_unit_noenc_encoder_best.pt").exists()

    trainer.models_folder = None
    trainer._save_checkpoints()

    assert trainer._select_loss(1.0, 2.0, 3.0, 4.0, "rmse") == pytest.approx(2.0)
    assert trainer._select_loss(1.0, 2.0, 3.0, 4.0, "mae") == pytest.approx(3.0)
    assert trainer._select_loss(1.0, 2.0, 3.0, 4.0, "huber") == pytest.approx(4.0)
    assert trainer._select_loss(1.0, 2.0, 3.0, 4.0, "mse") == pytest.approx(1.0)

    monkeypatch.setattr(trainer, "_gradnorm_total", lambda losses: torch.tensor(float(len(losses))))
    trainer.loss_balancing = "gradnorm"
    trainer.device = torch.device("cpu")

    gradnorm_total = trainer._combine_losses(
        rec_loss=torch.tensor(2.0),
        energy_loss_val=torch.tensor(1.0),
        lambda_rec=1.0,
        lambda_energy=1.0,
        kld=torch.tensor(0.1),
        beta_vae=0.5,
        l2_penalty=torch.tensor(0.2),
        lambda_l2=0.5,
        contractive=torch.tensor(0.3),
        lambda_contractive=0.5,
    )
    assert gradnorm_total.item() > 2.0


@pytest.mark.order(442)
def test_future_aetrainer_uncertainty_and_fixed_combine_losses():
    model = _build_autoencoder(use_vae=False)
    trainer = ocfaetrainer.AETrainer(
        model=model,
        config={"optimization": {"loss_balancing": "uncertainty"}},
        device=torch.device("cpu"),
        verbose=False,
    )

    total_uncertainty = trainer._combine_losses(
        rec_loss=torch.tensor(2.0),
        energy_loss_val=torch.tensor(1.0),
        lambda_rec=1.0,
        lambda_energy=1.0,
        kld=torch.tensor(0.0),
        beta_vae=0.0,
        l2_penalty=torch.tensor(0.0),
        lambda_l2=0.0,
        contractive=torch.tensor(0.0),
        lambda_contractive=0.0,
    )
    assert total_uncertainty.item() > 0.0

    trainer.loss_balancing = "fixed"
    trainer.uncertainty = None
    total_fixed = trainer._combine_losses(
        rec_loss=torch.tensor(2.0),
        energy_loss_val=None,
        lambda_rec=0.0,
        lambda_energy=1.0,
        kld=torch.tensor(0.0),
        beta_vae=0.0,
        l2_penalty=torch.tensor(0.0),
        lambda_l2=0.0,
        contractive=torch.tensor(0.0),
        lambda_contractive=0.0,
    )
    assert total_fixed.item() == pytest.approx(0.0)
