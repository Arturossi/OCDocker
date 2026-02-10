#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore DNN future losses, metrics, datasets, and models.
'''

# Imports
###############################################################################
import numpy as np
import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

import OCDocker.OCScore.DNN.future.datasets as ocfdatasets
import OCDocker.OCScore.DNN.future.losses as ocflosses
import OCDocker.OCScore.DNN.future.metrics as ocfmetrics
import OCDocker.OCScore.DNN.future.models as ocfmodels

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

## Public ##

@pytest.mark.order(433)
def test_future_dnn_losses_uncertainty_focal_rank_and_supcon():
    model = ocflosses.UncertaintyWeighting(["reg", "cls"], init_log_var=0.0)

    total, weights = model({"reg": torch.tensor(2.0), "cls": torch.tensor(1.0)})
    assert total.item() > 0.0
    assert set(weights.keys()) == {"reg", "cls"}

    total2, weights2 = model({"reg": torch.tensor(1.5)})
    assert total2.item() > 0.0
    assert "cls" not in weights2

    logits = torch.tensor([0.2, -0.5, 1.2], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    loss_mean = ocflosses.focal_binary_loss(logits, targets, reduction="mean")
    loss_sum = ocflosses.focal_binary_loss(logits, targets, reduction="sum")
    loss_none = ocflosses.focal_binary_loss(logits, targets, reduction="none")
    assert loss_mean.item() > 0.0
    assert loss_sum.item() > 0.0
    assert loss_none.shape[0] == 3

    assert ocflosses._safe_k(0.5, 10) == 5
    assert ocflosses._safe_k(100, 3) == 3
    assert ocflosses._safe_k(1, 0) == 0

    single_rank = ocflosses._lambda_rank_loss_single_k(torch.tensor([0.3]), torch.tensor([1.0]), 0.5)
    assert single_rank.item() == pytest.approx(0.0)

    rank_loss = ocflosses.lambda_rank_ndcg_loss(
        torch.tensor([0.9, 0.3, 0.2, 0.1], dtype=torch.float32),
        torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        k_fractions=(0.25, 0.50),
        weights=(1.0,),
    )
    assert rank_loss.item() >= 0.0

    no_pos = ocflosses.supervised_contrastive_loss(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        torch.tensor([0, 1]),
    )
    assert no_pos.item() == pytest.approx(0.0)

    with_pos = ocflosses.supervised_contrastive_loss(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32),
        torch.tensor([1, 1, 0, 0]),
    )
    assert with_pos.ndim == 0

    none_embed = ocflosses.supervised_contrastive_loss(None, torch.tensor([0, 1]))  # type: ignore[arg-type]
    assert none_embed.item() == pytest.approx(0.0)


@pytest.mark.order(434)
def test_future_dnn_metrics_full_and_guard_paths(monkeypatch):
    monkeypatch.setattr(ocfmetrics.ocrank, "enrichment_factor", lambda y, s, frac: float(np.mean(y) + frac))

    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], dtype=float)
    target_ids = np.array(["t1", "t1", "t2", "t2", "t3", "t3"])

    metrics = ocfmetrics.compute_classification_metrics(
        y_true=y_true,
        y_score=y_score,
        target_ids=target_ids,
        k_fractions=(0.25, 0.50),
    )
    assert "AUC" in metrics
    assert "EF@25%" in metrics
    assert "NDCG@50%" in metrics

    group_empty = ocfmetrics.compute_group_metrics(y_true, y_score, target_ids, k_fractions=())
    assert group_empty == {}

    y_true_single = np.array([1, 1, 1, 0], dtype=int)
    y_score_single = np.array([0.8, 0.7, 0.6, 0.5], dtype=float)
    target_single = np.array(["a", "a", "b", "b"])
    group_metrics = ocfmetrics.compute_group_metrics(y_true_single, y_score_single, target_single, k_fractions=(0.5,))
    assert "EF@50%" in group_metrics
    assert "NDCG@50%" in group_metrics

    assert ocfmetrics.ndcg_at_k(np.array([]), np.array([]), 1) == 0.0
    assert ocfmetrics.ndcg_at_k(np.array([1, 0]), np.array([0.9, 0.1]), 1) == pytest.approx(1.0)

    assert ocfmetrics.partial_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) == 0.0
    assert ocfmetrics.partial_auc(y_true, y_score, max_fpr=0.05) >= 0.0

    monkeypatch.setattr(ocfmetrics, "roc_curve", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    assert ocfmetrics.partial_auc(y_true, y_score) == 0.0

    monkeypatch.setattr(ocfmetrics, "roc_auc_score", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(ocfmetrics, "log_loss", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(ocfmetrics, "average_precision_score", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")))
    assert ocfmetrics.safe_auc(y_true, y_score) == 0.0
    assert ocfmetrics.safe_log_loss(y_true, y_score) == float("inf")
    assert ocfmetrics.safe_pr_auc(y_true, y_score) == 0.0


@pytest.mark.order(435)
def test_future_dnn_datasets_and_sampler_paths():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [4.0, 5.0]], dtype=float)
    energies = np.array([0.2, 0.4, 0.6], dtype=float)
    mask = np.array([1.0, 0.0], dtype=float)

    energy_ds = ocfdatasets.EnergyDataset(features, energies, mask=mask)
    assert len(energy_ds) == 3
    x0, y0 = energy_ds[0]
    assert x0.shape[0] == 2
    assert x0[1].item() == pytest.approx(0.0)
    assert y0.shape == (1,)

    with pytest.raises(ValueError, match="same length"):
        ocfdatasets.TargetRankingDataset(features, np.array([1, 0], dtype=float), target_ids=["a", "b", "c"])

    rank_ds = ocfdatasets.TargetRankingDataset(
        features,
        np.array([1.0, 0.0, 1.0], dtype=float),
        target_ids=["t1", "t2", "t1"],
        mask=np.array([1.0, 1.0], dtype=float),
    )
    assert len(rank_ds) == 3
    assert rank_ds.target_to_index == {"t1": 0, "t2": 1}
    assert rank_ds.target_to_indices[0] == [0, 2]
    r0 = rank_ds[0]
    assert len(r0) == 3

    sampler_full = ocfdatasets.TargetBatchSampler(rank_ds.target_to_indices, batch_size=None, shuffle=False)
    batches_full = list(iter(sampler_full))
    assert batches_full == [[0, 2], [1]]
    assert len(sampler_full) == 2

    random.seed(0)
    sampler_single = ocfdatasets.TargetBatchSampler(rank_ds.target_to_indices, batch_size=1, shuffle=False, split_target_batches=False)
    batches_single = list(iter(sampler_single))
    assert len(batches_single) == 2
    assert all(len(b) == 1 for b in batches_single)

    random.seed(0)
    sampler_split = ocfdatasets.TargetBatchSampler(rank_ds.target_to_indices, batch_size=1, shuffle=False, split_target_batches=True)
    batches_split = list(iter(sampler_split))
    assert len(batches_split) == 3
    assert len(sampler_split) == 3


@pytest.mark.order(436)
def test_future_dnn_model_helpers_and_parser_paths():
    with pytest.raises(ValueError, match="non-empty"):
        ocfmodels.MLP(input_size=2, layer_sizes=[], activations="ReLU")

    with pytest.raises(ValueError, match="length must match"):
        ocfmodels.MLP(input_size=2, layer_sizes=[3, 1], activations=[("ReLU", {})])

    assert isinstance(ocfmodels._build_activation("LeakyReLU", {"negative_slope": 0.2}), nn.LeakyReLU)
    assert isinstance(ocfmodels._build_activation("GELU", {"approximate": "tanh"}), nn.GELU)
    assert isinstance(ocfmodels._build_activation("Mish", {}), nn.Mish)
    assert isinstance(ocfmodels._build_activation("SELU", {}), nn.SELU)
    assert isinstance(ocfmodels._build_activation("Identity", {}), nn.Identity)
    assert isinstance(ocfmodels._build_activation("UNKNOWN", {}), nn.ReLU)

    dims, activs = ocfmodels.parse_encoder_params({"encoding_dim": 8, "encoder_activation": "GELU", "approximate_encoder": "tanh"})
    assert dims == [8]
    assert activs[0][0] == "GELU"
    assert activs[0][1]["approximate"] == "tanh"

    parsed_dims, parsed_acts = ocfmodels.parse_encoder_params(
        {
            "n_layers_encoder": 2,
            "n_units_layer_0_encoder": 6,
            "n_units_layer_1_encoder": 4,
            "activation_function_0_encoder": "LeakyReLU",
            "activation_function_1_encoder": "ReLU",
            "negative_slope_0_encoder": 0.15,
        }
    )
    assert parsed_dims == [6, 4]
    assert parsed_acts[0][0] == "LeakyReLU"
    assert parsed_acts[0][1]["negative_slope"] == pytest.approx(0.15)
    assert parsed_acts[1][0] == "ReLU"

    with pytest.raises(ValueError, match="at least one layer"):
        ocfmodels.parse_encoder_params({})


@pytest.mark.order(437)
def test_future_dnn_multitask_model_forward_paths():
    x = torch.randn(4, 3)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

    model = ocfmodels.MultiTaskModel(
        input_size=3,
        encoder_params=None,
        shared_sizes=[5],
        shared_activation="ReLU",
        decoder_sizes=[3],
        head_sizes=[4],
        embedding_dim=2,
        dropout=0.0,
        batch_norm=False,
        mask=mask,
    )
    out = model(x, return_reconstruction=True)
    assert out["latent"] is not None
    assert out["energy"] is not None
    assert out["activity"] is not None
    assert out["embedding"] is not None
    assert out["reconstruction"] is not None
    assert out["energy"].shape == (4, 1)
    assert out["activity"].shape == (4, 1)

    legacy_model = ocfmodels.MultiTaskModel(
        input_size=3,
        encoder_params={"encoding_dim": 4, "encoder_activation": "ReLU"},
        shared_sizes=[3],
        decoder_sizes=None,
        head_sizes=[2],
        embedding_dim=None,
        batch_norm=False,
    )
    out_legacy = legacy_model(x, return_reconstruction=False)
    assert out_legacy["embedding"] is None
    assert out_legacy["reconstruction"] is None
