#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.DNN.DNNOptimizer core branches.

Usage:

pytest tests/test_ocscore_dnn_optimizer_module.py
'''

# Imports
###############################################################################
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import OCDocker.OCScore.DNN.DNNOptimizer as ocdnn

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


class _FakeTrial:
    def __init__(self, prune=False):
        self.user_attrs = {}
        self.reports = []
        self._prune = prune

    def suggest_float(self, name, low, high):  # noqa: ARG002
        if name == "lr":
            return 1e-3
        if name == "weight_decay":
            return 1e-6
        if name.startswith("negative_slope"):
            return 0.1
        if name == "clip_grad":
            return 0.5
        return low

    def suggest_int(self, name, low, high):  # noqa: ARG002
        if name == "n_layers":
            return 1
        if name == "epochs":
            return 1
        return low

    def suggest_categorical(self, name, choices):
        if name.startswith("n_units_layer_"):
            return choices[0]
        if name.startswith("activation_function_"):
            return "ReLU"
        if name == "optimizer":
            return "Adam"
        if name == "batch_size":
            return choices[0]
        if name.startswith("approximate_"):
            return choices[0]
        return choices[0]

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._prune


class _FakeStudy:
    def __init__(self):
        self.optimize_calls = []
        self.best_params = {"lr": 0.01}
        self.best_trial = "best-trial"

    def optimize(self, objective, n_trials=1, n_jobs=1):
        self.optimize_calls.append((objective, n_trials, n_jobs))


class _FakeNeuralNet:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        self.validation_auc = 0.9
        self.pr_auc = 0.8
        self.log_loss_value = 0.7
        self.mae = 0.6
        self.rmse = 0.55

    def train_model(self, *args, **kwargs):  # noqa: ARG002
        return None


# Functions
###############################################################################
## Private ##


def _nn_params():
    return {
        "activation_function_0": "ReLU",
        "n_units_layer_0": 4,
        "batch_size": 2,
        "epochs": 1,
        "lr": 1e-2,
        "clip_grad": 1.0,
        "optimizer": "Adam",
        "weight_decay": 0.0,
    }


def _single_branch_data():
    x_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.5, 2.0]], dtype=float)
    y_train = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    x_test = np.array([[0.2, 0.8], [1.1, 0.1]], dtype=float)
    y_test = np.array([0.0, 1.0], dtype=float)
    return x_train, y_train, x_test, y_test


def _multibranch_data():
    x1 = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=float)
    x2 = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=float)
    x3 = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float)
    y = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
    return [x1, x2, x3], y


## Public ##


@pytest.mark.order(464)
def test_custom_dataset_converts_inputs_and_supports_indexing():
    dataset = ocdnn.CustomDataset([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])
    assert len(dataset) == 2
    features, target = dataset[1]
    assert torch.is_tensor(features)
    assert torch.is_tensor(target)
    assert target.item() == pytest.approx(1.0)

    dataset_tensor = ocdnn.CustomDataset(
        torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )
    assert len(dataset_tensor) == 2


@pytest.mark.order(465)
def test_multibranch_custom_dataset_converts_inputs_and_supports_indexing():
    dataset = ocdnn.MultiBranchCustomDataset(
        [[1.0], [2.0]],
        [[3.0], [4.0]],
        [[5.0], [6.0]],
        [0.0, 1.0],
    )
    assert len(dataset) == 2
    f1, f2, f3, tgt = dataset[0]
    assert f1.shape == torch.Size([1])
    assert f2.shape == torch.Size([1])
    assert f3.shape == torch.Size([1])
    assert tgt.item() == pytest.approx(0.0)


@pytest.mark.order(466)
def test_dynamicnn_mask_paths_forward_and_private_ablation_setter():
    model = ocdnn.DynamicNN(
        input_size=2,
        output_size=1,
        hidden_layers=[3],
        activation_data=[(nn.ReLU, {})],
        mask=[1, 0],
        device=torch.device("cpu"),
    )
    out = model(torch.tensor([[1.0, 5.0], [2.0, 6.0]], dtype=torch.float32))
    assert out.shape == torch.Size([2, 1])
    assert model.mask is not None

    model._DynamicNN__set_ablation_mask(torch.tensor([]))
    assert model.mask is None
    model._DynamicNN__set_ablation_mask(np.array([0, 1]))
    assert isinstance(model.mask, torch.Tensor)


@pytest.mark.order(467)
def test_dynamicnn_with_encoder_and_activation_param_processing():
    encoder = [
        ("Linear", 2, 2),
        ("BatchNorm1d", 2),
        ("Activation", nn.ReLU()),
    ]
    model = ocdnn.DynamicNN(
        input_size=2,
        output_size=1,
        hidden_layers=[2],
        activation_data=[(nn.GELU, {"approximate_0": "tanh"})],
        encoder=encoder,
        device=torch.device("cpu"),
    )
    assert model.input_layer_size == 2
    out = model(torch.tensor([[0.5, 0.5], [1.0, 1.0]], dtype=torch.float32))
    assert out.shape == torch.Size([2, 1])


@pytest.mark.order(468)
def test_multibranch_dynamicnn_requires_list_encoder():
    with pytest.raises(ValueError, match="encoder should be a list"):
        _ = ocdnn.MultiBranchDynamicNN(2, 1, [4], encoders=None)  # type: ignore[arg-type]


@pytest.mark.order(469)
def test_multibranch_dynamicnn_forward_validation_and_success():
    encoders = [
        [("Linear", 2, 2)],
        [("Linear", 1, 1)],
        [("Linear", 1, 1)],
    ]
    model = ocdnn.MultiBranchDynamicNN(
        input_size=[2, 1, 1],
        output_size=1,
        hidden_layers=[4],
        activation_data=[(nn.ReLU, {})],
        encoders=encoders,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="list of tensors"):
        _ = model(torch.tensor([[1.0, 2.0]], dtype=torch.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="number of inputs"):
        _ = model([torch.tensor([[1.0, 2.0]], dtype=torch.float32)])

    out = model(
        [
            torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            torch.tensor([[0.5], [0.6]], dtype=torch.float32),
            torch.tensor([[0.7], [0.8]], dtype=torch.float32),
        ]
    )
    assert out.shape == torch.Size([2, 1])


@pytest.mark.order(470)
def test_neuralnet_get_model_and_train_model_single_class_validation():
    model = ocdnn.NeuralNet(
        input_size=2,
        output_size=1,
        encoder_params=None,
        nn_params=_nn_params(),
        use_gpu=False,
        verbose=False,
    )
    assert isinstance(model.get_model(), nn.Module)

    x_train, y_train, x_test, y_test = _single_branch_data()
    x_validation = np.array([[0.2, 0.1], [0.7, 0.8]], dtype=float)
    y_validation = np.array([1.0, 1.0], dtype=float)
    model.train_model(x_train, y_train, x_test, y_test, x_validation, y_validation)

    assert model.validation_auc == 0.0
    assert np.isinf(model.log_loss_value)
    assert model.prediction is not None


@pytest.mark.order(471)
def test_neuralnet_train_model_requires_y_validation_when_x_validation_provided():
    model = ocdnn.NeuralNet(
        input_size=2,
        output_size=1,
        encoder_params=None,
        nn_params=_nn_params(),
        use_gpu=False,
        verbose=False,
    )
    x_train, y_train, x_test, y_test = _single_branch_data()
    with pytest.raises(ValueError, match="y_validation must be provided"):
        model.train_model(x_train, y_train, x_test, y_test, X_validation=np.array([[0.1, 0.2]]), y_validation=None)


@pytest.mark.order(472)
def test_dnnoptimizer_init_single_branch_with_none_validation_list():
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        use_gpu=False,
        verbose=False,
    )
    assert isinstance(optimizer.X_train, torch.Tensor)
    assert optimizer.X_validation is None
    assert optimizer.y_validation is None


@pytest.mark.order(473)
def test_dnnoptimizer_init_multibranch_with_encoder_tuple():
    x_train, y_train = _multibranch_data()
    x_test, y_test = _multibranch_data()
    x_validation, y_validation = _multibranch_data()
    encoder_params = (
        {"encoder_activation": "ReLU", "encoding_dim": 2},
        {"encoder_activation": "ReLU", "encoding_dim": 1},
        {"encoder_activation": "ReLU", "encoding_dim": 1},
    )

    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=x_validation,
        y_validation=y_validation,
        encoder_params=encoder_params,
        use_gpu=False,
        verbose=False,
    )
    assert isinstance(optimizer.input_size, list)
    assert isinstance(optimizer.encoder, list)
    assert len(optimizer.encoder) == 3


@pytest.mark.order(474)
def test_dnnoptimizer_init_raises_for_encoder_without_activation():
    x_train, y_train, x_test, y_test = _single_branch_data()
    with pytest.raises(ValueError, match="at least one activation function"):
        _ = ocdnn.DNNOptimizer(
            x_train,
            y_train,
            x_test,
            y_test,
            X_validation=[None, None, None],
            y_validation=None,
            encoder_params={"n_layers_encoder": 1},
            use_gpu=False,
            verbose=False,
        )


@pytest.mark.order(475)
def test_dnnoptimizer_ablate_success_and_error_path(monkeypatch):
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        use_gpu=False,
        verbose=True,
    )
    created = {}

    def _create_study(**kwargs):
        created["kwargs"] = kwargs
        study = _FakeStudy()
        created["study"] = study
        return study

    monkeypatch.setattr(ocdnn.optuna, "create_study", _create_study)
    optimizer.ablate({"epochs": 1}, n_trials=2, study_name="ablation", load_if_exists=False, n_jobs=3)
    assert optimizer.network_params["epochs"] == 1
    assert created["kwargs"]["study_name"] == "ablation"
    assert created["study"].optimize_calls[0][1:] == (2, 3)

    msgs = []
    monkeypatch.setattr(ocdnn.optuna, "create_study", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))  # noqa: ARG005
    monkeypatch.setattr(ocdnn.ocprint, "print_error", lambda msg: msgs.append(msg))
    optimizer.ablate({"epochs": 1})
    assert msgs and "An error occurred" in msgs[-1]


@pytest.mark.order(476)
def test_dnnoptimizer_objective_raises_when_loaders_not_initialized(monkeypatch):
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        use_gpu=False,
        verbose=False,
    )
    monkeypatch.setattr(ocdnn, "DataLoader", lambda *args, **kwargs: None)  # noqa: ARG005
    trial = _FakeTrial()
    with pytest.raises(ValueError, match="Train/test loaders were not initialized"):
        optimizer.objective(trial)


@pytest.mark.order(477)
def test_dnnoptimizer_objective_sets_validation_attrs_for_single_class(monkeypatch):
    x_train, y_train, x_test, y_test = _single_branch_data()
    x_validation = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=float)
    y_validation = np.array([1.0, 1.0], dtype=float)
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=x_validation,
        y_validation=y_validation,
        use_gpu=False,
        verbose=False,
    )
    monkeypatch.setattr(ocdnn.DNNOptimizer, "train_test_model", lambda *args, **kwargs: 0.123)
    trial = _FakeTrial()
    score = optimizer.objective(trial)
    assert score == pytest.approx(0.123)
    assert trial.user_attrs["AUC"] == 0.0
    assert trial.user_attrs["pr_auc"] == 0.0
    assert np.isinf(trial.user_attrs["log_loss"])
    assert "mae" in trial.user_attrs


@pytest.mark.order(478)
def test_dnnoptimizer_objective_ablation_paths(monkeypatch):
    # Multi-branch encoder path should raise NotImplementedError
    x_train_mb, y_train_mb = _multibranch_data()
    x_test_mb, y_test_mb = _multibranch_data()
    encoder_params = (
        {"encoder_activation": "ReLU", "encoding_dim": 2},
        {"encoder_activation": "ReLU", "encoding_dim": 1},
        {"encoder_activation": "ReLU", "encoding_dim": 1},
    )
    optimizer_mb = ocdnn.DNNOptimizer(
        x_train_mb,
        y_train_mb,
        x_test_mb,
        y_test_mb,
        X_validation=[None, None, None],
        y_validation=None,
        encoder_params=encoder_params,
        use_gpu=False,
        verbose=False,
    )
    with pytest.raises(NotImplementedError, match="not supported for MultiBranchDynamicNN"):
        optimizer_mb.objective_ablation(_FakeTrial())

    # Single-branch path should set user attrs and return rmse
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        mask=[True, False],
        use_gpu=False,
        verbose=False,
    )
    optimizer.network_params = _nn_params()
    monkeypatch.setattr(ocdnn, "NeuralNet", _FakeNeuralNet)
    trial = _FakeTrial()
    result = optimizer.objective_ablation(trial)
    assert result == pytest.approx(0.55)
    assert trial.user_attrs["Feature_Mask"] == "10"
    assert trial.user_attrs["random_seed"] == optimizer.random_seed


@pytest.mark.order(479)
def test_dnnoptimizer_optimize_and_gpu_seed_branch(monkeypatch):
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        use_gpu=False,
        verbose=True,
    )
    created = {}

    def _create_study(**kwargs):
        created["kwargs"] = kwargs
        study = _FakeStudy()
        created["study"] = study
        return study

    logs = []
    monkeypatch.setattr(ocdnn.optuna, "create_study", _create_study)
    monkeypatch.setattr(ocdnn.ocprint, "printv", lambda msg: logs.append(msg))
    optimizer.optimize(direction="minimize", n_trials=4, study_name="study", load_if_exists=False, n_jobs=2)
    assert created["kwargs"]["direction"] == "minimize"
    assert created["kwargs"]["study_name"] == "study"
    assert created["study"].optimize_calls[0][1:] == (4, 2)
    assert any("Best Hyperparameters" in str(msg) for msg in logs)

    # Explicitly exercise GPU seed branch without allocating CUDA tensors
    called = {"seed": None}
    optimizer.use_gpu = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: called.__setitem__("seed", seed))
    optimizer.set_random_seed()
    assert called["seed"] == optimizer.random_seed
    assert optimizer.device.type == "cuda"


@pytest.mark.order(480)
def test_train_test_model_single_and_multibranch_with_pruning():
    x_train, y_train, x_test, y_test = _single_branch_data()
    optimizer = ocdnn.DNNOptimizer(
        x_train,
        y_train,
        x_test,
        y_test,
        X_validation=[None, None, None],
        y_validation=None,
        use_gpu=False,
        verbose=False,
    )
    model = nn.Sequential(nn.Linear(2, 1))
    train_loader = DataLoader(ocdnn.CustomDataset(x_train, y_train), batch_size=2, shuffle=False)
    test_loader = DataLoader(ocdnn.CustomDataset(x_test, y_test), batch_size=2, shuffle=False)
    criterion = nn.MSELoss()
    optimizer_t = torch.optim.Adam(model.parameters(), lr=1e-3)

    trial_ok = _FakeTrial(prune=False)
    rmse = optimizer.train_test_model(model, train_loader, test_loader, optimizer_t, criterion, 1.0, trial_ok, epochs=1)
    assert rmse >= 0.0
    assert trial_ok.reports

    trial_prune = _FakeTrial(prune=True)
    with pytest.raises(ocdnn.optuna.exceptions.TrialPruned):
        optimizer.train_test_model(model, train_loader, test_loader, optimizer_t, criterion, 1.0, trial_prune, epochs=1)

    x_mb, y_mb = _multibranch_data()
    mb_model = ocdnn.MultiBranchDynamicNN(
        input_size=[2, 1, 1],
        output_size=1,
        hidden_layers=[4],
        activation_data=[],
        encoders=[
            [("Linear", 2, 2)],
            [("Linear", 1, 1)],
            [("Linear", 1, 1)],
        ],
        device=torch.device("cpu"),
    )
    mb_train_loader = DataLoader(ocdnn.MultiBranchCustomDataset(x_mb[0], x_mb[1], x_mb[2], y_mb), batch_size=2, shuffle=False)
    mb_test_loader = DataLoader(ocdnn.MultiBranchCustomDataset(x_mb[0], x_mb[1], x_mb[2], y_mb), batch_size=2, shuffle=False)
    mb_optimizer = torch.optim.Adam(mb_model.parameters(), lr=1e-3)
    mb_rmse = optimizer.train_test_model(
        mb_model,
        mb_train_loader,
        mb_test_loader,
        mb_optimizer,
        criterion,
        1.0,
        _FakeTrial(prune=False),
        epochs=1,
    )
    assert mb_rmse >= 0.0
