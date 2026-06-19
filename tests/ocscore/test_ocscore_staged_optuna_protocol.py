#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for the current staged OCScore Optuna protocol.
'''

# Imports
###############################################################################
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

from OCDocker.OCScore.Optimization.Protocol import ProtocolContext, StagedProtocol
from OCDocker.OCScore.Optimization.StagedOptuna import DUDEzOptunaConfig, DUDEzOptunaStage
from OCDocker.OCScore.Optimization.StagedOptuna import PDBbindOptunaConfig, PDBbindOptunaStage
from OCDocker.OCScore.Optimization.StagedOptuna import TransferFeatureExtractorStage
from OCDocker.OCScore.Utils.DUDEzSplit import DUDEzSplitConfig
from OCDocker.OCScore.Utils.FixedOuterSplit import build_fixed_outer_split_assignment

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
    def __init__(self):
        self.categorical_calls = {}
        self.float_calls = {}
        self.int_calls = {}
        self.user_attrs = {}

    def suggest_categorical(self, name, choices):
        self.categorical_calls[name] = list(choices)
        if name == "encoder_depth":
            return 4
        if name.startswith("encoder_hidden_"):
            return max(choices)
        if name == "decoder_lambda_rec":
            return 0.0
        return choices[0]

    def suggest_float(self, name, low, high, log=False):  # noqa: ARG002
        self.float_calls[name] = (low, high, log)
        return low

    def suggest_int(self, name, low, high):  # noqa: ARG002
        self.int_calls[name] = (low, high)
        return low

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _RecorderStage:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def run(self, context):
        self.calls.append(self.name)
        context.stage_results[self.name] = {"called": True}
        return context


# Functions
###############################################################################
## Private ##

def _context(tmp_path):
    selected = [f"f{i}" for i in range(64)]
    pdbbind_df = _pdbbind_df(selected)
    dudez_df = _dudez_df(selected)
    fixed_outer_split = build_fixed_outer_split_assignment(
        outer_split_seed=7,
        pdbbind_train_indices=list(range(18)),
        pdbbind_validation_indices=list(range(18, 24)),
        pdbbind_test_indices=list(range(24, 30)),
        dudez_train_indices=list(range(43)),
        dudez_validation_indices=list(range(43, 58)),
        dudez_test_indices=list(range(58, 72)),
        feature_selection_fit_row_count=18,
        selected_features=selected,
        removed_features=[],
    )
    return ProtocolContext(
        pdbbind_df=pdbbind_df,
        dudez_df=dudez_df,
        selected_features=selected,
        output_dir=str(tmp_path),
        random_seed=7,
        metadata={
            "pdbbind_path": "synthetic_pdbbind",
            "dudez_path": "synthetic_dudez",
            "fixed_outer_split": fixed_outer_split.to_dict(),
        },
    )


def _dudez_df(selected):
    rows = []
    rng = np.random.default_rng(11)
    for target_idx in range(6):
        for row_idx in range(12):
            is_ligand = row_idx < 3
            values = rng.normal(loc=1.0 if is_ligand else -0.4, scale=0.2, size=len(selected))
            row = {feature: float(value) for feature, value in zip(selected, values)}
            row.update({
                "receptor": f"T{target_idx}",
                "ligand": f"L{target_idx}_{row_idx}",
                "name": f"dudez_{target_idx}_{row_idx}",
                "kind": "ligands" if is_ligand else "decoys",
                "db": "DUDEZ",
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _pdbbind_df(selected):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(30, len(selected)))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=30)
    df = pd.DataFrame(X, columns=selected)
    df["experimental"] = y
    df["receptor"] = [f"P{i % 5}" for i in range(len(df))]
    df["ligand"] = [f"LP{i}" for i in range(len(df))]
    df["name"] = [f"pdb_{i}" for i in range(len(df))]
    df["db"] = "PDBBIND"
    return df


def _fast_pdbbind_config():
    return PDBbindOptunaConfig(
        n_trials=1,
        epochs=1,
        storage=None,
        use_gpu=False,
        validation_size=0.2,
        test_size=0.2,
        study_name="pdbbind_test",
    )


def _fast_dudez_config():
    return DUDEzOptunaConfig(
        n_trials=1,
        epochs=1,
        storage=None,
        use_gpu=False,
        validation_size=0.2,
        test_size=0.2,
        study_name="dudez_test",
        split_config=DUDEzSplitConfig(
            strategy="receptor_stratified_kind",
            train_size=0.6,
            validation_size=0.2,
            test_size=0.2,
            random_seed=7,
            relaxed_split=False,
        ),
    )


def _run_pdbbind_transfer(tmp_path):
    context = _context(tmp_path)
    context = PDBbindOptunaStage(_fast_pdbbind_config()).run(context)
    context = TransferFeatureExtractorStage().run(context)
    return context


## Public ##

@pytest.mark.order(480)
def test_pdbbind_optuna_objective_optimizes_rmse_only(tmp_path):
    context = _context(tmp_path)
    result = PDBbindOptunaStage(_fast_pdbbind_config()).run(context)
    stage = result.stage_results["pdbbind_optuna"]

    assert stage["objective_metric"] == "RMSE"
    assert stage["direction"] == "minimize"
    assert "MAE" in stage["report_only_metrics"]
    assert "AUC" not in stage["objective_metric"]
    assert stage["validation_metrics"]["RMSE"] == pytest.approx(stage["best_value"])


@pytest.mark.order(480)
def test_dudez_trial_test_metrics_are_stored_on_optuna_trial():
    class _FakeTrial:
        def __init__(self) -> None:
            self.user_attrs: dict[str, object] = {}

        def set_user_attr(self, key: str, value: object) -> None:
            self.user_attrs[key] = value

    trial = _FakeTrial()
    metrics = {
        "BEDROC": 0.42,
        "ROC-AUC": 0.71,
        "score_std": 0.15,
        "n_unique_scores": 12.0,
        "ranking_metrics_valid": 1.0,
        "n_groups_total": 6.0,
        "n_groups_used": 6.0,
    }
    ocstaged._set_dudez_trial_test_metric_attrs(trial, metrics, "grouped", "BEDROC")

    assert trial.user_attrs["test_BEDROC"] == 0.42
    assert trial.user_attrs["test_ROC-AUC"] == 0.71
    assert trial.user_attrs["test_objective_metric_name"] == "BEDROC"
    assert trial.user_attrs["test_objective_metric_value"] == 0.42
    assert trial.user_attrs["test_n_groups_used"] == 6.0


@pytest.mark.order(481)
def test_dudez_optuna_objective_does_not_compute_test_metrics(tmp_path, monkeypatch):
    context = _run_pdbbind_transfer(tmp_path)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("test metrics must not be computed inside Optuna objective")

    monkeypatch.setattr(ocstaged, "_set_dudez_trial_test_metric_attrs", _forbidden)

    result = DUDEzOptunaStage(_fast_dudez_config()).run(context)
    stage = result.stage_results["dudez_optuna"]

    assert stage["test_metrics"]
    assert stage["validation_metrics"]


@pytest.mark.order(481)
def test_dudez_optuna_objective_does_not_access_pdbbind_rmse(tmp_path, monkeypatch):
    context = _run_pdbbind_transfer(tmp_path)

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("DUDEz stage must not evaluate PDBbind RMSE")

    monkeypatch.setattr(ocstaged, "evaluate_regression_metrics", _raise_if_called)
    result = DUDEzOptunaStage(_fast_dudez_config()).run(context)
    stage = result.stage_results["dudez_optuna"]

    assert stage["objective_metric"] in {"BEDROC", "PR-AUC"}
    assert "RMSE" not in stage["objective_metric"]
    assert "pdbbind_rmse" not in json.dumps(stage).lower()


@pytest.mark.order(482)
def test_invalid_mixed_objective_is_rejected(tmp_path):
    context = _context(tmp_path)
    bad_pdb = PDBbindOptunaConfig(objective_metric="RMSE - AUC", n_trials=1, epochs=1, storage=None, use_gpu=False)
    with pytest.raises(ValueError, match="Mixed regression/screening objective"):
        PDBbindOptunaStage(bad_pdb).run(context)

    bad_dudez = DUDEzOptunaConfig(primary_metric="RMSE - BEDROC", n_trials=1, epochs=1, storage=None, use_gpu=False)
    with pytest.raises(ValueError, match="Mixed regression/screening objective"):
        DUDEzOptunaStage(bad_dudez).run(context)

    bad_pdb_direction = PDBbindOptunaConfig(direction="maximize", n_trials=1, epochs=1, storage=None, use_gpu=False)
    with pytest.raises(ValueError, match="PDBbind Optuna direction must be minimize"):
        PDBbindOptunaStage(bad_pdb_direction).run(context)

    bad_dudez_direction = DUDEzOptunaConfig(direction="minimize", n_trials=1, epochs=1, storage=None, use_gpu=False)
    with pytest.raises(ValueError, match="DUDEz Optuna direction must be maximize"):
        DUDEzOptunaStage(bad_dudez_direction).run(context)


@pytest.mark.order(483)
def test_feature_extractor_transfer_replaces_final_regression_head(tmp_path):
    context = _run_pdbbind_transfer(tmp_path)
    config = _fast_dudez_config()
    config.allow_scratch = False
    dudez_context = DUDEzOptunaStage(config).run(context)

    pdbbind_model = dudez_context.artifacts["pdbbind_model"]
    dudez_model = dudez_context.artifacts["dudez_model"]
    transferred = dudez_context.artifacts["transferred_feature_extractor"]

    assert hasattr(pdbbind_model, "regression_head")
    assert hasattr(dudez_model, "classifier_head")
    assert not hasattr(dudez_model, "regression_head")
    assert set(transferred.state_dict().keys()) == set(dudez_model.feature_extractor.state_dict().keys())


@pytest.mark.order(484)
def test_pdbbind_checkpoint_is_not_overwritten_by_dudez_finetuning(tmp_path):
    context = _context(tmp_path)
    protocol = StagedProtocol([
        PDBbindOptunaStage(_fast_pdbbind_config()),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(_fast_dudez_config()),
    ])
    result = protocol.run(context)

    pdb_checkpoint = result.stage_results["pdbbind_optuna"]["checkpoint_path"]
    dudez_checkpoint = result.stage_results["dudez_optuna"]["checkpoint_path"]
    assert pdb_checkpoint != dudez_checkpoint
    assert pdb_checkpoint.endswith("pdbbind_best.pt")
    assert dudez_checkpoint.endswith("dudez_best.pt")


@pytest.mark.order(485)
def test_architecture_sampler_enforces_monotonic_structured_encoder_shapes():
    trial = _FakeTrial()
    arch_idx, hidden_sizes, latent_dim = ocstaged.suggest_encoder_architecture(trial, input_dim=64)

    assert hidden_sizes == sorted(hidden_sizes, reverse=True)
    assert all(size >= latent_dim for size in hidden_sizes)
    assert latent_dim in ocstaged.DEFAULT_PDBBIND_SEARCH_SPACE.encoder.latent_dim_options
    assert "encoder_architecture_index" in trial.int_calls
    assert arch_idx == 0


@pytest.mark.order(485)
def test_architecture_sampler_prunes_expanding_encoders():
    # Expansion is impossible-by-construction; assert a monotonic-by-rank sample never expands.
    trial = _FakeTrial()
    _arch_idx, hidden_sizes, latent_dim = ocstaged.suggest_encoder_architecture(trial, input_dim=512)
    assert hidden_sizes == sorted(hidden_sizes, reverse=True)
    assert all(size >= latent_dim for size in hidden_sizes)
    assert "encoder_hidden_rank_delta_1" not in trial.int_calls
    assert "encoder_latent_rank" not in trial.int_calls


@pytest.mark.order(486)
def test_encoder_architecture_has_static_optuna_categorical_space(tmp_path):
    study = ocstaged.optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{tmp_path}/architecture.db",
    )

    def objective(trial):
        _arch_idx, hidden_sizes, latent_dim = ocstaged.suggest_encoder_architecture(trial, input_dim=64)
        assert hidden_sizes == sorted(hidden_sizes, reverse=True)
        assert all(size >= latent_dim for size in hidden_sizes)
        return 0.0

    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2


@pytest.mark.order(487)
def test_optuna_sqlite_lock_retry_keeps_trial_budget(monkeypatch):
    class _LockedOnceStudy:
        def __init__(self):
            self.trials = []
            self.calls = []
            self.locked = False

        def optimize(self, objective, *, n_trials, n_jobs):
            self.calls.append((n_trials, n_jobs))
            if not self.locked:
                self.locked = True
                self.trials.extend([object(), object(), object()])
                raise RuntimeError("database is locked")
            self.trials.extend([object() for _ in range(n_trials)])

    study = _LockedOnceStudy()
    monkeypatch.setattr(ocstaged.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv(ocstaged.OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(ocstaged.OPTUNA_SQLITE_LOCK_RETRY_SECONDS_ENV, "0")

    ocstaged._optimize_with_sqlite_lock_retry(
        study,
        lambda _trial: 0.0,
        n_trials=10,
        n_jobs=2,
        stage_label="unit:optimize",
    )

    assert study.calls == [(10, 2), (7, 2)]
    assert len(study.trials) == 10


@pytest.mark.order(488)
def test_optuna_create_study_retries_sqlite_lock(monkeypatch):
    calls = []
    sentinel = object()

    def _create_study(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return sentinel

    monkeypatch.setattr(ocstaged.optuna, "create_study", _create_study)
    monkeypatch.setattr(ocstaged.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv(ocstaged.OPTUNA_SQLITE_LOCK_RETRY_ATTEMPTS_ENV, "3")
    monkeypatch.setenv(ocstaged.OPTUNA_SQLITE_LOCK_RETRY_SECONDS_ENV, "0")

    result = ocstaged._create_study_with_sqlite_lock_retry(
        "unit:create_study",
        direction="minimize",
        study_name="study",
    )

    assert result is sentinel
    assert len(calls) == 2


@pytest.mark.order(489)
def test_optuna_retry_does_not_mask_non_lock_errors():
    class _BrokenStudy:
        trials = []

        def optimize(self, objective, *, n_trials, n_jobs):
            raise RuntimeError("invalid model")

    with pytest.raises(RuntimeError, match="invalid model"):
        ocstaged._optimize_with_sqlite_lock_retry(
            _BrokenStudy(),
            lambda _trial: 0.0,
            n_trials=3,
            n_jobs=1,
            stage_label="unit:optimize",
        )


@pytest.mark.order(490)
def test_optuna_search_space_uses_structured_categorical_layer_sizes():
    summary = ocstaged.pdbbind_search_space_summary()
    assert summary["encoder_hidden_size_options"] == list(
        ocstaged.DEFAULT_PDBBIND_SEARCH_SPACE.encoder.hidden_size_options
    )
    assert summary["encoder_latent_dim_options"] == list(
        ocstaged.DEFAULT_PDBBIND_SEARCH_SPACE.encoder.latent_dim_options
    )

    trial = _FakeTrial()
    params = ocstaged.suggest_pdbbind_trial_params(trial, input_dim=64)
    assert all(
        size in ocstaged.DEFAULT_PDBBIND_SEARCH_SPACE.encoder.hidden_size_options
        or size in ocstaged._encoder_allowed_hidden_sizes(64, min_hidden=32)
        for size in params["encoder_hidden_sizes"]
    )
    assert "encoder_architecture_index" in trial.int_calls
    assert "encoder_latent_rank" not in trial.int_calls


@pytest.mark.order(486)
def test_encoder_allowed_hidden_sizes_use_lower_pow2_ladder():
    assert ocstaged._encoder_allowed_hidden_sizes(360, min_hidden=32) == [256, 128, 64, 32]
    assert ocstaged._encoder_allowed_hidden_sizes(600, min_hidden=32) == [512, 256, 128, 64, 32]


@pytest.mark.order(486)
def test_encoder_small_input_candidates_use_adaptive_compression_ladder():
    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=28,
        depth_options=[2, 3, 4],
        latent_dim_options=[8, 16, 32, 64, 128],
        min_hidden=32,
        plateaus_allowed=True,
        max_hidden_layers=4,
    )

    assert ocstaged._encoder_allowed_hidden_sizes(28, min_hidden=32) == []
    assert ocstaged._encoder_small_input_hidden_sizes(28, min_hidden=32) == [16, 8, 4, 2]
    assert candidates

    hidden_options = {tuple(candidate["encoder_hidden_sizes"]) for candidate in candidates}
    assert (16,) in hidden_options
    assert (16, 8) in hidden_options
    assert (16, 8, 4) in hidden_options
    assert all(
        all(left > right for left, right in zip(hidden, hidden[1:]))
        for hidden in hidden_options
    )
    assert all(
        int(candidate["encoder_latent_dim"]) <= int(candidate["encoder_hidden_sizes"][-1])
        for candidate in candidates
    )


@pytest.mark.order(486)
def test_encoder_standard_candidates_do_not_add_small_input_fallbacks():
    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=600,
        depth_options=[2, 3, 4],
        latent_dim_options=[8, 16, 32, 64, 128],
        min_hidden=32,
        plateaus_allowed=True,
        max_hidden_layers=4,
    )

    assert candidates
    assert all(
        int(size) >= 32
        for candidate in candidates
        for size in candidate["encoder_hidden_sizes"]
    )


@pytest.mark.order(486)
def test_encoder_candidate_generation_is_monotonic_and_latent_is_bounded():
    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=600,
        depth_options=[2, 3],
        latent_dim_options=[8, 16, 32, 64, 128],
        min_hidden=32,
        plateaus_allowed=True,
        max_hidden_layers=4,
    )
    assert candidates
    for cand in candidates:
        hidden = list(cand["encoder_hidden_sizes"])
        depth = int(cand["encoder_depth"])
        latent = int(cand["encoder_latent_dim"])
        assert hidden == sorted(hidden, reverse=True)
        assert latent <= int(hidden[-1])
        assert [32, 512] != hidden
        assert all(int(v) > 0 for v in hidden)
        assert ocstaged._hidden_sizes_respect_equal_chain_limit(hidden, depth=depth)
        if depth == 2:
            assert hidden[0] != hidden[1]


@pytest.mark.order(486)
def test_encoder_candidates_reject_long_equal_chains():
    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=600,
        depth_options=[3],
        latent_dim_options=[32],
        min_hidden=32,
        plateaus_allowed=True,
    )
    assert candidates
    assert not any(
        list(c["encoder_hidden_sizes"]) == [256, 256, 256]
        for c in candidates
    )
    assert any(
        list(c["encoder_hidden_sizes"]) == [256, 256, 128]
        for c in candidates
    )


@pytest.mark.order(486)
def test_encoder_plateau_block_rules():
    assert ocstaged._count_equal_plateau_blocks([256, 128, 128, 64]) == 1
    assert ocstaged._count_equal_plateau_blocks([256, 256, 128, 128]) == 2
    assert ocstaged._hidden_sizes_respect_equal_chain_limit(
        [256, 128, 128, 64],
        depth=4,
    )
    assert not ocstaged._hidden_sizes_respect_equal_chain_limit(
        [256, 256, 128, 128],
        depth=4,
    )

    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=600,
        depth_options=[4],
        latent_dim_options=[32],
        min_hidden=32,
        plateaus_allowed=True,
    )
    assert any(
        list(c["encoder_hidden_sizes"]) == [256, 128, 128, 64]
        for c in candidates
    )
    assert not any(
        list(c["encoder_hidden_sizes"]) == [256, 256, 128, 128]
        for c in candidates
    )


@pytest.mark.order(486)
def test_pdbbind_trial_params_include_readable_encoder_fields_and_user_attrs():
    trial = _FakeTrial()
    params = ocstaged.suggest_pdbbind_trial_params(trial, input_dim=360)
    assert "encoder_architecture_index" in params
    assert "encoder_hidden_sizes" in params
    assert "encoder_latent_dim" in params
    assert "encoder_depth" in params
    assert "encoder_is_monotonic" in params
    assert trial.user_attrs.get("encoder_architecture_index") == params["encoder_architecture_index"]
    assert trial.user_attrs.get("encoder_hidden_sizes") == params["encoder_hidden_sizes"]
    assert trial.user_attrs.get("encoder_latent_dim") == params["encoder_latent_dim"]
    assert trial.user_attrs.get("encoder_depth") == params["encoder_depth"]
    assert trial.user_attrs.get("encoder_is_monotonic") is True


@pytest.mark.order(488)
def test_lambda_rec_zero_disables_reconstruction_loss_cleanly():
    prediction = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[1.5], [2.5]])
    features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    reconstruction = torch.full_like(features, 100.0)
    criterion = nn.MSELoss()

    loss = ocstaged.compute_regression_reconstruction_loss(
        prediction,
        target,
        reconstruction,
        features,
        criterion,
        criterion,
        lambda_rec=0.0,
    )

    assert loss.item() == pytest.approx(criterion(prediction, target).item())

    params = {
        "encoder_hidden_sizes": [32, 16],
        "encoder_latent_dim": 8,
        "projection_dim": 0,
        "encoder_activation": "ReLU",
        "encoder_dropout": 0.0,
        "decoder_lambda_rec": 0.0,
        "decoder_hidden_sizes": None,
        "decoder_depth": 0,
    }
    model = ocstaged.build_pdbbind_model(input_size=2, params=params)
    assert model.decoder is None


@pytest.mark.order(489)
def test_metadata_columns_are_rejected_as_selected_model_features(tmp_path):
    context = _context(tmp_path)
    context.selected_features = ["experimental", *context.selected_features[1:]]

    with pytest.raises(ValueError, match="metadata/target columns"):
        PDBbindOptunaStage(_fast_pdbbind_config()).run(context)


@pytest.mark.order(490)
def test_selected_features_are_consumed_consistently_by_both_tasks(tmp_path):
    context = _context(tmp_path)
    result = StagedProtocol([
        PDBbindOptunaStage(_fast_pdbbind_config()),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(_fast_dudez_config()),
    ]).run(context)

    assert "selected_features" not in result.stage_results["pdbbind_optuna"]
    assert "selected_features" not in result.stage_results["dudez_optuna"]
    pdb_payload = torch.load(
        result.stage_results["pdbbind_optuna"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    dudez_payload = torch.load(
        result.stage_results["dudez_optuna"]["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    assert pdb_payload["selected_features"] == context.selected_features
    assert dudez_payload["selected_features"] == context.selected_features


@pytest.mark.order(491)
def test_metrics_are_reported_separately_by_task(tmp_path):
    context = _context(tmp_path)
    result = StagedProtocol([
        PDBbindOptunaStage(_fast_pdbbind_config()),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(_fast_dudez_config()),
    ]).run(context)

    pdb_metrics = result.stage_results["pdbbind_optuna"]["validation_metrics"]
    dudez_metrics = result.stage_results["dudez_optuna"]["validation_metrics"]
    assert {"RMSE", "MAE", "Pearson r", "Spearman rho", "R2"}.issubset(pdb_metrics)
    assert {"ROC-AUC", "PR-AUC", "BEDROC", "EF1%", "EF5%", "NDCG@1%", "NDCG@5%"}.issubset(dudez_metrics)
    assert "RMSE" not in dudez_metrics
    assert "ROC-AUC" not in pdb_metrics


@pytest.mark.order(492)
def test_reproducibility_log_contains_protocol_metadata(tmp_path):
    context = _context(tmp_path)
    result = StagedProtocol([
        PDBbindOptunaStage(_fast_pdbbind_config()),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(_fast_dudez_config()),
    ]).run(context)

    log_path = tmp_path / "protocol_log.json"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["random_seed"] == 7
    assert "selected_features" not in payload
    assert "feature_selection" not in payload
    assert payload["checkpoints"]["pdbbind"] == result.stage_results["pdbbind_optuna"]["checkpoint_path"]
    assert payload["checkpoints"]["dudez"] == result.stage_results["dudez_optuna"]["checkpoint_path"]
    assert result.stage_results["pdbbind_optuna"]["search_space"]
    assert result.stage_results["dudez_optuna"]["search_space"]
    assert result.stage_results["pdbbind_optuna"]["best_trial"] == 0
    assert result.stage_results["dudez_optuna"]["best_trial"] == 0


@pytest.mark.order(493)
def test_protocol_context_passes_artifacts_between_stages(tmp_path):
    context = _context(tmp_path)
    result = StagedProtocol([
        PDBbindOptunaStage(_fast_pdbbind_config()),
        TransferFeatureExtractorStage(),
        DUDEzOptunaStage(_fast_dudez_config()),
    ]).run(context)

    assert "pdbbind_model" in result.artifacts
    assert "transferred_feature_extractor" in result.artifacts
    assert "dudez_model" in result.artifacts
    assert result.stage_results["transfer_feature_extractor"]["excluded_components"] == ["regression_head", "decoder"]


@pytest.mark.order(494)
def test_staged_protocol_calls_stages_and_not_optuna_owning_full_workflow(tmp_path):
    calls = []
    context = _context(tmp_path)
    protocol = StagedProtocol([
        _RecorderStage("one", calls),
        _RecorderStage("two", calls),
    ], write_protocol_log=False)
    result = protocol.run(context)

    assert calls == ["one", "two"]
    assert result.stage_results["one"]["called"] is True
    assert result.stage_results["two"]["called"] is True


@pytest.mark.order(495)
def test_archived_optuna_namespace_removed():
    '''Legacy four-study Optuna modules were dropped from the package.'''

    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("OCDocker.OCScore.Optimization.legacy")


@pytest.mark.order(496)
def test_bedroc_missing_falls_back_to_pr_auc():
    metrics = {"BEDROC": float("nan"), "PR-AUC": 0.42}
    assert ocstaged.resolve_dudez_primary_metric("BEDROC", metrics) == "PR-AUC"

@pytest.mark.order(497)
def test_fixed_outer_split_metadata_accepts_bool_plus_payload(tmp_path):
    context = _context(tmp_path)
    payload = context.metadata["fixed_outer_split"]
    context.metadata["fixed_outer_split"] = True
    context.metadata["fixed_outer_split_metadata"] = payload

    fixed = ocstaged._require_fixed_outer_split(context)

    assert fixed.selected_features_hash == payload["selected_features_hash"]


@pytest.mark.order(498)
def test_fixed_outer_split_metadata_rejects_bool_without_payload(tmp_path):
    context = _context(tmp_path)
    context.metadata["fixed_outer_split"] = True
    context.metadata.pop("fixed_outer_split_metadata", None)

    with pytest.raises(ValueError, match="fixed_outer_split metadata"):
        ocstaged._require_fixed_outer_split(context)
