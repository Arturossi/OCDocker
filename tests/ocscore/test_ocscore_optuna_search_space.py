#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for centralized staged Optuna search-space configuration."""

# Imports
###############################################################################
import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

import OCDocker.OCScore.Optimization.OptunaSearchSpace as ocsearch
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

from OCDocker.OCScore.Optimization.OptunaSearchSpace import DecoderSearchSpace
from OCDocker.OCScore.Optimization.OptunaSearchSpace import EncoderSearchSpace
from OCDocker.OCScore.Optimization.OptunaSearchSpace import PDBbindSearchSpaceConfig
from OCDocker.OCScore.Optimization.OptunaSearchSpace import pdbbind_search_space_for_phase
from OCDocker.OCScore.Optimization.OptunaSearchSpace import validate_pdbbind_search_phase

# License
###############################################################################
"""
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
"""


# Classes
###############################################################################

class _FakeTrial:
    def __init__(self):
        self.categorical_calls = {}
        self.float_calls = {}
        self.int_calls = {}

    def suggest_categorical(self, name, choices):
        self.categorical_calls[name] = list(choices)
        if name == "encoder_depth":
            return 3
        if name == "encoder_latent_dim":
            return 16
        if name.startswith("encoder_hidden_"):
            return max(choices)
        if name.startswith("decoder_hidden_"):
            return max(choices)
        if name == "decoder_lambda_rec":
            return 0.1
        if name == "decoder_depth":
            return 2
        return choices[0]

    def suggest_float(self, name, low, high, log=False):  # noqa: ARG002
        self.float_calls[name] = (low, high, log)
        return low

    def suggest_int(self, name, low, high):  # noqa: ARG002
        self.int_calls[name] = (low, high)
        return low


# Functions
###############################################################################
## Public ##

@pytest.mark.order(264)
def test_expanded_activation_options_are_accepted():
    for name in ocsearch.DEFAULT_ACTIVATION_OPTIONS:
        assert ocsearch.activation_is_available(name)
        assert isinstance(ocsearch.build_activation_module(name), nn.Module)


@pytest.mark.order(265)
def test_optuna_parameter_names_use_clear_prefixes():
    trial = _FakeTrial()
    custom_space = PDBbindSearchSpaceConfig(
        activation_options=("ReLU", "GELU"),
        encoder=EncoderSearchSpace(hidden_size_options=(64, 32), depth_options=(2,)),
        decoder=DecoderSearchSpace(
            lambda_rec_options=(0.0, 0.1),
            hidden_size_options=(16, 32, 64),
            depth_options=(2,),
        ),
    )
    params = ocstaged.suggest_pdbbind_trial_params(trial, input_dim=64, search_space=custom_space)

    assert "encoder_architecture_index" in trial.int_calls
    assert "encoder_activation" in trial.categorical_calls
    assert "optimizer_learning_rate" in trial.float_calls
    assert "decoder_lambda_rec" in trial.categorical_calls
    assert "pdbbind_regression_loss" in trial.categorical_calls
    assert params["decoder_lambda_rec"] == 0.1
    assert params["decoder_hidden_sizes"] is not None
    if params["decoder_lambda_rec"] > 0.0:
        assert "decoder_architecture_index" in trial.int_calls


@pytest.mark.order(266)
def test_encoder_expansion_is_rejected_by_monotonic_sampler():
    candidates = ocstaged._encoder_architecture_candidates(
        input_dim=128,
        depth_options=(3,),
        latent_dim_options=(16,),
        min_hidden=32,
        plateaus_allowed=True,
    )
    assert candidates
    for candidate in candidates:
        hidden_sizes = list(candidate["encoder_hidden_sizes"])
        assert hidden_sizes == sorted(hidden_sizes, reverse=True)
        assert all(left >= right for left, right in zip(hidden_sizes, hidden_sizes[1:]))


@pytest.mark.order(267)
def test_decoder_expansion_is_allowed():
    class _ExpandingDecoderTrial(_FakeTrial):
        def suggest_categorical(self, name, choices):
            self.categorical_calls[name] = list(choices)
            if name == "decoder_lambda_rec":
                return 0.1
            if name == "decoder_depth":
                return 2
            if name == "decoder_hidden_1":
                return min(choices)
            if name == "decoder_hidden_2":
                return max(choices)
            return choices[0]

    trial = _ExpandingDecoderTrial()
    decoder_sizes, lambda_rec = ocstaged.suggest_decoder_hidden_sizes(
        trial,
        search_space=DecoderSearchSpace(
            hidden_size_options=(8, 32, 64),
            depth_options=(2,),
            lambda_rec_options=(0.1,),
        ),
        latent_dim=8,
        projection_dim=0,
        input_dim=64,
    )
    assert lambda_rec == 0.1
    assert decoder_sizes is not None
    assert decoder_sizes[-1] > decoder_sizes[0]


@pytest.mark.order(268)
def test_search_space_values_can_be_changed_from_central_config():
    custom = PDBbindSearchSpaceConfig(
        encoder=EncoderSearchSpace(latent_dim_options=(4, 8)),
        optimizer=ocsearch.OptimizerSearchSpace(batch_size_options=(16,)),
    )
    summary = ocsearch.search_space_to_summary(custom)
    assert summary["encoder_latent_dim_options"] == [4, 8]
    assert summary["optimizer_batch_size_options"] == [16]


@pytest.mark.order(269)
def test_lambda_rec_zero_disables_decoder_and_reconstruction_loss():
    class _ZeroLambdaTrial(_FakeTrial):
        def suggest_categorical(self, name, choices):
            self.categorical_calls[name] = list(choices)
            if name == "decoder_lambda_rec":
                return 0.0
            return choices[0]

    trial = _ZeroLambdaTrial()
    decoder_sizes, lambda_rec = ocstaged.suggest_decoder_hidden_sizes(
        trial,
        search_space=DecoderSearchSpace(lambda_rec_options=(0.0, 0.1)),
        latent_dim=8,
        projection_dim=0,
        input_dim=64,
    )
    assert lambda_rec == 0.0
    assert decoder_sizes is None


@pytest.mark.order(269)
def test_projection_does_not_expand_encoder_and_decoder_remains_expanding():
    class _ProjectionExpansionTrial(_FakeTrial):
        def __init__(self):
            super().__init__()
            self.user_attrs = {}

        def suggest_categorical(self, name, choices):
            self.categorical_calls[name] = list(choices)
            if name == "projection_dim":
                return 128
            if name == "decoder_lambda_rec":
                return 0.2
            return choices[0]

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    assert ocstaged._decoder_architecture_candidates(
        decoder_start_dim=128,
        input_dim=28,
        depth_options=(1, 2, 3),
        hidden_size_options=(8, 16, 32, 64, 128, 256, 512),
    ) == []

    trial = _ProjectionExpansionTrial()
    params = ocstaged.suggest_pdbbind_trial_params(
        trial,
        input_dim=28,
        search_space=PDBbindSearchSpaceConfig(
            projection=ocsearch.ProjectionSearchSpace(projection_dim_options=(0, 128)),
            decoder=DecoderSearchSpace(lambda_rec_options=(0.0, 0.2)),
        ),
    )

    assert trial.categorical_calls["projection_dim"] == [0, 128]
    assert trial.user_attrs["projection_dim_sampled"] == 128
    assert trial.user_attrs["projection_dim_effective"] == 0
    assert trial.user_attrs["projection_disabled_reason"] == "projection_would_expand_encoder"
    assert params["projection_dim"] == 0
    assert params["decoder_lambda_rec"] == 0.2
    assert params["decoder_hidden_sizes"] is not None
    assert params["decoder_hidden_sizes"] == sorted(params["decoder_hidden_sizes"])
    assert trial.user_attrs["decoder_is_monotonic_increasing"] is True
    assert trial.user_attrs["decoder_is_monotonic_decreasing"] is False

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


@pytest.mark.order(267)
def test_encoder_regression_phase_search_space_is_restricted():
    phase_space = pdbbind_search_space_for_phase("encoder_regression")
    assert phase_space.encoder.latent_dim_options == (16, 32, 64)
    assert phase_space.encoder.depth_options == (2, 3)
    assert phase_space.projection.projection_dim_options == (0,)
    assert phase_space.decoder.lambda_rec_options == (0.0,)
    with pytest.raises(ValueError, match="Unknown PDBbind search phase"):
        validate_pdbbind_search_phase("decoder_only")


@pytest.mark.order(268)
def test_encoder_regression_phase_skips_decoder_and_dae_sampling():
    class _Phase1Trial(_FakeTrial):
        def suggest_categorical(self, name, choices):
            self.categorical_calls[name] = list(choices)
            return choices[0]

        def suggest_int(self, name, low, high):  # noqa: ARG002
            self.int_calls[name] = (low, high)
            if name == "encoder_architecture_index":
                return 0
            return low

    trial = _Phase1Trial()
    space = pdbbind_search_space_for_phase("encoder_regression")
    params = ocstaged.suggest_pdbbind_trial_params(trial, input_dim=360, search_space=space)

    assert params["decoder_lambda_rec"] == 0.0
    assert params["projection_dim"] == 0
    assert params["decoder_hidden_sizes"] is None
    assert params["dae_noise_type"] == "none"
    assert "decoder_architecture_index" not in trial.int_calls
    assert "dae_mask_prob" not in trial.float_calls


@pytest.mark.order(269)
def test_median_pruner_default_thresholds():
    assert ocstaged._median_pruner_n_startup_trials_default(15) == 5
    assert ocstaged._median_pruner_n_startup_trials_default(100) == 10
    assert ocstaged._median_pruner_n_warmup_steps_default(100) == 10
    assert ocstaged._median_pruner_n_warmup_steps_default(50) == 10

    settings = ocstaged._resolve_pdbbind_pruner_settings(
        ocstaged.PDBbindOptunaConfig(
            n_trials=15,
            epochs=100,
            enable_pruning=True,
        )
    )
    assert settings == {
        "type": "MedianPruner",
        "n_startup_trials": 5,
        "n_warmup_steps": 10,
    }


@pytest.mark.order(270)
def test_pdbbind_phase1_experiment_config_defaults():
    optuna = pytest.importorskip("optuna")
    config = ocstaged.pdbbind_phase1_experiment_config()
    assert config.search_phase == "encoder_regression"
    assert config.n_trials == 40
    assert config.enable_pruning is False
    assert config.study_name == "PDBbind_EncoderRegression_Phase1"
    pruner = ocstaged._build_pdbbind_pruner(config)
    assert isinstance(pruner, optuna.pruners.NopPruner)
