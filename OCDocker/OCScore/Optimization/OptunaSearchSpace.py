#!/usr/bin/env python3

# Description
###############################################################################
'''
Centralized Optuna search-space definitions for staged OCScore optimization.

Edit the dataclasses in this module to expand or restrict hyperparameter search
spaces without modifying sampler logic in ``StagedOptuna.py``.
'''

# Imports
###############################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch.nn as nn

import OCDocker.Error as ocerror

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

PDBBIND_SEARCH_PHASE_FULL = "full"
PDBBIND_SEARCH_PHASE_ENCODER_REGRESSION = "encoder_regression"
PDBBIND_SEARCH_PHASES: tuple[str, ...] = (
    PDBBIND_SEARCH_PHASE_FULL,
    PDBBIND_SEARCH_PHASE_ENCODER_REGRESSION,
)

DEFAULT_ACTIVATION_OPTIONS: tuple[str, ...] = (
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "SiLU",
    "Mish",
)

_ACTIVATION_FACTORIES: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
}


# Classes
###############################################################################

@dataclass
class OptimizerSearchSpace:
    """Optimizer and training batch search space.

    Parameters
    ----------
    learning_rate_min : float, optional
        Minimum learning rate, by default ``1e-5``.
    learning_rate_max : float, optional
        Maximum learning rate, by default ``1e-3``.
    weight_decay_min : float, optional
        Minimum weight decay, by default ``1e-6``.
    weight_decay_max : float, optional
        Maximum weight decay, by default ``1e-3``.
    batch_size_options : tuple[int, ...], optional
        Batch-size candidates, by default ``(32, 64, 128, 256)``.
    """

    learning_rate_min: float = 1e-5
    learning_rate_max: float = 1e-3
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-3
    batch_size_options: tuple[int, ...] = (32, 64, 128, 256)


@dataclass
class EncoderSearchSpace:
    """Encoder / feature-extractor search space.

    The encoder is constrained to be monotonic (non-increasing widths). Same-size
    plateaus are allowed; expansion between encoder layers is not sampled.

    Parameters
    ----------
    hidden_size_options : tuple[int, ...], optional
        Candidate hidden widths, by default ``(32, 64, 128, 256, 512)``.
    depth_options : tuple[int, ...], optional
        Candidate encoder depths, by default ``(2, 3, 4)``.
    latent_dim_options : tuple[int, ...], optional
        Candidate latent dimensions, by default ``(8, 16, 32, 64, 128)``.
    dropout_min : float, optional
        Minimum encoder dropout, by default ``0.0``.
    dropout_max : float, optional
        Maximum encoder dropout, by default ``0.3``.
    max_hidden_layers : int, optional
        Maximum number of ``encoder_hidden_*`` Optuna parameters, by default ``4``.
    """

    hidden_size_options: tuple[int, ...] = (32, 64, 128, 256, 512)
    depth_options: tuple[int, ...] = (2, 3, 4)
    latent_dim_options: tuple[int, ...] = (8, 16, 32, 64, 128)
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    max_hidden_layers: int = 4


@dataclass
class ProjectionSearchSpace:
    """Projection block search space after the encoder.

    Parameters
    ----------
    projection_dim_options : tuple[int, ...], optional
        Candidate projection dimensions. ``0`` disables the projection block.
    """

    projection_dim_options: tuple[int, ...] = (0, 16, 32, 64, 128)


@dataclass
class DecoderSearchSpace:
    """Optional PDBbind reconstruction decoder search space.

    The decoder is a PDBbind-only auxiliary branch. It is not transferred to
    DUDEz. Decoder hidden layers may expand toward the input dimension.

    Parameters
    ----------
    depth_options : tuple[int, ...], optional
        Candidate decoder depths, by default ``(1, 2, 3)``.
    hidden_size_options : tuple[int, ...], optional
        Candidate explicit decoder hidden widths.
    lambda_rec_options : tuple[float, ...], optional
        Reconstruction-loss weights. ``0.0`` disables the decoder.
    """

    depth_options: tuple[int, ...] = (1, 2, 3)
    hidden_size_options: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    lambda_rec_options: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)


@dataclass
class PDBbindHeadSearchSpace:
    """PDBbind regression-head search space.

    Parameters
    ----------
    regression_loss_options : tuple[str, ...], optional
        Regression loss candidates, by default ``("huber", "mse")``.
    huber_delta_min : float, optional
        Minimum Huber delta when Huber loss is selected.
    huber_delta_max : float, optional
        Maximum Huber delta when Huber loss is selected.
    """

    regression_loss_options: tuple[str, ...] = ("huber", "mse")
    huber_delta_min: float = 0.1
    huber_delta_max: float = 2.0


@dataclass
class DUDEzHeadSearchSpace:
    """DUDEz classifier-head and transfer search space.

    Parameters
    ----------
    classifier_hidden_size_options : tuple[int, ...], optional
        Classifier hidden-size candidates.
    classifier_dropout_min : float, optional
        Minimum classifier dropout.
    classifier_dropout_max : float, optional
        Maximum classifier dropout.
    fine_tuning_mode_options : tuple[str, ...], optional
        Feature-extractor fine-tuning modes.
    num_unfrozen_layers_options : tuple[int, ...], optional
        Candidate numbers of unfrozen encoder layers in partial mode.
    use_transfer_options : tuple[bool, ...], optional
        Whether from-scratch extractors are allowed when ``allow_scratch`` is True.
    use_class_weighting_options : tuple[bool, ...], optional
        Class-weighting candidates when class weighting is tunable.
    """

    classifier_hidden_size_options: tuple[int, ...] = (32, 64, 128, 256)
    classifier_dropout_min: float = 0.0
    classifier_dropout_max: float = 0.3
    fine_tuning_mode_options: tuple[str, ...] = ("frozen", "partial", "full")
    num_unfrozen_layers_options: tuple[int, ...] = (1, 2, 3)
    use_transfer_options: tuple[bool, ...] = (True, False)
    use_class_weighting_options: tuple[bool, ...] = (True, False)


@dataclass
class SharedNeuralSearchSpace:
    """Search-space blocks shared by PDBbind and DUDEz stages.

    Parameters
    ----------
    activation_options : tuple[str, ...], optional
        Activation candidates for encoder/projection/decoder/classifier blocks.
    encoder : EncoderSearchSpace, optional
        Encoder search space.
    projection : ProjectionSearchSpace, optional
        Projection-block search space.
    optimizer : OptimizerSearchSpace, optional
        Optimizer search space.
    """

    activation_options: tuple[str, ...] = DEFAULT_ACTIVATION_OPTIONS
    encoder: EncoderSearchSpace = field(default_factory=EncoderSearchSpace)
    projection: ProjectionSearchSpace = field(default_factory=ProjectionSearchSpace)
    optimizer: OptimizerSearchSpace = field(default_factory=OptimizerSearchSpace)


@dataclass
class PDBbindSearchSpaceConfig(SharedNeuralSearchSpace):
    """Full PDBbind staged Optuna search space.

    Parameters
    ----------
    decoder : DecoderSearchSpace, optional
        Optional reconstruction-decoder search space.
    pdbbind_head : PDBbindHeadSearchSpace, optional
        Regression-head search space.
    """

    decoder: DecoderSearchSpace = field(default_factory=DecoderSearchSpace)
    pdbbind_head: PDBbindHeadSearchSpace = field(default_factory=PDBbindHeadSearchSpace)


@dataclass
class DUDEzSearchSpaceConfig(SharedNeuralSearchSpace):
    """Full DUDEz staged Optuna search space.

    Parameters
    ----------
    dudez_head : DUDEzHeadSearchSpace, optional
        Classifier-head and transfer search space.
    """

    dudez_head: DUDEzHeadSearchSpace = field(default_factory=DUDEzHeadSearchSpace)


# Functions
###############################################################################
## Public ##

def validate_pdbbind_search_phase(phase: str) -> str:
    '''Return a normalized PDBbind search-phase name.

    Parameters
    ----------
    phase : str
        Requested phase (``full`` or ``encoder_regression``).

    Returns
    -------
    str
        Validated phase string.

    Raises
    ------
    ValueError
        If the phase is unknown.
    '''

    normalized = str(phase).strip().lower()
    if normalized not in PDBBIND_SEARCH_PHASES:
        raise ValueError(
            f"Unknown PDBbind search phase {phase!r}. "
            f"Expected one of: {list(PDBBIND_SEARCH_PHASES)}"
        )
    return normalized


def pdbbind_search_space_for_phase(
        phase: str,
        *,
        base: PDBbindSearchSpaceConfig | None = None,
    ) -> PDBbindSearchSpaceConfig:
    '''Build a PDBbind search space for a staged search phase.

    Parameters
    ----------
    phase : str
        ``full`` uses the default wide space. ``encoder_regression`` restricts
        PDBbind to encoder + regression only (no decoder, DAE, or projection).
    base : PDBbindSearchSpaceConfig | None, optional
        Optional base configuration copied before phase overrides.

    Returns
    -------
    PDBbindSearchSpaceConfig
        Phase-specific search-space configuration.
    '''

    validated = validate_pdbbind_search_phase(phase)
    if validated == PDBBIND_SEARCH_PHASE_FULL:
        return base or PDBbindSearchSpaceConfig()

    root = base or PDBbindSearchSpaceConfig()
    encoder = EncoderSearchSpace(
        hidden_size_options=root.encoder.hidden_size_options,
        depth_options=(2, 3),
        latent_dim_options=(16, 32, 64),
        dropout_min=root.encoder.dropout_min,
        dropout_max=root.encoder.dropout_max,
        max_hidden_layers=min(int(root.encoder.max_hidden_layers), 3),
    )
    return PDBbindSearchSpaceConfig(
        activation_options=root.activation_options,
        encoder=encoder,
        projection=ProjectionSearchSpace(projection_dim_options=(0,)),
        optimizer=root.optimizer,
        decoder=DecoderSearchSpace(
            depth_options=root.decoder.depth_options,
            hidden_size_options=root.decoder.hidden_size_options,
            lambda_rec_options=(0.0,),
        ),
        pdbbind_head=root.pdbbind_head,
    )


def available_activation_options(
        requested: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
    '''Return activation names supported by the installed PyTorch build.

    Parameters
    ----------
    requested : Sequence[str] | None, optional
        Candidate activation names. Defaults to :data:`DEFAULT_ACTIVATION_OPTIONS`.

    Returns
    -------
    tuple[str, ...]
        Activation names that can be constructed at runtime.

    Raises
    ------
    ValueError
        If no requested activation is available.
    '''

    candidates = tuple(requested or DEFAULT_ACTIVATION_OPTIONS)
    available = tuple(name for name in candidates if activation_is_available(name))
    if not available:
        raise ValueError(
            "No requested activation functions are available in the installed PyTorch build: "
            f"{list(candidates)}"
        )
    return available


def activation_is_available(name: str) -> bool:
    '''Return whether one activation name is supported.

    Parameters
    ----------
    name : str
        Activation name.

    Returns
    -------
    bool
        True when the activation can be constructed.
    '''

    try:
        build_activation_module(name)
    except ValueError:
        return False
    return True


def build_activation_module(name: str) -> nn.Module:
    '''Build one activation module from a centralized search-space name.

    Parameters
    ----------
    name : str
        Activation name from :data:`DEFAULT_ACTIVATION_OPTIONS`.

    Returns
    -------
    torch.nn.Module
        Instantiated activation module.

    Raises
    ------
    ValueError
        If the activation is unknown or unavailable in the installed PyTorch build.
    '''

    if name == "Mish":
        if hasattr(nn, "Mish"):
            return nn.Mish()
        _ = ocerror.Error.value_error(
            "Mish activation requires PyTorch with torch.nn.Mish available (PyTorch >= 1.9)."
        )
        raise ValueError("Mish activation is not available in the installed PyTorch build.")

    factory = _ACTIVATION_FACTORIES.get(name)
    if factory is None:
        _ = ocerror.Error.value_error(f"Unsupported activation function: {name}")
        raise ValueError(f"Unsupported activation function: {name}")
    return factory()


def search_space_to_summary(space: SharedNeuralSearchSpace) -> dict[str, Any]:
    '''Convert one search-space dataclass into a JSON-compatible summary.

    Parameters
    ----------
    space : SharedNeuralSearchSpace
        Search-space configuration object.

    Returns
    -------
    dict[str, Any]
        JSON-compatible search-space summary.
    '''

    summary: dict[str, Any] = {
        "activation_options": list(available_activation_options(space.activation_options)),
        "encoder_hidden_size_options": list(space.encoder.hidden_size_options),
        "encoder_depth_options": list(space.encoder.depth_options),
        "encoder_latent_dim_options": list(space.encoder.latent_dim_options),
        "encoder_dropout": [space.encoder.dropout_min, space.encoder.dropout_max],
        "projection_dim_options": list(space.projection.projection_dim_options),
        "optimizer_learning_rate": [space.optimizer.learning_rate_min, space.optimizer.learning_rate_max],
        "optimizer_weight_decay": [space.optimizer.weight_decay_min, space.optimizer.weight_decay_max],
        "optimizer_batch_size_options": list(space.optimizer.batch_size_options),
    }
    if isinstance(space, PDBbindSearchSpaceConfig):
        summary.update({
            "decoder_depth_options": list(space.decoder.depth_options),
            "decoder_hidden_size_options": list(space.decoder.hidden_size_options),
            "decoder_lambda_rec_options": list(space.decoder.lambda_rec_options),
            "pdbbind_regression_loss_options": list(space.pdbbind_head.regression_loss_options),
            "pdbbind_huber_delta": [
                space.pdbbind_head.huber_delta_min,
                space.pdbbind_head.huber_delta_max,
            ],
        })
    if isinstance(space, DUDEzSearchSpaceConfig):
        summary.update({
            "dudez_classifier_hidden_size_options": list(space.dudez_head.classifier_hidden_size_options),
            "dudez_classifier_dropout": [
                space.dudez_head.classifier_dropout_min,
                space.dudez_head.classifier_dropout_max,
            ],
            "dudez_fine_tuning_mode_options": list(space.dudez_head.fine_tuning_mode_options),
            "dudez_num_unfrozen_layers_options": list(space.dudez_head.num_unfrozen_layers_options),
            "dudez_use_transfer_options": list(space.dudez_head.use_transfer_options),
            "dudez_use_class_weighting_options": list(space.dudez_head.use_class_weighting_options),
        })
    return summary
