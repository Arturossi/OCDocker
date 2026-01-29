#!/usr/bin/env python3

# Description
###############################################################################
'''Models for the future DNN pipeline (shared encoder + multi-head).'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MLP(nn.Module):
    '''Simple MLP with optional batch norm and dropout.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    layer_sizes : list[int]
        Hidden layer sizes (final size is last).
    activations : list[tuple[str, dict]] | str
        Activation configuration for each layer or a single activation name.
    dropout : float, optional
        Dropout probability applied after activation, by default 0.0.
    batch_norm : bool, optional
        Use BatchNorm1d after Linear, by default True.
    '''

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            activations: Union[str, List[Tuple[str, Dict[str, Any]]]] = "GELU",
            dropout: float = 0.0,
            batch_norm: bool = True
        ) -> None:
        super(MLP, self).__init__()

        if not layer_sizes:
            raise ValueError("layer_sizes must be a non-empty list")

        if isinstance(activations, str):
            activations = [(activations, {}) for _ in layer_sizes]
        else:
            if len(activations) != len(layer_sizes):
                raise ValueError("activations length must match layer_sizes")

        layers: List[nn.Module] = []
        prev = input_size

        for size, (act_name, act_params) in zip(layer_sizes, activations):
            layers.append(nn.Linear(prev, size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(_build_activation(act_name, act_params))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = size

        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskModel(nn.Module):
    '''Shared encoder with heads for energy, activity, embedding and reconstruction.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    encoder_params : dict | None
        Encoder params (old-style).
    shared_sizes : list[int]
        Sizes for shared encoder when encoder_params is None.
    shared_activation : str, optional
        Activation for shared encoder, by default GELU.
    decoder_sizes : list[int] | None, optional
        Decoder sizes for reconstruction. If None, decoder is disabled.
    head_sizes : list[int]
        Hidden sizes for heads.
    embedding_dim : int | None
        Output dimension for embedding head.
    dropout : float, optional
        Dropout probability.
    batch_norm : bool, optional
        Use BatchNorm1d.
    mask : torch.Tensor | None
        Feature mask.
    '''

    def __init__(
            self,
            input_size: int,
            encoder_params: Union[None, Dict[str, Any]],
            shared_sizes: List[int],
            shared_activation: str = "GELU",
            decoder_sizes: Union[List[int], None] = None,
            head_sizes: List[int] = [128, 64],
            embedding_dim: Union[int, None] = 64,
            dropout: float = 0.0,
            batch_norm: bool = True,
            mask: Union[torch.Tensor, None] = None
        ) -> None:
        super(MultiTaskModel, self).__init__()

        self.mask = mask

        if encoder_params is not None:
            layer_sizes, activations = parse_encoder_params(encoder_params)
            self.encoder = MLP(
                input_size=input_size,
                layer_sizes=layer_sizes,
                activations=activations,
                dropout=dropout,
                batch_norm=batch_norm
            )
            latent_dim = layer_sizes[-1]
        else:
            self.encoder = MLP(
                input_size=input_size,
                layer_sizes=shared_sizes,
                activations=[(shared_activation, {}) for _ in shared_sizes],
                dropout=dropout,
                batch_norm=batch_norm
            )
            latent_dim = shared_sizes[-1]

        self.energy_head = MLP(
            input_size=latent_dim,
            layer_sizes=head_sizes + [1],
            activations=[(shared_activation, {}) for _ in head_sizes] + [("Identity", {})],
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.activity_head = MLP(
            input_size=latent_dim,
            layer_sizes=head_sizes + [1],
            activations=[(shared_activation, {}) for _ in head_sizes] + [("Identity", {})],
            dropout=dropout,
            batch_norm=batch_norm
        )

        if embedding_dim is not None and embedding_dim > 0:
            self.embedding_head = MLP(
                input_size=latent_dim,
                layer_sizes=[embedding_dim],
                activations=[("Identity", {})],
                dropout=0.0,
                batch_norm=False
            )
        else:
            self.embedding_head = None

        if decoder_sizes is not None:
            self.decoder = MLP(
                input_size=latent_dim,
                layer_sizes=decoder_sizes,
                activations=[(shared_activation, {}) for _ in decoder_sizes],
                dropout=dropout,
                batch_norm=batch_norm
            )
        else:
            self.decoder = None


    def forward(self, x: torch.Tensor, return_reconstruction: bool = False) -> Dict[str, torch.Tensor]:
        if self.mask is not None:
            x = x * self.mask
        latent = self.encoder(x)

        energy = self.energy_head(latent)
        activity = self.activity_head(latent)

        if self.embedding_head is not None:
            embedding = self.embedding_head(latent)
            embedding = F.normalize(embedding, dim=1)
        else:
            embedding = None

        reconstruction = None
        if return_reconstruction and self.decoder is not None:
            reconstruction = self.decoder(latent)

        return {
            "latent": latent,
            "energy": energy,
            "activity": activity,
            "embedding": embedding,
            "reconstruction": reconstruction
        }


# Methods
###############################################################################


def _build_activation(name: str, params: Dict[str, Any]) -> nn.Module:
    name = name or "ReLU"

    if name == "LeakyReLU":
        negative_slope = float(params.get("negative_slope", params.get("negative_slope_encoder", 0.01)))
        return nn.LeakyReLU(negative_slope=negative_slope)

    if name == "GELU":
        approximate = params.get("approximate", params.get("approximate_encoder", "none"))
        return nn.GELU(approximate=approximate)

    if name == "Mish":
        return nn.Mish()

    if name == "SELU":
        return nn.SELU()

    if name == "Identity":
        return nn.Identity()

    return nn.ReLU()


def parse_encoder_params(encoder_params: Dict[str, Any]) -> Tuple[List[int], List[Tuple[str, Dict[str, Any]]]]:
    '''Parse old-style encoder params into layer sizes and activation configs.'''

    if "encoding_dim" in encoder_params:
        layer_sizes = [int(encoder_params["encoding_dim"])]
        act_name = encoder_params.get("encoder_activation", "ReLU")
        act_params = {}
        if act_name == "LeakyReLU":
            act_params["negative_slope"] = encoder_params.get("negative_slope_encoder", 0.01)
        if act_name == "GELU":
            act_params["approximate"] = encoder_params.get("approximate_encoder", "none")
        activations = [(act_name, act_params)]
        return layer_sizes, activations

    layer_sizes: List[int] = []
    activations: List[Tuple[str, Dict[str, Any]]] = []

    if "n_layers_encoder" in encoder_params:
        n_layers = int(encoder_params["n_layers_encoder"])
    else:
        # Fallback to scanning keys
        n_layers = 0
        while f"n_units_layer_{n_layers}_encoder" in encoder_params:
            n_layers += 1

    for i in range(n_layers):
        layer_sizes.append(int(encoder_params[f"n_units_layer_{i}_encoder"]))
        act_name = encoder_params.get(f"activation_function_{i}_encoder", encoder_params.get("encoder_activation", "ReLU"))
        act_params: Dict[str, Any] = {}
        if act_name == "LeakyReLU":
            act_params["negative_slope"] = encoder_params.get(f"negative_slope_{i}_encoder", encoder_params.get("negative_slope_encoder", 0.01))
        if act_name == "GELU":
            act_params["approximate"] = encoder_params.get(f"approximate_{i}_encoder", encoder_params.get("approximate_encoder", "none"))
        activations.append((act_name, act_params))

    if not layer_sizes:
        raise ValueError("encoder_params must define at least one layer")

    return layer_sizes, activations
