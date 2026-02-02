#!/usr/bin/env python3

# Description
###############################################################################
'''Autoencoder models for the future Dimensionality pipeline.'''

# Imports
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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


class MLP(nn.Module):
    """Simple MLP with normalization and dropout.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    layer_sizes : list[int]
        Layer sizes for the MLP (last size is output).
    activations : str | list[tuple[str, dict]]
        Activation name or per-layer activation config.
    dropout : float, optional
        Dropout probability, by default 0.0.
    norm : str, optional
        Normalization type: 'batch', 'layer', or 'none'. Default 'batch'.
    output_activation : str, optional
        Activation for the last layer (if provided). Default 'Identity'.

    Example
    -------
    >>> mlp = MLP(input_size=128, layer_sizes=[256, 64, 10], activations="ReLU", dropout=0.1)
    >>> out = mlp(torch.randn(32, 128))
    """

    def __init__(
            self,
            input_size: int,
            layer_sizes: List[int],
            activations: Union[str, List[Tuple[str, Dict[str, Any]]]] = "GELU",
            dropout: float = 0.0,
            norm: str = "batch",
            output_activation: str = "Identity"
        ) -> None:
        '''Initialize MLP.

        Parameters
        ----------
        input_size : int
            Input feature dimension.
        layer_sizes : list[int]
            Layer sizes for the MLP.
        activations : str | list[tuple[str, dict]], optional
            Activation configuration, by default "GELU".
        dropout : float, optional
            Dropout probability, by default 0.0.
        norm : str, optional
            Normalization type, by default "batch".
        output_activation : str, optional
            Output activation name, by default "Identity".
        '''

        super(MLP, self).__init__()

        if not layer_sizes:
            raise ValueError("layer_sizes must be a non-empty list")

        if isinstance(activations, str):
            activations = [(activations, {}) for _ in layer_sizes]
        elif len(activations) != len(layer_sizes):
            raise ValueError("activations length must match layer_sizes")

        layers: List[nn.Module] = []
        prev = input_size

        for idx, (size, (act_name, act_params)) in enumerate(zip(layer_sizes, activations)):
            layers.append(nn.Linear(prev, size))
            norm_layer = _build_norm(norm, size)
            if norm_layer is not None:
                layers.append(norm_layer)

            if idx == len(layer_sizes) - 1:
                # Allow a distinct output activation for the final layer.
                layers.append(_build_activation(output_activation or act_name, {}))
            else:
                layers.append(_build_activation(act_name, act_params))

            if dropout > 0.0:
                # Dropout after activation to regularize hidden representations.
                layers.append(nn.Dropout(dropout))

            prev = size

        self.net = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        '''

        return self.net(x)


class EncoderModule(nn.Module):
    """Encoder module with optional VAE heads.
    
    Parameters
    ----------
    input_size : int
        Input feature dimension.
    encoder_hidden_sizes : list[int]
        Hidden layer sizes for the encoder (excluding latent).
    latent_dim : int
        Latent embedding dimension.
    activation : str, optional
        Activation for encoder hidden layers, by default 'GELU'.
    latent_activation : str, optional
        Activation for latent layer, by default 'Identity'.
    dropout : float, optional
        Dropout probability, by default 0.0.
    latent_dropout : float, optional
        Dropout applied to latent embedding, by default 0.0.
    norm : str, optional
        Normalization type: 'batch', 'layer', or 'none'. Default 'batch'.
    use_vae : bool, optional
        If True, use VAE reparameterization, by default False.

    Example
    -------
    >>> encoder = EncoderModule(input_size=256, encoder_hidden_sizes=[512, 256], latent_dim=64)
    >>> z = encoder(torch.randn(8, 256))
    """

    def __init__(
            self,
            input_size: int,
            encoder_hidden_sizes: List[int],
            latent_dim: int,
            activation: str = "GELU",
            latent_activation: str = "Identity",
            dropout: float = 0.0,
            latent_dropout: float = 0.0,
            norm: str = "batch",
            use_vae: bool = False
        ) -> None:
        '''Initialize encoder module.

        Parameters
        ----------
        input_size : int
            Input feature dimension.
        encoder_hidden_sizes : list[int]
            Encoder hidden layer sizes.
        latent_dim : int
            Latent dimension.
        activation : str, optional
            Encoder activation, by default "GELU".
        latent_activation : str, optional
            Latent activation, by default "Identity".
        dropout : float, optional
            Dropout probability, by default 0.0.
        latent_dropout : float, optional
            Latent dropout probability, by default 0.0.
        norm : str, optional
            Normalization type, by default "batch".
        use_vae : bool, optional
            Enable VAE heads, by default False.
        '''

        super(EncoderModule, self).__init__()

        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)
        self.use_vae = bool(use_vae)

        self.encoder_body = None
        if encoder_hidden_sizes:
            self.encoder_body = MLP(
                input_size=self.input_size,
                layer_sizes=encoder_hidden_sizes,
                activations=[(activation, {}) for _ in encoder_hidden_sizes],
                dropout=dropout,
                norm=norm,
                output_activation=activation
            )

        encoder_out = encoder_hidden_sizes[-1] if encoder_hidden_sizes else self.input_size

        if self.use_vae:
            self.fc_mu = nn.Linear(encoder_out, self.latent_dim)
            self.fc_logvar = nn.Linear(encoder_out, self.latent_dim)
        else:
            self.fc_latent = nn.Linear(encoder_out, self.latent_dim)

        self.latent_norm = _build_norm(norm, self.latent_dim)
        self.latent_activation = _build_activation(latent_activation, {})
        self.latent_dropout = nn.Dropout(latent_dropout) if latent_dropout > 0.0 else nn.Identity()


    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        '''Encode inputs through the encoder body.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tensor.
        '''

        if self.encoder_body is None:
            return x
        return self.encoder_body(x)


    def _latent_transform(self, z: torch.Tensor) -> torch.Tensor:
        '''Apply normalization/activation to latent tensor.

        Parameters
        ----------
        z : torch.Tensor
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Transformed latent tensor.
        '''

        if self.latent_norm is not None:
            # Latent normalization stabilizes scale for downstream heads.
            z = self.latent_norm(z)
        z = self.latent_activation(z)
        return z


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        '''Reparameterization trick for VAE.

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean tensor.
        logvar : torch.Tensor
            Latent log-variance tensor.

        Returns
        -------
        torch.Tensor
            Sampled latent tensor.
        '''

        std = torch.exp(0.5 * logvar)
        # Random noise ensures stochastic sampling from the posterior.
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(
            self,
            x: torch.Tensor,
            sample: bool = False,
            return_stats: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''Forward pass for the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample : bool, optional
            If True, sample from posterior when VAE is enabled, by default False.
        return_stats : bool, optional
            If True, return (z, mu, logvar), by default False.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Latent tensor (and optional stats).
        '''

        h = self._encode_features(x)

        if self.use_vae:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar) if sample else mu
        else:
            # Zero stats keep downstream loss code consistent without VAE.
            mu = torch.zeros(h.shape[0], self.latent_dim, device=h.device)
            logvar = torch.zeros_like(mu)
            z = self.fc_latent(h)

        z = self._latent_transform(z)

        if return_stats:
            return z, mu, logvar

        return z


class Autoencoder(nn.Module):
    """Denoising autoencoder with optional energy head and VAE support.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    encoder_hidden_sizes : list[int]
        Hidden layer sizes for the encoder (excluding latent).
    latent_dim : int
        Latent embedding dimension.
    decoder_sizes : list[int] | None, optional
        Decoder sizes (including output). If None, mirror encoder. Default None.
    activation : str, optional
        Activation for encoder/decoder hidden layers, by default 'GELU'.
    latent_activation : str, optional
        Activation for latent layer, by default 'Identity'.
    decoder_output_activation : str, optional
        Activation for decoder output, by default 'Identity'.
    dropout : float, optional
        Dropout probability, by default 0.0.
    latent_dropout : float, optional
        Dropout applied to latent embedding, by default 0.0.
    norm : str, optional
        Normalization type: 'batch', 'layer', or 'none'. Default 'batch'.
    use_vae : bool, optional
        If True, use VAE reparameterization, by default False.
    energy_head_sizes : list[int] | None, optional
        Hidden sizes for energy head. If None, energy head is disabled.
    device : torch.device, optional
        Device to place the module on.

    Example
    -------
    >>> model = Autoencoder(input_size=256, encoder_hidden_sizes=[512, 256], latent_dim=64)
    >>> z = model.encode(torch.randn(8, 256))

    Notes
    -----
    Forward returns a dictionary with:
    - reconstruction: decoded input
    - latent: latent embedding
    - mu/logvar: VAE statistics (zeros when use_vae=False)
    - energy: optional energy head output (None if disabled)
    """

    def __init__(
            self,
            input_size: int,
            encoder_hidden_sizes: List[int],
            latent_dim: int,
            decoder_sizes: Optional[List[int]] = None,
            activation: str = "GELU",
            latent_activation: str = "Identity",
            decoder_output_activation: str = "Identity",
            dropout: float = 0.0,
            latent_dropout: float = 0.0,
            norm: str = "batch",
            use_vae: bool = False,
            energy_head_sizes: Optional[List[int]] = None,
            device: torch.device = torch.device("cpu")
        ) -> None:
        '''Initialize autoencoder.

        Parameters
        ----------
        input_size : int
            Input feature dimension.
        encoder_hidden_sizes : list[int]
            Encoder hidden layer sizes.
        latent_dim : int
            Latent dimension.
        decoder_sizes : list[int] | None, optional
            Decoder layer sizes, by default None.
        activation : str, optional
            Encoder/decoder activation, by default "GELU".
        latent_activation : str, optional
            Latent activation, by default "Identity".
        decoder_output_activation : str, optional
            Output activation for decoder, by default "Identity".
        dropout : float, optional
            Dropout probability, by default 0.0.
        latent_dropout : float, optional
            Latent dropout probability, by default 0.0.
        norm : str, optional
            Normalization type, by default "batch".
        use_vae : bool, optional
            Enable VAE mode, by default False.
        energy_head_sizes : list[int] | None, optional
            Energy head hidden sizes, by default None.
        device : torch.device, optional
            Device to place the module on, by default CPU.
        '''

        super(Autoencoder, self).__init__()

        self.input_size = int(input_size)
        self.latent_dim = int(latent_dim)
        self.use_vae = bool(use_vae)
        self.device = device

        if decoder_sizes is None:
            if encoder_hidden_sizes:
                # Mirror encoder sizes for a symmetric decoder by default.
                decoder_sizes = list(reversed(encoder_hidden_sizes)) + [self.input_size]
            else:
                decoder_sizes = [self.input_size]

        self.encoder = EncoderModule(
            input_size=self.input_size,
            encoder_hidden_sizes=encoder_hidden_sizes,
            latent_dim=self.latent_dim,
            activation=activation,
            latent_activation=latent_activation,
            dropout=dropout,
            latent_dropout=latent_dropout,
            norm=norm,
            use_vae=self.use_vae
        )

        self.decoder = MLP(
            input_size=self.latent_dim,
            layer_sizes=decoder_sizes,
            activations=[(activation, {}) for _ in decoder_sizes],
            dropout=dropout,
            norm=norm,
            output_activation=decoder_output_activation
        )

        if energy_head_sizes is not None and len(energy_head_sizes) > 0:
            # Optional energy head for auxiliary regression supervision.
            self.energy_head = MLP(
                input_size=self.latent_dim,
                layer_sizes=energy_head_sizes + [1],
                activations=[(activation, {}) for _ in energy_head_sizes] + [("Identity", {})],
                dropout=dropout,
                norm=norm,
                output_activation="Identity"
            )
        else:
            self.energy_head = None

        self.to(self.device)


    def encode(
            self,
            x: torch.Tensor,
            sample: bool = False,
            return_stats: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        '''Encode inputs into latent embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample : bool, optional
            If True and VAE enabled, sample from posterior, by default False.
        return_stats : bool, optional
            If True, return (z, mu, logvar), by default False.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Latent tensor (and optional stats).
        '''

        return self.encoder(x, sample=sample, return_stats=return_stats)


    def reconstruct(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        '''Reconstruct inputs through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample : bool, optional
            If True and VAE enabled, sample from posterior, by default False.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor.
        '''

        z = self.encode(x, sample=sample)
        z = self.encoder.latent_dropout(z)
        return self.decoder(z)


    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        '''Forward pass returning reconstruction and auxiliary outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        sample : bool, optional
            If True and VAE enabled, sample from posterior, by default True.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with reconstruction, latent, and auxiliary outputs.

        Notes
        -----
        When use_vae is False, mu/logvar are zero tensors for API consistency.
        When energy_head is disabled, energy is returned as None.
        '''

        if self.use_vae:
            z, mu, logvar = self.encode(x, sample=sample, return_stats=True)
        else:
            z = self.encode(x, sample=False, return_stats=False)
            # Use zero stats to keep downstream loss handling consistent.
            mu = torch.zeros_like(z)
            logvar = torch.zeros_like(z)

        z_used = self.encoder.latent_dropout(z)
        # Reconstruction uses possibly dropped-out latents for robustness.
        reconstruction = self.decoder(z_used)
        energy = self.energy_head(z_used) if self.energy_head is not None else None

        return {
            "reconstruction": reconstruction,
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "energy": energy
        }


    def get_encoder_topology(self) -> List[str]:
        '''Return encoder topology description.

        Returns
        -------
        list[str]
            Encoder topology tokens.
        '''

        return ["Linear", "Norm", "Activation"]


    def get_decoder_topology(self) -> List[str]:
        '''Return decoder topology description.

        Returns
        -------
        list[str]
            Decoder topology tokens.
        '''

        return ["Linear", "Norm", "Activation"]


    def get_encoder(self) -> nn.Module:
        '''Return the encoder module.

        Returns
        -------
        nn.Module
            Encoder module.
        '''

        return self.encoder


    def get_decoder(self) -> nn.Module:
        '''Return the decoder module.

        Returns
        -------
        nn.Module
            Decoder module.
        '''
        return self.decoder


    def save_encoder(self, path: str) -> None:
        '''Save encoder weights to a file.

        Parameters
        ----------
        path : str
            Output path for encoder state dict.
        '''

        torch.save(self.encoder.state_dict(), path)


    def load_encoder(self, path: str, map_location: Optional[str] = None) -> None:
        '''Load encoder weights from a file.

        Parameters
        ----------
        path : str
            Path to encoder state dict.
        map_location : str | None, optional
            Torch map location override, by default None.
        '''

        state = torch.load(path, map_location=map_location)
        self.encoder.load_state_dict(state, strict=False)


    def sanity_check(self, batch_size: int = 4) -> Dict[str, object]:
        '''Run a lightweight sanity check on shapes and reconstruction.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for the synthetic check, by default 4.

        Returns
        -------
        Dict[str, object]
            Dictionary with shapes and reconstruction RMSE.
        '''

        self.eval()
        # Synthetic batch validates tensor shapes without external data.
        x = torch.randn(batch_size, self.input_size, device=self.device)
        with torch.no_grad():
            out = self.forward(x, sample=False)

        recon = out["reconstruction"]
        latent = out["latent"]
        recon_error = torch.mean((recon - x) ** 2).sqrt().item()

        return {
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "reconstruction_shape": tuple(recon.shape),
            "reconstruction_rmse": float(recon_error)
        }

# Methods
###############################################################################
def _build_activation(name: str, params: Dict[str, Any]) -> nn.Module:
    '''Build activation module from name/params.

    Parameters
    ----------
    name : str
        Activation name.
    params : Dict[str, Any]
        Activation parameters.

    Returns
    -------
    nn.Module
        Activation module.
    '''

    name = name or "ReLU"

    if name == "LeakyReLU":
        negative_slope = float(params.get("negative_slope", 0.01))
        return nn.LeakyReLU(negative_slope=negative_slope)

    if name == "GELU":
        approximate = params.get("approximate", "none")
        return nn.GELU(approximate=approximate)

    if name == "Mish":
        return nn.Mish()

    if name == "SELU":
        return nn.SELU()

    if name == "Identity":
        return nn.Identity()

    return nn.ReLU()


def _build_norm(norm: str, num_features: int) -> Optional[nn.Module]:
    '''Build normalization module from name.

    Parameters
    ----------
    norm : str
        Normalization type ("batch", "layer", "none").
    num_features : int
        Feature dimension for normalization.

    Returns
    -------
    nn.Module | None
        Normalization module or None.
    '''

    if norm == "batch":
        return nn.BatchNorm1d(num_features)
    if norm == "layer":
        return nn.LayerNorm(num_features)
    return None
