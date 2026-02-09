#!/usr/bin/env python3

# Description
###############################################################################
'''
Build neural network models for SHAP analysis in OCScore.

Usage:

from OCDocker.OCScore.Analysis.SHAP.Model import build_neural_net
'''

# Imports
###############################################################################
from __future__ import annotations
import torch

from typing import Dict, List, Optional, Union, cast

from OCDocker.OCScore.DNN.DNNOptimizer import NeuralNet


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
def build_neural_net(
    input_dim: int,
    autoencoder_params: Dict[str, Union[int, float, str, bool]],
    nn_params: Dict[str, Union[int, float, str, bool]],
    seed: int,
    mask: Optional[Union[list[int], list[bool]]] = None,
    use_gpu: Optional[bool] = None,
    verbose: bool = False,
) -> NeuralNet:
    '''Build and configure a neural network for SHAP analysis.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    autoencoder_params : Dict[str, Union[int, float, str, bool]]
        Parameters for the autoencoder component.
    nn_params : Dict[str, Union[int, float, str, bool]]
        Parameters for the neural network component.
    seed : int
        Random seed for reproducibility.
    mask : Optional[list[int] | list[bool]], optional
        Feature mask to apply. Default is None.
    use_gpu : Optional[bool], optional
        Whether to use GPU. If None, auto-detects CUDA availability. Default is None.
    verbose : bool, optional
        Whether to print verbose output. Default is False.
    
    Returns
    -------
    NeuralNet
        Configured neural network in evaluation mode.
    '''
    
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    mask_typed = cast(Optional[List[Union[int, bool]]], mask)
    neural = NeuralNet(
        input_dim,
        1,
        autoencoder_params,
        nn_params,
        random_seed=seed,
        use_gpu=use_gpu,
        verbose=verbose,
        mask=mask_typed,
    )
    neural.NN.eval()
    return neural
