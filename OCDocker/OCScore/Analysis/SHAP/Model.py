
from __future__ import annotations
from typing import Dict, Union, Optional
import torch
from OCDocker.OCScore.DNN.DNNOptimizer import NeuralNet


def build_neural_net(
    input_dim: int,
    autoencoder_params: Dict[str, Union[int, float, str, bool]],
    nn_params: Dict[str, Union[int, float, str, bool]],
    seed: int,
    mask: Optional = None,
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
    mask : Optional
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

    neural = NeuralNet(
        input_dim,
        1,
        autoencoder_params,
        nn_params,
        random_seed=seed,
        use_gpu=use_gpu,
        verbose=verbose,
        mask=mask,
    )
    neural.NN.eval()
    return neural
