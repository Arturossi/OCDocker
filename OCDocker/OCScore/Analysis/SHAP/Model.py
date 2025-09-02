
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
