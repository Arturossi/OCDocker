""" UNetOptimizer.py - Optimize a UNet model using Optuna.

This module contains the UNetOptimizer class, which is used to optimize a UNet model using Optuna.

Classes:
    CustomDataset - Create a custom dataset for the PyTorch DataLoader.
    UNet - The UNet model.
    UNetOptimizer - Optimize a UNet model using Optuna.
"""

import math
import optuna
import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from optuna.samplers import TPESampler
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset, DataLoader
from typing import Union

#from OCDocker.Initialise import *

class CustomDataset(Dataset):
    """ Create a custom dataset for the PyTorch DataLoader. """
    def __init__(self, features: list, target: list) -> None:
        ''' Initialize the dataset.
        
        Parameters
        ----------
        features : list
            The features.
        target : list
            The target.
        '''

        self.features = features
        self.target = target

        return None

    def __len__(self) -> int:
        ''' Get the length of the dataset.	

        Returns
        -------
        int
            The length of the dataset.
        '''

        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        ''' Get the item at the index.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        tuple
            The features and the target.
        '''
        
        return self.features[idx], self.target[idx]

import torch
import torch.nn as nn

class ReshapePreprocessingLayer(nn.Module):
    def __init__(self, input_features, target_channels = 1):
        super(ReshapePreprocessingLayer, self).__init__()
        self.input_features = input_features
        self.target_side_length = math.ceil(math.sqrt(input_features / target_channels))
        self.total_size = self.target_side_length ** 2 * target_channels
        self.padding_size = self.total_size - input_features
        self.target_channels = target_channels

    def forward(self, x):
        x = torch.cat((x, torch.zeros(x.size(0), self.padding_size, device=x.device)), dim=1)
        x = x.view(x.size(0), self.target_channels, self.target_side_length, self.target_side_length)
        return x

class UNet(nn.Module):
    def __init__(self, 
            in_channels: int,
            out_channels: int = 1,
            n_layers: int = 3, 
            starting_channel_size: int = 64, 
            encoder_data: list[tuple[bool, float, list]] = [],
            decoder_data: list[tuple[bool, float, list]] = [],
            bottleneck_activation_functions: list = [nn.ReLU(inplace = True), nn.ReLU(inplace = True)],
            init_type: str = 'zeros',
            init_params: dict = {},
            default_prob_encoder: float = 0.5,
            default_activation_encoder = nn.ReLU,
            default_prob_decoder: float = 0.5,
            default_activation_decoder = nn.ReLU,
            random_seed: int = 42,
            use_gpu: bool = True, # TODO: Add this
            verbose: bool = False
        ):
        super(UNet, self).__init__()

        self.use_gpu = use_gpu
        self.random_seed = random_seed

        self.set_random_seed()

        self.preprocess = ReshapePreprocessingLayer(in_channels)

        # Define the number of activation functions for the encoder, decoder, and bottleneck
        n_activation_functions_encoder = 2
        n_activation_functions_decoder = 2
        n_activation_functions_bottleneck = 2

        # Check if the length of the activation functions for the bottleneck is correct
        if len(bottleneck_activation_functions) != n_activation_functions_bottleneck:
            raise ValueError(f"[WARNING] The bottleneck_activation_functions list does not have {n_activation_functions_bottleneck} elements. It has {len(bottleneck_activation_functions)} elements.")

        # If the dropout encoder list is empty, fill it with False for all layers
        if not encoder_data:
            encoder_data = [(False, default_prob_encoder, [default_activation_encoder] * n_activation_functions_encoder)] * n_layers

        # If the dropout decoder list is empty, fill it with False for all layers
        if not decoder_data:
            decoder_data = [(False, default_prob_decoder, [default_activation_decoder] * n_activation_functions_decoder)] * n_layers

        # If any tuple inside the dropout encoder list does not have enough elements, add the missing data
        for i in range(len(encoder_data)):
            if len(encoder_data[i]) < 1:
                encoder_data[i] = (
                    False, 
                    default_prob_encoder, 
                    [default_activation_encoder] * n_activation_functions_encoder
                )

                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'
                
                print(f"[WARNING] The {i}{suffix} element of the encoder_data list has less than 1 element. Filling it with the default values.")
            elif len(encoder_data[i]) < 2:
                encoder_data[i] = (
                    encoder_data[i][0], 
                    default_prob_encoder, 
                    [default_activation_encoder] * n_activation_functions_encoder
                )

                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'
                
                print(f"[WARNING] The {i}{suffix} element of the encoder_data list has less than 2 elements. Filling it with the default values.")
            elif len(encoder_data[i]) < 3:
                encoder_data[i] = (
                    encoder_data[i][0], 
                    encoder_data[i][1], 
                    [default_activation_encoder] * n_activation_functions_encoder
                )
                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'

                print(f"[WARNING] The {i}{suffix} element of the encoder_data list has less than 3 elements. Filling it with the default values.")
            
        # If any tuple inside the dropout decoder list does not have enough elements, add the missing data
        for i in range(len(decoder_data)):
            if len(decoder_data[i]) < 1:
                decoder_data[i] = (
                    False, 
                    default_prob_decoder, 
                    [default_activation_decoder] * n_activation_functions_decoder
                )

                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'

                print(f"[WARNING] The {i}{suffix} element of the decoder_data list has less than 1 element. Filling it with the default values.")
            elif len(decoder_data[i]) < 2:
                decoder_data[i] = (
                    decoder_data[i][0], 
                    default_prob_decoder, 
                    [default_activation_decoder] * n_activation_functions_decoder
                )

                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'

                print(f"[WARNING] The {i}{suffix} element of the decoder_data list has less than 2 elements. Filling it with the default values.")
            elif len(decoder_data[i]) < 3:
                decoder_data[i] = (
                    decoder_data[i][0], 
                    decoder_data[i][1], 
                    [default_activation_decoder] * n_activation_functions_decoder
                )

                if i == 1:
                    suffix = 'st'
                elif i == 2:
                    suffix = 'nd'
                elif i == 3:
                    suffix = 'rd'
                else:
                    suffix = 'th'

                print(f"[WARNING] The {i}{suffix} element of the decoder_data list has less than 3 elements. Filling it with the default values.")

        #region Encoder
        
        # Create the encoder list (contracting path)
        self.encoder = []

        # Check if the data activation function is valid
        if len(encoder_data) != n_activation_functions_encoder:
            raise ValueError(f"[WARNING] The length of the encoder_data is not compatible with the number of activation functions. It has {len(encoder_data)} elements. It should have {n_activation_functions_encoder} elements.")

        # For each layer
        for i in range(n_activation_functions_encoder):

            # If it is the first layer
            if i == 0:
                in_layer_channels = 1
                out_layer_channels = starting_channel_size
            else:
                in_layer_channels = starting_channel_size * 2 ** (i - 1)
                out_layer_channels = starting_channel_size * 2 ** i
    
            self.encoder.append(
                self.contracting_block(
                    in_layer_channels,
                    out_layer_channels,
                    apply_pooling = True, 
                    use_dropout = encoder_data[i][0],
                    dropout_prob = encoder_data[i][1],
                    activation_functions = encoder_data[i][2]
                )
            )
        
        #endregion

        bottleneck_channels = starting_channel_size * 2 ** (n_layers)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_layer_channels, bottleneck_channels, kernel_size = 3, padding = 1),
            bottleneck_activation_functions[0],
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size = 3, padding = 1),
            bottleneck_activation_functions[1]
        ).to(self.device)

        #region Decoder

        # Create the decoder list (expansive path)
        self.decoder = []

        # Check if the data activation function is valid
        if len(decoder_data) != n_activation_functions_decoder:
            raise ValueError(f"[WARNING] The length of the decoder_data is not compatible with the number of activation functions. It has {len(decoder_data)} elements. It should have {n_activation_functions_decoder} elements.")

        # For each layer
        for i in range(n_activation_functions_decoder):
            # If it is the first layer
            if i == 0:
                in_layer_channels = starting_channel_size * 2 ** (n_layers - 1)
                mid_layer_channels = starting_channel_size * 2 ** (n_layers + 1)
                out_layer_channels = out_channels
            if i == n_activation_functions_decoder - 1:
                in_layer_channels = starting_channel_size * 2 ** n_layers
                mid_layer_channels = starting_channel_size * 2 ** (n_layers - 1)
                out_layer_channels = starting_channel_size * 2 ** (n_layers - 2)
            else:
                in_layer_channels = starting_channel_size * 2 ** n_layers
                mid_layer_channels = starting_channel_size * 2 ** (n_layers - 1)
                out_layer_channels = starting_channel_size * 2 ** (n_layers - 2) + (n_activation_functions_encoder - i) ** 2

            self.decoder.append(
                self.expansive_block(
                    in_layer_channels, 
                    mid_layer_channels,
                    out_layer_channels,
                    use_dropout = decoder_data[i][0],
                    dropout_prob = decoder_data[i][1],
                    activation_functions = decoder_data[i][2]
                )
            )

        # Reverse the decoder list
        self.decoder.reverse()

        #endregion

        self.init_functions = {
            'xavier_uniform': init.xavier_uniform_,
            'glorot_uniform': init.xavier_uniform_,
            'he_uniform': init.kaiming_uniform_,
            'kaiming_uniform': init.kaiming_uniform_,
            'xavier_normal': init.xavier_normal_,
            'glorot_normal': init.xavier_normal_,
            'he_normal': init.kaiming_normal_,
            'kaiming_normal': init.kaiming_normal_,
            'zeros': init.zeros_,
            'ones': init.ones_,
            'orthogonal': init.orthogonal_,
            'normal': init.normal_,
            'uniform': init.uniform_,
            'constant': init.constant_,
            'eye': init.eye_,
            'sparse': init.sparse_
        }

        # Other parameters
        self.init_type = init_type
        self.init_params = init_params

        # Initialize weights
        self.initialize_weights()

        if verbose:
            # Print the model
            print(self)

    def set_random_seed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)

        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.random_seed)
        else:
            self.device = torch.device('cpu')
        
        #torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def initialize_weights(self):
        if self.init_type in self.init_functions.keys():
            init_func = self.init_functions[self.init_type]
        else:
            raise ValueError('Unknown initialization function')

        # Apply the initialization to all linear layers in the model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type in ['zeros', 'ones', 'eye']:
                    init_func(m.weight)
                elif self.init_type in ['constant']:
                    init_func(m.weight, **self.init_params)
                else:
                    init_func(m.weight, **self.init_params, generator = self.generator)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def contracting_block(self, 
            in_channels, 
            out_channels, 
            apply_pooling = True,
            use_dropout = False, 
            dropout_prob = 0.5, 
            activation_functions = [nn.ReLU(inplace = True), nn.ReLU(inplace = True)]
        ):

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1).to(self.device),
            nn.BatchNorm2d(out_channels).to(self.device),  # Batch normalization before activation
            activation_functions[0].to(self.device)
        ]

        if use_dropout:
            layers.append(nn.Dropout2d(dropout_prob).to(self.device))  # Dropout after activation
        
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1).to(self.device))
        layers.append(nn.BatchNorm2d(out_channels).to(self.device))  # Another batch normalization
        layers.append(activation_functions[1].to(self.device))
        
        if apply_pooling:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2).to(self.device))

        return nn.Sequential(*layers)

    def expansive_block(self, 
            in_channels, 
            mid_channels, 
            out_channels, 
            use_dropout = False, 
            dropout_prob = 0.5, 
            activation_functions = [nn.ReLU(inplace = True), nn.ReLU(inplace = True)]
        ):

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1).to(self.device),
            nn.BatchNorm2d(mid_channels).to(self.device),
            activation_functions[0].to(self.device),
            nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, padding = 1).to(self.device),
            nn.BatchNorm2d(mid_channels).to(self.device),
            activation_functions[1].to(self.device)
        ]

        if use_dropout:
            layers.append(nn.Dropout2d(dropout_prob).to(self.device))

        layers.append(nn.ConvTranspose2d(mid_channels, out_channels, kernel_size = 2, stride = 2, output_padding = 1).to(self.device))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        ## Preprocessing
        x = self.preprocess(x)
        print("After preprocessing:", x.shape)  # Debug: Check the shape after preprocessing

        ## Encoder
        encoder = []

        # For each encoder layer
        for layer in self.encoder:
            x = layer(x).to(self.device)
            encoder.append(x)
            print("Encoder output shape:", x.shape)  # Debug: Check the output shape of each encoder layer

        ## Bottleneck
        bottleneck = self.bottleneck(encoder[-1]).to(self.device)  # Put the last encoder layer in the bottleneck
        print("Bottleneck output shape:", bottleneck.shape)  # Debug: Check the bottleneck output shape

        ## Decoder
        decoder = []

        for enc in encoder:
            print("Encoder shape:", enc.shape)  # Debug
        
        print(self.decoder)

        # For each layer in the decoder
        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(bottleneck).to(self.device)
            else:
                # Concatenation of the feature maps from the encoder and the previous layer of the decoder
                concat_features = torch.cat([x, encoder[-i-1]], dim=1)
                print(f"Shape before layer {i} in decoder (after concatenation):", concat_features.shape)  # Debug
                x = layer(concat_features).to(self.device)

            decoder.append(x)
            print(f"Decoder layer {i} output shape:", x.shape)  # Debug

        return decoder[-1]


    def forward2(self, x):
        ## Preprocessing
        x = self.preprocess(x)

        ## Encoder

        encoder = []

        # For each encoder layer
        for layer in self.encoder:
            x = layer(x).to(self.device)
            encoder.append(x)

        ## Bottleneck

        bottleneck = self.bottleneck(encoder[-1]).to(self.device) # Put the last encoder layer in the bottleneck
        
        ## Decoder

        decoder = []

        # For each layer in the decoder
        for i, layer in enumerate(self.decoder):
            if i == 0:
                x = layer(bottleneck).to(self.device)
            else:
                x = layer(torch.cat([x, encoder[-i-1]], dim = 1)).to(self.device)

            decoder.append(x).to(self.device)
        
        return decoder[-1]

class UNetOptimizer:
    """ Optimize a UNet model using Optuna. 
    
    Parameters
    ----------
    X_train : np.array
        The training features.
    y_train : np.array
        The training target.
    X_test : np.array
        The test features.
    y_test : np.array
        The test target.
    X_validation : np.array, optional
        The validation features. Default is None.
    y_validation : np.array, optional
        The validation target. Default is None.
    max_nodes : int, optional
        The maximum number of nodes. POWERS OF TWO ONLY. Greater values means the possibility for deeper models, which comes with a higher computational cost. Avoid to set this too low, it may cause the search not work as intended. Default is 2048.
    storage : str, optional
        The storage for the optimization. Default is 'sqlite:///UNetoptimization.db'.
    output_size : int, optional
        The output size. Default is 1.
    random_seed : int, optional
        The random seed. Default is 42.
    use_gpu : bool, optional
        Whether to use the GPU. Default is True.
    verbose : bool, optional
        Whether to print the results. Default is False.
    """

    def __init__(self, 
            X_train: np.array, y_train: np.array, 
            X_test: np.array, y_test: np.array, 
            X_validation: Union[np.array, None] = None, y_validation: Union[np.array, None] = None, 
            max_nodes: int = 2048, 
            storage: str = 'sqlite:///UNetoptimization.db', 
            output_size: int = 1, 
            random_seed: int = 42, 
            use_gpu: bool = True, 
            verbose: bool = False
        ):
        ''' Initialize the optimizer.
        
        Parameters
        ----------
        X_train : np.array
            The training features.
        y_train : np.array
            The training target.
        X_test : np.array
            The test features.
        y_test : np.array
            The test target.
        X_validation : np.array, optional
            The validation features. Default is None.
        y_validation : np.array, optional
            The validation target. Default is None.
        max_nodes : int, optional
            The maximum number of nodes. POWERS OF TWO ONLY. Greater values means the possibility for deeper models, which comes with a higher computational cost. Avoid to set this too low, it may cause the search not work as intended. Default is 2048.
        storage : str, optional
            The storage for the optimization. Default is 'sqlite:///UNetoptimization.db'.
        output_size : int, optional
            The output size. Default is 1.
        random_seed : int, optional
            The random seed. Default is 42.
        use_gpu : bool, optional
            Whether to use the GPU. Default is True.
        verbose : bool, optional
            Whether to print the results. Default is False.
        '''

        self.random_seed = random_seed
        self.use_gpu = use_gpu
        self.set_random_seed()

        # Handling data
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)
        self.test_loader = None

        if X_validation is not None and y_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)
        else:
            self.X_validation = None
            self.y_validation = None

        self.validation_loader = None

        self.output_size = output_size
        self.verbose = verbose
        self.storage = storage

        # Check if max_nodes is a power of 2
        if not max_nodes or max_nodes % 2 != 0:
            raise ValueError('max_nodes must be a power of 2')
        
        # Check if max_nodes is below or equal to 128
        if max_nodes <= 128:
            # Show a warning
            print('[WARNING] max_nodes is below or equal to 256. This may cause the search not work as intended. Consider increasing the value of max_nodes.')

        self.max_nodes = max_nodes

        self.activation_functions = [nn.LeakyReLU, nn.ReLU, nn.SELU]
        self.activation_functions_str = ['LeakyReLU', 'ReLU', 'SELU']

    def set_random_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.use_gpu:
            torch.cuda.manual_seed_all(self.random_seed)

    def train_test_model(self, model, train_loader, test_loader, optimizer, criterion, clip_grad, trial, batch_size, epochs = 100):
        if self.verbose:
            torch.autograd.set_detect_anomaly(True)

        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            for _, (inputs, labels) in enumerate(train_loader):
                outputs = model(inputs)

                # Ensure the labels are of the correct type (float for regression)
                labels = labels.float()
                
                # Compute the loss
                loss = criterion(outputs, labels.view_as(outputs))

                # Zero the gradients
                optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Clip the gradients
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # Optimizer step
                optimizer.step()

                # Accumulate the loss
                running_loss += loss.item()

            # Set the model to evaluation mode
            model.eval()

            running_loss = 0.0

            all_predictions = []
            all_labels = []

            for inputs, labels in test_loader:
                # Get the predictions
                predicted = model(inputs)

                # Compute the loss
                loss = criterion(predicted, labels.view_as(outputs))

                # Accumulate the loss
                running_loss += loss.item()
                
                # Append the predictions and the labels
                all_predictions.extend(predicted.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        # Get the RMSE
        average_loss = running_loss / len(test_loader) # type: ignore
        rmse = np.sqrt(average_loss)

        if self.verbose:
            print(f'Test Loss: {average_loss}')
            print(f'Test RMSE: {rmse}')

        #trial.report(rmse, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return rmse

    def objective(self, trial):
        # Set encoder, decoder, and bottleneck size
        n_encoder_activation = 2
        n_decoder_activation = 2
        n_bottleneck_activation = 2

        # Suggest hyperparameters
        starting_channel_size = trial.suggest_categorical('starting_channel_size', [16, 32, 64]) # Number of channels in the first layer

        # Get the maximum limit for the depth based on the starting channels and going up to the maximum number of nodes
        max_depth = int(np.log2(self.max_nodes / starting_channel_size)) + 1 # How many powers of 2 can we go up to reach the maximum number of nodes

        if max_depth < 2:
            print('[WARNING] The maximum depth is below 2. This may cause the search not work as intended. Setting the max_depth as 2 Consider increasing the value of starting_channels.')

        # Find the maximum number of layers (based on the maximum number of nodes) (depth)
        n_layers = trial.suggest_int('n_layers', 2, max_depth) # Number of layers (accounting for the starting layer)

        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        #region Encoder

        encoder_data = []

        # Dropout for the encoder

        for i in range(n_encoder_activation):

            use_dropout_encoder = trial.suggest_categorical(f'use_dropout_encoder_{i}', [True, False])
            if use_dropout_encoder:
                dropout_prob_encoder = trial.suggest_float(f'dropout_prob_encoder_{i}', 0.1, 0.5)
            else:
                dropout_prob_encoder = 0.5

            # Activation functions for the encoder

            activation_functions_encoder = []

            # Suggest the activation functions for the encoder
            for j in range(n_encoder_activation):
                activation_function_str = trial.suggest_categorical(f'activation_function_{i}_{j}_encoder', self.activation_functions_str)
                activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

                # Now suggest the parameters for the activation function
                if activation_function == nn.LeakyReLU:
                    activation_functions_encoder.append(
                        activation_function(
                            negative_slope = trial.suggest_float(f'negative_slope_{i}_{j}_encoder', 0.01, 0.5),
                            inplace = True
                        )
                    )
                else:
                    activation_functions_encoder.append(
                        activation_function(inplace = True)
                    )
                
            encoder_data.append((use_dropout_encoder, dropout_prob_encoder, activation_functions_encoder))

        #endregion

        #region Bottleneck

        # Activation functions for the bottleneck

        activation_functions_bottleneck = []

        # Suggest the activation functions for the bottleneck
        for i in range(n_bottleneck_activation):
            activation_function_str = trial.suggest_categorical(f'activation_function_{i}_bottleneck', self.activation_functions_str)
            activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                activation_functions_bottleneck.append(
                    activation_function(
                        negative_slope = trial.suggest_float(f'negative_slope_{i}_bottleneck', 0.01, 0.5),
                        inplace = True
                    )
                )
            else:
                activation_functions_bottleneck.append(
                    activation_function(inplace = True)
                )
        
        #endregion Bottleneck

        #region Decoder

        decoder_data = []

        # Dropout for the decoder
        
        for i in range(n_decoder_activation):
            use_dropout_decoder = trial.suggest_categorical(f'use_dropout_decoder_{i}', [True, False])
            if use_dropout_decoder:
                dropout_prob_decoder = trial.suggest_float(f'dropout_prob_decoder_{i}', 0.1, 0.5)
            else:
                dropout_prob_decoder = 0.5

            # Activation functions for the decoder
            
            activation_functions_decoder = []

            # Suggest the activation functions for the decoder
            for j in range(n_decoder_activation):
                activation_function_str = trial.suggest_categorical(f'activation_function_{i}_{j}_decoder', self.activation_functions_str)
                activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

                # Now suggest the parameters for the activation function
                if activation_function == nn.LeakyReLU:
                    activation_functions_decoder.append(
                        activation_function(
                            negative_slope = trial.suggest_float(f'negative_slope_{i}_{j}_decoder', 0.01, 0.5),
                            inplace = True
                            )
                    )
                else:
                    activation_functions_decoder.append(
                        activation_function(inplace = True)
                    )
            
            decoder_data.append((use_dropout_decoder, dropout_prob_decoder, activation_functions_decoder))
            
        #endregion

        #region Initialization
        
        # Suggest the initialization type
        init_type = trial.suggest_categorical('init_type', ['zeros', 'orthogonal', 'normal', 'uniform', 'constant', 'xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform', 'sparse', 'eye'])

        # If the initialization typem requires parameters, suggest them
        if init_type in ['normal']:
            mean = trial.suggest_float('mean', -1, 1)
            std = trial.suggest_float('std', 0.1, 1)
            init_params = {'mean': mean, 'std': std}
        elif init_type in ['uniform']:
            a = trial.suggest_float('a', -1, 1)
            b = trial.suggest_float('b', 0.1, 1)
            init_params = {'a': a, 'b': b}
        elif init_type in ['constant']:
            val = trial.suggest_float('val', -1, 1)
            init_params = {'val': val}
        elif init_type in ['sparse']:
            sparsity = trial.suggest_float('sparsity', 0.1, 1)
            init_params = {'sparsity': sparsity}
        elif init_type in ['orthogonal', 'xavier_uniform', 'xavier_normal']:
            init_params = {'gain': init.calculate_gain('relu')}
        elif init_type in ['he_uniform', 'he_normal']:
            a = trial.suggest_float('a', 0, 1)
            nonlinearity = trial.suggest_categorical('nonlinearity', ['relu', 'leaky_relu', 'tanh', 'sigmoid'])
            init_params = {'a': a, 'nonlinearity': nonlinearity}
        else:
            init_params = {}

        #endregion

        # Create the model
        model = UNet(
            in_channels = self.X_train.shape[1], 
            out_channels = self.output_size,
            n_layers = n_layers, 
            starting_channel_size = starting_channel_size,
            encoder_data = encoder_data,
            decoder_data = decoder_data,
            bottleneck_activation_functions = activation_functions_bottleneck,
            init_type = init_type,
            init_params = init_params,
            random_seed = self.random_seed,
            use_gpu = self.use_gpu,
            verbose = self.verbose
        )

        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 10, 100)

        criterion = nn.MSELoss()

        self.train_loader = DataLoader(
                dataset = CustomDataset(self.X_train, self.y_train), 
                batch_size = batch_size, 
                shuffle = True,
                drop_last = True
            )
        
        self.test_loader = DataLoader(
                dataset = CustomDataset(self.X_test, self.y_test), 
                batch_size = batch_size,
                drop_last = True
            )

        # If a validation set has been provided, create the validation loader
        if self.X_validation is not None:
            self.validation_loader = DataLoader(
                dataset = CustomDataset(self.X_validation, self.y_validation), 
                batch_size = batch_size, 
                shuffle = True,
                drop_last = True
            )
        
        # Suggestions for clipping the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)

        test_loss = self.train_test_model(model, self.train_loader, self.test_loader, optimizer, criterion, clip_grad, trial, batch_size, epochs = epochs)

        # If a validation set has been provided, calculate the AUC
        if self.validation_loader is not None:
            # Set the model to evaluation mode
            model.eval()
            
            # Get the predictions for the validation set
            validation_predictions = model(self.X_validation)

            # Convert the predictions and the labels to numpy
            validation_predictions_np = validation_predictions.detach().cpu().numpy()
            y_validation_np = self.y_validation.cpu().numpy() # type: ignore

            # If there is a nan in the predictions, set the AUC to 0
            if np.isnan(validation_predictions_np).any():
                validation_auc = 0
            else:
                # Calculate the ROC
                fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np) # type: ignore
                validation_auc = auc(fpr, tpr)
            # Set the optuna user attrs
            trial.set_user_attr('AUC', validation_auc)
        else:
            validation_auc = None

        return test_loss

    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "UNet_Optimization", load_if_exists = True, sampler: optuna.samplers.BaseSampler = TPESampler(), n_jobs = 1):
        if self.verbose:
            print(f'Optimizing the model for {n_trials} trials')

        # Add a pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = n_trials // 10, # Start pruning after 10% of the trials to allow better exploration
            n_warmup_steps = 15,               # Prune should act only if after 15 steps the value is still not in the median
        )

        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler,
            pruner = pruner
        )

        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)
        
        best_params = study.best_params

        print("Best Hyperparameters:", best_params)

        return best_params
