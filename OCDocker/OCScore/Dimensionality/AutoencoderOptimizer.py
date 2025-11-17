#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the Autoencoder. 

It is imported as:

from OCDocker.OCScore.NN.AutoencoderOptimizer import AutoencoderOptimizer
'''

# Imports
###############################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Dataset
from typing import Any, Union

import optuna
import random
import re

import OCDocker.Toolbox.Printing as ocprint

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


class AutoencoderDataset(Dataset):
    '''Dataset class for the Autoencoder. It is used to create the DataLoader for the training and testing of the Autoencoder.

    Parameters
    ----------
    features : torch.Tensor
        The features to be used in the Autoencoder. It should be a torch.Tensor of shape (n_samples, n_features).
    '''


    def __init__(self, features: torch.Tensor) -> None:
        '''Constructor for the AutoencoderDataset class. It is used to create the DataLoader for the training and testing of the Autoencoder.

        Parameters
        ----------
        features : torch.Tensor
            The features to be used in the Autoencoder. It should be a torch.Tensor of shape (n_samples, n_features).
        '''


        self.features = features









    def __len__(self) -> int:
        '''Returns the length of the dataset. It is used by the DataLoader to know how many samples are in the dataset.

        Returns
        -------
        int
            The length of the dataset. It is used by the DataLoader to know how many samples are in the dataset.
        '''


        return len(self.features)








    def __getitem__(self, idx: int) -> tuple:
        '''Returns the features and the target for the given index. It is used by the DataLoader to get the samples from the dataset.

        Parameters
        ----------
        idx : int
            The index of the sample to be returned.
        Returns
        -------
        tuple
            The features and the target for the given index. It is used by the DataLoader to get the samples from the dataset.
        '''

        return self.features[idx], self.features[idx]


class Autoencoder(nn.Module):
    '''Autoencoder class. It is used to create the Autoencoder model. It is a subclass of nn.Module.
    It is used to create the Autoencoder model.
    
    Parameters
    ----------
    input_size : int
        The size of the input. It should be a positive integer.
    encoding_dim : list
        The size of the encoding. It should be a list of integers.
    encoder_activation_fn : list[tuple(nn.Module, dict[str, Any]]
        The activation functions to be used in the encoder. It should be a list of tuples where each tuple will be the activation function and its parameters.
    decoder_activation_fn : list[tuple(nn.Module, dict[str, Any]]
        The activation functions to be used in the decoder. It should be a list of tuples where each tuple will be the activation function and its parameters.
    decoding_dim : list
        The size of the decoding. It should be a list of integers. 
    device : torch.device, optional
        The device to be used. It should be a torch.device. Default is torch.device("cpu").
    '''


    def __init__(self,
                 input_size : int,
                 encoding_dim : list,
                 encoder_activation_fn : list[tuple[nn.Module, dict[str, Any]]],
                 decoder_activation_fn : list[tuple[nn.Module, dict[str, Any]]],
                 decoding_dim : list,
                 device : torch.device = torch.device("cpu")
                ) -> None:
        ''' Constructor for the Autoencoder class. It is used to create the Autoencoder model.

        Parameters
        ----------
        input_size : int
            The size of the input. It should be a positive integer.
        encoding_dim : list
            The size of the encoding. It should be a list of integers.
        encoder_activation_fn : list[tuple[nn.Module, dict[str, Any]]]
            The activation functions to be used in the encoder. It should be a list of tuples where each tuple will be the activation function and its parameters.
        decoder_activation_fn : list[tuple[nn.Module, dict[str, Any]]]
            The activation functions to be used in the decoder. It should be a list of tuples where each tuple will be the activation function and its parameters.
        decoding_dim : list
            The size of the decoding. It should be a list of integers.
        device : torch.device, optional
            The device to be used. It should be a torch.device. Default is torch.device("cpu").
        '''

        super(Autoencoder, self).__init__()

        self.device = device

        # If the encoder is a list
        if isinstance(encoder_activation_fn, list):
            # Then the encoding_dim should be a list as well
            if not isinstance(encoding_dim, list):
                raise ValueError("If the encoder_activation_fn is a list, then the encoding_dim should be a list as well.")
            
            # Create the encoder layers to be added to the ModuleList
            encoder_layers = []
        
            # For each element in the list
            for i in range(len(encoder_activation_fn)):
                if len(encoder_activation_fn[i]) == 1:
                    act_func = encoder_activation_fn[i][0]().to(self.device)
                else:
                    pre_act_func, act_params = encoder_activation_fn[i]
                
                    # Create a new dictionary with the trailing numbers removed from the keys (also removing _encoder)
                    processed_act_params = {re.sub(r'_\d+$', '', k.replace('_encoder', '')): v for k, v in act_params.items()}

                    # Create the activation function
                    act_func = pre_act_func(**processed_act_params).to(self.device)

                # If it is the first element
                if i == 0:
                    # Add the first layer
                    encoder_layers.extend([
                        nn.Linear(input_size, encoding_dim[i]).to(self.device),
                        nn.BatchNorm1d(encoding_dim[i]).to(self.device),
                        act_func.to(self.device)
                    ])
                else:
                    # Add the rest of the layers
                    encoder_layers.extend([
                        nn.Linear(encoding_dim[i-1], encoding_dim[i]).to(self.device),
                        nn.BatchNorm1d(encoding_dim[i]).to(self.device),
                        act_func.to(self.device)
                    ])

            # Create the encoder as a ModuleList
            self.encoder = nn.ModuleList(encoder_layers)
            
            # Create the decoder layers to be added to the ModuleList
            decoder_layers = []

            # For each decoder layer
            for i in range(len(decoding_dim)):
                if len(decoder_activation_fn[i]) == 1:
                    act_func = decoder_activation_fn[i][0]().to(self.device)
                else:
                    pre_act_func, act_params = decoder_activation_fn[i]
                
                    # Create a new dictionary with the trailing numbers removed from the keys (also removing _decoder)
                    processed_act_params = {re.sub(r'_\d+$', '', k.replace('_decoder', '')): v for k, v in act_params.items()}

                    # Create the activation function
                    act_func = pre_act_func(**processed_act_params).to(self.device)

                # If it is the first element
                if i == 0:
                    # Add the first layer
                    decoder_layers.extend([
                        nn.Linear(encoding_dim[-1], decoding_dim[i]).to(self.device),
                        act_func.to(self.device)
                    ])
                else:
                    # Add the rest of the layers
                    decoder_layers.extend([
                        nn.Linear(decoding_dim[i-1], decoding_dim[i]).to(self.device),
                        act_func.to(self.device)
                    ])

            # Create the decoder as a ModuleList
            self.decoder = nn.ModuleList(decoder_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_size, encoding_dim).to(self.device),
                nn.BatchNorm1d(encoding_dim).to(self.device),
                encoder_activation_fn.to(self.device)
            ).to(self.device)

            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, input_size).to(self.device),
                decoder_activation_fn.to(self.device)

            ).to(self.device)








    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward pass of the Autoencoder. It is used to pass the input through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            The input to be passed through the Autoencoder. It should be a torch.Tensor of shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            The output of the Autoencoder. It should be a torch.Tensor of shape (n_samples, n_features).
        '''

        # If the encoder is an nn.Sequential
        if isinstance(self.encoder, nn.Sequential):
            # Add the encoder
            x = self.encoder(x)
        else:
            # Add the encoder
            for layer in self.encoder:
                x = layer(x)

        # If the decoder is an nn.Sequential
        if isinstance(self.decoder, nn.Sequential):
            # Add the decoder
            x = self.decoder(x)
        else:
            # Add the decoder
            for layer in self.decoder:
                x = layer(x)


        return x








    def get_encoder_topology(self) -> list:
        '''Get the topology of the encoder. It is used to get the layers of the encoder.
        
        Returns
        -------
        list
            The topology of the encoder. It is used to get the layers of the encoder.
        '''


        return ['Linear', 'BatchNorm1d']








    def get_decoder_topology(self) -> list:
        '''Get the topology of the decoder. It is used to get the layers of the decoder.

        Returns
        -------
        list
            The topology of the decoder. It is used to get the layers of the decoder.
        '''


        return ['Linear']








    def get_encoder(self) -> nn.Module:
        '''Get the encoder. It is used to get the encoder of the Autoencoder.

        Returns
        -------
        nn.Module
            The encoder of the Autoencoder. It is used to get the encoder of the Autoencoder.
        '''


        return self.encoder








    def get_decoder(self) -> nn.Module:
        '''Get the decoder. It is used to get the decoder of the Autoencoder.

        Returns
        -------
        nn.Module
            The decoder of the Autoencoder. It is used to get the decoder of the Autoencoder.
        '''

        return self.decoder


class AutoencoderOptimizer:
    '''AutoencoderOptimizer class. It is used to optimize the Autoencoder using Optuna. It is used to create the AutoencoderOptimizer object.

    Parameters
    ----------
    X_train : Union[np.ndarray, pd.DataFrame, pd.Series]
        The training data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series.
    X_test : Union[np.ndarray, pd.DataFrame, pd.Series]
        The testing data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series.
    X_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]], optional
        The validation data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series. Default is None.
    encoding_dims : tuple, optional
        The dimensions of the encoding. It should be a tuple of two integers. Default is (16, 256).
    storage : str, optional
        The storage string for the study. It should be a string. Default is "sqlite:///autoencoder.db".
    models_folder : str, optional
        The folder where the models will be saved. It should be a string. Default is "./models/Autoencoder/".
    random_seed : int, optional
        The random seed to be used in the Autoencoder. It should be a positive integer. Default is 42.
    use_gpu : bool, optional
        If True, the Autoencoder will use the GPU. It should be a boolean. Default is True.
    verbose : bool, optional
        If True, the Autoencoder will print the training and testing information. It should be a boolean. Default is False.
    '''

    def __init__(self, 
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            encoding_dims: tuple = (16, 256),
            storage: str = "sqlite:///autoencoder.db",
            models_folder: str = "./models/Autoencoder/",
            random_seed: int = 42, 
            use_gpu: bool = True,
            verbose: bool = False
        ) -> None:
        '''Constructor for the AutoencoderOptimizer class. It is used to create the AutoencoderOptimizer object.

        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame, pd.Series]
            The training data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series.
        X_test : Union[np.ndarray, pd.DataFrame, pd.Series]
            The testing data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series.
        X_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            The validation data to be used in the Autoencoder. It should be a numpy array, pandas DataFrame or pandas Series. Default is None.
        encoding_dims : tuple, optional
            The dimensions of the encoding. It should be a tuple of two integers. Default is (16, 256).
        storage : str, optional
            The storage string for the study. It should be a string. Default is "sqlite:///autoencoder.db".
        models_folder : str, optional
            The folder where the models will be saved. It should be a string. Default is "./models/Autoencoder/".
        random_seed : int, optional
            The random seed to be used in the Autoencoder. It should be a positive integer. Default is 42.
        use_gpu : bool, optional
            If True, the Autoencoder will use the GPU. It should be a boolean. Default is True.
        verbose : bool, optional
            If True, the Autoencoder will print the training and testing information. It should be a boolean. Default is False.
        '''

        # Set the random seed
        self.random_seed = random_seed

        # Set the models folder
        self.models_folder = models_folder

        # Set the gpu flag
        self.use_gpu = use_gpu

        # Define all the random seeds
        self.set_random_seed()    
        
        # Convert the data do np.ndarray then to torch.Tensor
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        self.test_loader = None

        # Only convert the validation data to torch.Tensor if it is not None
        if X_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.validation_loader = None
        else:
            self.X_validation = None
            self.validation_loader = None

        # Set the input size
        self.input_size = self.X_train.shape[1]

        # Set the encoding dimensions
        self.encoding_dims = encoding_dims

        # Set the verbose flag
        self.verbose = verbose

        # Set the best rmse to infinity
        self.best_rmse = np.inf

        # Set the storage string for the study
        self.storage = storage

        # Define the power of two options 
        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256, 512, 1024, 2048, 4096

        # Define the activation functions and their names
        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]

        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']








    def set_random_seed(self) -> None:
        '''Set the random seed for the Autoencoder. It is used to set the random seed for the Autoencoder.'''

        # Set the random seed for numpy and random
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # If using GPU, set the seed for GPU as well
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.random_seed)
        else:
            self.device = torch.device('cpu')

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)
        
        # This is not recommended for performance since it will disable the cudnn auto-tuner (reason why it is commented)
        #torch.backends.cudnn.enabled = False

        # Set the backends for reproducibility
        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True








    def train_autoencoder(self,
                          model : nn.Module,
                          optimizer : optim.Optimizer,
                          criterion : nn.Module,
                          clip_grad : float,
                          epochs : int,
                          trial : optuna.Trial
                          ) -> tuple:
        '''Train the Autoencoder. It is used to train the Autoencoder.

        Parameters
        ----------
        model : nn.Module
            The Autoencoder model to be trained. It should be a nn.Module.
        optimizer : optim.Optimizer
            The optimizer to be used in the Autoencoder. It should be a optim.Optimizer.
        criterion : nn.Module
            The loss function to be used in the Autoencoder. It should be a nn.Module.
        clip_grad : float
            The gradient clipping value to be used in the Autoencoder. It should be a float.
        epochs : int
            The number of epochs to be used in the Autoencoder. It should be a positive integer.
        trial : optuna.Trial
            The Optuna trial to be used in the Autoencoder. It should be a optuna.Trial.
        
        Returns
        -------
        tuple
            The best validation and training RMSE. It is used to get the best validation and training RMSE.
        '''

        # Set the best validation and training rmse to infinity
        best_validation_rmse = np.inf
        best_train_rmse = np.inf

        # Set the model to training mode (this is important for some layers which behave differently during training and evaluation)
        model.train()

        # Loop over the epochs
        for epoch in range(epochs):
            # Print verbose information
            if self.verbose:
                ocprint.printv(f"Epoch {epoch+1}/{epochs}")

            # Set the running loss to 0.0 for this epoch
            running_loss = 0.0

            # Loop over the training data
            for data, _ in self.train_loader: # type: ignore
                # Check if the data is on the same device as the model
                data = data.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                reconstruction = model(data)

                # Compute the loss            
                loss = criterion(reconstruction, data)

                # Backward pass and optimization
                loss.backward()

                # Clip the gradients
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                # Update the weights
                optimizer.step()

                # Update the running loss
                running_loss = running_loss + loss.item()

            # Compute the average loss
            average_loss = running_loss / len(self.train_loader) # type: ignore

            # Compute the RMSE
            rmse = np.sqrt(average_loss)

            # Validation phase
            if self.validation_loader is not None:
                val_rmse = self.evaluate_autoencoder(model, criterion, self.validation_loader)

                trial.set_user_attr('val_rmse', val_rmse)
                
                if self.verbose:
                    ocprint.printv(f"Epoch {epoch+1}, Validation Loss: {val_rmse}")

                # Check for improvement
                if val_rmse < best_validation_rmse:
                    best_train_rmse = rmse
                    best_validation_rmse = val_rmse
            
            # Print some more verbose information
            if self.verbose:
                ocprint.printv(f'Test Loss: {average_loss}')
                ocprint.printv(f'Test RMSE: {rmse}')

            # Report the intermediate value to Optuna
            trial.report(rmse, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            

        return best_validation_rmse, best_train_rmse








    def evaluate_autoencoder(self,
                             model : nn.Module,
                             criterion : nn.Module,
                             loader : Union[None, DataLoader] = None
                            ) -> float:
        '''Evaluate the Autoencoder. It is used to evaluate the Autoencoder.

        Parameters
        ----------
        model : nn.Module
            The Autoencoder model to be evaluated. It should be a nn.Module.
        criterion : nn.Module
            The loss function to be used in the Autoencoder. It should be a nn.Module.
        loader : Union[None, DataLoader], optional
            The DataLoader to be used in the Autoencoder. It should be a DataLoader. Default is None.

        Returns
        -------
        float
            The RMSE of the Autoencoder. It is used to get the RMSE of the Autoencoder.
        '''

        # Set the random seed (this is done here because the user might want to just evaluate the model)
        self.set_random_seed()

        # Set the model to evaluation mode (this is important for some layers which behave differently during training and evaluation)
        model.eval()

        # Set the total loss to 0.0
        total_loss = 0
        
        # If the loader is None, use the test_loader
        if loader is None:
            loader = self.test_loader

        # Set the torch.no_grad() to avoid computing gradients
        # This is important for the evaluation phase to save memory and computations
        with torch.no_grad():
            # Loop over the data
            for data, _ in loader: # type: ignore
                # Check if the data is on the same device as the model
                data = data.to(self.device)

                # Compute the reconstruction
                reconstruction = model(data)

                # Compute the loss
                loss = criterion(reconstruction, data)

                # Update the total loss
                total_loss = total_loss + loss.item()

        # Compute the average loss
        average_loss = total_loss / len(loader) # type: ignore

        # Compute the RMSE
        rmse = np.sqrt(average_loss)
        

        return rmse








    def objective(self, trial : optuna.Trial) -> float:
        '''Objective function for the Optuna optimization. It is used to optimize the Autoencoder.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial to be used in the Autoencoder. It should be a optuna.Trial.
        
        Returns
        -------
        float
            The RMSE of the Autoencoder. It is used to get the RMSE of the Autoencoder.
        '''

        # Set the random seed (for safety)
        self.set_random_seed()
        
        # Suggest the learning rate, batch size, clip_grad, and epochs values
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)
        epochs = trial.suggest_int('epochs', 20, 100)

        # Suggest the number of hidden layers and the number of units in each layer for the encoder
        encoder_hidden_layers = []

        # Suggest the number of layers for the encoder
        encoder_nlayers = trial.suggest_int('n_layers_encoder', 1, 2)
        
        # For each layer
        for i in range(encoder_nlayers):
            # If is the last layer
            if i == encoder_nlayers - 1:
                # Its size should be smaller than the input size, so respect the limits imposed by the encoding_dims tuple
                encoder_hidden_layers.append(trial.suggest_int(f'n_units_layer_{i}_encoder', self.encoding_dims[0], self.encoding_dims[1]))
            else:
                # Otherwise, suggest a power of two
                encoder_hidden_layers.append(trial.suggest_int(f'n_units_layer_{i}_encoder', self.power_of_two_options[0], self.power_of_two_options[-1]))
        
        # Suggestions for the activation functions of the encoder
        encoder_activation_data = []

        for i in range(len(encoder_hidden_layers)):
            activation_function_str = trial.suggest_categorical(f'activation_function_{i}_encoder', self.activation_functions_str)
            activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                encoder_activation_data.append((activation_function, {
                    f'negative_slope_{i}': trial.suggest_float(f'negative_slope_{i}_encoder', 0.01, 0.5)
                }))
            elif activation_function == nn.GELU:
                encoder_activation_data.append((activation_function, {
                    f'approximate_{i}': trial.suggest_categorical(f'approximate_{i}_encoder', ['none', 'tanh'])
                }))  
            else:
                encoder_activation_data.append((activation_function, {}))

        # Suggest the number of hidden layers and the number of units in each layer for the decoder
        decoder_hidden_layers = []

        # If the encoder have more than one layer
        if encoder_nlayers > 1:
            # The decoder should have at least 2 layers
            decoder_nlayers = trial.suggest_int('n_layers_decoder', 2, 2)
        else:
            # It should have only one layer
            decoder_nlayers = trial.suggest_int('n_layers_decoder', 1, 1)
        
        # For each layer
        for i in range(decoder_nlayers):
            # If is the last layer
            if i == decoder_nlayers - 1:
                # Its size should be the input size
                decoder_hidden_layers.append(self.input_size)
            # If is the first layer
            elif i == 0:
                # It should be the same as the last layer of the encoder
                decoder_hidden_layers.append(encoder_hidden_layers[-1 - i])
            else:
                # Otherwise, suggest a power of two
                decoder_hidden_layers.append(trial.suggest_categorical(f'n_units_layer_{i}_decoder', self.power_of_two_options))

        # Suggestions for the activation functions of the decoder
        decoder_activation_data = []

        # For each layer in the encoder (same number of layers as the decoder)
        for i in range(len(encoder_hidden_layers)):
            activation_function_str = trial.suggest_categorical(f'activation_function_{i}_decoder', self.activation_functions_str)
            activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                decoder_activation_data.append((activation_function, {
                    f'negative_slope_{i}': trial.suggest_float(f'negative_slope_{i}_decoder', 0.01, 0.5)
                }))
            elif activation_function == nn.GELU:
                decoder_activation_data.append((activation_function, {
                    f'approximate_{i}': trial.suggest_categorical(f'approximate_{i}_decoder', ['none', 'tanh'])
                }))  
            else:
                decoder_activation_data.append((activation_function, {}))

        # Create the Autoencoder model
        model = Autoencoder(self.input_size, encoder_hidden_layers, encoder_activation_data, decoder_activation_data, decoder_hidden_layers, self.device).to(self.device)

        # Suggest the optimizer and weight decay
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)

        # Set the optimizer params
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Set the criterion
        criterion = nn.MSELoss()

        # Check if self.train_dataset and self.test_dataset are not None
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("train_dataset and test_dataset must be set before calling the objective function.")

        # Create the DataLoader for the training and testing datasets
        self.train_loader = DataLoader(
            dataset = self.train_dataset, 
            batch_size = batch_size, 
            shuffle = True
        )

        self.test_loader = DataLoader(
            dataset = self.test_dataset, 
            batch_size = batch_size
        )

        # If the validation dataset is not None, create the DataLoader for the validation dataset
        if self.validation_dataset is not None:
            self.validation_loader = DataLoader(
                dataset = self.validation_dataset, 
                batch_size = batch_size
            )

        # Train the Autoencoder
        best_validation_rmse, best_train_rmse = self.train_autoencoder(model, optimizer, criterion, clip_grad, epochs, trial = trial) # type: ignore

        # Evaluate the Autoencoder
        evaluate_rmse = self.evaluate_autoencoder(model, criterion)

        # Set the improvement threshold
        improvement_threshold = 0.0 # 0% improvement (if is better than the best, it will be logged)

        # Check if the best validation RMSE is better than the best RMSE with the improvement threshold
        is_promising = best_validation_rmse < self.best_rmse * (1 - improvement_threshold)

        # If the model is promising
        if is_promising:
            # Save the model (commented due to potential high disk usage)
            #torch.save(model.state_dict(), f'{self.models_folder}/autoencoder_{trial.number}.pt')
            # Set its rmse
            self.best_rmse = best_validation_rmse


        return evaluate_rmse








    def optimize(self,
                 direction: str = "maximize",
                 n_trials : int = 10,
                 study_name : str = "NN_Optimization",
                 load_if_exists : bool = True,
                 sampler : optuna.samplers.BaseSampler = TPESampler(),
                 n_jobs : int = 1
                ) -> optuna.study.Study:
        '''Optimize the Autoencoder. It is used to optimize the Autoencoder.

        Parameters
        ----------
        direction : str, optional
            The direction of the optimization. It should be a string. Default is "maximize".
        n_trials : int, optional
            The number of trials to be used in the Autoencoder. It should be a positive integer. Default is 10.
        study_name : str, optional
            The name of the study. It should be a string. Default is "NN_Optimization".
        load_if_exists : bool, optional
            If True, the study will be loaded if it exists. It should be a boolean. Default is True.
        sampler : optuna.samplers.BaseSampler, optional
            The sampler to be used in the Autoencoder. It should be a optuna.samplers.BaseSampler. Default is TPESampler().
        n_jobs : int, optional
            The number of jobs to be used in the Autoencoder. It should be a positive integer. Default is 1.
        
        Returns
        -------
        optuna.study.Study
            The Optuna study. It is used to get the study of the Autoencoder.
        '''
        
        # Set the training and testing datasets
        self.train_dataset = AutoencoderDataset(self.X_train)
        self.test_dataset = AutoencoderDataset(self.X_test)

        # Set the validation dataset if it is not None
        if self.X_validation is not None:
            self.validation_dataset = AutoencoderDataset(self.X_validation)

        # Add a pruner (Optuna optimizations to kill unpromising trials)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = n_trials // 10, # Start pruning after 10% of the trials
            n_warmup_steps = 15,               # After at least 15 epochs
        )

        # Create the study
        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler,
            pruner = pruner
        )

        # Perform the optimization
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)
        
        # If verbose, print the best trial data
        if self.verbose:
            ocprint.printv("Best trial:")

            trial = study.best_trial

            ocprint.printv(f"  Value:  {trial.value}" )
            ocprint.printv("  Params: ")

            for key, value in trial.params.items():
                ocprint.printv(f"    {key}: {value}")

        return study

# Methods
###############################################################################
