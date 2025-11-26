#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the Neural Network. 

It is imported as:

from OCDocker.OCScore.DNN.DNNOptimizer import DNNOptimizer
'''

# Imports
###############################################################################

import optuna
import random
import re

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from optuna.samplers import TPESampler
from sklearn.metrics import (
    auc, 
    log_loss,
    mean_absolute_error,
    precision_recall_curve, 
    roc_curve
)
from typing import Any, Union

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


class NeuralNet(nn.Module):
    ''' Neural Network class for the optimization of the neural network.

    Parameters
    ----------
    input_size : int
        Size of the input layer
    output_size : int
        Size of the output layer
    encoder_params : Union[None, dict, tuple[dict, dict, dict]]
        Parameters for the encoder
    nn_params : dict
        Parameters for the neural network
    random_seed : int, optional
        Random seed for the neural network, by default 42
    use_gpu : bool, optional
        Use GPU for the neural network, by default True
    verbose : bool, optional
        Verbose mode for the neural network, by default False
    mask : Union[None, list[Union[int, bool]], np.ndarray], optional
        Mask for the neural network, by default None
    '''


    def __init__(self, 
            input_size : int,
            output_size : int,
            encoder_params : Union[None, dict, tuple[dict, dict, dict]],
            nn_params : dict,
            random_seed : int = 42,
            use_gpu : bool = True,
            verbose :bool = False,
            mask : Union[None, list[Union[int, bool]], np.ndarray] = None
            ) -> None:
        '''Initialize the NeuralNet class

        Parameters
        ----------
        input_size : int
            Size of the input layer
        output_size : int
            Size of the output layer
        encoder_params : Union[None, dict, tuple[dict, dict, dict]]
            Parameters for the encoder
        nn_params : dict
            Parameters for the neural network
        random_seed : int, optional
            Random seed for the neural network, by default 42
        use_gpu : bool, optional
            Use GPU for the neural network, by default True
        verbose : bool, optional
            Verbose mode for the neural network, by default False
        mask : Union[None, list[Union[int, bool]], np.ndarray], optional
            Mask for the neural network, by default None
        '''

        # Call the parent constructor
        super(NeuralNet, self).__init__()

        # Set the random seed
        self.random_seed = random_seed

        # Set the gpu flag
        self.use_gpu = use_gpu

        # Set the input size
        self.input_size = input_size

        # Set the random seed to ensure reproducibility
        self.set_random_seed()

        # Define the activation functions and its string names
        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']
        
        # Define the optimizer functions and its string names
        self.optimizer_functions = [optim.Adam, optim.RMSprop, optim.SGD]
        self.optimizer_functions_str = ['Adam', 'RMSprop', 'SGD']

        # Define the input layer
        self.layers = nn.ModuleList()

        # Set the mask
        #self.mask = mask if mask != None else []
        self.mask = mask if mask is not None else []

        # Process the activation functions
        hidden_layers = []
        activation_data_dict = {}
        
        # For each key, value pair in the nn_params
        for key, value in nn_params.items():
            # Check if the key is an activation function
            if key.startswith('activation_function'):
                # Get the index of the activation function
                index = int(key.split('_')[-1])
                
                # Get the activation function and its parameters
                activation_data_dict[index] = [self.activation_functions[self.activation_functions_str.index(nn_params[f'activation_function_{index}'])]]
            
            # Check if the key is the number of units in a layer
            elif key.startswith('n_units_layer'):
                hidden_layers.append(value)
            
            # Check if the key is a parameter for an activation function (ends with a number)
            elif re.search(r'_\d+$', key):
                # Get the index of the activation function parameter
                index = int(key.split('_')[-1])
                
                # Remove the index from the key
                key = re.sub(r'_\d+$', '', key)
                
                # Add the parameter to the second element of the list dict, creating the dict if it doesn't exist
                if index in activation_data_dict:
                    activation_data_dict[index].append({key: value})
                else:
                    activation_data_dict[index] = [{key: value}]

            # Convert the activation_data_dict to a list while keeping the order
            activation_data = [v for _, v in activation_data_dict.items()]

        ''' TODO: Remove this or fully implement it (preliminary test have shown that it does not produce good results, but further testing is advised)
        # If the encoder is instance of list (multi branch model)
        if isinstance(encoder_params, list):
            self.encoder = []
            # Loop through the encoder_params (one branch at a time)
            for _, encoder_param in enumerate(encoder_params):
                self.encoder.append(self.__build_encoder_layer(encoder_param))
        elif encoder_params is not None:
            # Build the one branch encoder
            self.encoder = self.__build_encoder_layer(encoder_params)
        else:
            # No encoder
            self.encoder = None
        '''

        # If the encoder is instance of dict (single branch model)
        if encoder_params is not None:
            if isinstance(encoder_params, dict):
                self.encoder = self.__build_encoder(encoder_params)
            else: # It is a tuple
                # Split the tuple into 3 parts
                sf_encoder_params, lig_encoder_params, rec_encoder_params = encoder_params
                self.encoder = [
                    self.__build_encoder(sf_encoder_params), 
                    self.__build_encoder(lig_encoder_params), 
                    self.__build_encoder(rec_encoder_params)
                ]
        else:
            self.encoder = None
        
        # If the there are multiple branches
        if isinstance(encoder_params, list):
            # Create the MultiBranchDynamicNN
            self.NN = MultiBranchDynamicNN(input_size, output_size, hidden_layers, activation_data, self.encoder, self.device)
        else:
            # Create the DynamicNN
            self.NN = DynamicNN(input_size, output_size, hidden_layers, activation_data, self.encoder, self.device, mask = self.mask)

        # Set the parameters: batch size, epochs, learning rate, clip grad.
        self.batch_size = nn_params['batch_size']
        self.epochs = nn_params['epochs']
        self.lr = nn_params['lr']
        self.clip_grad = nn_params['clip_grad']

        # Set the optimizer by searching in the optimizer_functions list at the same index as the optimizer_functions_str and then set the parameters of it
        self.optimizer = self.optimizer_functions[self.optimizer_functions_str.index(nn_params['optimizer'])](
            self.NN.parameters(),
            weight_decay = nn_params['weight_decay'], 
            lr = nn_params['lr']
        )

        # Set the neural network parameters
        self.nn_params = nn_params

        # Set the AUC and rmse as nan
        self.validation_auc = np.nan
        self.rmse = np.nan
        self.pr_auc = np.nan
        self.log_loss_value = np.nan
        self.mae = np.nan

        # Set the verbose flag
        self.verbose = verbose

        # Since it is the instantiation of the class, set the prediction to None
        self.prediction = None

        # If verbose is True, print the neural network
        if verbose:

            ocprint.printv(self.NN) # type: ignore









    def __build_encoder(self, encoder_params : dict) -> list:
        ''' Build the encoder for the neural network
        
        Parameters
        ----------
        encoder_params : dict
            Parameters for the encoder
            
        Returns
        -------
        list
            List of tuples with the encoder layers
        '''

        # If the encoder_params has the key 'encoder_activation'
        if 'encoder_activation' in encoder_params:
            # Find which activation function to use and set the parameters accordingly
            # If the activation function is LeakyReLU, set the negative_slope
            if encoder_params['encoder_activation'] == 'LeakyReLU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](negative_slope = encoder_params['negative_slope_encoder'])
            # If the activation function is GELU, set the approximate
            elif encoder_params['encoder_activation'] == 'GELU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](approximate = encoder_params['approximate_encoder'])
            # Otherwise, just set the activation function since it does not have any parameters to be set
            else:
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])]()

            # Build just the encoder
            return [("Linear", self.input_size, encoder_params['encoding_dim']), ("BatchNorm1d", encoder_params['encoding_dim']), ("Activation", encoder_activation)]
        
        # Create an empty list to store the encoder
        encoder = []

        # Get all the keys from the encoder_params which starts with 'activation_function'
        activation_keys = [key for key in encoder_params.keys() if key.startswith('activation_function') and key.endswith('encoder')]
        
        # If there are no activation functions
        if not activation_keys:
            raise ValueError("The encoder_params should have at least one activation function")

        # Process the activation functions
        for i in range(encoder_params['n_layers_encoder']):
            # Now suggest the parameters for the activation function
            if encoder_params[f'activation_function_{i}_encoder'] == 'LeakyReLU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])](negative_slope = encoder_params[f'negative_slope_{i}_encoder'])
            elif encoder_params[f'activation_function_{i}_encoder'] == 'GELU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])](approximate = encoder_params[f'approximate_{i}_encoder'])
            else:
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])]()
            
            # If it is the first layer
            if i == 0:
                # Add the encoder layer to the encoder list
                encoder.extend([
                    ("Linear", self.input_size, encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("BatchNorm1d", encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("Activation", encoder_activation)
                ])
            else:
                # Add the encoder layer to the encoder list
                encoder.extend([
                    ("Linear", encoder_params[f'n_units_layer_{i-1}_encoder'], encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("BatchNorm1d", encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("Activation", encoder_activation)
                ])
            

        return encoder








    def __build_encoder_layer(self, encoder_params : dict) -> list:
        ''' Build the encoder for the neural network

        Parameters
        ----------
        encoder_params : dict
            Parameters for the encoder
        
        Returns
        -------
        list
            List of tuples with the encoder layers
        '''

        # Create an empty list to store the encoder layers
        encoder_layer = []

        # For each key in the encoder_param
        for key in encoder_params.keys():
            # Check if the key is an activation function for the encoder
            if key.startswith('activation_function') and key.endswith('encoder'):
                # Get the index of the activation function (index -2 since -1 will be 'encoder')
                index = int(key.split('_')[-2])
                
                if encoder_params[f'activation_function_{index}_encoder'] == 'LeakyReLU':
                    encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{index}_encoder'])](negative_slope = encoder_params[f'negative_slope_{index}_encoder'])
                elif encoder_params[f'activation_function_{index}_encoder'] == 'GELU':
                    encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{index}_encoder'])](approximate = encoder_params[f'approximate_{index}_encoder'])
                else:
                    encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{index}_encoder'])]()

                if index == 0:
                    # Add the encoder layer to the encoder list
                    encoder_layer.append([
                        ("Linear", self.input_size, encoder_params[f'n_units_layer_{index}_encoder']), 
                        ("BatchNorm1d", encoder_params[f'n_units_layer_{index}_encoder']), 
                        ("Activation", encoder_activation)
                    ])
                else:
                    # Add the encoder layer to the encoder list
                    encoder_layer.append([
                        ("Linear", encoder_params[f'n_units_layer_{index - 1}_encoder'], encoder_params[f'n_units_layer_{index}_encoder']), 
                        ("BatchNorm1d", encoder_params[f'n_units_layer_{index}_encoder']), 
                        ("Activation", encoder_activation)
                    ])

        # If the encoder_layer has only one element, return the element
        if len(encoder_layer) == 1:
            return encoder_layer[0]
        
        # Otherwise, return the list

        return encoder_layer








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








    def train_model(self,
                    X_train : Union[list, torch.Tensor, np.ndarray],
                    y_train : Union[list, torch.Tensor, np.ndarray],
                    X_test : Union[list, torch.Tensor, np.ndarray],
                    y_test : Union[list, torch.Tensor, np.ndarray],
                    X_validation : Union[list, torch.Tensor, np.ndarray, None] = None,
                    y_validation : Union[list, torch.Tensor, np.ndarray, None] = None,
                    criterion: nn.Module = nn.MSELoss()) -> None:
        '''Train the neural network

        Parameters
        ----------
        X_train : Union[list, torch.Tensor, np.ndarray]
            Training data
        y_train : Union[list, torch.Tensor, np.ndarray]
            Training labels
        X_test : Union[list, torch.Tensor, np.ndarray]
            Testing data
        y_test : Union[list, torch.Tensor, np.ndarray]
            Testing labels
        X_validation : Union[list, torch.Tensor, np.ndarray, None], optional
            Validation data, by default None
        y_validation : Union[list, torch.Tensor, np.ndarray, None], optional
            Validation labels, by default None
        criterion : nn.Module, optional
            Loss function, by default nn.MSELoss()
        '''

        self.set_random_seed()

        # Convert the data to torch.Tensor
        if isinstance(X_train, list):
            X_train = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_train]
        elif isinstance(X_train, torch.Tensor):
            X_train = X_train.to(self.device)
        else:
            X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)

        # If y_train is already a tensor, do not convert it and just move it to the device
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.to(self.device)
        else:
            y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)

        if isinstance(X_test, list):
            X_test = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_test]
        elif isinstance(X_test, torch.Tensor):
            X_test = X_test.to(self.device)
        else:
            X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)

        # If y_test is already a tensor, do not convert it and just move it to the device
        if isinstance(y_test, torch.Tensor):
            y_test = y_test.to(self.device)
        else:
            y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)

        if X_validation is not None and y_validation is not None:
            if isinstance(X_validation, list):
                X_validation = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_validation]
            elif isinstance(X_validation, torch.Tensor):
                X_validation = X_validation.to(self.device)
            else:
                X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            
            # If y_validation is already a tensor, do not convert it and just move it to the device
            if isinstance(y_validation, torch.Tensor):
                y_validation = y_validation.to(self.device)
            else:
                y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)

        # If the input is a list create the train and test loaders
        if isinstance(X_train, list):
            train_loader = DataLoader(
                dataset = MultiBranchCustomDataset(X_train[0], X_train[1], X_train[2], y_train), 
                batch_size = self.batch_size, 
                shuffle = True
            )
        else:
            train_loader = DataLoader(
                dataset = CustomDataset(X_train, y_train), 
                batch_size = self.batch_size, 
                shuffle = True
            )

        # If the input is a list create the train and test loaders
        if isinstance(X_test, list):
            test_loader = DataLoader(
                dataset = MultiBranchCustomDataset(X_test[0], X_test[1], X_test[2], y_test), 
                batch_size = self.batch_size, 
                shuffle = True
            )
        else:
            test_loader = DataLoader(
                dataset = CustomDataset(X_test, y_test), 
                batch_size = self.batch_size
            )

        # If a validation set has been provided, create the validation loader (if X_validation is not None, y_validation is not None as well)
        if X_validation is not None:
            if isinstance(X_validation, list):
                validation_loader = DataLoader(
                    dataset = MultiBranchCustomDataset(X_validation[0], X_validation[1], X_validation[2], y_validation), # type: ignore
                    batch_size = self.batch_size, 
                    shuffle = True
                )
            else:
                validation_loader = DataLoader(
                    dataset = CustomDataset(X_validation, y_validation), # type: ignore
                    batch_size = self.batch_size, 
                    shuffle = True
                )

        # For each epoch
        for epoch in range(self.epochs):
            # Set the model to training mode
            self.NN.train()

            # Set the running loss to 0            
            running_loss = 0.0

            # If the train loader is a multi branch dataset
            if isinstance(train_loader.dataset, MultiBranchCustomDataset):
                for i, (inputs1, inputs2, inputs3, labels) in enumerate(train_loader):
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    outputs = self.NN([inputs1, inputs2, inputs3])                 # Forward pass
                    loss = criterion(outputs, labels.view(-1, 1))                  # Calculate the loss
                    loss.backward()                                                # Backward pass
                    nn.utils.clip_grad_norm_(self.NN.parameters(), self.clip_grad) # Clip the gradients
                    self.optimizer.step()                                          # Update weights

                    running_loss = running_loss + loss.item()
            else:                
                for i, (inputs, labels) in enumerate(train_loader):
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    outputs = self.NN(inputs)                                      # Forward pass
                    loss = criterion(outputs, labels.view(-1, 1))                  # Calculate the loss
                    loss.backward()                                                # Backward pass
                    nn.utils.clip_grad_norm_(self.NN.parameters(), self.clip_grad) # Clip the gradients
                    self.optimizer.step()                                          # Update weights

                    running_loss = running_loss + loss.item()
        
            # Set the model to evaluation mode
            self.NN.eval()

            running_loss = 0.0

            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                # If the test loader is a multi branch dataset
                if isinstance(test_loader.dataset, MultiBranchCustomDataset):
                    for inputs1, inputs2, inputs3, labels in test_loader:
                        predicted = self.NN([inputs1, inputs2, inputs3])
                        loss = criterion(predicted, labels.view(-1, 1))
                        running_loss = running_loss + loss.item()
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                else:
                    for inputs, labels in test_loader:
                        predicted = self.NN(inputs)
                        loss = criterion(predicted, labels.view(-1, 1))
                        running_loss = running_loss + loss.item()
                        
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

            average_loss = running_loss / len(test_loader)
            rmse = np.sqrt(average_loss)

            if self.verbose:
                ocprint.printv(f'Epoch {epoch + 1}/{self.epochs}')
                ocprint.printv(f'Average Loss: {average_loss}')
                ocprint.printv(f'RMSE: {rmse}')

            # If a validation set has been provided, calculate the AUC
            if X_validation is not None:
                # Set the model to evaluation mode
                self.NN.eval()

                # If the validation loader is a multi branch dataset
                if isinstance(validation_loader.dataset, MultiBranchCustomDataset):
                    validation_predictions = self.NN([X_validation[0], X_validation[1], X_validation[2]])
                else:
                    validation_predictions = self.NN(X_validation)

                # Convert the predictions and the labels to numpy
                validation_predictions_np = validation_predictions.detach().cpu().numpy()
                y_validation_np = y_validation.cpu().numpy() # type: ignore

                self.prediction = validation_predictions_np

                # If there is a nan in the predictions, set the AUC to 0
                if np.isnan(validation_predictions_np).any():
                    validation_auc = 0
                    precision = 0
                    log_loss_value = 0
                    mae = 0
                else:
                    # Check if validation set has only one class (can't compute AUC/log_loss)
                    unique_classes = np.unique(y_validation_np)
                    if len(unique_classes) == 1:
                        # Only one class in validation set - can't compute classification metrics
                        validation_auc = 0.0
                        pr_auc = 0.0
                        log_loss_value = np.inf
                        mae = mean_absolute_error(y_validation_np, validation_predictions_np)
                        if self.verbose:
                            ocprint.printv(f'Warning: Validation set contains only one class ({unique_classes[0]}). Cannot compute AUC/log_loss.') # type: ignore
                    else:
                        # Calculate the ROC
                        fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np)
                        validation_auc = auc(fpr, tpr)
                   
                        # Calculate the PR AUC
                        precision, recall, _ = precision_recall_curve(y_validation_np, validation_predictions_np)
                        pr_auc = auc(recall, precision)
                        
                        # Calculate the log loss (with error handling for edge cases)
                        try:
                            log_loss_value = log_loss(y_validation_np, validation_predictions_np)
                        except ValueError as e:
                            # Handle cases where log_loss fails (e.g., single class, invalid predictions)
                            log_loss_value = np.inf
                            if self.verbose:
                                ocprint.printv(f'Warning: Could not compute log_loss: {e}') # type: ignore
        
                        # Calculate the Mean Absolute Error
                        mae = mean_absolute_error(y_validation_np, validation_predictions_np)
    
        # Set the optuna user attrs
        self.rmse = rmse
        self.validation_auc = validation_auc
        self.pr_auc = pr_auc
        self.log_loss_value = log_loss_value

        self.mae = mae








    def get_model(self):
        return self.NN


class DynamicNN(nn.Module):
    ''' Dynamic Neural Network class for the optimization of the neural network.
    
    Parameters
    ----------
    input_size : int
        Size of the input layer
    output_size : int
        Size of the output layer
    hidden_layers : list
        List of hidden layers
    activation_data : list, optional
        List of activation functions, by default []
    encoder : Union[None, list], optional
        Encoder for the neural network, by default None
    device : torch.device, optional
        Device for the neural network, by default torch.device('cpu')
    mask : Union[None, list[Union[int, bool]], np.ndarray], optional
        Mask for the neural network, by default None
    '''


    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: list,
            activation_data: list = [],
            encoder: Union[None, list] = None,
            device: torch.device = torch.device('cpu'),
            mask: Union[None, list[Union[int, bool]], np.ndarray] = None
        ) -> None:
        '''Initialize the DynamicNN class

        Parameters
        ----------
        input_size : int
            Size of the input layer
        output_size : int
            Size of the output layer
        hidden_layers : list
            List of hidden layers
        activation_data : list, optional
            List of activation functions, by default []
        encoder : Union[None, list], optional
            Encoder for the neural network, by default None
        device : torch.device, optional
            Device for the neural network, by default torch.device('cpu')
        mask : Union[None, list[Union[int, bool]], np.ndarray], optional
            Mask for the neural network, by default None
        '''

        # Call the parent constructor        
        super(DynamicNN, self).__init__()

        # If the mask is None, set it to an empty list
        #if mask == None:
        if mask is None:
            mask = []

        # Set input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # Initialize the layers
        self.layers = nn.ModuleList()

        # Set the device
        self.device = device

        # If the mask is not empty, set it
        if len(mask) > 0:
            self.__set_ablation_mask(mask)
        else:
            self.mask = []
        
        # If an encoder has been provided, add it to the layers
        if encoder is not None:
            # For each encoder layer
            for encoder_layer in encoder:
                # If the encoder is Linear, BatchNorm1d or Activation
                if encoder_layer[0] == "Linear":
                    self.layers.append(nn.Linear(encoder_layer[1], encoder_layer[2]).to(self.device))
                elif encoder_layer[0] == "BatchNorm1d":
                    self.layers.append(nn.BatchNorm1d(encoder_layer[1]).to(self.device))
                elif encoder_layer[0] == "Activation":
                    self.layers.append(encoder_layer[1].to(self.device))
            # Set the input layer size to the encoder size
            self.input_layer_size = encoder[0][2]
        else:
            # Set the input layer size to the input size
            self.input_layer_size = input_size

        # Create a list of the sizes of the layers
        self.layer_sizes = [self.input_layer_size] + hidden_layers + [self.output_size]

        # For each layer in the layer_sizes
        for i in range(len(self.layer_sizes) - 1):
            # Add a linear layer to the layers
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]).to(self.device))

            # Add batch normalization layer
            if i < len(self.layer_sizes) - 2:  # No batch norm for output layer
                self.layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]).to(self.device))
                
            # If the activation data is not empty and the index is less than the length of the activation data
            if activation_data and i < len(activation_data):
                # If the activation data has only one element, it is a function without parameters
                if len(activation_data[i]) == 1:
                    # Get the activation function
                    act_func = activation_data[i][0]

                    # Append the activation function to the layers while setting the device
                    self.layers.append(act_func().to(self.device))
                else:
                    # Get the activation function and its parameters
                    act_func, act_params = activation_data[i]
                
                    # Create a new dictionary with the trailing numbers removed from the keys
                    processed_act_params = {re.sub(r'_\d+$', '', k): v for k, v in act_params.items()}

                    # Append the activation function to the layers while setting the device
                    self.layers.append(act_func(**processed_act_params).to(self.device))


        return None








    def __set_ablation_mask(self, mask: list) -> None:
        ''' Set the mask for the ablation study

        Parameters
        ----------
        mask : list
            List of 1s and 0s to set the mask
        '''

        # If the mask is not empty
        if len(mask) > 0:
            # Convert the mask to a tensor
            self.mask = torch.tensor(mask, dtype = torch.float32).to(self.device)
        else:
            self.mask = []


        return None








    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        '''
        
        # Flag to check if its the first layer of the encoder (to apply the mask for ablation study)
        first_layer = True

        for layer in self.layers:
            # If the mask is not empty and it is the first layer
            if len(self.mask) > 0 and first_layer:
                # Apply the mask to the tensor
                x = x * self.mask

            x = layer(x.to(self.device))

            # Set the first_layer flag to False
            first_layer = False

        return x


class MultiBranchDynamicNN(nn.Module):
    ''' Multi Branch Dynamic Neural Network class for the optimization of the neural network.

    Parameters
    ----------
    input_size : Union[int, list[int]]
        Size of the input layer
    output_size : int
        Size of the output layer
    hidden_layers : list
        List of hidden layers
    activation_data : list, optional
        List of activation functions, by default []
    encoders : Union[None, list], optional
        Encoder for the neural network, by default None
    device : torch.device, optional
        Device for the neural network, by default torch.device('cpu')
    '''


    def __init__(self,
            input_size: Union[int, list[int]],
            output_size: int,
            hidden_layers: list,
            activation_data: list = [],
            encoders: Union[None, list] = None,
            device: torch.device = torch.device('cpu')
        ) -> None:
        ''' Initialize the MultiBranchDynamicNN class
        
        Parameters
        ----------
        input_size : Union[int, list[int]]
            Size of the input layer
        output_size : int
            Size of the output layer
        hidden_layers : list
            List of hidden layers
        activation_data : list, optional
            List of activation functions, by default []
        encoders : Union[None, list], optional
            Encoder for the neural network, by default None
        device : torch.device, optional
            Device for the neural network, by default torch.device('cpu')
        
        Raises
        ------
        ValueError
            If the encoder is not a list
        '''

        # Call the parent constructor
        super(MultiBranchDynamicNN, self).__init__()

        # Set the input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # Set the encoder list
        self.encoders = []

        # Set the layers list
        self.layers = nn.ModuleList()

        # Set the device
        self.device = device

        # If the encoder is a list
        if isinstance(encoders, list):
            # For each encoder in the encoders
            for encoder in encoders:
                # Set the modulelist for the specific encoder
                encoder_modules = nn.ModuleList()
                
                # Check if the encoder dict is not empty
                if not encoder:
                    # For each encoder layer
                    for encoder_layer in encoder:
                        # Add to encoder_modules with the corresponding kind and its parameters while setting the device
                        if encoder_layer[0] == "Linear":
                            encoder_modules.append(nn.Linear(encoder_layer[1], encoder_layer[2]).to(self.device))
                        elif encoder_layer[0] == "BatchNorm1d":
                            encoder_modules.append(nn.BatchNorm1d(encoder_layer[1]).to(self.device))
                        elif encoder_layer[0] == "Activation":
                            encoder_modules.append(encoder_layer[1].to(self.device))
                else:
                    # Add an identity layer (no encoder)
                    encoder_modules.append(nn.Identity().to(self.device))

                # Append the encoder to the encoders list
                self.encoders.append({
                    "input_size" : encoder[0][2], 
                    "encoder" : encoder_modules
                    })
        else:
            # Encoder should be a list
            raise ValueError("The encoder should be a list")

        # Set the sizes of the layers
        self.layer_sizes = hidden_layers + [self.output_size]

        # For each layer in the layer_sizes (excluding one layer)
        for i in range(len(self.layer_sizes) - 1):

            # Add a linear layer to the layers
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]).to(self.device))

            # Add batch normalization layer
            if i < len(self.layer_sizes) - 2:  # No batch norm for output layer
                self.layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]).to(self.device))
            
            # If the activation data is not empty and the index is less than the length of the activation data
            if activation_data and i < len(activation_data):
                # If the activation data has only one element, it is a function without parameters
                if len(activation_data[i]) == 1:
                    # Get the activation function
                    act_func = activation_data[i][0]

                    # Append the activation function to the layers while setting the device
                    self.layers.append(act_func().to(self.device))
                else:
                    # Get the activation function and its parameters
                    act_func, act_params = activation_data[i]

                    # Create a new dictionary with the trailing numbers removed from the keys
                    processed_act_params = {re.sub(r'_\d+$', '', k): v for k, v in act_params.items()}

                    # Append the activation function to the layers while setting the device
                    self.layers.append(act_func(**processed_act_params).to(self.device))


        return None








    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass through the network.

        Parameters
        ----------
        xs : list[torch.Tensor]
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        
        Raises
        ------
        ValueError
            If the input is not a list of tensors
        ValueError
            If the input is not a list or if the number of inputs is not equal to the number of encoders
        '''

        # Check if xs is a list
        if not isinstance(xs, list):
            raise ValueError("The input should be a list of tensors")
        
        # Check if the length of xs is the same as the number of encoders
        if len(xs) != len(self.encoders):
            raise ValueError("The number of inputs should be the same as the number of encoders")

        # Process each input tensor through its corresponding encoder
        encoded_outputs = []
        
        # For each input tensor and its corresponding encoder
        for x, encoder in zip(xs, self.encoders):
            # Check if the input tensor is a torch.Tensor
            for layer in encoder["encoder"]:
                x = layer(x.to(self.device))  # Update x with the output of the current layer
            encoded_outputs.append(x)  # Store the encoded output for each input tensor

        
        # Concatenate the encoded outputs into a single tensor
        x = torch.cat(encoded_outputs, dim=1)

        # Perform a BatchNorm1d on the concatenated tensor
        x = nn.BatchNorm1d(x.shape[1]).to(self.device)(x)

        # Add a linear layer to the concatenated tensor to match the input size of the first hidden layer
        x = nn.Linear(x.shape[1], self.layer_sizes[0]).to(self.device)(x)

        # For each layer in the layers
        for layer in self.layers:
            # Pass the tensor through the layer
            x = layer(x.to(self.device))

        # Return the tensor
        return x


class CustomDataset(Dataset):
    ''' Custom dataset class for the neural network

    Parameters
    ----------
    features : torch.Tensor
        Features tensor
    target : torch.Tensor
        Target tensor
    '''


    def __init__(self, features: torch.Tensor, target: torch.Tensor) -> None:
        ''' Initialize the CustomDataset class

        Parameters
        ----------
        features : torch.Tensor
            Features tensor
        target : torch.Tensor
            Target tensor
        '''

        self.features = features
        self.target = target


        return None








    def __len__(self) -> int:
        ''' Get the length of the dataset
        
        Returns
        -------
        int
            Length of the dataset
        '''


        return len(self.features)








    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ''' Get the item at the specified index

        Parameters
        ----------
        idx : int
            Index of the item

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of features and target at the specified index
        '''

        return self.features[idx], self.target[idx]


class MultiBranchCustomDataset(Dataset):
    ''' Custom dataset class for the multi branch neural network.

    Parameters
    ----------
    features1 : torch.Tensor
        Features tensor for the first branch
    features2 : torch.Tensor
        Features tensor for the second branch
    features3 : torch.Tensor
        Features tensor for the third branch
    target : torch.Tensor
        Target tensor
    '''


    def __init__(self,
                 features1 : torch.Tensor,
                 features2 : torch.Tensor,
                 features3 : torch.Tensor,
                 target : torch.Tensor
                ) -> None:
        ''' Initialize the MultiBranchCustomDataset class
        
        Parameters
        ----------
        features1 : torch.Tensor
            Features tensor for the first branch
        features2 : torch.Tensor
            Features tensor for the second branch
        features3 : torch.Tensor
            Features tensor for the third branch
        target : torch.Tensor
            Target tensor
        '''

        self.features1 = features1
        self.features2 = features2
        self.features3 = features3

        self.target = target








    def __len__(self) -> int:
        ''' Get the length of the dataset

        Returns
        -------
        int
            Length of the dataset
        '''


        return len(self.features1)








    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ''' Get the item at the specified index

        Parameters
        ----------
        idx : int
            Index of the item
        
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of features1, features2, features3 and target at the specified index
        '''

        return self.features1[idx], self.features2[idx], self.features3[idx], self.target[idx]


class DNNOptimizer:
    ''' Dynamic Neural Network Optimizer class for the optimization of the neural network.

    Parameters
    ----------
    X_train : Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]]
        Training data
    y_train : Union[np.ndarray, pd.DataFrame, pd.Series]
        Training labels
    X_test : Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]]
        Testing data
    y_test : Union[np.ndarray, pd.DataFrame, pd.Series]
        Testing labels
    X_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series], list[Union[None, np.ndarray, pd.DataFrame, pd.Series]]], optional
        Validation data, by default None
    y_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]], optional
        Validation labels, by default None
    mask : Union[list[Union[int, bool]], np.ndarray], optional
        Mask for the neural network, by default []
    storage : str, optional
        Storage string for the study, by default "sqlite:///NNoptimization.db"
    encoder_params : Union[None, dict, tuple[dict, dict, dict]], optional
        Encoder parameters for the neural network, by default None
    output_size : int, optional
        Size of the output layer, by default 1
    random_seed : int, optional
        Random seed for the neural network, by default 42
    use_gpu : bool, optional
        Use GPU for the neural network, by default True
    verbose : bool, optional
        Verbose mode for the neural network, by default False
    '''


    def __init__(self,
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series], list[Union[None, np.ndarray, pd.DataFrame, pd.Series]]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            mask: Union[list[Union[int, bool]], np.ndarray] = [],
            storage: str = "sqlite:///NNoptimization.db",
            encoder_params: Union[None, dict, tuple[dict, dict, dict]] = None,
            output_size: int = 1,
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False
        ) -> None:
        ''' Constructor for the DNNOptimizer class

        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]]
            Training data
        y_train : Union[np.ndarray, pd.DataFrame, pd.Series]
            Training labels
        X_test : Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]]
            Testing data
        y_test : Union[np.ndarray, pd.DataFrame, pd.Series]
            Testing labels
        X_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series], list[Union[None, np.ndarray, pd.DataFrame, pd.Series]]], optional
            Validation data, by default None
        y_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]], optional
            Validation labels, by default None
        mask : Union[list[Union[int, bool]], np.ndarray], optional
            Mask for the neural network, by default []
        storage : str, optional
            Storage string for the study, by default "sqlite:///NNoptimization.db"
        encoder_params : Union[None, dict, tuple[dict, dict, dict]], optional
            Encoder parameters for the neural network, by default None
        output_size : int, optional
            Size of the output layer, by default 1
        random_seed : int, optional
            Random seed for the neural network, by default 42
        use_gpu : bool, optional
            Use GPU for the neural network, by default True
        verbose : bool, optional
            Verbose mode for the neural network, by default False
        '''

        # Set the gpu flag and random seed
        self.use_gpu = use_gpu
        self.random_seed = random_seed

        # Set the device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set the activation functions
        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']

        # Set the mask and encoder parameters
        self.mask = mask
        self.encoder_params = encoder_params

        # Set the random seed
        self.set_random_seed()

        # If X_train is a list
        if isinstance(X_train, list):
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            self.X_train = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_train]

            # Set the input size to the size of the element in index 1 of the X_train object
            self.input_size = [x.shape[1] for x in self.X_train]
        else:
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            try:
                self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
            except Exception as e:
                ocprint.print_error(e) # type: ignore
            
            # Set the input size to the size of the X_train object shape (number of features)
            self.input_size = self.X_train.shape[1] # type: ignore

        # Convert y_train to np.ndarray then to torch.Tensor and move it to the device
        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)

        # Set the train loader to None
        self.train_loader = None

        # If the X_test is a list
        if isinstance(X_test, list):
            # Convert it to np.ndarray then to torch.Tensor and move it to the device then put it in a list
            self.X_test = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_test]
        else:
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)

        # Convert y_test to np.ndarray then to torch.Tensor and move it to the device
        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)

        # Set the test loader to None
        self.test_loader = None

        # Check if the validation set has been provided or if any of its elements are None
        if (X_validation is not None and y_validation is not None) or not (isinstance(X_validation, list) and any(x is None for x in X_validation)):
            # If the X_validation is a list
            if isinstance(X_validation, list):
                # Convert it to np.ndarray then to torch.Tensor and move it to the device then put it in a list
                self.X_validation = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_validation]
            else:
                # Convert it to np.ndarray then to torch.Tensor and move it to the device
                self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)

            # Convert y_validation to np.ndarray then to torch.Tensor and move it to the device
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)

            # Set the validation loader to None
            self.validation_loader = None
        else:
            # If the validation set has not been provided, set it all to None
            self.X_validation = None
            self.y_validation = None
            self.validation_loader = None

        # Set the output size
        self.output_size = output_size

        # Define the allowed power of two options
        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
        
        # Set the verbose flag
        self.verbose = verbose

        # If there are encoder parameters
        if encoder_params is not None:
            # If the encoder_params is a dictionary
            if isinstance(encoder_params, dict):
                # Build the encoder and set it
                self.encoder = self.__build_encoder(encoder_params)
            else: # It is a tuple
                # Split the tuple into 3 parts (scoring function, ligand and receptor)
                sf_encoder_params, lig_encoder_params, rec_encoder_params = encoder_params

                # Build the encoder for each part and set it
                self.encoder = [
                    self.__build_encoder(sf_encoder_params), 
                    self.__build_encoder(lig_encoder_params), 
                    self.__build_encoder(rec_encoder_params)
                ]
        else:
            # Set the encoder to None
            self.encoder = None
        
        # Set the storage string for the study

        self.storage = storage








    def __build_encoder(self, encoder_params : dict) -> list:
        ''' Build the encoder for the neural network

        Parameters
        ----------
        encoder_params : dict
            Encoder parameters for the neural network

        Returns
        -------
        list
            List of encoder layers

        Raises
        ------
        ValueError
            If the encoder_params have not at least one activation function
        '''

        # Build the encoder #

        # If the encoder_params has the key 'encoder_activation'
        if 'encoder_activation' in encoder_params:
            # If the activation function is LeakyReLU
            if encoder_params['encoder_activation'] == 'LeakyReLU':
                # Set its parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](negative_slope = encoder_params['negative_slope_encoder'])
            # If the activation function is GELU
            elif encoder_params['encoder_activation'] == 'GELU':
                # Set its parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](approximate = encoder_params['approximate_encoder'])
            # If the activation function is not in the list
            else:
                # Do not set any parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])]()

            # Build just the encoder
            return [("Linear", self.input_size, encoder_params['encoding_dim']), ("BatchNorm1d", encoder_params['encoding_dim']), ("Activation", encoder_activation)]
        
        # Create an empty list to store the encoder
        encoder = []

        # Get all the keys from the encoder_params which starts with 'activation_function'
        activation_keys = [key for key in encoder_params.keys() if key.startswith('activation_function') and key.endswith('encoder')]
        
        # If there are no activation functions
        if not activation_keys:
            raise ValueError("The encoder_params should have at least one activation function")

        # Build the Network #

        # Process the activation functions
        for i in range(encoder_params['n_layers_encoder']):
            # If the activation function is LeakyReLU
            if encoder_params[f'activation_function_{i}_encoder'] == 'LeakyReLU':
                # Set its parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])](negative_slope = 
                encoder_params[f'negative_slope_{i}_encoder'])
            # If the activation function is GELU
            elif encoder_params[f'activation_function_{i}_encoder'] == 'GELU':
                # Set its parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])](approximate = encoder_params[f'approximate_{i}_encoder'])
            else:
                # Do not set any parameters
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params[f'activation_function_{i}_encoder'])]()
            
            # If it is the first layer
            if i == 0:
                # Add the encoder layer to the encoder list
                encoder.extend([
                    ("Linear", self.input_size, encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("BatchNorm1d", encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("Activation", encoder_activation)
                ])
            else:
                # Add the encoder layer to the encoder list
                encoder.extend([
                    ("Linear", encoder_params[f'n_units_layer_{i-1}_encoder'], encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("BatchNorm1d", encoder_params[f'n_units_layer_{i}_encoder']), 
                    ("Activation", encoder_activation)
                ])
            

        return encoder








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








    def train_test_model(self,
                         model : nn.Module,
                         train_loader : DataLoader,
                         test_loader : DataLoader,
                         optimizer : optim.Optimizer,
                         criterion : nn.Module,
                         clip_grad : float,
                         trial : optuna.Trial,
                         epochs : int = 100) -> float:
        ''' Train and test the model

        Parameters
        ----------
        model : nn.Module
            Model to train and test
        train_loader : DataLoader
            DataLoader for the training data
        test_loader : DataLoader
            DataLoader for the testing data
        optimizer : optim.Optimizer
            Optimizer for the model
        criterion : nn.Module
            Loss function for the model
        clip_grad : float
            Gradient clipping value
        trial : optuna.Trial
            Optuna trial object
        epochs : int, optional
            Number of epochs to train the model, by default 100
        
        Returns
        -------
        float
            RMSE of the model
        '''

        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            # If the train loader is a multi branch dataset
            if isinstance(train_loader.dataset, MultiBranchCustomDataset):
                # For each batch in the train loader
                for i, (inputs1, inputs2, inputs3, labels) in enumerate(train_loader):
                    # Zero the gradients
                    optimizer.zero_grad()

                    outputs = model([inputs1, inputs2, inputs3])             # Forward pass
                    loss = criterion(outputs, labels.view(-1, 1))            # Calculate the loss
                    loss.backward()                                          # Backward pass
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Clip the gradients
                    optimizer.step()                                         # Update weights

                    # Accumulate the loss
                    running_loss = running_loss + loss.item()
            else:
                # For each batch in the train loader
                for i, (inputs, labels) in enumerate(train_loader):
                    # Zero the gradients
                    optimizer.zero_grad()

                    outputs = model(inputs)                                  # Forward pass
                    loss = criterion(outputs, labels.view(-1, 1))            # Calculate the loss
                    loss.backward()                                          # Backward pass
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Clip the gradients
                    optimizer.step()                                         # Update weights

                    # Accumulate the loss
                    running_loss = running_loss + loss.item()

            # Set the model to evaluation mode
            model.eval()

            # Set the running loss to 0.0
            running_loss = 0.0

            # Set the predictions and labels to empty lists
            all_predictions = []
            all_labels = []

            # If the test loader is a list
            if isinstance(test_loader.dataset, MultiBranchCustomDataset):
                # For each batch in the test loader
                for inputs1, inputs2, inputs3, labels in test_loader:
                    # Get the predictions
                    predicted = model([inputs1, inputs2, inputs3])

                    # Calculate the loss
                    loss = criterion(predicted, labels.view(-1, 1))

                    # Accumulate the loss
                    running_loss = running_loss + loss.item()
                    
                    # Extend the predictions and labels lists while detaching them from the GPU
                    all_predictions.extend(predicted.cpu().detach().numpy())

                    # Do the same for the labels
                    all_labels.extend(labels.cpu().detach().numpy())
            else:
                # For each batch in the test loader
                for inputs, labels in test_loader:
                    # Get the predictions
                    predicted = model(inputs)

                    # Calculate the loss
                    loss = criterion(predicted, labels.view(-1, 1))

                    # Accumulate the loss
                    running_loss = running_loss + loss.item()
                    
                    # Extend the predictions and labels lists while detaching them from the GPU
                    all_predictions.extend(predicted.cpu().detach().numpy())

                    # Do the same for the labels
                    all_labels.extend(labels.cpu().detach().numpy())

        # Get the RMSE
        average_loss = running_loss / len(test_loader) # type: ignore

        # Calculate the RMSE
        rmse = np.sqrt(average_loss)

        # If the verbose flag is set to True, print the results
        if self.verbose:
            ocprint.printv(f'Test Loss: {average_loss}')
            ocprint.printv(f'Test RMSE: {rmse}')

        # Report the RMSE to Optuna
        trial.report(rmse, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


        return rmse








    def objective(self, trial : optuna.Trial) -> float:
        ''' Objective function for the Optuna study

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
            
        Returns
        -------
        float
            RMSE of the model
        '''

        # Set the random seed
        self.set_random_seed()

        # Suggest the learning rate
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Suggest the number of hidden layers and the number of units in each layer
        hidden_layers = []

        # Suggest the number of layers and then iterate for each layer
        for i in range(trial.suggest_int('n_layers', 1, 5)):
            # Suggest the number of units in the layer
            hidden_layers.append(trial.suggest_categorical(f'n_units_layer_{i}', self.power_of_two_options))
        
        # Suggestions for the activation functions
        activation_data = []

        # Suggest the activation functions for each layer
        for i in range(len(hidden_layers)):
            # Suggest the activation function
            activation_function_str = trial.suggest_categorical(f'activation_function_{i}', self.activation_functions_str)
            activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]

            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                # Suggest the negative slope for LeakyReLU
                activation_data.append((activation_function, {
                    f'negative_slope_{i}': trial.suggest_float(f'negative_slope_{i}', 0.01, 0.5)
                }))
            # If the activation function is GELU
            elif activation_function == nn.GELU:
                # Suggest the approximate parameter for GELU
                activation_data.append((activation_function, {
                    f'approximate_{i}': trial.suggest_categorical(f'approximate_{i}', ['none', 'tanh'])
                }))
            else:
                activation_data.append((activation_function, {}))

        # If the first element in the encoder is a list
        if self.encoder != None and isinstance(self.encoder[0], list):
            # Create the NN model with multiple branches
            model = MultiBranchDynamicNN(self.input_size, self.output_size, hidden_layers, activation_data, self.encoder, self.device)
        else:
            # Create the NN model
            model = DynamicNN(self.input_size, self.output_size, hidden_layers, activation_data, self.encoder, self.device, self.mask) # type: ignore

        # Print the model architecture
        if self.verbose:
            ocprint.printv(model) # type: ignore

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Suggest the batch size
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        # If the input is a list create the train and test loaders
        if isinstance(self.X_train, list):
            # Create the train loader for the multi branch dataset
            self.train_loader = DataLoader(
                dataset = MultiBranchCustomDataset(self.X_train[0], self.X_train[1], self.X_train[2], self.y_train), 
                batch_size = batch_size, 
                shuffle = True
            )
        else:
            # Create the train loader for the single branch dataset
            self.train_loader = DataLoader(
                dataset = CustomDataset(self.X_train, self.y_train), 
                batch_size = batch_size, 
                shuffle = True
            )

        # Create the train and test loaders #

        # If the input is a list create the test loader
        if isinstance(self.X_test, list):
            # Create the test loader for the multi branch dataset
            self.test_loader = DataLoader(
                dataset = MultiBranchCustomDataset(self.X_test[0], self.X_test[1], self.X_test[2], self.y_test), 
                batch_size = batch_size
            )
        else:
            # Create the test loader for the single branch dataset
            self.test_loader = DataLoader(
                dataset = CustomDataset(self.X_test, self.y_test), 
                batch_size = batch_size
            )

        # If a validation set has been provided, create the validation loader (If X_validation is not None, y_validation is not None as well)
        if self.X_validation is not None:
            # If the input is a list create the validation loader
            if isinstance(self.X_validation, list):
                # Create the validation loader for the multi branch dataset
                self.validation_loader = DataLoader(
                    dataset = MultiBranchCustomDataset(self.X_validation[0], self.X_validation[1], self.X_validation[2], self.y_validation),  # type: ignore
                    batch_size = batch_size, 
                    shuffle = True
                )
            else:
                # Create the validation loader for the single branch dataset
                self.validation_loader = DataLoader(
                    dataset = CustomDataset(self.X_validation, self.y_validation), # type: ignore
                    batch_size = batch_size, 
                    shuffle = True
                )

        # Suggestions for the epochs
        epochs = trial.suggest_int('epochs', 100, 1000)

        # Suggestions for clipping the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)

        # Use Root Mean Squared Error as the loss function
        criterion = nn.MSELoss()

        test_loss = self.train_test_model(model, self.train_loader, self.test_loader, optimizer, criterion, clip_grad, trial, epochs = epochs)

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
                pr_auc = 0
                log_loss_value = 0
                mae = 0
            else:
                # Check if validation set has only one class (can't compute AUC/log_loss)
                unique_classes = np.unique(y_validation_np)
                if len(unique_classes) == 1:
                    # Only one class in validation set - can't compute classification metrics
                    validation_auc = 0.0
                    pr_auc = 0.0
                    log_loss_value = np.inf
                    mae = mean_absolute_error(y_validation_np, validation_predictions_np)
                else:
                    # Calculate the ROC
                    fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np) # type: ignore
                    validation_auc = auc(fpr, tpr)

                    # Calculate the PR AUC
                    precision, recall, _ = precision_recall_curve(y_validation_np, validation_predictions_np)
                    pr_auc = auc(recall, precision)
                    
                    # Calculate the log loss (with error handling for edge cases)
                    try:
                        log_loss_value = log_loss(y_validation_np, validation_predictions_np)
                    except ValueError as e:
                        # Handle cases where log_loss fails (e.g., single class, invalid predictions)
                        log_loss_value = np.inf

                    # Calculate the Mean Absolute Error
                    mae = mean_absolute_error(y_validation_np, validation_predictions_np)

            # Set the optuna user attrs
            trial.set_user_attr('AUC', validation_auc)
            trial.set_user_attr('pr_auc', pr_auc)
            trial.set_user_attr('log_loss', log_loss_value)
            trial.set_user_attr('mae', float(mae))


        return test_loss








    def objective_ablation(self, trial : optuna.Trial) -> float:
        ''' Objective function for the Optuna study for the ablation study

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object
        
        Returns
        -------
        float
            RMSE of the model
        '''

        # Set the random seed
        self.set_random_seed()

        # If the first element in the encoder is a list
        if self.encoder != None and isinstance(self.encoder[0], list):
            raise NotImplementedError("Ablation study is not supported for MultiBranchDynamicNN yet")
        else:
            # Create the NN model
            model = NeuralNet(
                self.X_train.shape[1], # type: ignore
                self.output_size,          
                self.encoder_params,
                self.network_params,
                random_seed = self.random_seed,
                use_gpu = self.use_gpu, 
                verbose = self.verbose,
                mask = self.mask
            )

        # Reset the random seeds
        self.set_random_seed()

        # Print the model architecture
        if self.verbose:
            ocprint.printv(model) # type: ignore

        _ = model.train_model(
            self.X_train, 
            self.y_train, 
            self.X_test, 
            self.y_test, 
            self.X_validation, 
            self.y_validation
        )

        # Set the optuna user attrs
        trial.set_user_attr('AUC', model.validation_auc)
        trial.set_user_attr('pr_auc', model.pr_auc)
        trial.set_user_attr('log_loss', model.log_loss_value)
        trial.set_user_attr('mae', float(model.mae))

        # Convert mask to string and then store it
        mask = self.mask
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
    
        # Convert booleans to integers if necessary
        if mask.dtype == bool:
            mask = mask.astype(int)
        
        # Convert array of integers to a string
        mask = ''.join(mask.astype(str))

        trial.set_user_attr('Feature_Mask', mask)
        trial.set_user_attr('random_seed', self.random_seed)


        return model.rmse # type: ignore








    def optimize(self,
                 direction: str = "maximize",
                 n_trials : int = 10,
                 study_name : str = "NN_Optimization",
                 load_if_exists : bool = True,
                 sampler : optuna.samplers.BaseSampler = TPESampler(), 
                 n_jobs : int = 1) -> None:
        ''' Optimize the model using Optuna

        Parameters
        ----------
        direction : str, optional
            Direction of the optimization, by default "maximize"
        n_trials : int, optional
            Number of trials to run, by default 10
        study_name : str, optional
            Name of the study, by default "NN_Optimization"
        load_if_exists : bool, optional
            Load the study if it exists, by default True
        sampler : optuna.samplers.BaseSampler, optional
            Sampler for the study, by default TPESampler()
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1
        '''

        # If verbose is set to True, print the optimization process info
        if self.verbose:
            ocprint.printv(f'Optimizing the model for {n_trials} trials')

        # Add a pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = n_trials // 10, # Start pruning after 10% of the trials to allow better exploration
            n_warmup_steps = 15,               # Prune should act only if after 15 steps the value is still not in the median
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

        # Optimize the study
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)
        
        # If Verbose is set to True, print the optimization process info
        if self.verbose:
            best_params = study.best_params

            ocprint.printv(f"Best Hyperparameters: {best_params}")








    def ablate(self,
               network_params: dict[str, Any],
               n_trials : int = 1,
               study_name : str = "NN_Ablation_Optimization",
               load_if_exists : bool = True,
               n_jobs : int = 1
               ) -> None:
        ''' Perform an ablation study on the model. Here Optuna will not optimize the model, but will just run the trials with the given parameters and log them in the database.

        Parameters
        ----------
        network_params : dict[str, Any]
            Network parameters for the model
        n_trials : int, optional
            Number of trials to run, by default 1
        study_name : str, optional
            Name of the study, by default "NN_Ablation_Optimization"
        load_if_exists : bool, optional
            Load the study if it exists, by default True
        n_jobs : int, optional
            Number of jobs to run in parallel, by default 1
        '''

        # If verbose is set to True, print the optimization process info
        if self.verbose:
            ocprint.printv("Starting ablation study...")
        
        try:
            # Set the network parameters
            self.network_params = network_params

            # Create the study
            study = optuna.create_study(
                study_name=study_name, 
                storage=self.storage, 
                load_if_exists=load_if_exists,
            )

            # Optimize the study
            study.optimize(self.objective_ablation, n_trials=n_trials, n_jobs=n_jobs)

            # If Verbose is set to True, print the optimization process info
            if self.verbose:
                ocprint.printv("Finished Ablation Study. Best trial:")
                ocprint.printv(f"{study.best_trial}")
        except Exception as e:
            ocprint.print_error(f"An error occurred: {e}")

# Methods
###############################################################################
