#!/usr/bin/env python3

# Description
###############################################################################
''' Module to perform the optimization of the Transformer parameters model
using Optuna.

It is imported as:

from OCDocker.OCScore.Transformer.TransOptimizer import TransOptimizer
'''

# Imports
###############################################################################

import optuna
import random
import torch

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

from optuna.samplers import TPESampler
from sklearn.metrics import auc, roc_curve
from torch.utils.data import Dataset, DataLoader
from typing import Union



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


class CustomDataset(Dataset):
    ''' Create a custom dataset for the PyTorch DataLoader. '''
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


class TransformerModel(nn.Module):
    ''' Transformer-based neural network model with configurable initialization and structure.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    d_model : int
        The dimension of the model.
    output_dim : int
        The output dimension.
    nhead : int
        The number of heads in the multihead attention.
    num_encoder_layers : int
        The number of encoder layers.
    dim_feedforward : int
        The dimension of the feedforward network model.
    dropout : float, optional
        The dropout value (default is 0.1).
    init_type : str, optional
        The type of initialization (default is 'zeros').
    init_params : dict, optional
        The parameters for the initialization function (default is {}).
    random_seed : int, optional
        The random seed for reproducibility (default is 42).
    device : torch.device, optional
        The device to use (default is torch.device('cuda')).
    verbose : bool, optional
        If True, print the model summary (default is False).
    '''


    def __init__(self,
                 input_dim : int,
                 d_model : int,
                 output_dim : int,
                 nhead : int,
                 num_encoder_layers : int,
                 dim_feedforward : int,
                 dropout : float = 0.1,
                 init_type: str = 'zeros',
                 init_params: dict = {},
                 random_seed: int = 42,
                 device : torch.device = torch.device('cuda'),
                 verbose : bool = False
                ) -> None:
        ''' Constructor for the TransformerModel class.
        
        Parameters
        ----------
        input_dim : int
            The input dimension.
        d_model : int
            The dimension of the model.
        output_dim : int
            The output dimension.
        nhead : int
            The number of heads in the multihead attention.
        num_encoder_layers : int
            The number of encoder layers.
        dim_feedforward : int
            The dimension of the feedforward network model.
        dropout : float, optional
            The dropout value (default is 0.1).
        init_type : str, optional
            The type of initialization (default is 'zeros').
        init_params : dict, optional
            The parameters for the initialization function (default is {}).
        random_seed : int, optional
            The random seed for reproducibility (default is 42).
        device : torch.device, optional
            The device to use (default is torch.device('cuda')).
        verbose : bool, optional
            If True, print the model summary (default is False).
        '''

        # Call the parent constructor
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, d_model).to(device)

        # Normalization layer
        self.norm = nn.LayerNorm(d_model).to(device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, 
            nhead = nhead, 
            dim_feedforward = dim_feedforward, 
            dropout = dropout,
            batch_first = True
        ).to(device)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers).to(device)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim).to(device)

        # Set the supported initialization functions
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
        self.d_model = d_model
        self.device = device
        self.random_seed = random_seed
        self.generator = self.set_random_seed()

        # Initialize weights
        self.initialize_weights()

        if verbose:
            # Print the model

            ocprint.printv(self) # type: ignore

    def set_random_seed(self) -> torch.Generator:
        ''' Set the random seed for reproducibility. '''

        # Set the random seed for numpy and random
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)

        # Set the seed for GPU if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        # Create a generator for reproducibility
        generator = torch.Generator(device=self.device)

        # Set the seed for the generator
        generator.manual_seed(self.random_seed)

        # This is not recommended for performance since it will disable the cudnn auto-tuner (reason why it is commented)
        #torch.backends.cudnn.enabled = False

        # Set the backends for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        return generator

    def initialize_weights(self) -> None:
        ''' Initialize the weights of the model. '''

        # If init_type is defined in the init_functions, use it
        if self.init_type in self.init_functions.keys():
            init_func = self.init_functions[self.init_type]
        else:
            # User-facing error: invalid initialization function
            ocerror.Error.value_error(f"Unknown initialization function: '{self.init_type}'") # type: ignore
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

    def forward(self, src : torch.Tensor) -> torch.Tensor:
        ''' Forward pass through the model.

        Parameters
        ----------
        src : torch.Tensor
            The input tensor.
        '''

        # Embed the input
        src = self.embedding(src) * np.sqrt(self.d_model)

        # Add a normalization layer
        src = self.norm(src)

        # Pass through Transformer encoder
        output = self.transformer_encoder(src)

        # Apply final linear layer
        output = self.fc_out(output)  # This uses the complete feature vector

        return output


class Transformer(nn.Module):
    ''' Transformer-based neural network model with configurable initialization and structure.

    Parameters
    ----------
    input_size : int
        The input dimension.
    output_size : int
        The output dimension.
    trans_params : dict
        The parameters for the transformer model.
    random_seed : int, optional
        The random seed for reproducibility (default is 42).
    use_gpu : bool, optional
        If True, use GPU (default is True).
    verbose : bool, optional
        If True, print the model summary (default is False).
    '''

    def __init__(self, 
            input_size : int,
            output_size : int,
            trans_params : dict,
            random_seed : int = 42,
            use_gpu : bool = True,
            verbose : bool = False
        ) -> None:
        ''' Constructor for the Transformer class.

        Parameters
        ----------
        input_size : int
            The input dimension.
        output_size : int
            The output dimension.
        trans_params : dict
            The parameters for the transformer model.
        random_seed : int, optional
            The random seed for reproducibility (default is 42).
        use_gpu : bool, optional
            If True, use GPU (default is True).
        verbose : bool, optional
            If True, print the model summary (default is False).
        '''
        
        # Call the parent constructor
        super(Transformer, self).__init__()

        # Set the random seed value and the gpu flag
        self.random_seed = random_seed
        self.use_gpu = use_gpu

        # Set the device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Set the input
        self.input_size = input_size

        # Set the random seed
        self.set_random_seed()

        # Define the optimizer functions
        self.optimizer_functions = [optim.Adam, optim.RMSprop, optim.SGD]
        self.optimizer_functions_str = ['Adam', 'RMSprop', 'SGD']

        # Create the transformer model
        self.trans = TransformerModel(input_size, trans_params['d_model'], output_size, trans_params['nhead'], trans_params['num_encoder_layers'], trans_params['dim_feedforward'], trans_params['dropout'], device = self.device).to(self.device)

        # Set the parameters for the transformer model (batch size, epochs, learning rate, clip_grad)
        self.batch_size = trans_params['batch_size']
        self.epochs = trans_params['epochs']
        self.lr = trans_params['lr']
        self.clip_grad = trans_params['clip_grad']

        # Set the optimizer and its parameters
        self.optimizer = self.optimizer_functions[self.optimizer_functions_str.index(trans_params['optimizer'])](
            self.trans.parameters(),
            weight_decay = trans_params['weight_decay'], 
            lr = trans_params['lr']
        )

        # Set the transformer parameters
        self.trans_params = trans_params

        # Set the AUC and rmse as nan
        self.validation_auc = np.nan
        self.rmse = np.nan

        # Set the verbose flag
        self.verbose = verbose

        self.prediction = None

        # Print the model if verbose is True
        if verbose:

            ocprint.printv(self.trans) # type: ignore

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
                    X_train : Union[np.ndarray, pd.DataFrame, list],
                    y_train : Union[np.ndarray, pd.DataFrame, list],
                    X_test : Union[np.ndarray, pd.DataFrame, list],
                    y_test : Union[np.ndarray, pd.DataFrame, list],
                    X_validation : Union[np.ndarray, pd.DataFrame, list, None] = None,
                    y_validation : Union[np.ndarray, pd.DataFrame, list, None]= None,
                    criterion : nn.Module = nn.MSELoss()
                    ) -> bool:
        ''' Train the model.
        
        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame, list]
            The training features.
        y_train : Union[np.ndarray, pd.DataFrame, list]
            The training labels.
        X_test : Union[np.ndarray, pd.DataFrame, list]
            The test features.
        y_test : Union[np.ndarray, pd.DataFrame, list]
            The test labels.
        X_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
            The validation features (default is None).
        y_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
            The validation labels (default is None).
        criterion : nn.Module, optional
            The loss function (default is nn.MSELoss()).

        Returns
        -------
        bool
            True if the model was trained successfully, False otherwise.
        '''
    
        self.set_random_seed()

        # If X_train is a list
        if isinstance(X_train, list):
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            X_train = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_train]
        else:
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device) # type: ignore

        # If y_train is a list
        y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device) # type: ignore

        # If X_test is a list
        if isinstance(X_test, list):
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            X_test = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_test]
        else:
            # Convert it to np.ndarray then to torch.Tensor and move it to the device
            X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device) # type: ignore

        # Convert y_test to torch.Tensor and move it to the device
        y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device) # type: ignore

        # If X_validation is not none (y_validation is not none as well)
        if X_validation is not None:
            # If X_validation is a list
            if isinstance(X_validation, list):
                # Convert it to np.ndarray then to torch.Tensor and move it to the device
                X_validation = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_validation]
            else:
                # Convert it to np.ndarray then to torch.Tensor and move it to the device
                X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device) # type: ignore
            
            # Convert y_validation to torch.Tensor and move it to the device
            y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device) # type: ignore

        # Create the training loader
        train_loader = DataLoader(
            dataset = CustomDataset(X_train, y_train), # type: ignore
            batch_size = self.batch_size, 
            shuffle = True,
            drop_last=True
        )

        # Create the test loader
        test_loader = DataLoader(
            dataset = CustomDataset(X_test, y_test), # type: ignore
            batch_size = self.batch_size,
            drop_last=True
        )

        # If a validation set has been provided
        if X_validation is not None:
            # Create the validation loader
            validation_loader = DataLoader(
                dataset = CustomDataset(X_validation, y_validation), # type: ignore
                batch_size = self.batch_size, 
                shuffle = True,
                drop_last=True
            )

        # For each epoch
        for epoch in range(self.epochs):
            # Set the model to training mode
            self.trans.train()

            # Set the running loss to 0            
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                outputs = self.trans(inputs)                                      # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))                     # Calculate the loss
                loss.backward()                                                   # Backward pass
                nn.utils.clip_grad_norm_(self.trans.parameters(), self.clip_grad) # Clip the gradients
                self.optimizer.step()                                             # Update weights

                # Accumulate the loss
                running_loss = running_loss + loss.item()
        
            # Set the model to evaluation mode
            self.trans.eval()

            # Set the running loss to 0.0
            running_loss = 0.0

            # Initialize the lists for predictions and labels
            all_predictions = []
            all_labels = []
            
            # Set as no_grad to avoid tracking history
            with torch.no_grad():
                # For each batch in the test loader
                for inputs, labels in test_loader:
                    # Predict the labels
                    predicted = self.trans(inputs)

                    # Compute the loss
                    loss = criterion(predicted, labels.view(-1, 1))

                    # Accumulate the loss
                    running_loss = running_loss + loss.item()
                    
                    # Append the predictions and the labels to the lists
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Compute the average loss
            average_loss = running_loss / len(test_loader)

            # Compute the RMSE
            rmse = np.sqrt(average_loss)

            # If verbose, print some data to the user
            if self.verbose:
                ocprint.printv(f'Epoch {epoch + 1}/{self.epochs}')
                ocprint.printv(f'Average Loss: {average_loss}')
                ocprint.printv(f'RMSE: {rmse}')

            # If a validation set has been provided, calculate the AUC
            if X_validation is not None:
                # Set the model to evaluation mode
                self.trans.eval()
                
                validation_predictions = self.trans(X_validation)

                # Convert the predictions and the labels to numpy
                validation_predictions_np = validation_predictions.detach().cpu().numpy()
                y_validation_np = y_validation.cpu().numpy() # type: ignore

                # Set the prediction
                self.prediction = y_validation_np

                # If there is a nan in the predictions, set the AUC to 0
                if np.isnan(validation_predictions_np).any():
                    validation_auc = 0
                else:
                    # Calculate the ROC
                    fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np)
                    validation_auc = auc(fpr, tpr)

        # Set the rmse and validation_auc
        self.rmse = rmse
        self.validation_auc = validation_auc


        return True

    def get_model(self) -> nn.Module:
        ''' Get the model.

        Returns
        -------
        nn.Module
            The model.
        '''

        return self.trans


class TransOptimizer:
    ''' Class to optimize the Transformer model using Optuna.

    Parameters
    ----------
    X_train : Union[np.ndarray, pd.DataFrame, list]
        The training features.
    y_train : Union[np.ndarray, pd.DataFrame, list]
        The training labels.
    X_test : Union[np.ndarray, pd.DataFrame, list]
        The test features.
    y_test : Union[np.ndarray, pd.DataFrame, list]
        The test labels.
    X_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
        The validation features (default is None).
    y_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
        The validation labels (default is None).
    storage : str, optional
        The storage for the Optuna study (default is 'sqlite:///Transoptimization.db').
    output_size : int, optional
        The output size (default is 1).
    random_seed : int, optional
        The random seed for reproducibility (default is 42).
    use_gpu : bool, optional
        If True, use GPU (default is True).
    verbose : bool, optional
        If True, print the model summary (default is False).
    '''

    def __init__(self,
                 X_train : Union[np.ndarray, pd.DataFrame, list],
                 y_train : Union[np.ndarray, pd.DataFrame, list],
                 X_test : Union[np.ndarray, pd.DataFrame, list],
                 y_test : Union[np.ndarray, pd.DataFrame, list],
                 X_validation : Union[np.ndarray, pd.DataFrame, list, None] = None,
                 y_validation : Union[np.ndarray, pd.DataFrame, list, None] = None,
                 storage : str = 'sqlite:///Transoptimization.db',
                 output_size : int = 1,
                 random_seed : int = 42,
                 use_gpu : bool = True,
                 verbose : bool = False
                ) -> None:
        ''' Constructor for the TransOptimizer class.

        Parameters
        ----------
        X_train : Union[np.ndarray, pd.DataFrame, list]
            The training features.
        y_train : Union[np.ndarray, pd.DataFrame, list]
            The training labels.
        X_test : Union[np.ndarray, pd.DataFrame, list]
            The test features.
        y_test : Union[np.ndarray, pd.DataFrame, list]
            The test labels.
        X_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
            The validation features (default is None).
        y_validation : Union[np.ndarray, pd.DataFrame, list, None], optional
            The validation labels (default is None).
        storage : str, optional
            The storage for the Optuna study (default is 'sqlite:///Transoptimization.db').
        output_size : int, optional
            The output size (default is 1).
        random_seed : int, optional
            The random seed for reproducibility (default is 42).
        use_gpu : bool, optional
            If True, use GPU (default is True).
        verbose : bool, optional
            If True, print the model summary (default is False).
        '''

        # Set the random seed value and the gpu flag
        self.random_seed = random_seed
        self.use_gpu = use_gpu

        # Set the random seed
        self.set_random_seed()

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        
        # Se the X_train and y_train and move it to the device
        self.X_train = torch.tensor(np.asarray(X_train), dtype = torch.float32).to(self.device)
        self.y_train = torch.tensor(np.asarray(y_train), dtype = torch.float32).to(self.device)

        # Set the train loader to None
        self.train_loader = None

        # Set the X_test and y_test and move it to the device
        self.X_test = torch.tensor(np.asarray(X_test), dtype = torch.float32).to(self.device)
        self.y_test = torch.tensor(np.asarray(y_test), dtype = torch.float32).to(self.device)

        # Set the test loader to None
        self.test_loader = None

        # If X_validation is not none (y_validation is not none as well)
        if X_validation is not None:
            # Convert X_validation and y_validation to np.ndarray then to torch.Tensor and move it to the device
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype = torch.float32).to(self.device)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype = torch.float32).to(self.device)
        else:
            # Set the validation sets to None
            self.X_validation = None
            self.y_validation = None

        # Set the validation loader to None
        self.validation_loader = None

        # Set the output size
        self.output_size = output_size

        # Set the verbose flag
        self.verbose = verbose

        # Set the storage for the Optuna study

        self.storage = storage

    def set_random_seed(self) -> None:
        ''' Set the random seed for the Autoencoder. It is used to set the random seed for the Autoencoder.'''

        # Set the random seed for numpy and random
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU in torch
        torch.manual_seed(self.random_seed)

        # If using GPU, set the seed for GPU as well in torch
        if self.use_gpu:

            torch.cuda.manual_seed_all(self.random_seed)

    def train_test_model(self,
                         model : nn.Module,
                         train_loader : DataLoader,
                         test_loader : DataLoader,
                         optimizer : optim.Optimizer,
                         criterion : nn.Module,
                         clip_grad : float,
                         trial : optuna.Trial,
                         batch_size : int,
                         epochs : int = 100
                        ) -> float:
        ''' Train and test the model.

        Parameters
        ----------
        model : nn.Module
            The model to train and test.
        train_loader : DataLoader
            The training data loader.
        test_loader : DataLoader
            The test data loader.
        optimizer : optim.Optimizer
            The optimizer to use.
        criterion : nn.Module
            The loss function to use.
        clip_grad : float
            The gradient clipping value.
        trial : optuna.Trial
            The Optuna trial object.
        batch_size : int
            The batch size to use.
        epochs : int, optional
            The number of epochs to train for (default is 100).

        Returns
        -------
        float
            The RMSE of the model on the test set.
        '''

        # If verbose, set the autograd to detect anomalies        
        if self.verbose:
            # Set the autograd to detect anomalies
            torch.autograd.set_detect_anomaly(True) # type: ignore
            
        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            # For each batch in the training loader
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
                running_loss = running_loss + loss.item()

            # Set the model to evaluation mode
            model.eval()

            # Set the running loss to 0.0
            running_loss = 0.0

            # Set the predictions and labels to empty lists
            all_predictions = []
            all_labels = []

            # For each element in the test loader
            for inputs, labels in test_loader:
                # Get the predictions
                predicted = model(inputs)

                # Compute the loss
                loss = criterion(predicted, labels.view_as(outputs))

                # Accumulate the loss
                running_loss = running_loss + loss.item()
                
                # Append the predictions and the labels
                all_predictions.extend(predicted.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        # Compute the average loss
        average_loss = running_loss / len(test_loader) # type: ignore

        # Compute the RMSE
        rmse = np.sqrt(average_loss)

        # If verbose, print some data to the user
        if self.verbose:
            ocprint.printv(f'Test Loss: {average_loss}')
            ocprint.printv(f'Test RMSE: {rmse}')

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


        return rmse

    def objective(self, trial : optuna.Trial) -> float:
        ''' Objective function for the Optuna study.

        Parameters
        ----------
        trial : optuna.Trial
            The Optuna trial object.
        
        Returns
        -------
        float
            The RMSE of the model on the test set.
        '''

        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8, 16])
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 6)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048, 4096])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 10, 500)

        # Suggest the initialization type
        init_type = trial.suggest_categorical('init_type', ['zeros', 'orthogonal', 'normal', 'uniform', 'constant', 'xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform', 'sparse', 'eye'])

        # If the initialization typem requires parameters, suggest them
        if init_type in ['normal']:
            mean = trial.suggest_float('mean', -1, 1)
            std = trial.suggest_float('std', 0.1, 1)
            init_params = {'mean': mean, 'std': std}
        elif init_type in ['uniform']:
            a = trial.suggest_float('a', -1, 0.1)
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
        
        # Model setup
        model = TransformerModel(
            self.X_train.shape[-1],
            d_model,
            self.output_size,
            nhead,
            num_encoder_layers,
            dim_feedforward,
            dropout,
            init_type,
            init_params,
            self.random_seed,
            self.device
        )

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Set the criterion
        criterion = nn.MSELoss()

        # Initialize the train and test loaders
        self.train_loader = DataLoader(
                dataset = CustomDataset(self.X_train, self.y_train), # type: ignore
                batch_size = batch_size, 
                shuffle = True,
                drop_last=True
            )
        
        self.test_loader = DataLoader(
                dataset = CustomDataset(self.X_test, self.y_test), # type: ignore
                batch_size = batch_size,
                drop_last=True
            )

        # If a validation set has been provided, create the validation loader
        if self.X_validation is not None:
            self.validation_loader = DataLoader(
                dataset = CustomDataset(self.X_validation, self.y_validation), # type: ignore
                batch_size = batch_size, 
                shuffle = True,
                drop_last=True
            )
        
        # Suggestions for clipping the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 0.5)

        # Train and test the model
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

    def optimize(self,
                 direction: str = "maximize",
                 n_trials : int = 10,
                 study_name : str = "NN_Optimization",
                 load_if_exists : bool = True,
                 sampler : optuna.samplers.BaseSampler = TPESampler(),
                 n_jobs : int = 1) -> dict:
        ''' Optimize the model using Optuna.

        Parameters
        ----------
        direction : str, optional
            The direction of the optimization (default is "maximize").
        n_trials : int, optional
            The number of trials to run (default is 10).
        study_name : str, optional
            The name of the study (default is "NN_Optimization").
        load_if_exists : bool, optional
            If True, load the study if it exists (default is True).
        sampler : optuna.samplers.BaseSampler, optional
            The sampler to use (default is TPESampler()).
        n_jobs : int, optional
            The number of jobs to run in parallel (default is 1).

        Returns
        -------
        dict
            The best hyperparameters found by Optuna.
        '''
        
        # If verbose, print some information
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
        
        # Get the best hyperparameters
        best_params = study.best_params

        # If verbose, print the best hyperparameters
        if self.verbose:
            ocprint.printv(f"Best Hyperparameters: {best_params}")

        return best_params
