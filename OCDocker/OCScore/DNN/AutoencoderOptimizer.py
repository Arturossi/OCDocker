#!/usr/bin/env python3

# Description
###############################################################################
""" Module to perform the optimization of the Autoencoder. 

It is imported as:

from OCDocker.OCScore.NN.AutoencoderOptimizer import AutoencoderOptimizer
"""

# Imports
###############################################################################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Dataset
from typing import Union

import optuna
import random
import re

import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

class AutoencoderDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.features[idx]

class Autoencoder(nn.Module):
    def __init__(self,
                 input_size,
                 encoding_dim,
                 encoder_activation_fn,
                 decoder_activation_fn,
                 decoding_dim = None,
                 device = torch.device("cpu")
                ):
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

            # Check if the decoding_dim is not None
            if decoding_dim is None:
                raise ValueError("If the encoding_dim has more than one element, then the decoding_dim should be a list as well.")
            
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
    
    def forward(self, x):
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
    
    def get_encoder_topology(self):
        return ['Linear', 'BatchNorm1d']

    def get_decoder_topology(self):
        return ['Linear']

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

class AutoencoderOptimizer:
    def __init__(self, 
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            encoding_dims: tuple = (16, 256),
            storage: str = "sqlite:///autoencoder.db",
            models_folder: str = "./models/Autoencoder/",
            random_seed = 42, 
            use_gpu = True,
            verbose = False
        ):

        self.random_seed = random_seed
        
        self.models_folder = models_folder
        self.use_gpu = use_gpu

        self.set_random_seed()    
        
        # Convert the data do np.ndarray then to torch.Tensor
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        self.test_loader = None

        if X_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.validation_loader = None
        else:
            self.X_validation = None
            self.validation_loader = None

        self.input_size = self.X_train.shape[1]

        self.encoding_dims = encoding_dims

        self.verbose = verbose

        self.best_rmse = np.inf

        # Set the storage string for the study
        self.storage = storage

        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256

        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']

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

    def train_autoencoder(self, model, optimizer, criterion, clip_grad, epochs, trial):
        # Set the best validation and training rmse to infinity
        best_validation_rmse = np.inf
        best_train_rmse = np.inf
        # Set the epochs without improvement to 0
        epochs_without_improvement = 0
        # Set the early stopping patience as 20% of the epochs
        early_stopping_patience = epochs // 20

        model.train()
        for epoch in range(epochs):
            if self.verbose:
                ocprint.printv(f"Epoch {epoch+1}/{epochs}")

            running_loss = 0.0

            for data, _ in self.train_loader: # type: ignore
                optimizer.zero_grad()
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Clip the gradients
                optimizer.step()

                running_loss += loss.item()
            
            average_loss = running_loss / len(self.train_loader) # type: ignore
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
            
            if self.verbose:
                ocprint.printv(f'Test Loss: {average_loss}')
                ocprint.printv(f'Test RMSE: {rmse}')

            trial.report(rmse, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return best_validation_rmse, best_train_rmse

    def evaluate_autoencoder(self, model, criterion, loader = None):
        self.set_random_seed()
        model.eval()
        total_loss = 0
        
        if loader is None:
            loader = self.test_loader

        with torch.no_grad():
            for data, _ in loader: # type: ignore
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                total_loss += loss.item()

        average_loss = total_loss / len(loader) # type: ignore

        rmse = np.sqrt(average_loss)
        
        return rmse

    def objective(self, trial):
        self.set_random_seed()
        
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        # Suggestions for clipping the gradients
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

        model = Autoencoder(self.input_size, encoder_hidden_layers, encoder_activation_data, decoder_activation_data, decoder_hidden_layers, self.device).to(self.device)

        # Choose the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        criterion = nn.MSELoss()

        self.train_loader = DataLoader(
            dataset = self.train_dataset, 
            batch_size = batch_size, 
            shuffle = True
        )

        self.test_loader = DataLoader(
            dataset = self.test_dataset, 
            batch_size = batch_size
        )

        if self.validation_dataset is not None:
            self.validation_loader = DataLoader(
                dataset = self.validation_dataset, 
                batch_size = batch_size
            )

        best_validation_rmse, best_train_rmse = self.train_autoencoder(model, optimizer, criterion, clip_grad, epochs, trial = trial) # type: ignore

        evaluate_rmse = self.evaluate_autoencoder(model, criterion)

        improvement_threshold = 0.0 # 0% improvement (if is better than the best, it will be logged)
        is_promising = best_validation_rmse < self.best_rmse * (1 - improvement_threshold)

        if is_promising:
            # Save the model
            #torch.save(model.state_dict(), f'{self.models_folder}/autoencoder_{trial.number}.pt')
            self.best_rmse = best_validation_rmse

        return evaluate_rmse

    def objective_old(self, trial):
        self.set_random_seed()
        encoding_dim = trial.suggest_int('encoding_dim', self.encoding_dims[0], self.encoding_dims[1])
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        # Suggestions for clipping the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)
        epochs = trial.suggest_int('epochs', 20, 100)

        activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']
        
        encoder_activation_str = trial.suggest_categorical('encoder_activation', activation_functions_str)

        if encoder_activation_str == 'LeakyReLU':
            pre_encoder_params = {
                f'negative_slope_encoder': trial.suggest_float(f'negative_slope_encoder', 0.01, 0.5)
            }
            encoder_params = {k.replace('_encoder', ''): v for k, v in pre_encoder_params.items()}
            encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)](**encoder_params)
        elif encoder_activation_str == 'GELU':
            pre_encoder_params = {
                f'approximate_encoder': trial.suggest_categorical(f'approximate_encoder', ['none', 'tanh'])
            }
            encoder_params = {k.replace('_encoder', ''): v for k, v in pre_encoder_params.items()}
            encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)](**encoder_params)
        else:
            encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)]()

        decoder_activation_str = trial.suggest_categorical('decoder_activation', activation_functions_str)

        if decoder_activation_str == 'LeakyReLU':
            pre_decoder_params = {
                f'negative_slope_decoder': trial.suggest_float(f'negative_slope_decoder', 0.01, 0.5)
            }
            decoder_params = {k.replace('_decoder', ''): v for k, v in pre_decoder_params.items()}
            decoder_activation_fn = activation_functions[activation_functions_str.index(decoder_activation_str)](**decoder_params)
        elif decoder_activation_str == 'GELU':
            pre_decoder_params = {
                f'approximate_decoder': trial.suggest_categorical(f'approximate_decoder', ['none', 'tanh'])
            }
            decoder_params = {k.replace('_decoder', ''): v for k, v in pre_decoder_params.items()}
            decoder_activation_fn = activation_functions[activation_functions_str.index(decoder_activation_str)](**decoder_params)
        else:
            decoder_activation_fn = activation_functions[activation_functions_str.index(decoder_activation_str)]()

        model = Autoencoder(self.input_size, encoding_dim, encoder_activation_fn, decoder_activation_fn).to(self.device)

        # Choose the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        criterion = nn.MSELoss()

        self.train_loader = DataLoader(
            dataset = self.train_dataset, 
            batch_size = batch_size, 
            shuffle = True
        )

        self.test_loader = DataLoader(
            dataset = self.test_dataset, 
            batch_size = batch_size
        )

        if self.validation_dataset is not None:
            self.validation_loader = DataLoader(
                dataset = self.validation_dataset, 
                batch_size = batch_size
            )

        best_validation_rmse, best_train_rmse = self.train_autoencoder(model, optimizer, criterion, clip_grad, epochs, trial = trial) # type: ignore

        evaluate_rmse = self.evaluate_autoencoder(model, criterion)

        improvement_threshold = 0.0 # 0% improvement (if is better than the best, it will be logged)
        is_promising = best_validation_rmse < self.best_rmse * (1 - improvement_threshold)

        if is_promising:
            # Save the model
            torch.save(model.state_dict(), f'{self.models_folder}/autoencoder_{trial.number}.pt')
            self.best_rmse = best_validation_rmse

        return evaluate_rmse
    
    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "NN_Optimization", load_if_exists = True, sampler: optuna.samplers.BaseSampler = TPESampler(), n_jobs = 1):
        # Data preparation (example, replace with your actual data loading)
        self.train_dataset = AutoencoderDataset(self.X_train)
        self.test_dataset = AutoencoderDataset(self.X_test)

        if self.X_validation is not None:
            self.validation_dataset = AutoencoderDataset(self.X_validation)

        # Add a pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials = n_trials // 10, # Start pruning after 10% of the trials
            n_warmup_steps = 15,
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
