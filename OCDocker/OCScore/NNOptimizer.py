import optuna
import re

import numpy as np
import pandas as pd

from optuna.samplers import TPESampler
from sklearn.metrics import auc, roc_curve
from typing import Union

#from OCDocker.Initialise import *

import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class NeuralNet(nn.Module):
    def __init__(self, 
            input_size, 
            output_size, 
            encoder_params,
            nn_params,
            random_seed = 42,
            use_gpu = True,
            verbose = False
    ):
        super(NeuralNet, self).__init__()

        self.random_seed = random_seed
        self.use_gpu = use_gpu

        self.set_random_seed()

        # Define the activation functions
        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']
        
        self.optimizer_functions = [optim.Adam, optim.RMSprop, optim.SGD]
        self.optimizer_functions_str = ['Adam', 'RMSprop', 'SGD']

        # Define the input layer
        self.layers = nn.ModuleList()

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

        if encoder_params is not None:
            # Build the encoding and decoding functions
            if encoder_params['encoder_activation'] == 'LeakyReLU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](negative_slope = encoder_params['negative_slope_encoder'])
            elif encoder_params['encoder_activation'] == 'GELU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](approximate = encoder_params['approximate_encoder'])
            else:
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])]()

            # Build just the encoder
            self.encoder = [("Linear", input_size, encoder_params['encoding_dim']), ("BatchNorm1d", encoder_params['encoding_dim']), ("Activation", encoder_activation)]
        else:
            self.encoder = None
                
        # Create the DynamicNN
        self.NN = DynamicNN(input_size, output_size, hidden_layers, activation_data, self.encoder, self.device)

        self.batch_size = nn_params['batch_size']
        self.epochs = nn_params['epochs']
        self.lr = nn_params['lr']
        self.clip_grad = nn_params['clip_grad']

        self.optimizer = self.optimizer_functions[self.optimizer_functions_str.index(nn_params['optimizer'])](
            self.NN.parameters(),
            weight_decay = nn_params['weight_decay'], 
            lr = nn_params['lr']
        )

        self.nn_params = nn_params

        # Set the AUC and rmse as nan
        self.validation_auc = np.NaN
        self.rmse = np.NaN

        # Set the verbose flag
        self.verbose = verbose

        self.prediction = None

        if verbose:
            print(self.NN)

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
    def train_model(self, X_train, y_train, X_test, y_test, X_validation = None, y_validation = None, criterion = nn.MSELoss()):
        self.set_random_seed()
        # Convert the data to torch.Tensor
        X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)

        X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)

        if X_validation is not None and y_validation is not None:
            X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)

        # Create the train and test loaders
        train_loader = DataLoader(
            dataset = CustomDataset(X_train, y_train), 
            batch_size = self.batch_size, 
            shuffle = True
        )

        test_loader = DataLoader(
            dataset = CustomDataset(X_test, y_test), 
            batch_size = self.batch_size
        )

        # If a validation set has been provided, create the validation loader
        if X_validation is not None:
            validation_loader = DataLoader(
                dataset = CustomDataset(X_validation, y_validation), 
                batch_size = self.batch_size, 
                shuffle = True
            )

        # For each epoch
        for epoch in range(self.epochs):
            # Set the model to training mode
            self.NN.train()

            # Set the running loss to 0            
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()

                outputs = self.NN(inputs)
                loss = criterion(outputs, labels.view(-1, 1))

                loss.backward()
                nn.utils.clip_grad_norm_(self.NN.parameters(), self.clip_grad)
                self.optimizer.step()

                running_loss += loss.item()

        
            # Set the model to evaluation mode
            self.NN.eval()

            running_loss = 0.0

            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    predicted = self.NN(inputs)
                    loss = criterion(predicted, labels.view(-1, 1))
                    running_loss += loss.item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            average_loss = running_loss / len(test_loader)
            rmse = np.sqrt(average_loss)

            if self.verbose:
                print(f'Epoch {epoch + 1}/{self.epochs}')
                print(f'Average Loss: {average_loss}')
                print(f'RMSE: {rmse}')

            # If a validation set has been provided, calculate the AUC
            if X_validation is not None:
                # Set the model to evaluation mode
                self.NN.eval()
                # Get the predictions for the validation set
                validation_predictions = self.NN(X_validation)
                # Convert the predictions and the labels to numpy
                validation_predictions_np = validation_predictions.detach().cpu().numpy()
                y_validation_np = y_validation.cpu().numpy() # type: ignore

                self.prediction = y_validation_np

                # If there is a nan in the predictions, set the AUC to 0
                if np.isnan(validation_predictions_np).any():
                    validation_auc = 0
                else:
                    # Calculate the ROC
                    fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np)
                    validation_auc = auc(fpr, tpr)

        self.rmse = rmse
        self.validation_auc = validation_auc

        return True
    
    def get_model(self):
        return self.NN
        
class DynamicNN(nn.Module):
    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: list,
            activation_data: list = [],
            encoder: Union[None, list] = None,
            device: torch.device = torch.device('cpu')
        ):
        super(DynamicNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.ModuleList()

        self.device = device

        # If an encoder has been provided, add it to the layers
        if encoder is not None:
            for encoder_layer in encoder:
                if encoder_layer[0] == "Linear":
                    self.layers.append(nn.Linear(encoder_layer[1], encoder_layer[2]).to(self.device))
                elif encoder_layer[0] == "BatchNorm1d":
                    self.layers.append(nn.BatchNorm1d(encoder_layer[1]).to(self.device))
                elif encoder_layer[0] == "Activation":
                    self.layers.append(encoder_layer[1].to(self.device))

            self.input_layer_size = encoder[0][2]
        else:
            self.input_layer_size = input_size

        self.layer_sizes = [self.input_layer_size] + hidden_layers + [self.output_size]

        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]).to(self.device))

            # Add batch normalization layer
            if i < len(self.layer_sizes) - 2:  # No batch norm for output layer
                self.layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]).to(self.device))
                
            if activation_data and i < len(activation_data):
                if len(activation_data[i]) == 1:
                    act_func = activation_data[i][0]
                    self.layers.append(act_func().to(self.device))
                else:
                    act_func, act_params = activation_data[i]
                
                    # Create a new dictionary with the trailing numbers removed from the keys
                    processed_act_params = {re.sub(r'_\d+$', '', k): v for k, v in act_params.items()}
                    self.layers.append(act_func(**processed_act_params).to(self.device))

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

        for layer in self.layers:
            x = layer(x.to(self.device))
        return x

class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]
 
class NNOptimizer:
    def __init__(self,
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            storage: str = "sqlite:///NNoptimization.db",
            encoder_params: Union[None, dict] = None,
            output_size: int = 1,
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False
        ):

        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']

        self.random_seed = random_seed
        self.use_gpu = use_gpu

        self.set_random_seed()
        
        # Convert the data do np.ndarray then to torch.Tensor
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)
        self.test_loader = None

        if X_validation is not None and y_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)
            self.validation_loader = None
        else:
            self.X_validation = None
            self.y_validation = None
            self.validation_loader = None

        self.input_size = self.X_train.shape[1]
        self.output_size = output_size

        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256
        
        self.verbose = verbose

        if encoder_params is not None:
            # Build the encoding and decoding functions
            if encoder_params['encoder_activation'] == 'LeakyReLU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](negative_slope = encoder_params['negative_slope_encoder'])
            elif encoder_params['encoder_activation'] == 'GELU':
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])](approximate = encoder_params['approximate_encoder'])
            else:
                encoder_activation = self.activation_functions[self.activation_functions_str.index(encoder_params['encoder_activation'])]()

            # Build just the encoder
            self.encoder = [("Linear", self.input_size, encoder_params['encoding_dim']), ("BatchNorm1d", encoder_params['encoding_dim']), ("Activation", encoder_activation)]
        else:
            self.encoder = None
        
        # Set the storage string for the study
        self.storage = storage

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

    def train_test_model(self, model, train_loader, test_loader, optimizer, criterion, clip_grad, trial, epochs = 100):
        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                outputs = model(inputs)                                  # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))            # Calculate the loss
                loss.backward()                                          # Backward pass
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)  # Clip the gradients
                optimizer.step()                                         # Update weights

                running_loss += loss.item()

            # Set the model to evaluation mode
            model.eval()

            running_loss = 0.0

            all_predictions = []
            all_labels = []

            for inputs, labels in test_loader:
                predicted = model(inputs)
                loss = criterion(predicted, labels.view(-1, 1))
                running_loss += loss.item()
                
                all_predictions.extend(predicted.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

        # Get the RMSE
        average_loss = running_loss / len(test_loader)
        rmse = np.sqrt(average_loss)

        if self.verbose:
            print(f'Test Loss: {average_loss}')
            print(f'Test RMSE: {rmse}')

        trial.report(rmse, epoch)
        
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return rmse

    def objective(self, trial):
        self.set_random_seed()

        # Suggest the learning rate
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Suggest the number of hidden layers and the number of units in each layer
        hidden_layers = []
        for i in range(trial.suggest_int('n_layers', 1, 5)):
            hidden_layers.append(trial.suggest_categorical(f'n_units_layer_{i}', self.power_of_two_options))
        
        # Suggestions for the activation functions
        activation_data = []

        for i in range(len(hidden_layers)):
            activation_function_str = trial.suggest_categorical(f'activation_function_{i}', self.activation_functions_str)
            activation_function = self.activation_functions[self.activation_functions_str.index(activation_function_str)]
            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                activation_data.append((activation_function, {
                    f'negative_slope_{i}': trial.suggest_float(f'negative_slope_{i}', 0.01, 0.5)
                }))
            elif activation_function == nn.GELU:
                activation_data.append((activation_function, {
                    f'approximate_{i}': trial.suggest_categorical(f'approximate_{i}', ['none', 'tanh'])
                }))  
            else:
                activation_data.append((activation_function, {}))

        model = DynamicNN(self.input_size, self.output_size, hidden_layers, activation_data, self.encoder, self.device)

        # Print the model architecture
        if self.verbose:
            print(model)

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Suggest the batch size
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        # Create the train and test loaders
        self.train_loader = DataLoader(
            dataset = CustomDataset(self.X_train, self.y_train), 
            batch_size = batch_size, 
            shuffle = True
        )
        self.test_loader = DataLoader(
            dataset = CustomDataset(self.X_test, self.y_test), 
            batch_size = batch_size
        )

        # If a validation set has been provided, create the validation loader
        if self.X_validation is not None:
            self.validation_loader = DataLoader(
                dataset = CustomDataset(self.X_validation, self.y_validation), 
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
            else:
                # Calculate the ROC
                fpr, tpr, _ = roc_curve(y_validation_np, validation_predictions_np) # type: ignore
                validation_auc = auc(fpr, tpr)
            # Set the optuna user attrs
            trial.set_user_attr('AUC', validation_auc)
        else:
            validation_auc = None

        return test_loss

    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "NN_Optimization", load_if_exists = True, sampler: optuna.samplers.BaseSampler = TPESampler(), n_jobs = 1):
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
