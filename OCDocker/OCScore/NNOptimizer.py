import optuna

import cupy as cp
import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from numpy.random import default_rng
from optuna.samplers import CmaEsSampler, TPESampler
from sklearn.metrics import auc, roc_curve
from typing import Union
from urllib.parse import quote_plus

#from OCDocker.Initialise import *

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

class DynamicNN(nn.Module):
    def __init__(self,
            input_size: int,
            output_size: int,
            hidden_layers: list,
            activation_data: list = [],
            device: torch.device = torch.device('cpu')
        ):
        super(DynamicNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.layers = nn.ModuleList()

        self.input_layer_size = input_size

        self.layer_sizes = [input_size] + hidden_layers + [output_size]

        self.device = device

        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]).to(self.device))
            if activation_data and i < len(activation_data):
                act_func, act_params = activation_data[i]
                self.layers.append(act_func(**act_params).to(self.device))

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
    def __init__(self, X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            output_size: int = 1,
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False
        ):

        # Set the seed for CPU
        torch.manual_seed(random_seed)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(random_seed)
        else:
            self.device = torch.device('cpu')
        
        # Convert the data do np.ndarray then to torch.Tensor
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)
        self.train_loader = DataLoader(CustomDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)
        self.test_loader = DataLoader(CustomDataset(self.X_test, self.y_test), batch_size=32, shuffle=True)

        if X_validation is not None and y_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)
            self.validation_loader = DataLoader(CustomDataset(self.X_validation, self.y_validation), batch_size=32, shuffle=True)
        else:
            self.X_validation = None
            self.y_validation = None
            self.validation_loader = None

        self.input_size = self.X_train.shape[1]
        self.output_size = output_size

        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256
        
        self.verbose = verbose

        # Set the storage string for the study
        self.storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"

    def train_model(self, model, train_loader, optimizer, criterion, trial, epochs = 100):
        # Set the model to training mode
        model.train()

        # For each epoch
        for epoch in range(epochs):
            # Set the running loss to 0            
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))  # Calculate the loss

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)

            if self.verbose:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

            trial.report(average_loss, epoch)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Optionally, you might return some metric like average_loss or validation_loss
        return average_loss  # Adjust as needed
    
    def test_model(self, model, test_loader, criterion):
        model.eval() # Set the model to evaluation mode

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():  # No need to calculate gradients during testing
            for inputs, labels in test_loader:
                predicted = model(inputs)
                loss = criterion(predicted, labels.view(-1, 1))
                running_loss += loss.item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Get the RMSE
        average_loss = running_loss / len(test_loader)
        rmse = np.sqrt(average_loss)

        if self.verbose:
            print(f'Test Loss: {average_loss}')
            print(f'Test RMSE: {rmse}')

        return rmse, predicted

    def train_test_model(self, model, train_loader, test_loader, optimizer, criterion, trial, epochs = 100):
        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))  # Calculate the loss

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

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
        # Suggest the learning rate
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Suggest the number of hidden layers and the number of units in each layer
        hidden_layers = []
        for i in range(trial.suggest_int('n_layers', 1, 5)):
            hidden_layers.append(trial.suggest_categorical(f'n_units_layer{i}', self.power_of_two_options))
        
        # Suggestions for the activation functions
        activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU]
        activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU']
        activation_data = []
        for i in range(len(hidden_layers) - 1):
            activation_function_str = trial.suggest_categorical(f'activation_function{i}', activation_functions_str)
            activation_function = activation_functions[activation_functions_str.index(activation_function_str)]
            # Now suggest the parameters for the activation function
            if activation_function == nn.LeakyReLU:
                activation_data.append((activation_function, {'negative_slope': trial.suggest_float('negative_slope', 0.01, 0.5)}))
            elif activation_function == nn.GELU:
                activation_data.append((activation_function, {'approximate': trial.suggest_categorical('approximate', ['none', 'tanh'])}))
            else:
                activation_data.append((activation_function, {}))

        model = DynamicNN(self.input_size, self.output_size, hidden_layers, activation_data, self.device)

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Suggestions for the epochs
        epochs = trial.suggest_int('epochs', 100, 1000)

        # Use Root Mean Squared Error as the loss function
        criterion = nn.MSELoss()

        # Train the model
        #train_loss = self.train_model(model, self.train_loader, optimizer, criterion, epochs = epochs)

        # Get the test loss
        #test_loss, _ = self.test_model(model, self.test_loader, criterion)

        test_loss = self.train_test_model(model, self.train_loader, self.test_loader, optimizer, criterion, trial, epochs = epochs)

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

    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "NN_Optimization", load_if_exists = True, sampler: optuna.samplers._base.BaseSampler = TPESampler(), n_jobs = 1):
        if self.verbose:
            print(f'Optimizing the model for {n_trials} trials')

        # Add a pruner
        pruner = optuna.pruners.MedianPruner()

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

