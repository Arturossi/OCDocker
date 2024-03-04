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
        ):
        super(DynamicNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = nn.ModuleList()

        self.input_layer_size = input_size

        for layer_size in hidden_layers:
            self.hidden_layers.append(nn.Linear(self.input_layer_size, layer_size))
            self.input_layer_size = layer_size
        
        self.output_layer = nn.Linear(self.input_layer_size, self.output_size)

    def forward(self, x: torch.Tensor, activation_data: list[tuple] = []) -> torch.Tensor:
        '''
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        activation_data : list of tuples
            List with the parameters for the activation function

        Returns
        -------
        torch.Tensor
            Output tensor
        '''

        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                if activation_data:
                    # Remove the first element from the list
                    activation_function, activation_params = activation_data.pop(0)
                    # Set the activation function
                    x = activation_function(layer(x), **activation_params)

            # Set the output layer
            x = self.output_layer(x)
            return x
        else:
            # Throw an error if the model has not been built
            raise ValueError("Model has not been built")

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
        
        # Convert the data do np.ndarray then to torch.Tensor
        self.X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32)
        self.train_loader = DataLoader(CustomDataset(self.X_train, self.y_train), batch_size=32, shuffle=True)

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32)
        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32)
        self.test_loader = DataLoader(CustomDataset(self.X_test, self.y_test), batch_size=32, shuffle=True)

        if X_validation is not None and y_validation is not None:
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32)
            self.validation_loader = DataLoader(CustomDataset(self.X_validation, self.y_validation), batch_size=32, shuffle=True)
        else:
            self.X_validation = None
            self.y_validation = None
            self.validation_loader = None

        self.input_size = self.X_train.shape[1]
        self.output_size = output_size

        self.power_of_two_options = [2**i for i in range(4, 9)]  # 16, 32, 64, 128, 256

        # Set the seed for CPU
        torch.manual_seed(random_seed)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(random_seed)
        else:
            self.device = torch.device('cpu')
        
        self.verbose = verbose

        # Set the storage string for the study
        self.storage = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"

    def train_model(self, model, train_loader, optimizer, criterion, activation_data, epochs = 100):
        # Set the model to training mode
        model.train()

        # For each epoch
        for epoch in range(epochs):
            # Set the running loss to 0            
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                optimizer.zero_grad()

                outputs = model(inputs, activation_data)  # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))  # Calculate the loss

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)

            if self.verbose:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

        # Optionally, you might return some metric like average_loss or validation_loss
        return average_loss  # Adjust as needed
    
    def test_model(self, model, test_loader, criterion, activation_data):
        model.eval() # Set the model to evaluation mode

        running_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():  # No need to calculate gradients during testing
            for inputs, labels in test_loader:
                predicted = model(inputs, activation_data)
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
                activation_data.append((activation_function, {'negative_slope': trial.suggest_uniform('negative_slope', 0.01, 0.5)}))
            elif activation_function == nn.GELU:
                activation_data.append((activation_function, {'approximate': trial.suggest_categorical('approximate', ['none', 'tanh'])}))
            else:
                activation_data.append((activation_function, {}))

        model = DynamicNN(self.input_size, self.output_size, hidden_layers)

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        # Suggestions for the epochs
        epochs = trial.suggest_int('epochs', 5, 100)

        # Use Root Mean Squared Error as the loss function
        criterion = nn.MSELoss()

        # Train the model
        train_loss = self.train_model(model, self.train_loader, optimizer, criterion, activation_data, epochs = epochs)

        # Get the test loss
        test_loss, _ = self.test_model(model, self.test_loader, activation_data, criterion)

        # If a validation set has been provided, calculate the AUC
        if self.validation_loader is not None:
            # Set the model to evaluation mode
            model.eval()
            # Get the predictions for the validation set
            validation_predictions = model(self.X_validation, activation_data,)
            # Calculate the ROC
            fpr, tpr, _ = roc_curve(self.y_validation, validation_predictions) # type: ignore
            validation_auc = auc(fpr, tpr)
            # Set the optuna user attrs
            trial.set_user_attr('AUC', validation_auc)
        else:
            validation_auc = None

        return test_loss

    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "NN_Optimization", load_if_exists = True, sampler = TPESampler(), n_jobs = 1):
        if self.verbose:
            print(f'Optimizing the model for {n_trials} trials')

        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler
        )

        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)

