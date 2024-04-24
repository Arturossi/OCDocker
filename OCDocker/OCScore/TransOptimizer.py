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



class CustomDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import optuna
import random
from typing import Union, List

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.embedding(src)  # embedding the input
        output = self.transformer_encoder(src)
        output = self.fc_out(output.mean(dim=1))
        return output

class TransOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, X_validation=None, y_validation=None, storage='sqlite:///Transoptimization.db', output_size=1, random_seed=42, use_gpu=True, verbose=False):
        self.random_seed = random_seed
        self.use_gpu = use_gpu
        self.set_random_seed()

        # Handling data
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        
        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        self.test_loader = None

        if X_validation is not None and y_validation is not None:
            self.X_validation = torch.tensor(X_validation, dtype=torch.float32).to(self.device)
            self.y_validation = torch.tensor(y_validation, dtype=torch.float32).to(self.device)
        else:
            self.X_validation = None
            self.y_validation = None

        self.validation_loader = None

        self.output_size = output_size
        self.verbose = verbose
        self.storage = storage

    def set_random_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.use_gpu:
            torch.cuda.manual_seed_all(self.random_seed)

    def train_test_model(self, model, train_loader, test_loader, optimizer, criterion, clip_grad, trial, epochs = 100):
        # For each epoch
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # Set the running loss to 0            
            running_loss = 0.0

            for _, (inputs, labels) in enumerate(train_loader):
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
        average_loss = running_loss / len(test_loader) # type: ignore
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
        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8, 16])
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 6)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024, 2048])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 100)

        # Model setup
        model = TransformerModel(self.X_train.shape[-1], d_model, self.output_size, nhead, num_encoder_layers, dim_feedforward, dropout).to(self.device)

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        criterion = nn.MSELoss()

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
        
        # Suggestions for clipping the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)

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

        return best_params