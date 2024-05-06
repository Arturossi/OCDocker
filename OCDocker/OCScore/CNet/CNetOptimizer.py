"""
"""

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

class ONet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ONet, self).__init__()
        
        # Upper branch (Encoder-like)
        self.upper1 = self.contracting_block(in_channels, 64)
        self.upper2 = self.contracting_block(64, 128)
        
        # Lower branch (Decoder-like)
        self.lower_start = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.lower1 = self.expanding_block(64, 128)
        self.lower2 = self.expanding_block(128, 256)
        
        # Feature Integration and Final Layer
        self.integrate = nn.Conv2d(256 + 128, out_channels, kernel_size=1)  # Adjust channel sizes accordingly
        
    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Upper branch
        upper1 = self.upper1(x)
        upper2 = self.upper2(upper1)
        
        # Lower branch
        lower_start = self.lower_start(x)
        lower1 = self.lower1(lower_start)
        lower2 = self.lower2(lower1)
        
        # Combine features from both branches
        combined = torch.cat([upper2, lower2], dim=1)
        
        # Final integrated output
        output = self.integrate(combined)
        return output

# Example usage:
model = ONet(in_channels=3, out_channels=1)
print(model)


class TransOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test, X_validation=None, y_validation=None, storage='sqlite:///Transoptimization.db', output_size=1, random_seed=42, use_gpu=True, verbose=False):
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

    def set_random_seed(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.use_gpu:
            torch.cuda.manual_seed_all(self.random_seed)

    def train_test_model(self, model, train_loader, test_loader, optimizer, criterion, clip_grad, trial, batch_size, epochs = 100):
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
        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8, 16])
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 6)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [512, 1024, 2048, 4096])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        lr = trial.suggest_float('lr', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 10, 100)

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
        
        # Model setup
        model = TransformerModel(self.X_train.shape[-1], d_model, self.output_size, nhead, num_encoder_layers, dim_feedforward, dropout, init_type, init_params, self.random_seed, self.device)

        # Suggestions for the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        criterion = nn.MSELoss()

        self.train_loader = DataLoader(
                dataset = CustomDataset(self.X_train, self.y_train), 
                batch_size = batch_size, 
                shuffle = True,
                drop_last=True
            )
        
        self.test_loader = DataLoader(
                dataset = CustomDataset(self.X_test, self.y_test), 
                batch_size = batch_size,
                drop_last=True
            )

        # If a validation set has been provided, create the validation loader
        if self.X_validation is not None:
            self.validation_loader = DataLoader(
                dataset = CustomDataset(self.X_validation, self.y_validation), 
                batch_size = batch_size, 
                shuffle = True,
                drop_last=True
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