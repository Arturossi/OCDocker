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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, init_type: str = 'zeros', init_params: dict = {}, random_seed: int = 42, device=torch.device('cuda'), verbose = False):
        super(TransformerModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Linear(input_dim, d_model).to(device)

        # Normalization layer
        self.norm = nn.LayerNorm(d_model).to(device)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        ).to(device)

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers).to(device)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim).to(device)

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
            print(self)

    def set_random_seed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        # Create a generator for reproducibility
        generator = torch.Generator(device=self.device)

        # Set the seed for the generator
        generator.manual_seed(self.random_seed)

        return generator

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

    def forward(self, src):
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
    def __init__(self, 
            input_size, 
            output_size, 
            trans_params,
            random_seed = 42,
            use_gpu = True,
            verbose = False
    ):
        super(Transformer, self).__init__()

        self.random_seed = random_seed
        self.use_gpu = use_gpu

        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.input_size = input_size

        self.set_random_seed()

        self.optimizer_functions = [optim.Adam, optim.RMSprop, optim.SGD]
        self.optimizer_functions_str = ['Adam', 'RMSprop', 'SGD']

        # Create the transformer model
        self.trans = TransformerModel(input_size, trans_params['d_model'], output_size, trans_params['nhead'], trans_params['num_encoder_layers'], trans_params['dim_feedforward'], trans_params['dropout'], self.device).to(self.device)

        self.batch_size = trans_params['batch_size']
        self.epochs = trans_params['epochs']
        self.lr = trans_params['lr']
        self.clip_grad = trans_params['clip_grad']

        self.optimizer = self.optimizer_functions[self.optimizer_functions_str.index(trans_params['optimizer'])](
            self.trans.parameters(),
            weight_decay = trans_params['weight_decay'], 
            lr = trans_params['lr']
        )

        self.trans_params = trans_params

        # Set the AUC and rmse as nan
        self.validation_auc = np.NaN
        self.rmse = np.NaN

        # Set the verbose flag
        self.verbose = verbose

        self.prediction = None

        if verbose:
            print(self.trans)

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
        if isinstance(X_train, list):
            X_train = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_train]
        else:
            X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)

        y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)

        if isinstance(X_test, list):
            X_test = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_test]
        else:
            X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)

        y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)

        if X_validation is not None and y_validation is not None:
            if isinstance(X_validation, list):
                X_validation = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_validation]
            else:
                X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            
            y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)

        train_loader = DataLoader(
            dataset = CustomDataset(X_train, y_train), 
            batch_size = self.batch_size, 
            shuffle = True,
            drop_last=True
        )

        test_loader = DataLoader(
            dataset = CustomDataset(X_test, y_test),
            batch_size = self.batch_size,
            drop_last=True
        )

        # If a validation set has been provided, create the validation loader
        if X_validation is not None:
            validation_loader = DataLoader(
                dataset = CustomDataset(X_validation, y_validation), 
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

                outputs = self.trans(inputs)                                                    # Forward pass
                loss = criterion(outputs, labels.view(-1, 1))                                   # Calculate the loss
                loss.backward()                                                                 # Backward pass
                nn.utils.clip_grad_norm_(self.trans.parameters(), self.clip_grad, max_norm = 1) # Clip the gradients
                self.optimizer.step()                                                           # Update weights

                running_loss += loss.item()
        
            # Set the model to evaluation mode
            self.trans.eval()

            running_loss = 0.0

            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    predicted = self.trans(inputs)
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
                self.trans.eval()
                
                validation_predictions = self.trans(X_validation)

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
        return self.trans

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