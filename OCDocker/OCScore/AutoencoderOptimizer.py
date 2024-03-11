import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Dataset
import optuna
from typing import Union

#import mlflow

class AutoencoderDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    
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
                 device = torch.device("cpu")
                ):
        super(Autoencoder, self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            encoder_activation_fn
        ).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size),
            decoder_activation_fn
        ).to(self.device)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
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
            storage: str = "sqlite:///autoencoder.db",
            models_folder: str = "./models/Autoencoder/",
            random_seed = 42, 
            use_gpu = True,
            verbose = False
        ):
        
        # Set the seed for CPU
        torch.manual_seed(random_seed)

        self.models_folder = models_folder

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(random_seed)
        else:
            self.device = torch.device('cpu')
        
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

        self.verbose = verbose

        self.best_rmse = np.inf

        # Set the storage string for the study
        self.storage = storage

    def train_autoencoder(self, model, optimizer, criterion, epochs, trial):

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
                print(f"Epoch {epoch+1}/{epochs}")

            running_loss = 0.0

            for data, _ in self.train_loader: # type: ignore
                optimizer.zero_grad()
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            average_loss = running_loss / len(self.train_loader) # type: ignore
            rmse = np.sqrt(average_loss)

            # Validation phase
            if self.validation_loader is not None:
                val_rmse = self.evaluate_autoencoder(model, criterion)

                trial.set_user_attr('val_rmse', val_rmse)
                
                if self.verbose:
                    print(f"Epoch {epoch+1}, Validation Loss: {val_rmse}")

                # Check for improvement
                if val_rmse < best_validation_rmse:
                    best_train_rmse = rmse
                    best_validation_rmse = val_rmse
                '''
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping_patience:
                        if self.verbose:
                            print("Early stopping triggered.")
                        trial.report(rmse, epoch)

                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
                        break
                '''
            
            if self.verbose:
                print(f'Test Loss: {average_loss}')
                print(f'Test RMSE: {rmse}')

            trial.report(rmse, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            return best_validation_rmse, best_train_rmse

    def evaluate_autoencoder(self, model, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in self.test_loader: # type: ignore
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                total_loss += loss.item()
        average_loss = total_loss / len(self.test_loader) # type: ignore
        rmse = np.sqrt(average_loss)
        return rmse

    def objective(self, trial):
        
        encoding_dim = trial.suggest_int('encoding_dim', 16, 256)
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 20, 100)

        activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'None']
        
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

        best_validation_rmse, best_train_rmse = self.train_autoencoder(model, optimizer, criterion, epochs, trial = trial) # type: ignore

        evaluate_rmse = self.evaluate_autoencoder(model, criterion)

        improvement_threshold = 0.0 # 0% improvement (if is better than the best, it will be logged)
        is_promising = best_validation_rmse < self.best_rmse * (1 - improvement_threshold)

        if is_promising:
            # Save the model
            torch.save(model.state_dict(), f'{self.models_folder}/autoencoder_{trial.number}.pt')
            self.best_rmse = best_validation_rmse

        '''
        with mlflow.start_run(nested = True):
            # Log the parameters
            mlflow.log_param('encoding_dim', encoding_dim)
            mlflow.log_param('lr', lr)
            mlflow.log_param('batch_size', batch_size)
            mlflow.log_param('epochs', epochs)

            mlflow.log_param('encoder_activation', encoder_activation_str)
            if encoder_activation_str == 'LeakyReLU':
                mlflow.log_param('negative_slope_encoder', pre_encoder_params['negative_slope_encoder'])
            elif encoder_activation_str == 'GELU':
                mlflow.log_param('approximate_encoder', pre_encoder_params['approximate_encoder'])

            mlflow.log_param('decoder_activation', decoder_activation_str)
            if decoder_activation_str == 'LeakyReLU':
                mlflow.log_param('negative_slope_decoder', pre_decoder_params['negative_slope_decoder'])
            elif decoder_activation_str == 'GELU':
                mlflow.log_param('approximate_decoder', pre_decoder_params['approximate_decoder'])

            mlflow.log_param('optimizer', optimizer_name)
            mlflow.log_param('weight_decay', weight_decay)
            mlflow.log_metric('best_train_rmse', best_train_rmse)
            mlflow.log_metric('best_validation_rmse', best_validation_rmse)
            mlflow.log_metric('evaluate_rmse', evaluate_rmse)

            if is_promising:
                mlflow.pytorch.log_model(model, "model")
                self.best_rmse = best_validation_rmse
        '''

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

        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler,
            pruner = pruner
        )

##        mlflow.set_experiment(study_name)

        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs)
        
        print("Best trial:")

        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")

        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study
