import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import optuna
from sklearn.model_selection import train_test_split
from urllib.parse import quote_plus

# Assuming you have a dataset loaded into X
# X = ...

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

        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
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

class AutoencoderOptimizer:
    def __init__(self, 
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            random_seed = 42, 
            use_gpu = True
        ):
        self.input_size = train_dataset.features.shape[1] # TODO: Here
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.validation_dataset = validation_dataset

        # Set the random seed for reproducibility
        torch.manual_seed(random_seed)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(random_seed)
        else:
            self.device = torch.device("cpu")

    def train_autoencoder(self, model, train_loader, optimizer, criterion, epochs):
        model.train()
        for epoch in range(epochs):
            for data, _ in train_loader:
                optimizer.zero_grad()
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                loss.backward()
                optimizer.step()

    def evaluate_autoencoder(self, model, test_loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                reconstruction = model(data)
                loss = criterion(reconstruction, data)
                total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
        rmse = np.sqrt(average_loss)
        return rmse

    def objective(self, trial):
        encoding_dim = trial.suggest_int('encoding_dim', 16, 256)
        lr = trial.suggest_float('lr', 1e-4, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        epochs = trial.suggest_int('epochs', 20, 100)

        activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU]
        activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU']
        
        encoder_activation_str = trial.suggest_categorical('encoder_activation', activation_functions_str)
        encoder_activation_fn = activation_functions[activation_functions_str.index(encoder_activation_str)]()

        decoder_activation_str = trial.suggest_categorical('decoder_activation', activation_functions_str)
        decoder_activation_fn = activation_functions[activation_functions_str.index(decoder_activation_str)]()

        model = Autoencoder(self.input_size, encoding_dim, encoder_activation_fn, decoder_activation_fn).to(self.device)

        # Choose the optimizer
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr = lr, weight_decay = weight_decay)

        criterion = nn.MSELoss()

        train_loader = DataLoader(dataset=self.train_dataset, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size = batch_size)

        self.train_autoencoder(model, train_loader, optimizer, criterion, epochs)

        avg_loss = self.evaluate_autoencoder(model, test_loader, criterion)

        return avg_loss
    
    def optimize(self, n_trials = 100):
        # Data preparation (example, replace with your actual data loading)
        train_dataset = AutoencoderDataset(self.X_train)
        test_dataset = AutoencoderDataset(X_test)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        return study

if __name__ == "__main__":
    
