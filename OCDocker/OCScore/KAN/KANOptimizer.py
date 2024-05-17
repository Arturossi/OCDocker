import optuna
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optuna.samplers import TPESampler
from sklearn.metrics import auc, roc_curve
from typing import Callable, Union

#from OCDocker.Initialise import *

import kan
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


class KolmogorovArnoldNetwork():
    def __init__(self, 
            input_size: int, 
            hidden_layers_sizes: list[int],
            grid: int = 3,
            k: int = 3,
            noise_scale: float = 0.1,
            noise_scale_base: float = 0.1,
            base_fun: nn.Module = nn.SiLU(),
            symbolic_enabled: bool = True,
            bias_trainable: bool = True,
            grid_eps: float = 1.0,
            grid_range: list[float] = [-1.0, 1.0],
            sp_trainable: bool = True,
            sb_trainable: bool = True,
            output_size: int = 1,
            optimizer: str = "LBFGS",
            batch_size: int = 32,
            epochs: list[int] = [20],
            lr: float = 1e-3,
            clip_grad: float = 1.0,
            random_seed = 42,
            use_gpu = True,
            verbose = False
        ):

        self.optimizer_functions_str = ["LBFGS", "Adam"]

        # Check if the optimizer is in the list of optimizer functions
        if optimizer not in self.optimizer_functions_str:
            raise ValueError(f"The optimizer should be one of {self.optimizer_functions_str} and not {optimizer}")

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        self.random_seed = random_seed
        self.use_gpu = use_gpu

        self.input_size = input_size

        self.width = [self.input_size, *hidden_layers_sizes, output_size]
        self.grid = grid
        self.k = k
        self.noise_scale = noise_scale
        self.noise_scale_base = noise_scale_base
        self.base_fun = base_fun
        self.symbolic_enabled = symbolic_enabled
        self.bias_trainable = bias_trainable
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        self.seed = self.random_seed

        self.KAN = kan.KAN(
            width = self.width,
            grid = self.grid,
            k = self.k,
            noise_scale= self.noise_scale,
            noise_scale_base = self.noise_scale_base,
            base_fun = self.base_fun, # type: ignore
            symbolic_enabled = self.symbolic_enabled,
            bias_trainable = self.bias_trainable,
            grid_eps = self.grid_eps,
            grid_range = self.grid_range,
            sp_trainable = self.sp_trainable,
            sb_trainable = self.sb_trainable,
            device = self.device,
            seed = self.seed
        )

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.clip_grad = clip_grad

        self.optimizer = optimizer

        # Set the AUC and rmse as nan
        self.validation_auc = np.NaN
        self.rmse = np.NaN

        # Set the verbose flag
        self.verbose = verbose

        self.prediction = None

        if verbose:
            print(self.KAN)

    def set_random_seed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)

        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        #torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    def train_model(self, X_train, y_train, X_test, y_test, X_validation = None, y_validation = None, log=1, lamb=0.0, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.0, lamb_coefdiff=0.0, update_grid=True, grid_update_num=10, loss_fn=nn.MSELoss(), stop_grid_update_step=50, batch=-1, small_mag_threshold=1e-16, small_reg_factor=1.0, metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='plot', symbolic_enabled=True, plot_intermediate=False):
        self.set_random_seed()

        # Convert the data to torch.Tensor
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(self.device)

        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)
        
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)

        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)

        self.dataset = {
            'train_input': X_train,
            'train_label': y_train,
            'test_input': X_test,
            'test_label': y_test
        }

        # For each epoch element in the epochs
        for i, epoch in enumerate(self.epochs):
            # Train the model
            result = self.KAN.train(
                dataset = self.dataset,
                opt = self.optimizer,
                steps = epoch,
                log = log,
                lamb = lamb,
                lamb_l1 = lamb_l1,
                lamb_entropy = lamb_entropy,
                lamb_coef = lamb_coef,
                lamb_coefdiff = lamb_coefdiff,
                update_grid = update_grid,
                grid_update_num = grid_update_num,
                loss_fn = loss_fn,
                stop_grid_update_step = stop_grid_update_step,
                batch = batch,
                small_mag_threshold = small_mag_threshold,
                small_reg_factor = small_reg_factor,
                metrics = metrics,
                sglr_avoid = sglr_avoid,
                save_fig = save_fig,
                in_vars = in_vars,
                out_vars = out_vars,
                beta = beta,
                save_fig_freq = save_fig_freq,
                img_folder = img_folder,
                device = self.device
            )

            # If plot_intermediate is True
            if plot_intermediate:
                # Plot the model
                self.KAN.plot()

                plt.savefig(f'{img_folder}/KAN_epoch_{i}.png')

            # If it is not the last epoch (the last epoch can have the same value as any other epoch)
            if i != len(self.epochs) - 1:
                # Prune the model
                self.KAN = self.KAN.prune()

                # If plot_intermediate is True
                if plot_intermediate:
                    # Plot again after pruning
                    self.KAN.plot()

                    plt.savefig(f'{img_folder}/KAN_epoch_{i}_pruned.png')

        # If symbolic_enabled is True
        if symbolic_enabled:
            # Enable the symbolic mode
            self.symbolic = self.KAN.auto_symbolic()
        
        # Get the RMSE
        rmse = result['test_loss'].sum() # TODO: Sum???

        if X_validation is not None and y_validation is not None:
            if isinstance(X_validation, list):
                X_validation = [torch.tensor(np.asarray(x), dtype=torch.float32).to(self.device) for x in X_validation]
            else:
                X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            
            y_validation = np.asarray(y_validation)

            # Evaluate the model
            validation_predictions = self.KAN(X_validation)

            # Convert the predictions to numpy
            validation_predictions_np = validation_predictions.detach().cpu().numpy()

            # If there is a nan in the predictions, set the AUC to 0
            if np.isnan(validation_predictions_np).any():
                validation_auc = 0
            else:
                # Calculate the ROC
                fpr, tpr, _ = roc_curve(y_validation, validation_predictions_np)
                validation_auc = auc(fpr, tpr)

        self.rmse = rmse
        self.validation_auc = validation_auc

        return True
    
    def get_model(self):
        return self.KAN
        
class KANOptimizer:
    def __init__(self,
            X_train: Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]],
            y_train: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test: Union[np.ndarray, pd.DataFrame, pd.Series, list[Union[np.ndarray, pd.DataFrame, pd.Series]]],
            y_test: Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series], list[Union[None, np.ndarray, pd.DataFrame, pd.Series]]] = None,
            y_validation: Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            storage: str = "sqlite:///KANoptimization.db",
            output_size: int = 1,
            random_seed: int = 42,
            symbolic_enabled: bool = True,
            bias_trainable: bool = True,
            grid_range: list[float] = [-1.0, 1.0],
            log: int = 1, # Logging frequency,
            update_grid: bool = True,
            sglr_avoid=False, 
            save_fig=False, 
            in_vars=None, 
            out_vars=None, 
            beta=3, 
            save_fig_freq=1, 
            img_folder='plot', 
            plot_intermediate=False,
            use_gpu: bool = True,
            verbose: bool = False
        ):

        self.use_gpu = use_gpu
        self.random_seed = random_seed
        self.symbolic_enabled = symbolic_enabled
        self.bias_trainable = bias_trainable
        self.grid_range = grid_range
        self.log = log
        self.update_grid = update_grid
        self.sglr_avoid = sglr_avoid
        self.save_fig = save_fig
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.beta = beta
        self.save_fig_freq = save_fig_freq
        self.img_folder = img_folder
        self.plot_intermediate = plot_intermediate

        if self.use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.activation_functions = [nn.GELU, nn.LeakyReLU, nn.Mish, nn.ReLU, nn.SELU, nn.Identity]
        self.activation_functions_str = ['GELU', 'LeakyReLU', 'Mish', 'ReLU', 'SELU', 'Identity']

        self.set_random_seed()
        
        # Convert the data do np.ndarray then to torch.Tensor
        try:
            self.X_train = torch.tensor(np.asarray(X_train[:2286]), dtype=torch.float32).to(self.device)
        except Exception as e:
            print(e)

        self.input_size = self.X_train.shape[1]

        self.y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).to(self.device)
        self.train_loader = None

        self.X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(self.device)

        self.y_test = torch.tensor(np.asarray(y_test), dtype=torch.float32).to(self.device)
        self.test_loader = None

        # Check if the validation set has been provided or if any of its elements are None
        if (X_validation is not None and y_validation is not None) or not (isinstance(X_validation, list) and any(x is None for x in X_validation)):
            self.X_validation = torch.tensor(np.asarray(X_validation), dtype=torch.float32).to(self.device)
            self.y_validation = torch.tensor(np.asarray(y_validation), dtype=torch.float32).to(self.device)
            self.validation_loader = None
        else:
            self.X_validation = None
            self.y_validation = None
            self.validation_loader = None

        self.output_size = output_size

        self.power_of_two_options = [2**i for i in range(4, 12)]  # 16, 32, 64, 128, 256
        
        self.verbose = verbose

        self.model = None
        
        # Set the storage string for the study
        self.storage = storage

    def set_random_seed(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Set the seed for CPU
        torch.manual_seed(self.random_seed)

        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
        
        #torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def train_test_model(self, model, X_train, y_train, X_test, y_test, trial, X_validation = None, y_validation = None, log=1, lamb=0.0, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.0, lamb_coefdiff=0.0, update_grid=True, grid_update_num=10, loss_fn=nn.MSELoss(), stop_grid_update_step=50, batch=-1, small_mag_threshold=1e-16, small_reg_factor=1.0, metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='plot', symbolic_enabled=True, plot_intermediate=False):
        # Train the model
        results = model.train_model(X_train, y_train, X_test, y_test, X_validation, y_validation, log, lamb, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff, update_grid, grid_update_num, loss_fn, stop_grid_update_step, batch, small_mag_threshold, small_reg_factor, metrics, sglr_avoid, save_fig, in_vars, out_vars, beta, save_fig_freq, img_folder, symbolic_enabled, plot_intermediate)

        # Get the RMSE
        rmse = results['test_loss'].sum() # TODO: Sum???

        if self.verbose:
            print(f'Test RMSE: {rmse}')

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return rmse, model.validation_auc

    def objective(self, trial):
        self.set_random_seed()

        # Suggest the learning rate
        lr = trial.suggest_float('lr', 1e-5, 1e-1)

        # Suggest the number of hidden layers and the number of units in each layer
        hidden_layers = []
        for i in range(trial.suggest_int('n_layers', 1, 4)):
            hidden_layers.append(trial.suggest_int(f'n_units_layer_{i}', 2, 5))
        
        # Suggest the grid size
        grid = trial.suggest_int('grid', 3, 10)

        # Suggest the k value
        k = trial.suggest_int('k', 3, 10)

        # Suggest the noise scale
        noise_scale = trial.suggest_float('noise_scale', 0.01, 0.5)

        # Suggest the noise scale base
        noise_scale_base = trial.suggest_float('noise_scale_base', 0.01, 0.5)

        # Suggest the base function
        base_function_str = trial.suggest_categorical('base_function', self.activation_functions_str)

        if base_function_str == 'LeakyReLU':
            base_function = self.activation_functions[self.activation_functions_str.index(base_function_str)](negative_slope = trial.suggest_float('negative_slope', 0.01, 0.5))
        
        elif base_function_str == 'GELU':
            base_function = self.activation_functions[self.activation_functions_str.index(base_function_str)](approximate = trial.suggest_categorical('approximate', ['none', 'tanh']))
        else:
            base_function = self.activation_functions[self.activation_functions_str.index(base_function_str)]()

        # Suggest the grid epsilon
        grid_eps = trial.suggest_float('grid_eps', 0, 1.0)

        # Suggest the sp trainable
        sp_trainable = trial.suggest_categorical('sp_trainable', [True, False])

        # Suggest the sb trainable
        sb_trainable = trial.suggest_categorical('sb_trainable', [True, False])

        # Suggest the optimizer
        optimizer = trial.suggest_categorical('optimizer', ['LBFGS', 'Adam'])

        # Suggest the batch size
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        # Suggest the number of trainings
        n_trainings = trial.suggest_int('n_trainings', 1, 5)

        # Suggest the epochs for each training
        epochs = [trial.suggest_int(f'epochs_{i}', 20, 50) for i in range(n_trainings)]

        # Suggest the clipping of the gradients
        clip_grad = trial.suggest_float('clip_grad', 0.1, 1.0)

        # Suggest the lamb value
        lamb = trial.suggest_float('lamb', 0, 1.5)

        # Suggest the lamb l1 value
        lamb_l1 = trial.suggest_float('lamb_l1', 0, 1.5)

        # Suggest the lamb entropy value
        lamb_entropy = trial.suggest_float('lamb_entropy', 1, 5)

        # Suggest the lamb coef value
        lamb_coef = trial.suggest_float('lamb_coef', 0, 1.5)

        # Suggest the lamb coefdiff value
        lamb_coefdiff = trial.suggest_float('lamb_coefdiff', 0, 1.5)

        # Suggest the stop grid update step
        stop_grid_update_step = trial.suggest_int('stop_grid_update_step', 10, 25)

        # Suggest the small mag threshold
        small_mag_threshold = trial.suggest_float('small_mag_threshold', 1e-16, 1e-10)

        # Suggest the small reg factor
        small_reg_factor = trial.suggest_float('small_reg_factor', 0.5, 1.0)

        # Create the model
        self.model = KolmogorovArnoldNetwork(
            input_size = self.input_size,
            hidden_layers_sizes = hidden_layers,
            grid = grid,
            k = k,
            noise_scale = noise_scale,
            noise_scale_base = noise_scale_base,
            base_fun = base_function,
            symbolic_enabled = self.symbolic_enabled,
            bias_trainable = self.bias_trainable,
            grid_range = self.grid_range,
            sp_trainable = sp_trainable,
            sb_trainable = sb_trainable,
            output_size = self.output_size,
            optimizer = optimizer,
            batch_size = batch_size,
            epochs = epochs,
            lr = lr,
            clip_grad = clip_grad,
            random_seed = self.random_seed,
            use_gpu = self.use_gpu,
            verbose = self.verbose
        )

        # Train the model
        rmse, validation_auc = self.train_test_model(self.model, self.X_train, self.y_train, self.X_test, self.y_test, trial, self.X_validation, self.y_validation, log = self.log, lamb = lamb, lamb_l1 = lamb_l1, lamb_entropy = lamb_entropy, lamb_coef = lamb_coef, lamb_coefdiff = lamb_coefdiff, update_grid = self.update_grid, stop_grid_update_step = stop_grid_update_step, small_mag_threshold = small_mag_threshold, small_reg_factor = small_reg_factor, sglr_avoid = self.sglr_avoid, save_fig = self.save_fig, in_vars = self.in_vars, out_vars = self.out_vars, beta = self.beta, save_fig_freq = self.save_fig_freq, img_folder = self.img_folder, symbolic_enabled = self.symbolic_enabled, plot_intermediate = self.plot_intermediate)

        trial.set_user_attr('AUC', validation_auc)

        return rmse

    def optimize(self, direction: str = "maximize", n_trials = 10, study_name = "KAN_Optimization", load_if_exists = True, sampler: optuna.samplers.BaseSampler = TPESampler(), n_jobs = 1, show_progress_bar = False):
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

        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs, show_progress_bar = show_progress_bar)

        best_params = study.best_params
        print("Best Hyperparameters:", best_params)
