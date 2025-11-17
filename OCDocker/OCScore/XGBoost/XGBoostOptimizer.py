#!/usr/bin/env python3

# Description
###############################################################################
''' Module to run the Extreme Gradient Boost algorithm. 

It is imported as:

import OCDocker.OCScore.XGBoost.OCxgboost as OCxgboost
'''

# Imports
###############################################################################

import optuna

import numpy as np
import pandas as pd

from optuna.samplers import TPESampler
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import auc, roc_curve
from typing import Union

import OCDocker.OCScore.XGBoost.OCxgboost as OCxgboost
import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


class XGBoostOptimizer:
    def __init__(self, 
            X_train : Union[np.ndarray, pd.DataFrame, pd.Series],
            y_train : Union[np.ndarray, pd.DataFrame, pd.Series],
            X_test : Union[np.ndarray, pd.DataFrame, pd.Series],
            y_test : Union[np.ndarray, pd.DataFrame, pd.Series],
            X_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            y_validation : Union[None, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
            storage : str = "sqlite:///pre_xgboost.db",
            params : dict = {},
            early_stopping_rounds : int = 20,
            use_gpu : bool = False,
            random_state : int = 42,
            verbose : bool = False
        ) -> None:
        '''
        Initializes the PreXGBoostOptimizer with training data and configuration.

        Parameters
        ----------
        X_train : np.ndarray | pd.DataFrame | pd.Series
            The training dataset.
        y_train : np.ndarray | pd.DataFrame | pd.Series
            The training labels.
        X_test : np.ndarray | pd.DataFrame | pd.Series
            The test dataset.
        y_test : np.ndarray | pd.DataFrame | pd.Series
            The test labels.
        X_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation dataset and labels. Default is None.
        y_validation : np.ndarray | pd.DataFrame | pd.Series, optional
            The validation labels. Default is None.
        params : dict, optional
            The hyperparameters for the XGBoost model. Default is an empty dictionary.
        early_stopping_rounds : int, optional
            The number of early stopping rounds for the XGBoost model. Default is 50.
        use_gpu : bool, optional
            Whether to use the GPU for training the XGBoost model. Default is False.
        load_if_exists : bool, optional
            Whether to load the study if it exists. Default is True.
        random_state : int, optional
            The random state for reproducibility. Default is 42.
        verbose : bool, optional
            Whether to print the training logs. Default is False.
        '''

        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.X_test = np.asarray(X_test)
        self.y_test = np.asarray(y_test)

        # If the validation dataset is provided, convert it to numpy arrays
        if X_validation is not None and y_validation is not None:
            self.X_validation = np.asarray(X_validation)
            self.y_validation = np.asarray(y_validation)

            # If the use_gpu flag is set
            if use_gpu:
                import cupy as cp

                # Send the validation data to the GPU
                self.X_validation = cp.asarray(self.X_validation)
                self.y_validation = cp.asarray(self.y_validation)
        else:
            self.X_validation = None
            self.y_validation = None

        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbose = verbose
        self.use_gpu = use_gpu

        # If use_gpu is True, set the device to 'cuda'
        if use_gpu:
            self.params['device'] = 'cuda'

        # Set the storage string for the study

        self.storage = storage









    def objective(self,
            trial : optuna.trial.Trial
        ) -> Union[float, tuple[float, float]]:
        '''
        The objective function for Optuna optimization to tune XGBoost hyperparameters.

        Parameters
        ----------
        trial : optuna.trial._trial.Trial
            A single trial object which suggests hyperparameters.

        Returns
        -------
        float | tuple[float, float]
            The AUC of the model as a result of the suggested hyperparameters. If the validation dataset is provided, returns a tuple of AUC and RMSE.
        '''

        # Create a local copy of params for this trial to prevent side-effects
        trial_params = self.params.copy()

        # Check if the hyperparameters are already in the params dictionary, if not, suggest them
        if "max_depth" not in trial_params:
            trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 10)

        if "learning_rate" not in trial_params:
            trial_params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)

        if "n_estimators" not in trial_params:
            trial_params["n_estimators"] = trial.suggest_int("n_estimators", 75, 125)

        if "subsample" not in trial_params:
            trial_params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

        if "colsample_bytree" not in trial_params:
            trial_params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)

        if "reg_alpha" not in trial_params:
            trial_params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 1.0)

        if "reg_lambda" not in trial_params:
            trial_params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 1.0)

        if "min_child_weight" not in trial_params:
            trial_params["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)

        if "gamma" not in trial_params:
            trial_params["gamma"] = trial.suggest_float("gamma", 0.0, 1.0)

        if "tree_method" not in trial_params:
            trial_params["tree_method"] = "hist"

        if "objective" not in trial_params:
            trial_params["objective"] = "reg:squarederror"

        if "booster" not in trial_params:
            trial_params["booster"] = "gbtree"

        if "random_state" not in trial_params:
            trial_params["random_state"] = self.random_state

        if "eval_metric" not in trial_params:
            if self.X_validation is not None:
                trial_params["eval_metric"] = 'rmse'
            else:
                trial_params["eval_metric"] = 'auc'

            # Set validation for pruning based on AUC
            pruning_callback = XGBoostPruningCallback(trial, f"validation_0-{ trial_params['eval_metric'] }")

            # Add the pruning callback to the trial_params
            trial_params['callbacks'] = [ pruning_callback ]

        # Add the early stopping rounds to the trial_params
        trial_params['early_stopping_rounds'] = self.early_stopping_rounds

        # If the validation dataset is provided, use it to get the AUC score
        if self.X_validation is not None:
            # Train the model and get the AUC score
            model, metric = OCxgboost.run_xgboost(self.X_train, self.y_train, self.X_test, self.y_test, params = trial_params, verbose = self.verbose) # type: ignore

            # Predict the validation dataset
            y_pred = model.predict(self.X_validation)

            # If the use_gpu flag is set
            if self.use_gpu:
                # Convert the predictions to numpy arrays
                y_validation_np = self.y_validation.get() # type: ignore
            else:
                y_validation_np = self.y_validation


            # Get the AUC score of the validation dataset
            fpr, tpr, _ = roc_curve(y_validation_np, y_pred) # type: ignore

            # Calculate the AUC score
            roc_auc = auc(fpr, tpr)

            # Save the AUC score as a user attribute
            trial.set_user_attr("AUC", roc_auc)
        
        else:
            # Train the model and get the AUC score
            _, metric = OCxgboost.run_xgboost(self.X_train, self.y_train, self.X_test, self.y_test, params = trial_params, verbose = self.verbose) # type: ignore
    
        # Return the trained AUC score

        return metric








    def optimize(self, 
                 direction : str = "minimize",
                 n_trials : int = 1000,
                  n_jobs : int = 1,
                 study_name : str = "XGBoost pre-optimization",
                 load_if_exists : bool = True
                ) -> optuna.study.Study:
        '''
        Optimizes XGBoost hyperparameters using Optuna.

        Parameters
        ----------
        directions : str | list, optional
            The direction of the optimization. Default is "maximize".
        n_trials : int, optional
            The number of trials for Optuna optimization. Default is 100.

        Returns
        -------
        optuna.study.Study
            The Optuna study object.
        n_trials : int, optional
            The number of trials for Optuna optimization. Default is 1000.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is 1.
        study_name : str, optional
            The name of the study. Default is "XGBoost pre-optimization".
        dict
            The best hyperparameters.
        float
            The best AUC score.
        '''


        # Create the Sampler
        sampler = TPESampler(seed = self.random_state)

        # Create an Optuna study
        study = optuna.create_study(
            direction = direction, 
            study_name = study_name, 
            storage = self.storage, 
            load_if_exists = load_if_exists, 
            sampler = sampler
        )

        # Optimize the objective function
        study.optimize(self.objective, n_trials = n_trials, n_jobs = n_jobs) # type: ignore

        # Get the best hyperparameters and the best score
        best_params = study.best_params
        best_score = study.best_value

        if self.verbose:
            ocprint.printv(f"Best score: { best_score }")
            ocprint.printv(f"Best hyperparameters: {best_params}")

        return study

# Methods
###############################################################################
