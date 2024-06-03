#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage I/O operations in OCDocker in the context of scoring 
functions.

They are imported as:

import OCDocker.OCScore.Utils.Workers as ocscoreworkers
'''

# Imports
###############################################################################

import optuna
import time

import numpy as np

from OCDocker.OCScore.NN.AutoencoderOptimizer import AutoencoderOptimizer
from OCDocker.OCScore.NN.NNOptimizer import NNOptimizer
from OCDocker.OCScore.Transformer.TransOptimizer import TransOptimizer
from OCDocker.OCScore.XGBoost.XGBoostOptimizer import XGBoostOptimizer
from OCDocker.OCScore.XGBoost.GeneticAlgorithmFeatureSelector import GeneticAlgorithmFeatureSelector

from optuna.samplers import TPESampler
from typing import Union

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Methods
###############################################################################

def AEworker(
        pid: int,
        id: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        X_val: np.ndarray,
        encoding_dims: tuple,
        storage: str,
        models_folder: str,
        random_seed: int = 42,
        use_gpu: bool = True, 
        verbose: bool = False, 
        direction: str = "minimize", 
        n_trials: int = 250, 
        load_if_exists: bool = True, 
        n_jobs: int = 1,
        study_name: str = "Autoencoder_Optimization"
    ) -> optuna.study.Study:
    ''' Autoencoder optimization worker function.

    This function is used to run the optimization of an autoencoder model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Testing data.
    X_val : np.ndarray
        Validation data.
    encoding_dims : tuple
        Tuple with the encoding dimensions.
    storage : str
        Storage string.
    models_folder : str
        Folder to save the models.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    direction : str, optional
        Optimization direction. The default is "minimize".
    n_trials : int, optional
        Number of trials. The default is 250.
    load_if_exists : bool, optional
        Load if exists. The default is True.
    n_jobs : int, optional
        Number of jobs. The default is 1.
    study_name : str, optional
        Study name. The default is "Autoencoder_Optimization".

    Returns
    -------
    study : optuna.study.Study
        Study object.
    '''
    
    print(f"Process {pid} starting optimization")

    # Initialize the trainer
    trainer = AutoencoderOptimizer(
        X_train, 
        X_test, 
        X_val, 
        encoding_dims,
        storage,
        models_folder,
        random_seed = random_seed,
        use_gpu = use_gpu, 
        verbose = verbose
    )

    study = None
    
    # Run optimization
    study = trainer.optimize(
            direction = direction, 
            n_trials = n_trials,
            study_name = f"{study_name}_{id}", 
            load_if_exists = load_if_exists, 
            sampler = TPESampler(), 
            n_jobs = n_jobs
    )
    print(f"Process {id} completed optimization")

    return study

def GAWorker(
        pid: int,
        id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        X_validation: Union[np.ndarray, None] = None, 
        y_validation: Union[np.ndarray, None] = None, 
        best_params: dict = {}, 
        n_trials: int = 100, 
        study_name: str = "GA_Feature_Selection", 
        random_state: int = 42, 
        use_gpu: bool = True, 
        verbose: bool = False, 
        n_jobs: int = 1
    ) -> tuple[optuna.study.Study, dict, float]:
    ''' Feature selection worker function using Genetic Algorithms.

    This function is used to run the optimization of a feature selection model in
    a separate process. It is used to parallelize the optimization process.
    
    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_validation : Union[np.ndarray, None], optional
        Validation data. The default is None.
    y_validation : Union[np.ndarray, None], optional
        Validation labels. The default is None.
    best_params : dict, optional
        Best parameters. The default is {}.
    algorithm : str, optional
        Algorithm. The default is "ga".
    n_trials : int, optional
        Number of trials. The default is 100.
    study_name : str, optional
        Study name. The default is "GA_Feature_Selection".
    random_state : int, optional
        Random state. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.

    Returns
    -------
    study : optuna.study.Study
        Study object.
    best_features : list
        Best features.
    best_score : float
        Best score.
    '''

    # Setup unique to this instance, potentially using instance_id to differentiate setups
    print(f"Running instance {pid}")

    # Sleep pid seconds before starting
    time.sleep(pid)
    
    # Create the EvolutionaryFeatureSelectorCustom object
    evo = GeneticAlgorithmFeatureSelector(X_train, y_train, X_test, y_test, X_validation = X_validation, y_validation = y_validation, xgboost_params = best_params, use_gpu = use_gpu, random_state = random_state, verbose = verbose) # type: ignore
    
    # Run the optimization
    study, best_features, best_score = evo.optimize(study_name = f"{study_name}_{id}", direction = "minimize", n_trials = n_trials, n_jobs = n_jobs)

    return study, best_features, best_score

def NNworker(
        pid: int,
        id: int,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        storage: str,
        encoder_params: Union[dict, None] = None,
        output_size: int = 1,
        random_seed: int = 42,
        use_gpu: bool = True,
        verbose: bool = False,
        direction: str = "minimize",
        n_trials: int = 250,
        load_if_exists: bool = True,
        n_jobs: int = 1,
        study_name: str = "NN_Optimization"
    ) ->  None:
    ''' Neural network optimization worker function.
    
    This function is used to run the optimization of a neural network model in a
    separate process. It is used to parallelize the optimization process.
    
    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    X_val : np.ndarray
        Validation data.
    y_val : np.ndarray
        Validation labels.
    storage : str
        Storage string.
    encoder_params : Union[dict, None], optional
        Encoder parameters. The default is None.
    output_size : int, optional
        Output size. The default is 1.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    '''

    print(f"Process {pid} starting optimization")

    # Sleep pid seconds before starting
    time.sleep(pid)

    # Initialize the trainer
    trainer = NNOptimizer(
        X_train, y_train, 
        X_test, y_test, 
        X_val, y_val, 
        storage,
        encoder_params,
        output_size = output_size, 
        random_seed = random_seed,
        use_gpu = use_gpu, 
        verbose=verbose
    )

    # Run optimization
    trainer.optimize(
        direction = direction, 
        n_trials = n_trials, 
        study_name = f"{study_name}_{id}", 
        load_if_exists = load_if_exists, 
        sampler = TPESampler(), 
        n_jobs = n_jobs
    )

    print(f"Process {id} completed optimization")

    return

def Transworker(
            pid: int,
            id: int,
            X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            storage: str,
            output_size: int = 1,
            random_seed: int = 42,
            use_gpu: bool = True,
            verbose: bool = False,
            direction: str = "minimize",
            n_trials: int = 250,
            load_if_exists: bool = True,
            n_jobs: int = 1,
            study_name: str = "Trans_Optimization"
        ) -> None:
        ''' Transformer optimization worker function.

        This function is used to run the optimization of a transformer model in a
        separate process. It is used to parallelize the optimization process.

        Parameters
        ----------
        pid : int
            Process ID.
        id : int
            Instance ID.
        X_train : np.ndarray
            Training data.
        y_train : np.ndarray
            Training labels.
        X_test : np.ndarray
            Testing data.
        y_test : np.ndarray
            Testing labels.
        X_val : np.ndarray
            Validation data.
        y_val : np.ndarray
            Validation labels.
        storage : str
            Storage string.
        output_size : int, optional
            Output size. The default is 1.
        random_seed : int, optional
            Random seed. The default is 42.
        use_gpu : bool, optional
            Use GPU. The default is True.
        verbose : bool, optional
            Verbose. The default is False.
        direction : str, optional
            Optimization direction. The default is "minimize".
        n_trials : int, optional
            Number of trials. The default is 250.
        load_if_exists : bool, optional
            Load if exists. The default is True.
        n_jobs : int, optional
            Number of jobs. The default is 1.
        study_name : str, optional
            Study name. The default
        None
        '''

        if verbose:
            print(f"Process {pid} starting optimization")

        # Initialize the trainer
        trainer = TransOptimizer(
            X_train, y_train, 
            X_test, y_test, 
            X_val, y_val, 
            storage,
            output_size = output_size, 
            random_seed = random_seed,
            use_gpu = use_gpu, 
            verbose=verbose
        )

        # Run optimization
        trainer.optimize(
            direction = direction, 
            n_trials = n_trials, 
            study_name = f"{study_name}_{id}", 
            load_if_exists = load_if_exists, 
            sampler = TPESampler(), 
            n_jobs = n_jobs
        )

        if verbose:
            print(f"Process {id} completed optimization")

def XGBworker(
        pid: int, id: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_val: np.ndarray,
        storage: str,
        random_seed: int = 42,
        use_gpu: bool = True,
        verbose: bool = False,
        n_trials: int = 250,
        load_if_exists: bool = True,
        n_jobs: int = 10,
        study_name: str = "XGB_Optimization",
        early_stopping_rounds: int = 50,
        params: dict = {}
    ) -> optuna.study.Study:
    ''' XGBoost optimization worker function.

    This function is used to run the optimization of an XGBoost model in a
    separate process. It is used to parallelize the optimization process.

    Parameters
    ----------
    pid : int
        Process ID.
    id : int
        Instance ID.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Testing data.
    X_val : np.ndarray
        Validation data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Testing labels.
    y_val : np.ndarray
        Validation labels.
    storage : str
        Storage string.
    random_seed : int, optional
        Random seed. The default is 42.
    use_gpu : bool, optional
        Use GPU. The default is True.
    verbose : bool, optional
        Verbose. The default is False.
    n_trials : int, optional
        Number of trials. The default is 250.
    load_if_exists : bool, optional
        Load if exists. The default is True.
    n_jobs : int, optional
        Number of jobs. The default is 10.
    study_name : str, optional
        Study name. The default is "XGB_Optimization".
    early_stopping_rounds : int, optional
        Early stopping rounds. The default is 50.
    params : dict, optional
        Parameters. The default is {}.

    Returns
    -------
    study_pre : optuna.study.Study
        Study object.
    '''

    print(f"Process {pid} starting optimization")

    # Set direction based on X_val
    direction = "maximize" if X_val is None else "minimize"

    # Sleep pid seconds before starting
    time.sleep(pid)

    # Create the XGBoostOptimizer object
    xgb = XGBoostOptimizer(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        X_val, 
        y_val, 
        storage = storage,
        params = params, 
        use_gpu = use_gpu, 
        early_stopping_rounds = early_stopping_rounds, 
        random_state = random_seed, 
        verbose = verbose
    )

    # Run the pre-optimization for XGBoost
    study_pre = xgb.optimize(
        direction = direction,
        n_trials = n_trials,
        n_jobs = n_jobs,
        study_name = f"{study_name}_{id}",
        load_if_exists = load_if_exists,
    )

    print(f"Process {pid} completed optimization")

    return study_pre
