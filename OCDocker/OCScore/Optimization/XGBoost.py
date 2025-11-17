#!/usr/bin/env python3

# Description
###############################################################################
''' Module with a helper to perform the optimization of the Extreme Gradient
Boost (XGBoost) parameters model using Optuna.

It is imported as:

import OCDocker.OCScore.Optimization.XGBoost as ocxgb
'''

# Imports
###############################################################################

import optuna

from joblib import Parallel, delayed
from multiprocessing import Pool
from sklearn.decomposition import PCA
from typing import Union



import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.Workers as ocscoreworkers
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

# Methods
###############################################################################


def optimize_XGB(
        df_path: str,
        storage_id: int,
        base_models_folder: str,
        data: dict = {},
        storage: str = "sqlite:///XGB_optimization.db",
        use_pdb_train: bool = True,
        no_scores: bool = False,
        only_scores: bool = False,
        use_PCA: bool = False,
        pca_type: int = 95,
        pca_model: Union[str, PCA] = "",
        run_pre_XGB_optimization: bool = True,
        num_processes_pre_XGB: int = 8,
        total_trials_pre_XGB : int = 250,
        run_GA_optimization: bool = False,
        num_processes_GA: int = 8,
        total_trials_GA : int = 10,
        run_XGB_optimization: bool = True,
        num_processes_XGB: int = 8,
        total_trials_XGB : int = 10,
        early_stopping_rounds: int = 20, 
        random_seed: int = 42,
        load_if_exists: bool = True,
        use_gpu: bool = True,
        parallel_backend: str = "joblib",
        verbose: bool = False
    ) -> None:
    ''' Optimize the Extreme Gradient Boost using the given parameters.

    Parameters
    ----------
    df_path : str
        The path to the DataFrame.
    storage_id : int
        The storage ID to use.
    base_models_folder : str
        The base models folder to use.
    data : dict, optional
        The data dictionary. Default is {}. If not empty, the data dictionary will be used instead of loading the data. This is useful for multiprocessing to avoid loading the data multiple times.
    storage : str, optional
        The storage to use. Default is "sqlite:///XGB_optimization.db".
    use_pdb_train : bool, optional
        If True, use the PDBbind data for training. If False, use the DUDEz data for training. Default is True.
    no_scores : bool, optional
        If True, don't use the scoring functions for training. If False, use the scoring functions. Default is False. (Will override only_scores if True)
    only_scores : bool, optional
        If True, only use the scoring functions for training. If False, use all the features. Default is True.
    use_PCA : bool, optional
        If True, use PCA to reduce the number of features. If False, use all the features. Default is True.
    pca_type : int, optional
        The PCA type to use. Default is 80.
    pca_model : Union[str, PCA], optional
        The PCA model to use. Default is "".
    num_processes : int, optional
        The number of processes to use. Default is 8.
    run_pre_XGB_optimization : bool, optional
        If True, run the pre-XGBoost optimization. If False, don't run the pre-XGBoost optimization. Default is False.
    num_processes_pre_XGB : int, optional
        The number of processes to use for the pre-XGBoost optimization. Default is 8.
    n_trials_pre_XGB : int, optional
        The number of trials to use for the pre-XGBoost optimization. Default is 250.
    run_GA_optimization : bool, optional
        If True, run the Genetic Algorithm optimization. If False, don't run the Genetic Algorithm optimization. Default is False.
    num_processes_GA : int, optional
        The number of processes to use for the Genetic Algorithm optimization. Default is 8.
    run_XGB_optimization : bool, optional
        If True, run the Neural Network optimization. If False, don't run the Neural Network optimization. Default is True.
    random_seed : int, optional
        The random seed to use. Default is 42.
    load_if_exists : bool, optional
        If True, load the model if it exists. If False, don't load the model if it exists. Default is True.
    use_gpu : bool, optional
        If True, use the GPU. If False, don't use the GPU. Default is True.
    parallel_backend : str, optional
        The parallel backend to use. The default is "joblib". Options are "joblib" and "multiprocessing".
    verbose : bool, optional
        If True, print out more information. If False, print out less information. Default is False.
    '''

    # Check if the data dictionary is empty
    if not data:
        # Load the data
        data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = storage_id,
            df_path = df_path,
            optimization_type = "XGB",
            pca_model = pca_model,
            no_scores = no_scores,
            only_scores = only_scores,
            use_PCA = use_PCA,
            pca_type = pca_type,
            use_pdb_train = use_pdb_train,
            random_seed = random_seed
        )

    # Extract the data from the data dictionary object to the corresponding variables
    #models_folder = data["models_folder"]
    study_name = data["study_name"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    
    if run_pre_XGB_optimization:
        if verbose:
            ocprint.printv("Running XGBoost pre-optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_pre_XGB % num_processes_pre_XGB != 0:
            ocprint.print_warning("Warning: total_trials_pre_XGB is not divisible by num_processes_pre_XGB. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_pre_XGB = total_trials_pre_XGB // num_processes_pre_XGB

        # Check the parallel backend
        if parallel_backend == "joblib":
            # Run the optimization using joblib
            _ = Parallel(n_jobs = num_processes_pre_XGB)(
                delayed(ocscoreworkers.XGBworker)(
                    i, 
                    storage_id, 
                    X_train, 
                    X_test, 
                    X_val, 
                    y_train, 
                    y_test, 
                    y_val, 
                    storage, 
                    random_seed, 
                    use_gpu, 
                    verbose, 
                    n_trials_pre_XGB, 
                    load_if_exists, 
                    1, 
                    "Pre_XGB_Optimization", 
                    early_stopping_rounds, 
                    {}
                ) for i in range(num_processes_pre_XGB)
            )
        elif parallel_backend == "multiprocessing":
            # Run the optimization using multiprocessing
            with Pool(num_processes_pre_XGB) as p:
                # Run the optimization
                _ = p.starmap(ocscoreworkers.XGBworker, [(
                        i, 
                        storage_id, 
                        X_train, 
                        X_test, 
                        X_val, 
                        y_train, 
                        y_test, 
                        y_val, 
                        storage, 
                        random_seed, 
                        use_gpu, 
                        verbose, 
                        n_trials_pre_XGB, 
                        load_if_exists, 
                        1, 
                        "Pre_XGB_Optimization", 
                        early_stopping_rounds, 
                        {}
                    ) for i in range(num_processes_pre_XGB)
                ])
        else:
            # User-facing error: invalid parallel backend
            ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
            raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

        # Load the study
        pre_xgb_study = optuna.load_study(
            study_name = f"Pre_XGB_Optimization_{storage_id}",
            storage = storage
        )
        pre_xgb_df = pre_xgb_study.trials_dataframe()
        pre_xgb_df["combined_metric"] = pre_xgb_df["value"] - pre_xgb_df["user_attrs_AUC"]
        best_pre_xgb_df = pre_xgb_df.sort_values(
            by = ["combined_metric", "value", "user_attrs_AUC"],
            ascending = [True, True, False]
        )
        best_pre_xgb_trial = best_pre_xgb_df.iloc[0]
        best_xgb_trial = pre_xgb_study.trials[best_pre_xgb_trial.number]
        best_pre_xgb_params = best_xgb_trial.params
    else:
        best_pre_xgb_params = {}

    if run_GA_optimization:
        if verbose:
            ocprint.printv("Running feature selection...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_GA % num_processes_GA != 0:
            ocprint.print_warning("Warning: total_trials_GA is not divisible by num_processes_GA. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_GA = total_trials_GA // num_processes_GA

        # Check the parallel backend
        if parallel_backend == "joblib":
            # Run the optimization using joblib
            _ = Parallel(n_jobs = num_processes_GA)(
                delayed(ocscoreworkers.GAWorker)(
                    i, 
                    storage_id,
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    X_val, 
                    y_val, 
                    storage,
                    best_pre_xgb_params, 
                    n_trials_GA, 
                    "feature_selection", 
                    random_seed, 
                    use_gpu, 
                    verbose, 
                    1
                ) for i in range(num_processes_GA)
            )
        elif parallel_backend == "multiprocessing":
            # Run the optimization using multiprocessing
            with Pool(num_processes_GA) as p:
                # Run the optimization
                _ = p.starmap(ocscoreworkers.GAWorker, [(
                    i, 
                    storage_id,
                    X_train, 
                    y_train, 
                    X_test, 
                    y_test, 
                    X_val, 
                    y_val, 
                    storage,
                    best_pre_xgb_params, 
                    n_trials_GA, 
                    "feature_selection", 
                    random_seed, 
                    use_gpu, 
                    verbose, 
                    1
                ) for i in range(num_processes_GA)
                ])
        else:
            # User-facing error: invalid parallel backend
            ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
            raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

        # Load the study
        feature_selection_study = optuna.load_study(
            study_name = f"feature_selection_{storage_id}",
            storage = storage
        )
        feature_selection_df = feature_selection_study.trials_dataframe()
        feature_selection_df["combined_metric"] = feature_selection_df["value"] - feature_selection_df["user_attrs_best_AUC"]
        best_feature_selection_df = feature_selection_df.sort_values(
            by = ["combined_metric", "value", "user_attrs_best_AUC"],
            ascending = [True, True, False]
        )
        best_feature_selection_feature_mask = [bool(int(i)) for i in best_feature_selection_df.iloc[0]["user_attrs_best_individual"]]

        # Apply the best feature mask to the training and testing sets
        X_train_filtered = X_train.iloc[:, best_feature_selection_feature_mask]
        X_test_filtered = X_test.iloc[:, best_feature_selection_feature_mask]

        # If the validation set is not None, create a filtered validation set
        if X_val is not None:
            X_val_filtered = X_val.iloc[:, best_feature_selection_feature_mask]
        else:
            X_val_filtered = None
    else:
        X_train_filtered = X_train
        X_test_filtered = X_test
        X_val_filtered = X_val

    if run_XGB_optimization:
        if verbose:
            ocprint.printv("Running XGBoost final optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_XGB % num_processes_XGB != 0:
            ocprint.print_warning("Warning: total_trials_XGB is not divisible by num_processes_XGB. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_XGB = total_trials_XGB // num_processes_XGB

        # Check the parallel backend
        if parallel_backend == "joblib":
            # Run the optimization using joblib
            _ = Parallel(n_jobs = num_processes_XGB)(
                delayed(ocscoreworkers.XGBworker)(
                    i, 
                    storage_id, 
                    X_train_filtered, 
                    X_test_filtered, 
                    X_val_filtered, 
                    y_train, 
                    y_test, 
                    y_val, 
                    storage, 
                    random_seed, 
                    use_gpu, 
                    verbose, 
                    n_trials_XGB, 
                    load_if_exists, 
                    1, 
                    study_name, 
                    early_stopping_rounds, 
                    {}
                ) for i in range(num_processes_XGB)
            )
        elif parallel_backend == "multiprocessing":
            # Run the optimization using multiprocessing
            with Pool(num_processes_XGB) as p:
                # Run the optimization
                _ = p.starmap(ocscoreworkers.XGBworker, [(
                        i, 
                        storage_id, 
                        X_train_filtered, 
                        X_test_filtered, 
                        X_val_filtered, 
                        y_train, 
                        y_test, 
                        y_val, 
                        storage, 
                        random_seed, 
                        use_gpu, 
                        verbose, 
                        n_trials_XGB, 
                        load_if_exists, 
                        1, 
                        study_name, 
                        early_stopping_rounds, 
                        {}
                    ) for i in range(num_processes_XGB)
                ])
        else:
            # User-facing error: invalid parallel backend
            ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
            raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

    return None

# Alias the function
optimize = optimize_XGB
