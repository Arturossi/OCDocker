#!/usr/bin/env python3

# Description
###############################################################################
''' Module with a helper  to perform the optimization of the Transformer
parameters model using Optuna.

It is imported as:

import OCDocker.OCScore.Optimization.Transformer as octrans
'''

# Imports
###############################################################################

from joblib import Parallel, delayed
from multiprocessing import Pool
from sklearn.decomposition import PCA
from typing import Union



import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.IO as ocscoreio
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


def optimize_Transformer(
        df_path: str,
        storage_id: int,
        base_models_folder: str,
        data: dict = {},
        storage: str = "sqlite:///Transformer_optimization.db",
        use_pdb_train: bool = True,
        no_scores: bool = False,
        only_scores: bool = False,
        use_PCA: bool = False,
        pca_type: int = 95,
        pca_model: Union[str, PCA] = "",
        run_Trans_optimization: bool = True,
        num_processes_Trans: int = 4,
        total_trials_Trans: int = 1000,
        random_seed: int = 42,
        load_if_exists: bool = True,
        use_gpu: bool = True,
        parallel_backend: str = "joblib",
        verbose: bool = False
    ) -> None:
    ''' Function to optimize the Transformer model using Optuna.
    
    Parameters
    ----------
    df_path : str
        The path to the dataset file.
    storage_id : int
        The storage ID of the dataset.
    base_models_folder : str
        The base models folder.
    data : dict, optional
        The data dictionary. Default is {}. If not empty, the data dictionary will be used instead of loading the data. This is useful for multiprocessing to avoid loading the data multiple times.
    storage : str, optional
        The storage string for the database. The default is "sqlite:///Transformer_optimization.db".
    use_pdb_train : bool, optional
        Whether to use the PDB train dataset. The default is True.
    no_scores : bool, optional
        Whether to use the no scores dataset. The default is True.
    only_scores : bool, optional
        Whether to use the only scores dataset. The default is True.
    use_PCA : bool, optional
        Whether to use PCA. The default is True.
    pca_type : int, optional
        The PCA type to use. The default is 95.
    pca_model : Union[str, PCA], optional
        The PCA model to use. Default is "".
    run_Trans_optimization : bool, optional
        Whether to run the Transformer optimization. The default is False.
    num_processes : int, optional
        The number of processes to use. The default is 4.
    total_trials : int, optional
        The total number of trials to run. The default is 2000.
    random_seed : int, optional
        The random seed to use. The default is 42.
    load_if_exists : bool, optional
        Whether to load the study if it already exists. The default is True.
    use_gpu : bool, optional
        Whether to use the GPU. The default is True.
    parallel_backend : str, optional
        The parallel backend to use. The default is "joblib". Options are "joblib" and "multiprocessing". [ATTENTION] multiprocessing has shown to have some nasty bugs while testing this library. It is highly recommended to use joblib.
    verbose : bool, optional
        Whether to print verbose output. The default is False.
    
    Raises
    ------
    ValueError
        If the parallel backend is invalid.
    '''

    # Check if the data dictionary is empty
    if not data:
        # Load the data
        data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = storage_id,
            df_path = df_path,
            optimization_type = "Trans",
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

    if run_Trans_optimization:
        if verbose:
            ocprint.printv("Running Transformer optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_Trans % num_processes_Trans != 0:
            ocprint.print_warning("Warning: total_trials_Trans is not divisible by num_processes_Trans. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_Trans = total_trials_Trans // num_processes_Trans

        # Check the parallel backend
        if parallel_backend == "joblib":
            # Run the optimization using joblib
            Parallel(n_jobs=num_processes_Trans)(delayed(ocscoreworkers.Transworker)(
                    pid,
                    storage_id, 
                    X_train, y_train, 
                    X_test, y_test, 
                    X_val, y_val, 
                    storage,
                    1,              # output_size
                    random_seed,
                    use_gpu,
                    verbose,
                    "minimize",     # direction
                    n_trials_Trans,
                    load_if_exists,
                    1,              # n_jobs
                    study_name
                ) for pid in range(num_processes_Trans)
            )
        elif parallel_backend == "multiprocessing":
            # Run the optimization using multiprocessing
            with Pool(num_processes_Trans) as pool:
                # Each process will execute the 'Transworker' function with the datasets and optimizer parameters
                pool.starmap(ocscoreworkers.Transworker, [(
                    pid,
                    storage_id, 
                    X_train, y_train, 
                    X_test, y_test, 
                    X_val, y_val, 
                    storage,
                    1,              # output_size
                    random_seed,
                    use_gpu,
                    verbose,
                    "minimize",     # direction
                    n_trials_Trans,
                    load_if_exists,
                    1,              # n_jobs
                    study_name
                    ) for pid in range(num_processes_Trans)
                ])
        else:
            # User-facing error: invalid parallel backend
            ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
            raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

# Alias the function
optimize = optimize_Transformer
