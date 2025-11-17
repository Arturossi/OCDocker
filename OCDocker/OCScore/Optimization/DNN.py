#!/usr/bin/env python3

# Description
###############################################################################
''' Module with a helper to perform the optimization of the Neural Network
parameters model using Optuna.

It is imported as:

import OCDocker.OCScore.Optimization.DNN as ocdnn
'''

# Imports
###############################################################################

import itertools
import math
import optuna

import numpy as np
import pandas as pd

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


def perform_seed_ablation_study_NN(
        X_train : np.ndarray, y_train : np.ndarray,
        X_test : np.ndarray, y_test : np.ndarray, 
        X_val : np.ndarray, y_val : np.ndarray,
        id : int,
        num_processes : int,
        encoder_params : dict,
        best_params : dict,
        use_gpu : bool,
        verbose : bool,
        load_if_exists : bool,
        study_name : str,
        storage : str,
        mask : np.ndarray,
        seeds : list = [],
        output_size : int = 1,
        parallel_backend : str = "joblib",
        n_jobs : int = 1
    ) -> None:
    ''' Perform the ablation study for the Neural Network.

    Parameters
    ----------
    X_train : np.ndarray
        The training data.
    y_train : np.ndarray
        The training labels.
    X_test : np.ndarray
        The testing data.
    y_test : np.ndarray
        The testing labels.
    X_val : np.ndarray
        The validation data.
    y_val : np.ndarray
        The validation labels.
    id : int
        The ID of the study.
    num_processes : int
        The number of processes to use.
    encoder_params : dict
        The encoder parameters.
    best_params : dict
        The best parameters.
    use_gpu : bool
        If True, use the GPU.
    verbose : bool
        If True, print the output.
    load_if_exists : bool
        If True, load the model if it exists.
    study_name : str
        The study name.
    storage : str
        The storage to use.
    mask : np.ndarray
        The mask to be applied.
    seeds : list
        List of seeds to be applied. If empty, all seeds for scoring functions will be generated and used. This option is useful for splitting ablation in multiple computers. If empty, all seeds from 0 to 1000 will be used. The default is [].
    output_size : int, optional
        The output size. Default is 1.
    parallel_backend : str, optional
        The parallel backend to use. The default is "joblib". Options are "joblib" and "multiprocessing". [ATTENTION] multiprocessing has shown to have some nasty bugs while testing this library. It is highly recommended to use joblib.
    n_jobs : int, optional
        The number of jobs to use. Default is 1.
    
    Raises
    -------
    ValueError
        If the parallel backend is not "joblib" or "multiprocessing".
    '''

    # Check if seeds is empty
    if not seeds:
        # Create a list of seeds from 0 to 1000
        seeds = list(range(1000))
    
    # Adjust num_processes if the size of the seeds array is smaller
    if len(seeds) < num_processes:
        # If the number of seeds is smaller than the number of processes, set inner_num_processes to the number of seeds
        inner_num_processes = len(seeds)
    else:
        # Otherwise, set inner_num_processes to the number of processes
        inner_num_processes = num_processes

    # Split seeds into roughly equal parts for each process using Round Robin distribution
    split_seeds = [[] for _ in range(inner_num_processes)]

    # Distribute the seeds to the processes
    for i, seed in enumerate(seeds):
        # Append the seed to the corresponding process
        split_seeds[i % inner_num_processes].append(seed)

    # Check the parallel backend
    if parallel_backend == "joblib":
        # Create a pool of worker processes
        Parallel(n_jobs = inner_num_processes)(
            delayed(ocscoreworkers.NNSeedAblationworker)(
                pid,
                id,
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                X_val, 
                y_val,
                mask,
                storage,
                best_params,
                seed, 
                encoder_params,
                output_size,
                use_gpu, 
                verbose,
                load_if_exists,
                1,
                study_name
            ) for pid, seed in enumerate(split_seeds)
        )
    elif parallel_backend == "multiprocessing":
        # Create a pool of worker processes
        with Pool(inner_num_processes) as pool:
            # Each process will execute the 'NNAblationworker' function with the datasets and optimizer parameters
            pool.starmap(ocscoreworkers.NNSeedAblationworker, [(
                pid,
                id,
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                X_val, 
                y_val,
                mask,
                storage,
                best_params,
                seed, 
                encoder_params,
                output_size,
                use_gpu, 
                verbose,
                load_if_exists,
                n_jobs,
                study_name
                ) for pid, seed in enumerate(split_seeds)
            ])
    else:
        # User-facing error: invalid parallel backend
        ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
        raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

    return None


def perform_ablation_study_NN(
        X_train : pd.DataFrame, y_train : pd.DataFrame,
        X_test : pd.DataFrame, y_test : pd.DataFrame, 
        X_val : pd.DataFrame, y_val : pd.DataFrame,
        id : int,
        num_processes : int,
        encoder_params : dict,
        best_params : dict,
        random_seed : int,
        use_gpu : bool,
        verbose : bool,
        load_if_exists : bool,
        study_name : str,
        storage : str,
        masks : list = [],
        output_size : int = 1,
        parallel_backend : str = "joblib",
        n_jobs : int = 1
    ) -> None:
    ''' Perform the ablation study for the Neural Network.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training data.
    y_train : pd.Series
        The training labels.
    X_test : pd.DataFrame
        The testing data.
    y_test : pd.Series
        The testing labels.
    X_val : pd.DataFrame
        The validation data.
    y_val : pd.Series
        The validation labels.
    id : int
        The ID of the study.
    num_processes : int
        The number of processes to use.
    encoder_params : dict
        The encoder parameters.
    best_params : dict
        The best parameters.
    random_seed : int
        The random seed.
    use_gpu : bool
        If True, use the GPU.
    verbose : bool
        If True, print the output.
    load_if_exists : bool
        If True, load the model if it exists.
    study_name : str
        The study name.
    storage : str
        The storage to use.
    masks : list[], optional
        List of masks to be applied. If empty, all masks for scoring functions will be generated and used. This option is useful for splitting ablation in multiple computers. The default is [].
    output_size : int, optional
        The output size. Default is 1.
    parallel_backend : str, optional
        The parallel backend to use. The default is "joblib". Options are "joblib" and "multiprocessing". [ATTENTION] multiprocessing has shown to have some nasty bugs while testing this library. It is highly recommended to use joblib.
    n_jobs : int, optional
        The number of jobs to use. Default is 1.
    
    Raises
    -------
    ValueError
        If the parallel backend is not "joblib" or "multiprocessing".
    '''
    
    # If no masks are provided
    if not masks:
        # Filter the SFs
        sf = X_train.filter(regex = r"(VINA|SMINA|ODDT|PLANTS).*").columns.tolist()

        # Create the mask of zeros and ones for the ablation study (Brute force approach)
        feature_masks = list(itertools.product([0, 1], repeat=len(sf)))

        # Create a mask of ones for the full model
        full_mask = np.ones(X_train.shape[1], dtype=int)

        # Get the indexes for each sf
        sf_indexes = [X_train.columns.get_loc(col) for col in sf]

        # Set the evaluated masks to an empty list
        evaluated_masks = []

        try:
            # Try to load the study to check which masks have already been evaluated
            study = optuna.load_study(study_name = f"{study_name}_{id}", storage = storage)

            # Filter the trials to only include the ones that are complete
            trials = study.trials_dataframe()
            trials = trials[trials['state'] == 'COMPLETE']

            # Get the masks that have already been evaluated
            evaluated_masks = trials['user_attrs_Feature_Mask'].tolist()
        except (AttributeError, KeyError, ImportError):
            # Fallback if optuna study is not available or missing attributes
            evaluated_masks = []
        
        # Apply each feature mask to the full_mask
        masks = []
        for mask in feature_masks:
            # Start with a fresh copy of the full mask template
            modified_mask = full_mask.copy()
            # Set the specific feature indices according to the current mask
            for index, value in zip(sf_indexes, mask):
                modified_mask[index] = value
            if not evaluated_masks or "".join(map(str, modified_mask)) not in evaluated_masks:
                masks.append(modified_mask)

    # Adjust num_processes if the size of the masks array is smaller
    if len(masks) < num_processes:
        inner_num_processes = len(masks)
    else:
        inner_num_processes = num_processes

    # Split masks into roughly equal parts for each process using Round Robin distribution
    split_masks = [[] for _ in range(inner_num_processes)]
    for i, mask in enumerate(masks):
        split_masks[i % inner_num_processes].append(mask)

    # Check the parallel backend
    if parallel_backend == "joblib":
        # Create a pool of worker processes
        Parallel(n_jobs = inner_num_processes)(
            delayed(ocscoreworkers.NNAblationworker)(
                pid,
                id,
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                X_val, 
                y_val,
                mask,
                storage,
                best_params,
                encoder_params,
                output_size,
                random_seed, 
                use_gpu, 
                verbose,
                load_if_exists,
                1,
                study_name
            ) for pid, mask in enumerate(split_masks)
        )
    elif parallel_backend == "multiprocessing":
        # Create a pool of worker processes
        with Pool(inner_num_processes) as pool:
            # Each process will execute the 'NNAblationworker' function with the datasets and optimizer parameters
            pool.starmap(ocscoreworkers.NNAblationworker, [(
                pid,
                id,
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                X_val, 
                y_val,
                mask,
                storage,
                best_params,
                encoder_params,
                output_size,
                random_seed, 
                use_gpu, 
                verbose,
                load_if_exists,
                n_jobs,
                study_name,
                ) for pid, mask in enumerate(split_masks)
            ])
    else:
        # User-facing error: invalid parallel backend
        ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
        raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

    return None


def optimize_NN(
        df_path: str,
        storage_id: int,
        base_models_folder: str,
        data: dict = {},
        storage: str = "sqlite:///NN_optimization.db",
        use_pdb_train: bool = True,
        no_scores: bool = False,
        only_scores: bool = False,
        use_PCA: bool = False,
        best_ao_params: Union[dict, None] = None,
        pca_type: int = 80,
        pca_model: Union[str, PCA] = "",
        encoder_dims: tuple[int, int] = (16, 256),
        autoencoder: bool = True,
        multiencoder: bool = False,
        run_autoencoder_optimization: bool = True,
        num_processes_autoencoder: int = 8,
        total_trials_autoencoder: int = 2000,
        run_NN_optimization: bool = True,
        num_processes_NN: int = 8,
        total_trials_NN: int = 125,
        explained_variance: float = 0.95,
        random_seed: int = 42,
        load_if_exists: bool = True,
        use_gpu: bool = True,
        parallel_backend: str = "joblib",
        verbose: bool = False
    ) -> None:
    ''' Optimize the Neural Network using the given parameters.

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
        The storage to use. Default is "sqlite:///NN_optimization.db".
    use_pdb_train : bool, optional
        If True, use the PDBbind data for training. If False, use the DUDEz data for training. Default is True.
    no_scores : bool, optional
        If True, don't use the scoring functions for training. If False, use the scoring functions. Default is False. (Will override only_scores)
    only_scores : bool, optional
        If True, only use the scoring functions for training. If False, use all the features. Default is False.
    use_PCA : bool, optional
        If True, use PCA to reduce the number of features. If False, use all the features. Default is True.
    best_ao_params : dict, optional
        The best autoencoder parameters. Default is None.
    pca_type : int, optional
        The PCA type to use. Default is 80.
    pca_model : Union[str, PCA], optional
        The PCA model to use. Default is "".
    autoencoder : bool, optional
        If True, use the autoencoder. If False, don't use the autoencoder. Default is False.
    multiencoder : bool, optional
        If True, use the multiencoder. If False, don't use the multiencoder. Default is False.
    run_autoencoder_optimization : bool, optional
        If True, run the autoencoder optimization. If False, don't run the autoencoder optimization. Default is False.
    num_processes_autoencoder : int, optional
        The number of processes to use for the autoencoder. Default is 8.
    total_trials_autoencoder : int, optional
        The number of total trials to use for the autoencoder. Default is 2000.
    run_NN_optimization : bool, optional
        If True, run the Neural Network optimization. If False, don't run the Neural Network optimization. Default is True.
    num_processes_NN : int, optional
        The number of processes to use for the Neural Network. Default is 8.
    total_trials_NN : int, optional
        The number of trials to use for the Neural Network. Default is 1000.
    explained_variance : float, optional
        The explained variance to use. Default is 0.95.
    random_seed : int, optional
        The random seed to use. Default is 42.
    load_if_exists : bool, optional
        If True, load the model if it exists. If False, don't load the model if it exists. Default is True.
    use_gpu : bool, optional
        If True, use the GPU. If False, don't use the GPU. Default is True.
    parallel_backend : str, optional
        The parallel backend to use. The default is "joblib". Options are "joblib" and "multiprocessing". [ATTENTION] multiprocessing has shown to have some nasty bugs while testing this library. It is highly recommended to use joblib.
    verbose : bool, optional
        If True, print the output. If False, don't print the output. Default is False.

    Raises
    -------
    ValueError
        If the parallel backend is not "joblib" or "multiprocessing".
    '''

    # Check if the data dictionary is empty
    if not data:
        # Load the data
        data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = storage_id,
            df_path = df_path,
            optimization_type = "NN",
            pca_model = pca_model,
            no_scores = no_scores,
            only_scores = only_scores,
            use_PCA = use_PCA,
            pca_type = pca_type,
            use_pdb_train = use_pdb_train,
            random_seed = random_seed
        )

    # Extract the data from the data dictionary object to the corresponding variables
    models_folder = data["models_folder"]
    study_name = data["study_name"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    if autoencoder:
        if verbose:
            ocprint.printv("Running Auto Encoder optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_autoencoder % num_processes_autoencoder != 0:
            ocprint.print_warning("Warning: total_trials_autoencoder is not divisible by num_processes_autoencoder. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_autoencoder = total_trials_autoencoder // num_processes_autoencoder
        
        if multiencoder:
            # Set the classification for e
            sf = X_train.filter(regex = r"(VINA|SMINA|ODDT|PLANTS).*").columns.tolist()
            ligand = [f"AUTOCORR2D_{i}" for i in range(1, 193)] + \
                [f"BCUT2D_{attr}" for attr in ["CHGHI", "CHGLO", "LOGPHI", "LOGPLOW", "MRHI", "MRLOW", "MWHI", "MWLOW"]] + \
                [f"fr_{attr}" for attr in ["Al_COO", "Al_OH", "Al_OH_noTert", "ArN", "Ar_COO", "Ar_N", "Ar_NH", "Ar_OH", "COO", "COO2", "C_O", "C_O_noCOO", "C_S", "HOCCN", "Imine", "NH0", "NH1", "NH2", "N_O", "Ndealkylation1", "Ndealkylation2", "Nhpyrrole", "SH", "aldehyde", "alkyl_carbamate", "alkyl_halide", "allylic_oxid", "amide", "amidine", "aniline", "aryl_methyl", "azide", "azo", "barbitur", "benzene", "benzodiazepine", "bicyclic", "diazo", "dihydropyridine", "epoxide", "ester", "ether", "furan", "guanido", "halogen", "hdrzine", "hdrzone", "imidazole", "imide", "isocyan", "isothiocyan", "ketone", "ketone_Topliss", "lactam", "lactone", "methoxy", "morpholine", "nitrile", "nitro", "nitro_arom", "nitro_arom_nonortho", "nitroso", "oxazole", "oxime", "para_hydroxylation", "phenol", "phenol_noOrthoHbond", "phos_acid", "phos_ester", "piperdine", "piperzine", "priamide", "prisulfonamd", "pyridine", "quatN", "sulfide", "sulfonamd", "sulfone", "term_acetylene", "tetrazole", "thiazole", "thiocyan", "thiophene", "unbrch_alkane", "urea"]] + \
                [f"Chi{attr}" for attr in ["0", "0v", "0n", "1", "1v", "1n", "2v", "2n", "3v", "3n", "4v", "4n"]] + \
                [f"EState_VSA{i}" for i in range(1, 12)] + \
                [f"FpDensityMorgan{i}" for i in range(1, 4)] + \
                [f"Kappa{i}" for i in range(1, 4)] + \
                ["MolLogP", "MolMR", "MolWt", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumValenceElectrons", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3"] + \
                [f"PEOE_VSA{i}" for i in range(1, 15)] + \
                [f"SMR_VSA{i}" for i in range(1, 11)] + \
                [f"SlogP_VSA{i}" for i in range(1, 13)] + \
                [f"VSA_EState{i}" for i in range(1, 11)] + \
                ["BalabanJ", "BertzCT", "ExactMolWt", "FractionCSP3", "HallKierAlpha", "HeavyAtomMolWt", "HeavyAtomCount", "LabuteASA", "TPSA", "MaxAbsEStateIndex", "MaxEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "qed", "RingCount", "Asphericity", "Eccentricity", "InertialShapeFactor", "RadiusOfGyration", "SpherocityIndex", "NHOHCount", "NOCount"]
            receptor = [f"count{attr}" for attr in ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]] + \
                ["TotalAALength", "AvgAALength", "countChain", "SASA", "DipoleMoment", "IsoelectricPoint", "GRAVY", "Aromaticity", "InstabilityIndex"]

            # Extract the data
            sf_train_data = X_train[sf]
            ligand_train_data = X_train[ligand]
            receptor_train_data = X_train[receptor]

            sf_test_data = X_test[sf]
            ligand_test_data = X_test[ligand]
            receptor_test_data = X_test[receptor]

            if X_val is not None:
                sf_val_data = X_val[sf]
                ligand_val_data = X_val[ligand]
                receptor_val_data = X_val[receptor]
            else:
                ligand_val_data = None
                receptor_val_data = None
            
            new_X_train = [sf_train_data, ligand_train_data, receptor_train_data]
            new_X_test = [sf_test_data, ligand_test_data, receptor_test_data]
            new_X_val = [sf_val_data, ligand_val_data, receptor_val_data]

            # List to store the best topology for each set
            best_ao_params = [] # type: ignore
            
            if run_autoencoder_optimization:
                
                for name, AO_X_train, AO_X_test, AO_X_val in [
                    ("SF", sf_train_data, sf_test_data, sf_val_data), 
                    ("LIG", ligand_train_data, ligand_test_data, ligand_val_data), 
                    ("REC", receptor_train_data, receptor_test_data, receptor_val_data)
                ]:
                    # Compute the singular values for AO_X_train
                    singular_values = np.linalg.svd(AO_X_train, compute_uv = False)

                    # Compute the explained variance ratio
                    explained_variance_ratio = singular_values**2 / np.sum(singular_values**2)

                    # Compute the cumulative explained variance ratio
                    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

                    # Compute the number of components that explain 95% of the variance
                    n_components = np.argmax(cumulative_explained_variance_ratio >= explained_variance) + 1

                    # Get the number of dimensions for the encoding layer and round up to the nearest power of 2 + 1
                    encoding_dims = ( # Size should be the same size or smaller than the number of features to explain the desired variance
                        max(2 ** math.ceil(math.log2(n_components / 2) - 1), 4), # Minimum value
                        n_components
                    )

                    # Skip SF (for now) TODO: Check if this is necessary
                    if name == "SF":
                        continue

                    # Check the parallel backend
                    if parallel_backend == "joblib":
                        # Create a pool of worker processes
                        Parallel(n_jobs = num_processes_autoencoder)(
                            delayed(ocscoreworkers.AEworker)(
                                pid,
                                storage_id, 
                                AO_X_train,
                                AO_X_test,
                                AO_X_val,
                                encoding_dims,
                                storage,
                                models_folder,
                                random_seed,
                                use_gpu,
                                verbose,
                                "minimize",                     # direction
                                n_trials_autoencoder,
                                load_if_exists,
                                1,                              # n_jobs
                                f"Multi_AE_Optimization_{name}" # study_name
                            ) for pid in range(num_processes_autoencoder)
                        )
                    elif parallel_backend == "multiprocessing":
                        # Create a pool of worker processes
                        with Pool(num_processes_autoencoder) as pool:
                            # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
                            pool.starmap(ocscoreworkers.AEworker, [(
                                pid,
                                storage_id, 
                                AO_X_train,
                                AO_X_test,
                                AO_X_val,
                                encoding_dims,
                                storage,
                                models_folder,
                                random_seed,              # random_seed
                                use_gpu,                  # use_gpu
                                verbose,                  # verbose
                                "minimize",               # direction
                                n_trials_autoencoder,     # n_trials 
                                load_if_exists,           # load_if_exists
                                1,                        # n_jobs
                                f"Multi_AE_Optimization_{name}" # study_name
                                ) for pid in range(num_processes_autoencoder)
                            ])
                    else:
                        # User-facing error: invalid parallel backend
                        ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
                        raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

            for name in ["SF", "LIG", "REC"]:
                if name == "SF":
                    best_ao_params.append( # type: ignore
                        {
                            "n_layers_encoder": 1,
                            "activation_function_0_encoder": "Identity",
                            "n_units_layer_0_encoder": sf_train_data.shape[1]
                        })
                    continue

                # Load the study
                ao_multi_study = optuna.load_study(study_name = f"AO_Optimization_{name}_{storage_id}_TPE", storage = storage)
                ao_multi_df = ao_multi_study.trials_dataframe()
                ao_multi_df["combined_metric"] = abs(ao_multi_df["value"] - ao_multi_df["user_attrs_val_rmse"])

                best_ao_multi_df = ao_multi_df.sort_values(
                    by = ["combined_metric", "value", "user_attrs_val_rmse"],
                    ascending = [True, True, True]
                )

                # Recreate the autoencoder object for the best trial based on the best_ao_multi_df
                best_ao_multi_trial = best_ao_multi_df.iloc[0]

                # Select the trial by the best_ao_multi_trial number
                best_ao_multi_trial = ao_multi_study.trials[best_ao_multi_trial.number]

                # Pick the params from the best_ao_multi_trial
                best_ao_params.append(best_ao_multi_trial.params) # type: ignore

        else:
            if run_autoencoder_optimization:
                if parallel_backend == "joblib":
                    # Create a pool of worker processes
                    Parallel(n_jobs = num_processes_autoencoder)(
                        delayed(ocscoreworkers.AEworker)(
                            pid,
                            storage_id, 
                            X_train, 
                            X_test, 
                            X_val, 
                            encoder_dims,
                            storage,
                            models_folder,
                            random_seed,
                            use_gpu,
                            verbose,
                            "minimize",           # direction
                            n_trials_autoencoder,
                            load_if_exists,
                            1,                    # n_jobs
                            f"AO_Optimization"    # study_name
                        ) for pid in range(num_processes_autoencoder)
                    )
                elif parallel_backend == "multiprocessing":
                    # Create a pool of worker processes
                    with Pool(num_processes_autoencoder) as pool:
                        # Each process will execute the 'NNworker' function with the datasets and optimizer parameters
                        pool.starmap(ocscoreworkers.AEworker, [(
                            pid,
                            storage_id, 
                            X_train, 
                            X_test, 
                            X_val, 
                            encoder_dims,
                            storage,
                            models_folder,
                            random_seed,
                            use_gpu,
                            verbose,
                            "minimize",           # direction
                            n_trials_autoencoder,
                            load_if_exists,
                            1,                    # n_jobs
                            f"AO_Optimization"    # study_name
                            ) for pid in range(num_processes_autoencoder)
                        ])
                else:
                    # User-facing error: invalid parallel backend
                    ocerror.Error.value_error(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.") # type: ignore
                    raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

            # Load the study
            ao_study = optuna.load_study(
                study_name = f"AO_Optimization_{storage_id}",
                storage = storage
            )
            ao_df = ao_study.trials_dataframe()

            # Filter the trials to only include the ones that are complete
            ao_df = ao_df[ao_df["state"] == "COMPLETE"]
            
            best_ao_df = ao_df.sort_values(
                by = ["value", "user_attrs_val_rmse"],
                ascending = [True, True]
            )

            # Recreate the autoencoder object for the best trial based on the best_ao_df
            best_ao_trial = best_ao_df.iloc[0]

            # Select the trial by the best_ao_trial number
            best_ao_trial = ao_study.trials[best_ao_trial.number]

            # Pick the params from the best_ao_trial
            best_ao_params = best_ao_trial.params
            
            new_X_train = X_train
            new_X_test = X_test
            new_X_val = X_val
    else:
        new_X_train = X_train
        new_X_test = X_test
        new_X_val = X_val
        best_ao_params = None

    if run_NN_optimization:

        if verbose:
            ocprint.printv("Running Neural Network optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_NN % num_processes_NN != 0:
            ocprint.print_warning("Warning: total_trials_NN is not divisible by num_processes_NN. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_NN = total_trials_NN // num_processes_NN

        if parallel_backend == "joblib":
            # Create a pool of worker processes
            Parallel(n_jobs = num_processes_NN)(
                delayed(ocscoreworkers.NNworker)(
                    pid,
                    storage_id, 
                    new_X_train, y_train, 
                    new_X_test, y_test, 
                    new_X_val, y_val, 
                    storage,
                    best_ao_params,   # encoder
                    1,                # output_size
                    random_seed,
                    use_gpu,
                    verbose,
                    "minimize",       # direction
                    n_trials_NN,
                    load_if_exists,
                    1,                # n_jobs
                    study_name
                ) for pid in range(num_processes_NN)
            )
        elif parallel_backend == "multiprocessing":
            with Pool(num_processes_NN) as pool:
                # Each process will execute the "NNworker" function with the datasets and optimizer parameters
                pool.starmap(ocscoreworkers.NNworker, [(
                    pid,
                    storage_id, 
                    new_X_train, y_train, 
                    new_X_test, y_test, 
                    new_X_val, y_val, 
                    storage,
                    best_ao_params,   # encoder
                    1,                # output_size
                    random_seed,
                    use_gpu,
                    verbose,
                    "minimize",       # direction
                    n_trials_NN,
                    load_if_exists,
                    1,                # n_jobs
                    study_name
                    ) for pid in range(num_processes_NN)
                ])
        else:
            raise ValueError(f"Invalid parallel backend: '{parallel_backend}'. Please use 'joblib' or 'multiprocessing'.")

    return None

# Alias the function
optimize = optimize_NN
