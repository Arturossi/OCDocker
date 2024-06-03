#!/usr/bin/env python3

# Description
###############################################################################
""" Module to perform the optimization of the Neural Network parameters model
using Optuna."""

# Imports
###############################################################################

import math
import numpy as np

from typing import Union

import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.Workers as ocscoreworkers

import optuna

from multiprocessing import Pool
from urllib.parse import quote_plus

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

def optimize_NN(
        df_path: str,
        storage_id: int,
        use_pdb_train: bool = True,
        no_scores: bool = True,
        only_scores: bool = True,
        use_PCA: bool = True,
        best_ao_params: Union[dict, None] = None,
        pca_type: int = 80,
        storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}",
        encoder_dims: tuple[int, int] = (16, 256),
        base_models_folder: str = f"/data/hd4tb/OCDocker/data/ocdb/models",
        autoencoder: bool = False,
        multiencoder: bool = False,
        run_autoencoder_optimization: bool = False,
        num_processes_autoencoder: int = 8,
        total_trials_autoencoder: int = 2000,
        run_NN_optimization: bool = True,
        num_processes_NN: int = 8,
        total_trials_NN: int = 125,
        explained_variance: float = 0.95,
        random_seed: int = 42,
        load_if_exists: bool = True,
        use_gpu: bool = True,
        verbose: bool = False
    ) -> None:
    ''' Optimize the Neural Network using the given parameters.

    Parameters
    ----------
    df_path : str
        The path to the DataFrame.
    storage_id : int
        The storage ID to use.
    use_pdb_train : bool, optional
        If True, use the PDBbind data for training. If False, use the DUDEz data for training. Default is True.
    no_scores : bool, optional
        If True, don't use the scoring functions for training. If False, use the scoring functions. Default is False. (Will override only_scores)
    only_scores : bool, optional
        If True, only use the scoring functions for training. If False, use all the features. Default is True.
    use_PCA : bool, optional
        If True, use PCA to reduce the number of features. If False, use all the features. Default is True.
    best_ao_params : dict, optional
        The best autoencoder parameters. Default is None.
    pca_type : int, optional
        The PCA type to use. Default is 80.
    storage : str, optional
        The storage to use. Default is "mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}".
    base_models_folder : str, optional
        The base models folder to use. Default is "/data/hd4tb/OCDocker/data/ocdb/models".
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
    verbose : bool, optional
        If True, print the output. If False, don't print the output. Default is False.
    '''

    # Load the data
    data = ocscoredata.load_data(base_models_folder, storage_id, df_path, "NN", no_scores, only_scores, use_PCA, pca_type, use_pdb_train, random_seed)

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
            print("Running Auto Encoder optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_autoencoder % num_processes_autoencoder != 0:
            print("Warning: total_trials_autoencoder is not divisible by num_processes_autoencoder. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_autoencoder = total_trials_autoencoder // num_processes_autoencoder
        
        if multiencoder:
            # Set the classification for e
            sf = X_train.filter(regex = r"(VINA|SMINA|ODDT|PLANTS).*").columns.tolist()
            ligand = [f"AUTOCORR2D_{i}" for i in range(1, 193)] + \
                [f"BCUT2D_{attr}" for attr in ["CHGHI", "CHGLO", "LOGPHI", "LOGPLOW", "MRHI", "MRLOW", "MWHI", "MWLOW"]] + \
                [f"fr_{attr}" for attr in ["Al_COO", "Al_OH", "Al_OH_noTert", "ArN", "Ar_COO", "Ar_N", "Ar_NH", "Ar_OH", "COO", "COO2", "C_O", "C_O_noCOO", "C_S", "HOCCN", "Imine", "NH0", "NH1", "NH2", "N_O", "Ndealkylation1", "Ndealkylation2", "Nhpyrrole", "SH", "aldehyde", "alkyl_carbamate", "alkyl_halide", "allylic_oxid", "amide", "amidine", "aniline", "aryl_methyl", "azide", "azo", "barbitur", "benzene", "benzodiazepine", "bicyclic", "diazo", "dihydropyridine", "epoxide", "ester", "ether", "furan", "guanido", "halogen", "hdrzine", "hdrzone", "imidazole", "imide", "isocyan", "isothiocyan", "ketone", "ketone_Topliss", "lactam", "lactone", "methoxy", "morpholine", "nitrile", "nitro", "nitro_arom", "nitro_arom_nonortho", "nitroso", "oxazole", "oxime", "para_hydroxylation", "phenol", "phenol_noOrthoHbond", "phos_acid", "phos_ester", "piperdine", "piperzine", "priamide", "prisulfonamd", "pyridine", "quatN", "sulfide", "sulfonamd", "sulfone", "term_acetylene", "tetrazole", "thiazole", "thiocyan", "thiophene", "unbrch_alkane", "urea"]] + \
                [f"Chi{attr}" for attr in ["0", "0v", "0n", "1", "1v", "1n", "2v", "2n", "3v", "3n", "4v", "4n"]] + [f"EState_VSA{i}" for i in range(1, 12)] + [f"FpDensityMorgan{i}" for i in range(1, 4)] + \
                [f"Kappa{i}" for i in range(1, 4)] + \
                ["MolLogP", "MolMR", "MolWt", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRadicalElectrons", "NumRotatableBonds", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "NumValenceElectrons", "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "PEOE_VSA{i}" for i in range(1, 15)] + \
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
                        "AO_Optimization"     # study_name
                        ) for pid in range(num_processes_autoencoder)
                    ])

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
            print("Running Neural Network optimization...")

        # If total_trials is not divisible by num_processes, warn the user
        if total_trials_NN % num_processes_NN != 0:
            print("Warning: total_trials_NN is not divisible by num_processes_NN. The number of trials per process will be rounded down to the nearest perfect divisor integer.")

        n_trials_NN = total_trials_NN // num_processes_NN

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

    return None

"""
# Load the study
nn_study = optuna.load_study(study_name = f"{study_name}_{storage_id}_TPE", storage = storage)
nn_df = nn_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
nn_df = nn_df[nn_df['state'] == 'COMPLETE']

nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']

best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])

# Define the number of models to select
n_models = 5

# Get the best n models in the best_nn_df
best_nn_df.head(n_models)

# Build the models
models = []
predictions = []

for i in range(n_models):
    # Get the best trial
    best_trial = nn_study.trials[best_nn_df.iloc[i].number]

    # Pick the params from the best_trial
    best_params = best_trial.params

    # If the new_X_val is a list
    if isinstance(new_X_val, list):
        # Initialize the trainer
        NN_model = NeuralNet(
            [new_X_train[j].shape[0] for j in range(len(new_X_train))],
            1,          
            best_ao_params, 
            best_params,
            random_seed = random_seed,
            use_gpu = True, 
            verbose = False
        )
    else:
        # Initialize the trainer
        NN_model = NeuralNet(
            new_X_train.shape[1],
            1,          
            best_ao_params, 
            best_params,
            random_seed = random_seed,
            use_gpu = True, 
            verbose = False
        )

    # Reset the random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)

    # Train the model
    model = NN_model.train_model(
        new_X_train, 
        y_train, 
        new_X_test, 
        y_test, 
        new_X_val, 
        y_val
    )

    # Reset the random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    
    # Append the model to the list
    models.append(NN_model)

    # Make predictions
    if isinstance(new_X_val, list):
        predictions.append(
            NN_model.NN(
                [torch.tensor(
                    np.asarray(new_X_val[j]), 
                    dtype=torch.float32
                ).to(torch.device('cuda')) for j in range(len(new_X_val))]
            ).cpu().detach().numpy()
        )
    else:
        predictions.append(
            NN_model.NN(
                torch.tensor(
                    np.asarray(new_X_val), 
                    dtype=torch.float32
                ).to(torch.device('cuda'))
            ).cpu().detach().numpy()
        )

    # Save the model
    torch.save(model, f'/data/hd4tb/OCDocker/data/ocdb/models/NN_model_{i}.pt')

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(np.asarray(predictions).reshape(len(predictions), predictions[0].shape[0]).T)

def select_values(row):
    sorted_row = sorted(row)
    return pd.Series({
        'max': max(row),
        '2nd_highest': sorted_row[-2],
        'median': sorted_row[2],
        '4th_highest': sorted_row[1],
        'min': min(row)
    })

selected_values_df = predictions_df.apply(select_values, axis=1)
full_selected_values_df = pd.concat([predictions_df, selected_values_df], axis=1)
                                    
# Calculate the mean, median, std, min, max, and range for the predictions
full_selected_values_df['std'] = predictions_df.std(axis = 1)
full_selected_values_df['range'] = full_selected_values_df['max'] - full_selected_values_df['min']

# For each column in the full_selected_values_df, calculate the AUC
auc_dict = {}
for col in full_selected_values_df.columns:
    fpr, tpr, _ = roc_curve(y_val, full_selected_values_df[col]) # type: ignore
    auc_dict[col] = auc(fpr, tpr)

# make AUC_dict a DataFrame
auc_df = pd.DataFrame(auc_dict, index = ['AUC']).T

# Save the full_selected_values_df values to csv
full_selected_values_df.to_csv('full_selected_values_df_NN.csv')

'''
# Get the RMSE for each scoring function
rmse_dict = {}

# Get the AUC for each scoring function
auc_dict = {}

for col in score_columns:
    error = pdbbind_standard_norm_df[col] - pdbbind_standard_norm_df['experimental']
    rmse_dict[col] = np.sqrt(np.mean(error**2))

    # Map the Ligand and Decoy values to 1 and 0 respectively
    dudez_standard_norm_df['type_cat'] = dudez_standard_norm_df['type'].map({'ligand': 1, 'decoy': 0})

    fpr, tpr, _ = roc_curve(dudez_standard_norm_df['type_cat'], dudez_standard_norm_df[col])
    auc_dict[col] = auc(fpr, tpr)
'''
"""