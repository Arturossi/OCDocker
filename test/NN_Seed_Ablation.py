#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

import optuna

import numpy as np

from urllib.parse import quote_plus

import OCDocker.OCScore.Utils.Data as ocscoredata
from OCDocker.OCScore.Optimization.DNN import perform_seed_ablation_study_NN

from OCDocker.Initialise import *

ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"
base_models_folder: str = f"{base_path}/models"

num_proc = 8

max_seed = 2000

# WARNING: Only set this to True if NO machine is running the same study, otherwise you might end up with duplicate evaluations
filter_completed_jobs = True

# Id of the machine running the study
machine_id = 1
# Number of machines running the study (for splitting the masks)
num_machines = 4

# Set the study data here (Currently only for NN ablations)
study_number = 7
ao_study_name = f"AO_Optimization_{study_number}"
nn_study_name = f"NN_Optimization_{study_number}"

## Define the Autoencoder
##########################

# Load the study
ao_study = optuna.load_study(study_name = ao_study_name, storage = storage)
ao_df = ao_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
ao_df = ao_df[ao_df['state'] == 'COMPLETE']

#best_ao_df = ao_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])
best_ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])

# Recreate the autoencoder object for the best trial based on the best_ao_df
best_ao_trial = best_ao_df.iloc[0]

# Select the trial by the best_ao_trial number
best_ao_trial = ao_study.trials[best_ao_trial.number]

# Pick the params from the best_ao_trial
autoencoder_params = best_ao_trial.params

## Define the Topology
##########################

# Load the study
nn_study = optuna.load_study(study_name = nn_study_name, storage = storage)
nn_df = nn_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
nn_df = nn_df[nn_df['state'] == 'COMPLETE']

nn_df['combined_metric'] = nn_df['value'] - nn_df['user_attrs_AUC']

best_nn_df = nn_df.sort_values(by=['combined_metric'], ascending=[True])

best_nn_trial = best_nn_df.iloc[0]

best_nn_trial = nn_study.trials[best_nn_trial.number]

best_nn_params = best_nn_trial.params

# Get the mask
################

# Set the study data here (Currently only for NN ablations)
study_name = f"NN_Ablation_Optimization_1"

# Load the study
ablation_study = optuna.load_study(study_name = study_name, storage = storage)
ablation_df = ablation_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
ablation_df = ablation_df[ablation_df['state'] == 'COMPLETE']

# Reset data index
ablation_df = ablation_df.reset_index(drop=True)

# Rename the columns
# value is the RMSE
# user_attrs_Feature_Mask is the Feature Mask
# user_attrs_AUC is the AUC
ablation_df = ablation_df.rename(columns={
        'value': 'RMSE',
        'user_attrs_Feature_Mask': 'Feature_Mask',
        'user_attrs_AUC': 'AUC'
    }
)

# Compute the score (RMSE - AUC)
ablation_df['score'] = ablation_df['RMSE'] - ablation_df['AUC']

best_ablation_df = ablation_df.sort_values(by=['score'], ascending=[True])

# Recreate the autoencoder object for the best trial based on the best_ablation_df
#best_ablation_trial = best_ablation_df.iloc[0]

# Select the trial by the best_ablation_trial number
#best_ablation_trial = ablation_study.trials[best_ablation_trial.number]

# Pick the user_attrs_Feature_Mask (Feature_Mask) from the best_ablation_trial
mask = best_ablation_df.iloc[0]['Feature_Mask']

# Convert the mask to a numpy array of 0s and 1s
mask = np.array([int(x) for x in mask])

# Fetch data
################

# Get the dataframe and score columns for later use
_, df, score_columns = ocscoredata.preprocess_df(df_path)

# Drop the unecessary columns ignoring errors
df = df.drop(
        columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
        errors = "ignore"
    )

# Make the seed
pre_seeds = list(range(1, max_seed))

# If filter completed jobs is set to True, then we will only run the masks that have not been evaluated
if filter_completed_jobs:
    # Try to load the study to check which masks have already been evaluated
    try:
        study_name = f"NN_Seed_Ablation_Optimization_1"
        study = optuna.load_study(study_name = study_name, storage = storage)

        # Filter the trials to only include the ones that are complete
        trials = study.trials_dataframe()
        trials = trials[trials['state'] == 'COMPLETE']

        # Get the seeds that have already been evaluated
        evaluated_seeds = trials['user_attrs_random_seed'].tolist()

        # Apply each feature seed to the full_seed
        filtered_seeds = []

        for seed in pre_seeds:
            if not evaluated_seeds or seed not in evaluated_seeds:
                filtered_seeds.append(seed)

        # Chunk the seeds (all seeds here will come already without completed jobs)
        seeds = ocscoredata.chunkenize_dataset(filtered_seeds, machine_id, num_machines)
    except:
        print(f"Error while loading the study: {study_name}")
        evaluated_seeds = []
else:
    # Split the dataset based on the machine_id
    chunked_seeds = ocscoredata.chunkenize_dataset(pre_seeds, machine_id, num_machines)

    # Check if any study for this chunk has already been processed
    try:
        # Try to load the study to check which seeds have already been evaluated
        study = optuna.load_study(study_name = f"NN_Seed_Ablation_Optimization_1", storage = storage)

        # Filter the trials to only include the ones that are complete
        trials = study.trials_dataframe()
        trials = trials[trials['state'] == 'COMPLETE']

        # Get the seeds that have already been evaluated
        evaluated_seeds = trials['user_attrs_random_seed'].tolist()
    except:
        evaluated_seeds = []

    # Apply each feature seed to the full_seed
    seeds = []

    for seed in chunked_seeds:
        if not evaluated_seeds or seed not in evaluated_seeds:
            seeds.append(seed)

# Load the data to fetch the Xs, ys and validation
data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = study_number,
            df_path = df_path,
            optimization_type = "NN",
            no_scores = False,
            only_scores = False,
            use_PCA = False,
            use_pdb_train = True,
            random_seed = 42
        )

X_train = ocscoredata.invert_values_conditionally(data['X_train'])
X_test = ocscoredata.invert_values_conditionally(data['X_test'])
X_val = ocscoredata.invert_values_conditionally(data['X_val'])

# Perform the ablation
perform_seed_ablation_study_NN(
    X_train, 
    data['y_train'], 
    X_test, 
    data['y_test'], 
    X_val, 
    data['y_val'], 
    1, 
    num_proc, 
    autoencoder_params, 
    best_nn_params, 
    True, 
    False, 
    True, 
    "NN_Seed_Ablation_Optimization", 
    storage,
    mask = mask,
    seeds = seeds
)
