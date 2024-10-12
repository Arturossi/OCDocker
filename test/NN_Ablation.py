#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

import optuna

from urllib.parse import quote_plus

import OCDocker.OCScore.Utils.Data as ocscoredata
from OCDocker.OCScore.Optimization.DNN import perform_ablation_study_NN

from OCDocker.Initialise import *

ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"
base_models_folder: str = f"{base_path}/models"

machine_id = 1
num_machines = 2

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

# Get the dataframe and score columns for later use
_, df, score_columns = ocscoredata.preprocess_df(df_path)

# Drop the unecessary columns ignoring errors
df = df.drop(
        columns = ["receptor", "ligand", "name", "type", "db", "experimental"],
        errors = "ignore"
    )

# Make the mask
pre_masks = ocscoredata.generate_mask(df.columns, score_columns)

# Split the dataset based on the machine_id
chunked_masks = ocscoredata.chunkenize_dataset(pre_masks, machine_id, num_machines)

# Check if any study for this chunk has already been processed
try:
    # Try to load the study to check which masks have already been evaluated
    study = optuna.load_study(study_name = f"NN_Ablation_Optimization_1", storage = storage)

    # Filter the trials to only include the ones that are complete
    trials = study.trials_dataframe()
    trials = trials[trials['state'] == 'COMPLETE']

    # Get the masks that have already been evaluated
    evaluated_masks = trials['user_attrs_Feature_Mask'].tolist()
except:
    evaluated_masks = []

# Get the indexes for each sf
sf_indexes = [df.columns.get_loc(col) for col in score_columns]

# Apply each feature mask to the full_mask
masks = []

for mask in chunked_masks:
    # Start with a fresh copy of the full mask template
    modified_mask = mask.copy()
    if not evaluated_masks or "".join(map(str, modified_mask)) not in evaluated_masks:
        masks.append(modified_mask)

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

# Perform the ablation
perform_ablation_study_NN(
    data['X_train'], 
    data['y_train'], 
    data['X_test'], 
    data['y_test'], 
    data['X_val'], 
    data['y_val'], 
    1, 
    8, 
    autoencoder_params, 
    best_nn_params, 
    42, 
    True, 
    False, 
    True, 
    "NN_Ablation_Optimization", 
    storage,
    masks=masks
)
