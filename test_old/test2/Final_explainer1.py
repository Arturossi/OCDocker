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

# WARNING: Only set this to True if NO machine is running the same study, otherwise you might end up with duplicate evaluations
filter_completed_jobs = True

# Set the study data here (Currently only for NN ablations)
study_number = 7
ao_study_name = f"AO_Optimization_{study_number}"
nn_study_name = f"NN_Optimization_{study_number}"
seed_study_name = f"NN_Seed_Ablation_Optimization_2"
mask_study_name = f"NN_Ablation_Optimization_1"

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

## Define the seed
##########################

# Load the study
seed_study = optuna.load_study(study_name = seed_study_name, storage = storage)
seed_df = seed_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
seed_df = seed_df[seed_df['state'] == 'COMPLETE']

#best_seed_df = seed_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_auc'], ascending=[True, True, True])
best_seed_df = seed_df.sort_values(by=['value', 'user_attrs_AUC'], ascending=[True, False])

# Recreate the seed object for the best trial based on the best_seed_df
best_seed_trial = best_seed_df.iloc[0]

# Select the trial by the best_seed_trial number
best_seed_trial = seed_study.trials[best_seed_trial.number]

# Pick the params from the best_seed_trial
seed = best_seed_trial.user_attrs['random_seed']

## Define the mask
##########################

# Load the study
mask_study = optuna.load_study(study_name = mask_study_name, storage = storage)
mask_df = mask_study.trials_dataframe()

# Filter the trials to only include the ones that are complete
mask_df = mask_df[mask_df['state'] == 'COMPLETE']

#best_mask_df = mask_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_auc'], ascending=[True, True, True])
best_mask_df = mask_df.sort_values(by=['value', 'user_attrs_AUC'], ascending=[True, False])

# Recreate the mask object for the best trial based on the best_mask_df
best_mask_trial = best_mask_df.iloc[0]

# Select the trial by the best_mask_trial number
best_mask_trial = mask_study.trials[best_mask_trial.number]

# Pick the params from the best_mask_trial
mask = best_mask_trial.user_attrs['Feature_Mask']

# Convert the mask to a numpy array of 0s and 1s
mask = np.array([int(x) for x in mask])

# Fetch data
################

# Get the dataframe and score columns for later use
df_dudez, df_pdbbind, score_columns = ocscoredata.preprocess_df(df_path)

# Drop the unecessary columns ignoring errors
df_pdbbind = df_pdbbind.drop(
        columns = ["receptor", "ligand", "name", "type", "db", "experimental", "OCSCORE"],
        errors = "ignore"
    )

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
y_val = data["y_val"].values

from OCDocker.OCScore.DNN.DNNOptimizer import NeuralNet

# Create the neural network
neural = NeuralNet(
        data["X_train"].shape[1], 
        1, 
        autoencoder_params,
        best_nn_params,
        random_seed = seed,
        use_gpu = True,
        verbose = False,
        mask = mask
    )

import torch

neural.train_model(
    X_train= torch.tensor(X_train.values, dtype=torch.float32).to(neural.device),
    y_train=torch.tensor(data["y_train"].values, dtype=torch.float32).to(neural.device),
    X_test=torch.tensor(X_test.values, dtype=torch.float32).to(neural.device),
    y_test=torch.tensor(data["y_test"].values, dtype=torch.float32).to(neural.device),
    X_validation=torch.tensor(X_val.values, dtype=torch.float32).to(neural.device),
    y_validation=torch.tensor(y_val, dtype=torch.float32).to(neural.device)
)

neural.NN.eval()

import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Define a prediction function compatible with SHAP
def predict_model(X):
    if isinstance(X, pd.DataFrame):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(neural.device)  # fix here
    elif isinstance(X, np.ndarray):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(neural.device)
    else:
        raise ValueError("Input must be a pandas DataFrame or a numpy array.")
    with torch.no_grad():
        predictions = neural.NN(X_tensor).cpu().numpy().flatten()
    return predictions

# Define background (use a stratified sample from your training data)
background_size = 2000

# Create a stratified sample for background data based on DUDEz dataset
strat_labels = pd.Series(df_dudez['receptor']).astype(str) + "_" + pd.Series(df_dudez['type']).astype(str)

# Stratified sampling for background data
sss = StratifiedShuffleSplit(n_splits=1, train_size=background_size, random_state=42)
for background_idx, _ in sss.split(X_val, strat_labels):
    background_data = X_val.iloc[background_idx].values

exp = "DeepExplainer"

# Initialize KernelExplainer
if exp == "KernelExplainer":
    explainer = shap.KernelExplainer(predict_model, background_data)
    # Calculate SHAP values for validation dataset
    shap_values = explainer.shap_values(torch.tensor(X_val.values, dtype=torch.float32).to(neural.device), nsamples=background_size)
elif exp == "DeepExplainer":
    # Initialize DeepExplainer
    explainer = shap.DeepExplainer(neural.NN, torch.tensor(background_data, dtype=torch.float32).to(neural.device))
    # Calculate SHAP values for validation dataset
    shap_values = explainer.shap_values(torch.tensor(X_val.values, dtype=torch.float32).to(neural.device))
    shap_values = shap_values.squeeze(-1)
else:
    raise ValueError("Unknown explainer type. Use 'KernelExplainer' or 'DeepExplainer'.")

# Feature names (assuming df_pdbbind has feature columns aligned)
feature_names = df_pdbbind.columns.to_list()

# Plot global feature importance
plt.figure(figsize=(18, 28))
ax = plt.gca()
ax.xaxis.grid(True, linestyle='--', alpha=0.7) 
shap.summary_plot(
    shap_values, 
    X_val, 
    feature_names=feature_names, 
    plot_type="bar", 
    show=False,
    plot_size=(10, 8)
)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot detailed SHAP values
shap.summary_plot(
    shap_values,
    X_val,
    feature_names=feature_names,
    show=False,
    plot_size=(12, 10)
)
plt.tight_layout()
plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
plt.close()
