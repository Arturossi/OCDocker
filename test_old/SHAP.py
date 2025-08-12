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

neural.NN.eval()

import torch
import shap
import pandas as pd

def model_predict(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to("cuda")
    with torch.no_grad():
        output = neural.NN(x_tensor).cpu().numpy().squeeze()
    return output

samples = 1500

background = X_train.iloc[np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)]
background_tensor = torch.tensor(background.to_numpy(), dtype=torch.float32).to("cuda")

deep_explainer = shap.DeepExplainer(
    neural.NN, 
    background_tensor
)

#kernel_explainer = shap.KernelExplainer(
#    model_predict,
#    background.to_numpy()
#)

# Seleção de amostras de teste para explicar
test_sample = X_test.iloc[np.random.choice(X_test.shape[0], X_test.shape[0], replace=False)]
test_sample_tensor = torch.tensor(test_sample.to_numpy(), dtype=torch.float32).to("cuda")

# DeepExplainer
shap_values_deep = deep_explainer.shap_values(test_sample_tensor)

# KernelExplainer - Lento, não recomendado em alta dimensão
#shap_values_kernel = kernel_explainer.shap_values(test_sample.to_numpy())

# Conversão opcional para DataFrame, se quiser visualizar
shap_deep_df = pd.DataFrame(
    np.squeeze(shap_values_deep),  # Remove o eixo de saída
    columns=X_train.columns
)

#shap_kernel_df = pd.DataFrame(
#    np.squeeze(shap_values_kernel),
#    columns=X_train.columns
#)

import matplotlib.pyplot as plt

# Calcula importância relativa (%)
mean_abs_shap = np.abs(shap_values_deep[:, :, 0]).mean(axis=0)
relative_importance = (mean_abs_shap / mean_abs_shap.sum()) * 100

# Ordena
sorted_idx = np.argsort(relative_importance)[::-1]

# Plota
plt.figure(figsize=(10, 6))
plt.barh(
    y=np.array(X_train.columns)[sorted_idx][:20], 
    width=relative_importance[sorted_idx][:20]
)
plt.xlabel('Importância Relativa (%)')
plt.title('Importância dos Descritores (SHAP)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_feature_importance.png', dpi=300)
plt.close()


# Gera o beeswarm plot com tamanho ajustado
shap.summary_plot(
    shap_values_deep[:, :, 0], 
    test_sample.to_numpy(), 
    feature_names=X_train.columns,
    show=False,  # Não exibe na tela
    plot_size=(10, 6)  # <-- Aqui você controla (largura, altura)
)

plt.tight_layout()
plt.savefig('shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
plt.close()
