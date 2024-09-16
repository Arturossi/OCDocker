#!/usr/bin/env python3
import sys

sys.path.append("..")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Optimization.DNN as ocdnn # type: ignore
import OCDocker.OCScore.Dimensionality.PCA as ocpca # type: ignore

from OCDocker.Initialise import *

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
df_path: str = '/data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz'

# No dimensionality reduction
for i in tqdm(range(1, 6)):
    ocdnn.optimize_NN(
            df_path = df_path,
            storage_id = i,
            base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
            storage = storage,
            use_pdb_train = True,
            no_scores = False,
            only_scores = False,
            use_PCA = False,
            best_ao_params = None,
            pca_type = 80,
            encoder_dims = (16, 256),
            autoencoder = True,
            multiencoder = False,
            run_autoencoder_optimization = True,
            num_processes_autoencoder = 8,
            total_trials_autoencoder = 2000,
            run_NN_optimization = True,
            num_processes_NN = 8,
            total_trials_NN = 500,
            explained_variance = 0.95,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )

# Autoencoder
for i in tqdm(range(6, 11)):
    ocdnn.optimize_NN(
            df_path = df_path,
            storage_id = i,
            base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
            storage = storage,
            use_pdb_train = True,
            no_scores = False,
            only_scores = False,
            use_PCA = False,
            best_ao_params = None,
            pca_type = 80,
            encoder_dims = (16, 256),
            autoencoder = True,
            multiencoder = False,
            run_autoencoder_optimization = True,
            num_processes_autoencoder = 8,
            total_trials_autoencoder = 2000,
            run_NN_optimization = True,
            num_processes_NN = 8,
            total_trials_NN = 500,
            explained_variance = 0.95,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )

# PCA Type 80, 85, 90, 95 -> (PCA, start, end)
for pca_type, start_id, end_id in [(80, 11, 16), (85, 16, 21), (90, 21, 26), (95, 26, 31)]:
    # Define the path to save the PCA object
    pca_model = f"{pca_path}/pca{pca_type}.pkl"

    # If the PCA model does not exist
    if not os.path.exists(pca_model):
        # Create the PCA model
        pca_model = ocpca.run_pca(
            df_path = df_path,
            variance = pca_type / 100,
            pca_path = pca_path,
            verbose = True
        )

    for i in tqdm(range(start_id, end_id), desc=f"PCA Type: {pca_type}"):
        ocdnn.optimize_NN(
            df_path = df_path,
            storage_id = i,
            base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
            storage = storage,
            use_pdb_train = True,
            no_scores = False,
            only_scores = False,
            use_PCA = True,
            best_ao_params = None,
            pca_type = pca_type,
            encoder_dims = (16, 256),
            autoencoder = False,
            multiencoder = False,
            run_autoencoder_optimization = False,
            num_processes_autoencoder = 8,
            total_trials_autoencoder = 2000,
            run_NN_optimization = True,
            num_processes_NN = 8,
            total_trials_NN = 500,
            explained_variance = 0.95,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )
