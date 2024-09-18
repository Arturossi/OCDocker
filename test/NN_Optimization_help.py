#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Dimensionality.PCA as ocpca
import OCDocker.OCScore.Optimization.DNN as ocdnn
import OCDocker.OCScore.Utils.IO as ocscoreio

from OCDocker.Initialise import *

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
df_path: str = '/data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz'
base_models_folder: str = "/data/hd4tb/OCDocker/data/ocdb/models"

# PCA Type 80, 85, 90, 95 -> (PCA, start, end)
for pca_type, start_id, end_id in [(80, 11, 16), (85, 16, 21), (90, 21, 26), (95, 26, 31)]:
    
    # Path to save the PCA model
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
    
    # Load the PCA model
    pca_model = ocscoreio.load_object(pca_model)

    for i in tqdm(range(start_id, end_id), desc=f"PCA Type: {pca_type}"):
        # Load the data
        data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = i,
            df_path = df_path,
            optimization_type = "NN",
            pca_model = pca_model,
            no_scores = False,
            only_scores = False,
            use_PCA = True,
            pca_type = pca_type,
            use_pdb_train = True,
            random_seed = 42
        )

        ocdnn.optimize_NN(
            df_path = df_path,
            storage_id = i,
            base_models_folder = base_models_folder,
            data = data,
            storage = storage,
            use_pdb_train = True,
            no_scores = False,
            only_scores = False,
            use_PCA = True,
            best_ao_params = None,
            pca_type = pca_type,
            pca_model = pca_model,
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
        