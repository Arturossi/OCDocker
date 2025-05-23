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

ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"
base_models_folder: str = f"{base_path}/models"

for run, start_id, end_id in [("Descriptors only", 36, 41)]:
    for i in tqdm(range(start_id, end_id), desc=f"Optimizing {run}"):
        # Load the data
        data = ocscoredata.load_data(
            base_models_folder = base_models_folder,
            storage_id = i,
            df_path = df_path,
            optimization_type = "NN",
            no_scores = False if run == "SFs only" else True,
            only_scores = True if run == "SFs only" else False,
            use_PCA = False,
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
            no_scores = False if run == "SFs only" else True,
            only_scores = True if run == "SFs only" else False,
            use_PCA = False,
            best_ao_params = None,
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
