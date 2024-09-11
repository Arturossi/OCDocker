#!/usr/bin/env python3
import sys

sys.path.append("/data/hd4tb/OCDocker/OCDocker")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Optimization.DNN as ocdnn # type: ignore

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
df_path: str = '/data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz'


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
            autoencoder = False,
            multiencoder = False,
            run_autoencoder_optimization = False,
            num_processes_autoencoder = 8,
            total_trials_autoencoder = 2000,
            run_NN_optimization = True,
            num_processes_NN = 8,
            total_trials_NN = 125,
            explained_variance = 0.95,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )