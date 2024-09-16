#!/usr/bin/env python3
import sys

sys.path.append("/data/hd4tb/OCDocker/OCDocker")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Optimization.XGBoost as ocxgb # type: ignore
import OCDocker.OCScore.Dimensionality.PCA as ocpca # type: ignore

from OCDocker.Initialise import *

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@localhost:3306/optimization"
df_path: str = '/data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz'

# No dimensionality reduction
for i in tqdm(range(1, 6)):
    ocxgb.optimize_XGB(
        df_path = df_path,
        storage_id = i,
        base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
        storage = storage,
        use_pdb_train = True,
        no_scores = False,
        only_scores = False,
        use_PCA = False,
        pca_type = 80,
        run_pre_XGB_optimization = False,
        num_processes_pre_XGB = 8,
        total_trials_pre_XGB = 100,
        run_GA_optimization = False,
        num_processes_GA = 8,
        total_trials_GA = 10,
        run_XGB_optimization = True,
        num_processes_XGB = 8,
        total_trials_XGB = 10,
        early_stopping_rounds = 20,
        random_seed = 42,
        load_if_exists = True,
        use_gpu = True,
        verbose = False
    )

# Genetic Algorithm
for i in tqdm(range(6, 11)):
    ocxgb.optimize_XGB(
        df_path = df_path,
        storage_id = i,
        base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
        storage = storage,
        use_pdb_train = True,
        no_scores = False,
        only_scores = False,
        use_PCA = False,
        pca_type = 80,
        run_pre_XGB_optimization = True,
        num_processes_pre_XGB = 8,
        total_trials_pre_XGB = 100,
        run_GA_optimization = True,
        num_processes_GA = 8,
        total_trials_GA = 10,
        run_XGB_optimization = True,
        num_processes_XGB = 8,
        total_trials_XGB = 10,
        early_stopping_rounds = 20,
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
        ocxgb.optimize_XGB(
            df_path = df_path,
            storage_id = i,
            base_models_folder = "/data/hd4tb/OCDocker/data/ocdb/models",
            storage = storage,
            use_pdb_train = True,
            no_scores = False,
            only_scores = False,
            use_PCA = True,
            pca_type = pca_type,
            run_pre_XGB_optimization = True,
            num_processes_pre_XGB = 8,
            total_trials_pre_XGB = 100,
            run_GA_optimization = False,
            num_processes_GA = 8,
            total_trials_GA = 10,
            run_XGB_optimization = True,
            num_processes_XGB = 8,
            total_trials_XGB = 10,
            early_stopping_rounds = 20,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )