#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Dimensionality.PCA as ocpca
import OCDocker.OCScore.Optimization.XGBoost as ocxgb
import OCDocker.OCScore.Utils.Data as ocscoredata
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

for run, start_id, end_id in [("SFs only", 31, 36), ("Descriptors only", 36, 41)]:
    for i in tqdm(range(start_id, end_id), desc=f"Optimizing {run}"):
        ocxgb.optimize_XGB(
            df_path = df_path,
            storage_id = i,
            base_models_folder = base_models_folder,
            storage = storage,
            use_pdb_train = True,
            no_scores = False if run == "SFs only" else True,
            only_scores = True if run == "SFs only" else False,
            use_PCA = False,
            run_pre_XGB_optimization = False,
            num_processes_pre_XGB = 8,
            total_trials_pre_XGB = 100,
            run_GA_optimization = False,
            num_processes_GA = 8,
            total_trials_GA = 10,
            run_XGB_optimization = True,
            num_processes_XGB = 8,
            total_trials_XGB = 1000,
            early_stopping_rounds = 20,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            verbose = False
        )
