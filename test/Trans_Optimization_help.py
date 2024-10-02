#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

from tqdm import tqdm
from urllib.parse import quote_plus

import OCDocker.OCScore.Dimensionality.PCA as ocpca
import OCDocker.OCScore.Optimization.Transformer as octrans
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

run = "Descriptors only"

octrans.optimize(
    df_path = df_path,
    storage_id = 36,
    base_models_folder = base_models_folder,
    storage = storage,
    use_pdb_train = True,
    no_scores = False if run == "SFs only" else True,
    only_scores = True if run == "SFs only" else False,
    use_PCA = False,
    run_Trans_optimization = True,
    num_processes_Trans = 3,
    total_trials_Trans = 1000,
    random_seed = 42,
    load_if_exists = True,
    use_gpu = True,
    parallel_backend = "joblib",
    verbose = False
)

for run, start_id, end_id in [("Descriptors only", 37, 41)]:
    for i in tqdm(range(start_id, end_id), desc=f"Optimizing {run}"):
        octrans.optimize(
            df_path = df_path,
            storage_id = i,
            base_models_folder = base_models_folder,
            storage = storage,
            use_pdb_train = True,
            no_scores = False if run == "SFs only" else True,
            only_scores = True if run == "SFs only" else False,
            use_PCA = False,
            run_Trans_optimization = True,
            num_processes_Trans = 3,
            total_trials_Trans = 1000,
            random_seed = 42,
            load_if_exists = True,
            use_gpu = True,
            parallel_backend = "joblib",
            verbose = False
        )
