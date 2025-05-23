#!/usr/bin/env python3
import sys

sys.path.append("../OCDocker")

import optuna
import sqlalchemy

import numpy as np

from urllib.parse import quote_plus
import torch
from torch.utils.data import DataLoader

import OCDocker.OCScore.Utils.Data as ocscoredata
from OCDocker.OCScore.DNN.DNNOptimizer import CustomDataset, NeuralNet

from OCDocker.Initialise import *

ip: str = "192.168.101.2"
ip: str = "localhost"
port: int = 3306
base_path: str = "/data/hd8tb/OCDocker_data/ocdb"
base_path: str = "/data/hd4tb/OCDocker/data/ocdb"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/optimization"
df_path: str = f"{base_path}/OCDocker.csv.gz"

base_models_folder: str = f"{base_path}/models"

# If pt file does not exist, create it
if not os.path.exists(f"{base_path}/OCDocker.pt"):
    ## Get data from the seed ablation study
    ########################################

    study_name = "NN_Seed_Ablation_Optimization_1"

    # Load the study
    seed_study = optuna.load_study(study_name = study_name, storage = storage)
    seed_df = seed_study.trials_dataframe()

    # Filter the trials to only include the ones that are complete
    seed_df = seed_df[seed_df['state'] == 'COMPLETE']

    # Compute the combined metric 
    # - Lowest RMSE to find the model with the best performance 
    # - Highest AUC to find the model which were best at separating the classes
    # - Highest PRAUC to find the model which separate the classes with the highest precision
    seed_df["Combined_Metric"] = seed_df["value"] - seed_df["user_attrs_AUC"] - seed_df["user_attrs_pr_auc"]

    # Sort the trials by the combined_metric (the lower the better)
    best_seed_df = seed_df.sort_values(by=['Combined_Metric'], ascending=[True])

    # Select the trial by the best_ae_trial number
    best_seed_trial = seed_study.trials[best_seed_df.iloc[0].number]

    # Set the mask, seed
    seed = best_seed_trial.user_attrs["random_seed"]
    mask = best_seed_trial.user_attrs["Feature_Mask"]

    # Transform the mask to a list of 0s and 1s
    mask = [int(x) for x in mask]

    # Get Autoencoder and neural network parameters
    ###############################################

    # Set the study data here (Currently only for NN ablations)
    study_number = 7
    ae_study_name = f"AO_Optimization_{study_number}"
    nn_study_name = f"NN_Optimization_{study_number}"

    ## Define the Autoencoder
    ##########################

    # Load the study
    ao_study = optuna.load_study(study_name = ae_study_name, storage = storage)
    ao_df = ao_study.trials_dataframe()

    # Filter the trials to only include the ones that are complete
    ao_df = ao_df[ao_df['state'] == 'COMPLETE']

    #best_ao_df = ao_df.sort_values(by=['combined_metric', 'value', 'user_attrs_val_rmse'], ascending=[True, True, True])
    best_ao_df = ao_df.sort_values(by=['value', 'user_attrs_val_rmse'], ascending=[True, True])

    # Recreate the autoencoder object for the best trial based on the best_ao_df
    best_ae_trial = best_ao_df.iloc[0]

    # Select the trial by the best_ae_trial number
    best_ae_trial = ao_study.trials[best_ae_trial.number]

    # Pick the params from the best_ae_trial
    ae_params = best_ae_trial.params

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

    nn_params = best_nn_trial.params

    # DATA LOADING
    ##############

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


    # Training
    ##########
    neural = NeuralNet(
            data["X_train"].shape[1], 
            1, 
            ae_params,
            nn_params,
            random_seed = 42,
            use_gpu = True,
            verbose = False,
            mask = mask
        )

    model = neural.NN

    # Invert conditionally X_train
    X_train = ocscoredata.invert_values_conditionally(data['X_train']) 

    X_train = torch.tensor(np.asarray(X_train), dtype = torch.float32).to("cuda")

    # Save the model in the model folder compressed
    torch.save(model, f"{base_path}/OCDocker.pt")
else:
    # Load the model
    model = torch.load(f"{base_path}/OCDocker.pt", weights_only=False)

def read_vs():
    import sqlalchemy

    import pandas as pd

    from urllib.parse import quote_plus

    # Set the database connection
    ip: str = "192.168.101.2"
    ip: str = "localhost"
    port: int = 3306
    db: str = "tcpaqr"

    storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/{db}"

    # Connect to the database
    engine = sqlalchemy.create_engine(storage)

    # Read the complexes table where the ligand_id is tied to the id of the ligands table and the receptor_id is tied to the id of the receptors table and return all the columns from all the tables
    #query = sqlalchemy.text("SELECT * FROM complexes JOIN ligands ON complexes.ligand_id = ligands.id JOIN receptors ON complexes.receptor_id = receptors.id")

    query = sqlalchemy.text("""
        SELECT 
            complexes.name AS complex_name,
            ligands.name AS ligand_name, 
            receptors.name AS receptor_name, 
            complexes.*, 
            receptors.*,
            ligands.*
        FROM complexes
        JOIN receptors ON complexes.receptor_id = receptors.id
        JOIN ligands ON complexes.ligand_id = ligands.id;
    """)

    with engine.connect() as connection:
        result = connection.execute(query)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Set the columns to drop
    to_drop = [
        'created_at',
        'modified_at',
        'ligand_id',
        'receptor_id',
        'id',
        'name'
        ]

    # Drop the columns
    df.drop(columns=to_drop, inplace=True)

    # Check rows with NaN values
    df.isna().sum().sum()

    # Remove rows with NaN values
    df.dropna(inplace=True)

    return df

def parse_to_ml_input(df):
    # Extract the complex_name to a list (to be used as the index)
    complex_names = df["complex_name"].tolist()

    # Drop the complex_name, ligand_name and receptor_name columns
    df.drop(columns=["complex_name", "ligand_name", "receptor_name", "OCSCORE"], inplace=True)

    # Return the complex_names and the df as the X to be used in the model
    return complex_names, df

df = read_vs()

complex_names, X = parse_to_ml_input(df)

# Invert the values conditionally
X = ocscoredata.invert_values_conditionally(X)

if not isinstance(X, pd.DataFrame):
    print("No entries to parse")
    sys.exit(1)

import pandas as pd

import time
# Start timer
start = time.time()

tensor_X = torch.tensor(X.values, dtype = torch.float32).to("cuda")

# Predict
with torch.no_grad():
    y = model(tensor_X)

# End timer
end = time.time()

# Convert the predictions to a numpy array
y = y.cpu().detach().numpy()

# Destroy the model and tensor_X 
#del model
#del tensor_X

# Free the GPU memory
torch.cuda.empty_cache()

# Assign the predictions to the complex names in a new dataframe
pred_df = pd.DataFrame(y, index = complex_names, columns = ["OCScore"])

# Print the time taken
print(f"Time taken to parse {len(complex_names)} entries: {end - start} seconds")

# For each complex, add the prediction to the database
for idx, row in pred_df.iterrows():
    with engine.begin() as connection:  # Use 'begin' to handle transactions automatically
        query = sqlalchemy.text("UPDATE complexes SET OCSCORE = :ocscore WHERE name = :name")
        connection.execute(query, {"ocscore": row['OCScore'], "name": idx})
