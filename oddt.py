#!/usr/bin/env python3

import os
import sys

from glob import glob
from tqdm import tqdm

from OCDocker.Initialise import * # type: ignore

def preload_DUDEz(dudez_database_index_path: str, ignored_dudez_database_index_path: str):
    ''' Read the DUDEz database indexes from the specified files, filter the ignored indexes and return the dudez database indexes list.

    Parameters
    ----------
    dudez_database_index_path : str
        Path to the dudez database indexes file.
    ignored_dudez_database_index_path : str
        Path to the ignored dudez database indexes file.

    Returns
    -------
    dudez_targets : List[str]
        List of dudez database indexes.
    '''

    # If the pdb_database_index file exists
    if os.path.isfile(dudez_database_index_path):
        # Open the file
        with open(dudez_database_index_path, "r") as f:
            # Read the dudez database indexes from the specified file in the config file
            dudez_targets = list(set(line.strip() for line in f if line.strip()))
    else:
        sys.exit("The dudez_database_index file does not exist. Please, check the config file.")

    # If the ignored_pdb_database_index file exists
    if os.path.isfile(ignored_dudez_database_index_path):
        # Open the file
        with open(ignored_dudez_database_index_path, "r") as f:
            # Read the ignored dudez database indexes from the specified file in the config file
            ignored_dudez_targets = [line.strip() for line in f if line.strip()]
        
        # Remove from the dudez_targets list the ignored dudez database indexes
        dudez_targets = [x for x in dudez_targets if x not in ignored_dudez_targets]
    
    return dudez_targets

dudez_database_index = "/data/hd4tb/OCDocker/data/dudez_proteins.txt"

dudez_targets = preload_DUDEz(dudez_database_index, "/data/hd4tb/OCDocker/data/problematic_dudez_proteins.txt")

# Import the libraries
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
from OCDocker.DB.Models.Complexes import Complexes
from OCDocker.DB.Models.Ligands import Ligands
from OCDocker.DB.Models.Receptors import Receptors

cpu_cores = 16
available_cores = cpu_cores - 1 # The main thread is not counted
multiprocess = 1                # 0: single process; 1: multiprocess
generate_report = False         # Generate a report at the end of the pipeline
zip_output = False              # Zip the output files
update = False                  # Update the pipeline
overwrite = False               # Overwrite the output files

exit()

def find_mols(database, receptor, kind):
    """
    Find the molecules from the desired database.
    """

    mols = []

    # For each database
    for d in database:
        # For each receptor
        for r in receptor:
            # For each kind
            for k in kind:
                # Find its molecules
                mols += [os.path.join(ocdb_path, d, r, "compounds", k, t) for t in glob(os.path.join(ocdb_path, d, r, "compounds", k, "*"))]
    
    return mols

for mol in tqdm(find_mols(["DUDEz"], dudez_targets, ["ligands", "decoys", "compounds"]), desc="Molecules"):

    # Assemble the pkl name
    pkl = f"{mol}/payload.pkl"

    # Check if the file is empty
    if os.path.getsize(pkl) == 0:
        print(f"Skipping {mol} because it is empty")
        continue

    # Set the prepared receptor (3 parents up)
    preparedReceptor = f"{mol}/../../../prepared_receptor.pdbqt"

    # Determine the vina folder
    vina_folder = f"{mol}/vinaFiles"

    # Get the name of the chosen molecule (in snakemake with clustering) and clean it
    target = glob(f"{vina_folder}/*_rescoring.log")[0].replace("_rescoring.log", "").split("/")[-1].replace("_vinardo", "").replace("_vina", "")

    # Determine the chosen structure pose where it is needed to load the structure file
    if "ligand_entry_00" in target: # Plants
        chosen_pose = f"{mol}/plantsFiles/run/{target}.pdbqt"
        # Convert from mol2 to pdbqt
    else: # Vina
        chosen_pose = f"{mol}/vinaFiles/{target}.pdbqt"

    # Run ODDT
    df = ocoddt.run_oddt(preparedReceptor, [chosen_pose], target, f"{mol}/oddt") # type: ignore

    # Rename the columns to match the database (ODDT_ + uppercase name)
    df.columns = [f"ODDT_{col.upper()}" for col in df.columns]

    # Get the name of the receptor
    receptor_name = mol.split("/")[7]

    # Get the name of the ligand
    ligand_name = mol.split("/")[-1]

    # Convert the dataframe to a dictionary
    payload = df.to_dict("records")[0]

    # Update the database record where have name as f"{receptor_name}-{ligand_name}"
    Complexes.update(idorname = f"{receptor_name}-{ligand_name}", payload = payload) # type: ignore
