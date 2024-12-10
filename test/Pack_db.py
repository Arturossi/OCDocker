#!/usr/bin/env python3

# packs the output of the entire VS process in sdf files with metadata for the representative pose

import sys

sys.path.append("/data/hd8tb/OCDocker")

import os

import logging
import sqlalchemy

import pandas as pd

from glob import glob
from tqdm import tqdm
from urllib.parse import quote_plus
from rdkit import Chem
from typing import Union
from openbabel import openbabel

from OCDocker.Initialise import * # type: ignore

# Import the libraries
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Validation as ocvalidation
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust

cpu_cores = 18
available_cores = cpu_cores - 1 # The main thread is not counted
multiprocess = 1                # 0: single process; 1: multiprocess
generate_report = False         # Generate a report at the end of the pipeline
zip_output = False              # Zip the output files
update = False                  # Update the pipeline
overwrite = False               # Overwrite the output files

empty = 0

# Set the database connection
ip: str = "192.168.101.2"
port: int = 3306
db: str = "tcpaqr"

storage: str = f"mysql+pymysql://ocdocker:{quote_plus('@Kp3sRv9t@')}@{ip}:{port}/{db}"

# Set the output directory
output_dir = "/data/hd8tb/tcpaqr/molecule/vs_results"

# Safe create the output directory
ocff.safe_create_dir(output_dir)

# Create the engine
engine = sqlalchemy.create_engine(storage)

# Molecule list
sdf_molecules = []

def merge_sdfs_with_metadata(input_files, output_file):
    '''Merges multiple SDF files into a single SDF file while maintaining each molecule's metadata.

    Parameters
    ----------
    input_files : list of str
        List of SDF files to be merged.
    output_file : str
        Path to the output SDF file.
    '''
    # Open the output file
    writer = Chem.SDWriter(output_file)
    
    try:
        # Iterate over each input SDF file
        for input_file in input_files:
            # Open each SDF file
            suppl = Chem.SDMolSupplier(input_file)
            
            # Iterate over each molecule in the current SDF file
            for mol in suppl:
                if mol is not None:
                    # Write molecule with its metadata to the output SDF
                    writer.write(mol)
        
        print(f"Successfully merged SDF files into: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        writer.close()

def create_sdf_from_representative_pose(base_dir: str, ligand: str, receptor: str, output_dir: str) -> Union[str, None]:
    ''' Creates an sdf file with the representative pose of the ligand

    Parameters
    ----------
    base_dir : str
        The base directory where the molecule is located
    ligand : str
        The name of the ligand
    receptor : str
        The name of the receptor
    output_dir : str
        The directory where the output sdf file will be saved

    Returns
    -------
    str
        The path to the sdf file
    '''

    # Assemble the molecule path
    mol = os.path.join(base_dir, ligand)

    # Finished file (Temporary flag) TODO: Remove this
    finished_file = f"{mol}/finished_oddt"

    # Check if the finished flag file not exists
    if not os.path.isfile(finished_file):
        logging.warning(f"The file {finished_file} does not exist")
        return ""

    # Determine the docking folders
    vina_folder = f"{mol}/vinaFiles"
    plants_dir = f"{mol}/plantsFiles"

    plants_poses = ocplants.get_docked_poses(plants_dir)
    vina_poses = ocvina.get_docked_poses(vina_folder)

    # Concatenate the poses lists from vina and plants into a single list
    poses_list = vina_poses + plants_poses

    # Get the rmsd matrix from the poses list
    mols_mat = ocmolproc.get_rmsd_matrix(poses_list)

    # Get the rmsd matrix from the poses list
    rmsdMatrix = ocmolproc.get_rmsd_matrix(poses_list)

    # Get the clusters
    clusters = ocrmsdclust.cluster_rmsd(rmsdMatrix, algorithm = 'agglomerativeClustering', outputPlot = f"{mol}/medoids.png")

    # Get the medoids (The plot is just for visualization, it is not required)
    medoids = ocrmsdclust.get_medoids(rmsdMatrix, clusters, onlyBiggest = True) # type: ignore

    # If there is no medoid, return
    if len(medoids) == 0:
        print(f"No medoids found for '{mol}'")
        return None
    
    # If the length of the medoids is greater than 1, show a warning
    if len(medoids) > 1:
        print(f"Multiple medoids found for '{mol}'")

    # Get the target as the first medoid (ignore the rest) TODO: Make this more robust allowing multiple medoids
    target = medoids[0]

    # Get the extension of the input file
    inExtension = ocvalidation.validate_obabel_extension(target)

    # Convert the chosen pose to sdf and get the mol object
    mol = occonversion.convertMols(target, f"/tmp/a.sdf", return_molecule = True, overwrite = True)

    # Fetch its data from database
    query = sqlalchemy.text("""
        SELECT
            complexes.*, 
            receptors.*,
            ligands.*
        FROM complexes
        JOIN receptors ON complexes.receptor_id = receptors.id
        JOIN ligands ON complexes.ligand_id = ligands.id
        WHERE complexes.name = :complex_name;
        """)

    with engine.connect() as connection:
        result = connection.execute(query, {"complex_name": f"{receptor}-{ligand}"})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Set useless columns
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

    nans = df.isnull().sum().sum()

    if nans > 0:
        print(f"Found {nans} NaNs in the dataframe")
        return None

    # For each column in the dataframe
    for col in df.columns:
        # Get the value
        value = df[col].values[0]

        # If the value is not None
        if value is not None:
            # If mol is from openbabel
            if isinstance(mol, openbabel.OBMol):
                # Use OBPairData for Open Babel
                data = openbabel.OBPairData()
                # Set attribute (key) and value
                data.SetAttribute(col)
                data.SetValue(str(value))
                # Attach the metadata to the molecule
                mol.CloneData(data)
            else:
                print("Unknown molecule type")
                return None

    # Assemble the output file
    output_file = os.path.join(output_dir, f"{receptor}-{ligand}.sdf")

    # If mol is from openbabel
    if isinstance(mol, openbabel.OBMol):
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(inExtension, "sdf")
        # Write the molecule to the output file
        obConversion.WriteFile(mol, output_file)
    else:
        print("Unknown molecule type")
        return None

    return output_file

for mol in tqdm(glob("/data/hd8tb/tcpaqr/molecule/compounds/ligands/*"), desc="Molecules"):
    try:
        # Finished file
        finished_file = f"{mol}/finished_oddt"

        # Check if the finished flag file exists
        if not os.path.isfile(finished_file):
            logging.info(f"The file {finished_file} is not present, the protein has not been processed")
            continue

        # Split the molecule path from the base directory
        base_dir, ligand = os.path.split(mol)

        # Create the sdf file
        molecule = create_sdf_from_representative_pose(base_dir, ligand, "molecule", output_dir)

        # Assemble the pkl name
        pkl = f"{mol}/payload.pkl"

        # Check if molecule is not None or empty string
        if molecule is None or molecule == "":
            logging.warning(f"Skipping {mol} because the molecule is None")
            continue
        
        # Append the molecule to the list
        sdf_molecules.append(molecule)
    except Exception as e:
        with open(f"{mol}/../../../error_db", "a") as f:
            f.write(mol)
            f.write(str(e))
    
clean_sdf_molecules = []

# Create an OBConversion object for SDF format
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("sdf", "sdf")

for sdf in sdf_molecules:
    # Check if the sdf is the type of openbabel
    if isinstance(sdf, openbabel.OBMol):
        # Append the sdf to the list
        clean_sdf_molecules.append(sdf)

# If the list is not empty
if len(sdf_molecules) > 0:
    # Make a single sdf file
    merge_sdfs_with_metadata(sdf_molecules, "/data/hd8tb/tcpaqr/molecule/vs_results/VS_tcpaqr.sdf")

logging.info("Finished processing the molecules")
