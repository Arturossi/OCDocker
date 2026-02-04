#!/usr/bin/env python3
"""
Example 5: Python API - Vina docking
This example shows how to use OCDocker programmatically with Vina
"""

import os

from OCDocker.Docking.Vina import Vina
from OCDocker.Ligand import Ligand
from OCDocker.Receptor import Receptor

# Set configuration if needed
# os.environ['OCDOCKER_CONFIG'] = 'path/to/OCDocker.cfg'

# Create receptor object
receptor = Receptor(
    "./test_files/receptor.pdb",
    name="MyReceptor"
)

# Create ligand object
ligand = Ligand(
    "./test_files/compounds/ligands/ligand/ligand.smi",
    name="MyLigand"
)

# Define paths
ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Create Vina docking object
vina = Vina(
    config_path=f"{ligand_path}/vinaFiles/conf_vina.txt",
    box_file=f"{ligand_path}/boxes/box.pdb",
    receptor=receptor,
    prepared_receptor_path=f"{receptor_path}/prepared_receptor.pdbqt",
    ligand=ligand,
    prepared_ligand_path=f"{ligand_path}/prepared_ligand.pdbqt",
    vina_log=f"{ligand_path}/vinaFiles/vina.log",
    output_vina=f"{ligand_path}/vinaFiles/vina.pdbqt",
    name="Vina receptor-ligand"
)

# Prepare receptor (converts PDB to PDBQT)
vina.run_prepare_receptor()

# Prepare ligand (converts SMILES/SDF to PDBQT)
vina.run_prepare_ligand()

# Run docking
vina.run_docking()

# Split poses into individual files
vina.split_poses(f"{ligand_path}/vinaFiles", logFile="")

# Run rescoring (optional)
# Note: run_rescore requires both outPath and ligand parameters
# Use the output file from docking (contains all poses) for rescoring
vina.run_rescore(f"{ligand_path}/vinaFiles", vina.output_vina, skipDefaultScoring=True)

# Read docking results
docking_results = vina.read_log()
print("Docking results:", docking_results)

# Read rescoring results
rescoring_results = vina.read_rescore_logs(f"{ligand_path}/vinaFiles")
print("Rescoring results:", rescoring_results)

# Get docked poses
poses = vina.get_docked_poses()
print(f"Found {len(poses)} poses")
