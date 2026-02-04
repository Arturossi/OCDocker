#!/usr/bin/env python3
"""
Example 7: Python API - PLANTS docking
This example shows how to use OCDocker programmatically with PLANTS
"""

import os

from OCDocker.Docking.PLANTS import PLANTS
from OCDocker.Ligand import Ligand
from OCDocker.Receptor import Receptor

# Create receptor and ligand objects
receptor = Receptor("./test_files/receptor.pdb", name="MyReceptor")
ligand = Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="MyLigand")

# Define paths
ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Create PLANTS docking object
# Note: PLANTS uses MOL2 format instead of PDBQT
plants = PLANTS(
    config_path=f"{ligand_path}/plantsFiles/conf_plants.txt",
    box_file=f"{ligand_path}/boxes/box.pdb",
    receptor=receptor,
    prepared_receptor_path=f"{receptor_path}/prepared_receptor.mol2",
    ligand=ligand,
    prepared_ligand_path=f"{ligand_path}/prepared_ligand.mol2",
    plants_log=f"{ligand_path}/plantsFiles/plants.log",
    output_plants=f"{ligand_path}/plantsFiles",
    name="PLANTS receptor-ligand"
)

# Prepare receptor (converts to MOL2)
plants.run_prepare_receptor()

# Prepare ligand (converts to MOL2)
plants.run_prepare_ligand()

# Run docking
plants.run_docking()

# Read docking results
docking_results = plants.read_log(onlyBest=False)
print("PLANTS docking results:", docking_results)

# Get docked poses
docking_poses = plants.get_docked_poses()

# Write pose list for rescoring
pose_list = plants.write_pose_list()

# Run rescoring
plants.run_rescore(pose_list, logFile="", overwrite=False)

# Read rescoring results
rescoring_results = plants.read_rescore_logs(onlyBest=False)
print("PLANTS rescoring results:", rescoring_results)
