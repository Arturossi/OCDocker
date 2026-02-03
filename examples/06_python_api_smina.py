#!/usr/bin/env python3
"""
Example 6: Python API - Smina docking
This example shows how to use OCDocker programmatically with Smina
"""

import os

from OCDocker.Docking.Smina import Smina
from OCDocker.Ligand import Ligand
from OCDocker.Receptor import Receptor

# Create receptor and ligand objects
receptor = Receptor("./test_files/receptor.pdb", name="MyReceptor")
ligand = Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="MyLigand")

# Define paths
ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Create Smina docking object
smina = Smina(
    config_path=f"{ligand_path}/sminaFiles/conf_smina.txt",
    box_file=f"{ligand_path}/boxes/box.pdb",
    receptor=receptor,
    prepared_receptor_path=f"{receptor_path}/prepared_receptor.pdbqt",
    ligand=ligand,
    prepared_ligand_path=f"{ligand_path}/prepared_ligand.pdbqt",
    smina_log=f"{ligand_path}/sminaFiles/smina.log",
    output_smina=f"{ligand_path}/sminaFiles/smina.pdbqt",
    name="Smina receptor-ligand"
)

# Prepare receptor (check if already prepared to avoid unnecessary work)
if not os.path.isfile(f"{receptor_path}/prepared_receptor.pdbqt"):
    smina.run_prepare_receptor()

# Prepare ligand
if not os.path.isfile(f"{ligand_path}/prepared_ligand.pdbqt"):
    smina.run_prepare_ligand()

# Run docking
smina.run_docking()

# Run rescoring with Smina
# Note: run_rescore requires both outPath and ligand parameters
# Use the output file from docking (contains all poses) for rescoring
smina.run_rescore(f"{ligand_path}/sminaFiles", smina.output_smina, skipDefaultScoring=True)

# Read results
docking_results = smina.read_log()
rescoring_results = smina.read_rescore_logs(f"{ligand_path}/sminaFiles")

print("Smina docking results:", docking_results)
print("Smina rescoring results:", rescoring_results)

# Alternative: Use Smina only for rescoring (if docking was done with Vina)
# from OCDocker.Docking.Vina import Vina
# vina = Vina(...)
# vina.run_docking()
# docking_poses = vina.get_docked_poses()
# 
# # Rescore Vina poses with Smina
# from OCDocker.Docking.Smina import run_rescore
# for scoring_function in smina_scoring_functions:
#     run_rescore(
#         smina.config,
#         docking_poses,
#         f"{ligand_path}/sminaFiles",
#         scoring_function,
#         splitLigand=False
#     )
