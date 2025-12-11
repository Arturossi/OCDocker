#!/usr/bin/env python3
"""
Example 8: Python API - ODDT rescoring
This example shows how to use ODDT for rescoring docked poses
Note: ODDT is used only for rescoring, so docking must be done first
"""

import os
from OCDocker.Receptor import Receptor
from OCDocker.Ligand import Ligand
from OCDocker.Docking.Vina import Vina
from OCDocker.Rescoring.ODDT import run_oddt, df_to_dict

# First, perform docking (using Vina as example)
receptor = Receptor("./test_files/receptor.pdb", name="MyReceptor")
ligand = Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="MyLigand")

ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Create and run Vina docking
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

vina.run_prepare_receptor()
vina.run_prepare_ligand()
vina.run_docking()

# Get docked poses
docked_poses = vina.get_docked_poses()

# Run ODDT rescoring
# This will compute various scoring functions (RFScore, NNScore, PLEC, etc.)
# Note: run_oddt requires prepared receptor path, prepared ligand paths (list), ligand name, and output path
oddt_results_df = run_oddt(
    preparedReceptorPath=vina.prepared_receptor,
    preparedLigandPath=docked_poses,  # List of docked pose file paths
    ligandName=vina.input_ligand.name,
    outputPath=f"{ligand_path}/oddt"
)

# Convert DataFrame to dictionary if needed
oddt_results_dict = df_to_dict(oddt_results_df)

print("ODDT rescoring results (DataFrame):")
print(oddt_results_df)
print("\nODDT rescoring results (Dictionary):")
print(oddt_results_dict)

