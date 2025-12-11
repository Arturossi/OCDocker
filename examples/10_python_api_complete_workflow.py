#!/usr/bin/env python3
"""
Example 10: Complete workflow - Multi-engine docking, clustering, and rescoring
This example demonstrates a complete virtual screening workflow
"""

import os
from OCDocker.Receptor import Receptor
from OCDocker.Ligand import Ligand
from OCDocker.Docking.Vina import Vina
from OCDocker.Docking.Smina import Smina
from OCDocker.Processing.Preprocessing.RmsdClustering import cluster_rmsd, get_medoids
from OCDocker.Toolbox.MoleculeProcessing import get_rmsd_matrix

# Step 1: Create receptor and ligand objects
receptor = Receptor("./test_files/receptor.pdb", name="MyReceptor")
ligand = Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="MyLigand")

ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Step 2: Prepare receptor (only once, can be reused)
receptor_pdbqt = f"{receptor_path}/prepared_receptor.pdbqt"
if not os.path.isfile(receptor_pdbqt):
    # Use Vina to prepare receptor (any engine can do this)
    temp_vina = Vina(
        config_path=f"{ligand_path}/vinaFiles/conf_vina.txt",
        box_file=f"{ligand_path}/boxes/box.pdb",
        receptor=receptor,
        prepared_receptor_path=receptor_pdbqt,
        ligand=ligand,
        prepared_ligand_path=f"{ligand_path}/prepared_ligand.pdbqt",
        vina_log=f"{ligand_path}/vinaFiles/temp.log",
        output_vina=f"{ligand_path}/vinaFiles/temp.pdbqt",
        name="temp"
    )
    temp_vina.run_prepare_receptor()

# Step 3: Prepare ligand
ligand_pdbqt = f"{ligand_path}/prepared_ligand.pdbqt"
if not os.path.isfile(ligand_pdbqt):
    temp_vina = Vina(
        config_path=f"{ligand_path}/vinaFiles/conf_vina.txt",
        box_file=f"{ligand_path}/boxes/box.pdb",
        receptor=receptor,
        prepared_receptor_path=receptor_pdbqt,
        ligand=ligand,
        prepared_ligand_path=ligand_pdbqt,
        vina_log=f"{ligand_path}/vinaFiles/temp.log",
        output_vina=f"{ligand_path}/vinaFiles/temp.pdbqt",
        name="temp"
    )
    temp_vina.run_prepare_ligand()

# Step 4: Run docking with Vina
print("Running Vina docking...")
vina = Vina(
    config_path=f"{ligand_path}/vinaFiles/conf_vina.txt",
    box_file=f"{ligand_path}/boxes/box.pdb",
    receptor=receptor,
    prepared_receptor_path=receptor_pdbqt,
    ligand=ligand,
    prepared_ligand_path=ligand_pdbqt,
    vina_log=f"{ligand_path}/vinaFiles/vina.log",
    output_vina=f"{ligand_path}/vinaFiles/vina.pdbqt",
    name="Vina receptor-ligand"
)
vina.run_docking()
vina_results = vina.read_log()
vina_poses = vina.get_docked_poses()
print(f"Vina: Found {len(vina_poses)} poses")

# Step 5: Run docking with Smina
print("Running Smina docking...")
smina = Smina(
    config_path=f"{ligand_path}/sminaFiles/conf_smina.txt",
    box_file=f"{ligand_path}/boxes/box.pdb",
    receptor=receptor,
    prepared_receptor_path=receptor_pdbqt,
    ligand=ligand,
    prepared_ligand_path=ligand_pdbqt,
    smina_log=f"{ligand_path}/sminaFiles/smina.log",
    output_smina=f"{ligand_path}/sminaFiles/smina.pdbqt",
    name="Smina receptor-ligand"
)
smina.run_docking()
smina_results = smina.read_log()
smina_poses = smina.get_docked_poses()
print(f"Smina: Found {len(smina_poses)} poses")

# Step 6: Combine poses and cluster by RMSD
print("Clustering poses by RMSD...")
all_poses = vina_poses + smina_poses
rmsd_matrix = get_rmsd_matrix(all_poses)
clusters = cluster_rmsd(
    rmsd_matrix,
    algorithm='agglomerativeClustering',
    outputPlot="./workflow_clustering.png"
)

# Step 7: Get medoids (representative poses)
medoids = get_medoids(rmsd_matrix, clusters, onlyBiggest=True)
print(f"Found {len(medoids)} representative poses (medoids)")

# Step 8: Rescore medoids with Smina
print("Rescoring medoids with Smina...")
# Note: run_rescore requires both outPath and ligand parameters
smina.run_rescore(f"{ligand_path}/sminaFiles", smina.output_smina, skipDefaultScoring=True)
rescoring_results = smina.read_rescore_logs(f"{ligand_path}/sminaFiles")

# Step 9: Summary
print("\n=== Workflow Summary ===")
print(f"Total poses from all engines: {len(all_poses)}")
print(f"Number of clusters: {len(set(clusters.values()))}")
print(f"Selected medoids: {len(medoids)}")
print(f"Best Vina score: {min(vina_results) if vina_results else 'N/A'}")
print(f"Best Smina score: {min(smina_results) if smina_results else 'N/A'}")

print("\nWorkflow completed successfully!")

