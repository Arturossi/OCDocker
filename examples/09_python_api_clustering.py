#!/usr/bin/env python3
"""
Example 9: Python API - RMSD clustering
This example shows how to cluster docked poses from multiple engines using RMSD
"""

import os
from OCDocker.Receptor import Receptor
from OCDocker.Ligand import Ligand
from OCDocker.Docking.Vina import Vina
from OCDocker.Docking.PLANTS import PLANTS
from OCDocker.Processing.Preprocessing.RmsdClustering import cluster_rmsd, get_medoids
from OCDocker.Toolbox.MoleculeProcessing import get_rmsd_matrix

# Perform docking with multiple engines
receptor = Receptor("./test_files/receptor.pdb", name="MyReceptor")
ligand = Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="MyLigand")

ligand_path = "./test_files/compounds/ligands/ligand"
receptor_path = "./test_files"

# Run Vina docking
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
vina_docking_results = vina.read_log()
vina_poses = vina.get_docked_poses()

# Run PLANTS docking
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

plants.run_prepare_receptor()
plants.run_prepare_ligand()
plants.run_docking()
plants_docking_results = plants.read_log(onlyBest=False)
plants_poses = plants.get_docked_poses()

# Combine poses from all engines
all_poses = vina_poses + plants_poses

# Calculate RMSD matrix
rmsd_matrix = get_rmsd_matrix(all_poses)

# Perform clustering
clusters = cluster_rmsd(
    rmsd_matrix,
    algorithm='agglomerativeClustering',
    outputPlot="./clustering_plot.png"  # Optional: save visualization
)

# Get medoids (representative poses from each cluster)
medoids = get_medoids(rmsd_matrix, clusters, onlyBiggest=True)

print(f"Found {len(clusters)} clusters")
print(f"Selected {len(medoids)} medoids")

# Create dictionary mapping medoids to their docking scores
medoids_dict = {}
for medoid in medoids:
    if medoid in vina_poses:
        # Find index of medoid in vina_poses
        medoid_idx = vina_poses.index(medoid)
        medoids_dict[medoid] = {
            'engine': 'vina',
            'score': vina_docking_results[medoid_idx]
        }
    elif medoid in plants_poses:
        medoid_idx = plants_poses.index(medoid)
        medoids_dict[medoid] = {
            'engine': 'plants',
            'score': plants_docking_results[medoid_idx]
        }

print("\nMedoids with scores:")
for pose, info in medoids_dict.items():
    print(f"{pose}: {info}")

