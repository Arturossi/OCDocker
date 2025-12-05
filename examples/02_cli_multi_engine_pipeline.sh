#!/bin/bash
# Example 2: Multi-engine docking pipeline
# This example shows how to run docking across multiple engines with clustering and rescoring

# Run pipeline with multiple engines
ocdocker pipeline \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb \
  --engines vina,smina,plants \
  --outdir ./pipeline_output \
  --timeout 900

# Pipeline with only Vina and Smina
ocdocker pipeline \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb \
  --engines vina,smina \
  --outdir ./pipeline_output

# Pipeline with vina, plants, and rescore with all engines
ocdocker pipeline \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb \
  --engines vina,plants \
  --rescoring-engines vina,smina,plants,oddt \
  --outdir ./pipeline_output \
  --multiprocess

