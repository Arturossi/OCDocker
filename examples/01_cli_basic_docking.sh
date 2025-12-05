#!/bin/bash
# Example 1: Basic CLI docking with Vina
# This example shows how to run a simple docking job using the CLI

# Basic docking with Vina
ocdocker vs \
  --engine vina \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb \
  --outdir ./docking_output \
  --timeout 600

# Skip rescoring for faster execution
ocdocker vs \
  --engine vina \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb \
  --skip-rescore \
  --outdir ./docking_output

