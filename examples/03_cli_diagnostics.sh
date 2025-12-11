#!/bin/bash
# Example 3: Diagnostics and configuration
# This example shows how to check your OCDocker installation and create configuration files

# Run diagnostics to check installation
ocdocker doctor --conf OCDocker.cfg

# Create a new configuration file from example
ocdocker init-config --conf my_ocdocker.cfg

# Check version
ocdocker version

# Run with custom config file
ocdocker vs \
  --conf my_ocdocker.cfg \
  --engine vina \
  --receptor ./test_files/test_ptn1/receptor.pdb \
  --ligand ./test_files/test_ptn1/compounds/ligands/ligand/ligand.smi \
  --box ./test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb

