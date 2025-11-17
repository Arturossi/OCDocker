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
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.smi \
  --box path/to/box.pdb

