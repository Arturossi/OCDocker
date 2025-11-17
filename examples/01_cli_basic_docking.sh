#!/bin/bash
# Example 1: Basic CLI docking with Vina
# This example shows how to run a simple docking job using the CLI

# Basic docking with Vina
ocdocker vs \
  --engine vina \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.smi \
  --box path/to/box.pdb \
  --outdir ./docking_output \
  --timeout 600

# With rescoring enabled (default)
ocdocker vs \
  --engine vina \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --box path/to/box.pdb \
  --outdir ./docking_output \
  --store-db

# Skip rescoring for faster execution
ocdocker vs \
  --engine vina \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.smi \
  --box path/to/box.pdb \
  --skip-rescore \
  --outdir ./docking_output

