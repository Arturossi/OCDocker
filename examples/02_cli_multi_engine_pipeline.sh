#!/bin/bash
# Example 2: Multi-engine docking pipeline
# This example shows how to run docking across multiple engines with clustering and rescoring

# Run pipeline with multiple engines
ocdocker pipeline \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --box path/to/box.pdb \
  --engines vina,smina,plants \
  --outdir ./pipeline_output \
  --timeout 900 \
  --store-db

# Pipeline with only Vina and Smina
ocdocker pipeline \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --box path/to/box.pdb \
  --engines vina,smina \
  --outdir ./pipeline_output

# Pipeline with custom timeout
ocdocker pipeline \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --box path/to/box.pdb \
  --engines vina,smina,plants \
  --timeout 1200 \
  --outdir ./pipeline_output

