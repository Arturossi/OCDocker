#!/usr/bin/env python3
"""
Example 4: Using the interactive console
This example shows how to use the OCDocker interactive console for step-by-step workflows
"""

# To use the interactive console, run:
# ocdocker console --conf OCDocker.cfg

# The console provides:
# - Pre-imported OCDocker modules (Receptor, Ligand, Docking classes, etc.)
# - Tab completion for easier exploration
# - Step-by-step execution of docking workflows
# - Access to print_args() function to check configuration

# Example console session:
"""
$ ocdocker console --conf OCDocker.cfg

>>> print_args()  # Check current configuration
>>> print_args('paths')  # Check binary paths
>>> print_args('vina')  # Check Vina parameters

>>> import OCDocker.Receptor as ocr
>>> receptor = ocr.Receptor("path/to/receptor.pdb", name="MyReceptor")

>>> import OCDocker.Ligand as ocl
>>> ligand = ocl.Ligand("path/to/ligand.smi", name="MyLigand")

>>> import OCDocker.Docking.Vina as ocvina
>>> vina = ocvina.Vina(...)
>>> vina.run_prepare_receptor()
>>> vina.run_prepare_ligand()
>>> vina.run_docking()
"""

