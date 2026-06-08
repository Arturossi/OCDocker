#!/usr/bin/env python3
"""
Example 12: Running Python scripts with OCDocker pre-loaded

This example demonstrates how to use the 'ocdocker script' command to run
Python scripts with all OCDocker libraries automatically imported.

Usage:
    ocdocker script --conf OCDocker.cfg examples/12_cli_script_example.py

All OCDocker modules are pre-loaded, so you can use them directly without imports.
"""

# All OCDocker modules are already available - no imports needed!
# For example:
# - ocl (OCDocker.Ligand)
# - ocr (OCDocker.Receptor)
# - ocvina (OCDocker.Docking.Vina)
# - ocsmina (OCDocker.Docking.Smina)
# - ocplants (OCDocker.Docking.PLANTS)
# - ocoddt (OCDocker.Rescoring.ODDT)
# - ocmolproc (OCDocker.Toolbox.MoleculeProcessing)
# - ocrmsdclust (OCDocker.Processing.Preprocessing.RMSDClustering)
# - All symbols from OCDocker.Initialise
# - Standard library: os, sys, Path, glob, pprint

# Example: Simple script that uses OCDocker functionality
print("OCDocker Script Example")
print("=" * 50)

# Check if we have access to OCDocker modules
try:
    print(f"- OCDocker.Ligand available: {ocl is not None}")
    print(f"- OCDocker.Receptor available: {ocr is not None}")
    print(f"- OCDocker.Docking.Vina available: {ocvina is not None}")
    print(f"- OCDocker.Rescoring.ODDT available: {ocoddt is not None}")
    print(f"- Standard library (os) available: {os is not None}")
    print(f"- Path available: {Path is not None}")
except NameError as e:
    print(f"! Error: {e}")
    print("Make sure you're running this with: ocdocker script --conf OCDocker.cfg script.py")
    sys.exit(1)

# Example: Access script arguments
print("\nScript Arguments:")
print(f"  Script file: {sys.argv[0]}")
if len(sys.argv) > 1:
    print(f"  Additional arguments: {sys.argv[1:]}")
else:
    print("  No additional arguments provided")

# Example: Check configuration
try:
    from OCDocker.Config import get_config
    config = get_config()
    print(f"\nConfiguration loaded:")
    print(f"  Multiprocess: {config.multiprocess}")
    print(f"  Output level: {config.output_level}")
except Exception as e:
    print(f"\nNote: Could not access config: {e}")

# Example: Simple workflow
# NOTE: Update the paths below to point to your own receptor, ligand, and box files
# The paths shown use the test_files directory structure - adjust to match your setup

print("\n" + "=" * 50)
print("Running Docking Workflow Example")
print("=" * 50)

try:
    # Define paths (update these to your own files if desired)
    # Use absolute paths to avoid working directory issues
    test_files_dir = Path("./test_files/test_ptn1").resolve()
    receptor_path = str(test_files_dir / "receptor.pdb")
    ligand_path = str(test_files_dir / "compounds/ligands/ligand/ligand.smi")
    box_path = str(test_files_dir / "compounds/ligands/ligand/boxes/box0.pdb")
    ligand_dir = str(test_files_dir / "compounds/ligands/ligand")
    
    # Check if files exist
    print("\nChecking input files...")
    if not Path(receptor_path).exists():
        print(f"  ! Receptor file not found: {receptor_path}")
        print("  Please update receptor_path to point to your receptor.pdb file")
        raise FileNotFoundError(f"Receptor file not found: {receptor_path}")
    else:
        print(f"  - Receptor file found: {receptor_path}")
    
    if not Path(ligand_path).exists():
        print(f"  ! Ligand file not found: {ligand_path}")
        print("  Please update ligand_path to point to your ligand file")
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
    else:
        print(f"  - Ligand file found: {ligand_path}")
    
    if not Path(box_path).exists():
        print(f"  ! Box file not found: {box_path}")
        print("  Please update box_path to point to your box.pdb file")
        raise FileNotFoundError(f"Box file not found: {box_path}")
    else:
        print(f"  - Box file found: {box_path}")
    
    # Create required directories
    vina_files_dir = Path(f"{ligand_dir}/vinaFiles")
    vina_files_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - Created/verified output directory: {vina_files_dir}")
    
    # Create receptor and ligand objects
    print("\nCreating receptor and ligand objects...")
    receptor = ocr.Receptor(receptor_path, name="test_receptor")
    ligand = ocl.Ligand(ligand_path, name="test_ligand")
    print("  - Objects created")
    
    # Create Vina docking object
    print("\nCreating Vina docking object...")
    vina = ocvina.Vina(
        config_path=f"{ligand_dir}/vinaFiles/conf_vina.txt",
        box_file=box_path,
        receptor=receptor,
        prepared_receptor_path=f"{test_files_dir}/prepared_receptor.pdbqt",
        ligand=ligand,
        prepared_ligand_path=f"{ligand_dir}/prepared_ligand.pdbqt",
        vina_log=f"{ligand_dir}/vinaFiles/ligand.log",
        output_vina=f"{ligand_dir}/vinaFiles/ligand.pdbqt",
        name="Vina test"
    )
    print("  - Vina object created")
    
    # Run docking workflow
    print("\nRunning docking workflow...")
    
    # Helper function to check return codes
    def check_result(result, step_name):
        """Check if a docking step succeeded."""
        if isinstance(result, tuple):
            return_code = result[0]
            if return_code != 0:
                error_msg = result[1] if len(result) > 1 else ""
                raise RuntimeError(f"{step_name} failed with return code {return_code}. {error_msg}")
        elif isinstance(result, int):
            if result != 0:
                raise RuntimeError(f"{step_name} failed with return code {result}")
        return result
    
    print("  - Preparing receptor...")
    result = vina.run_prepare_receptor()
    check_result(result, "Receptor preparation")
    print("  - Receptor prepared")
    
    print("  - Preparing ligand...")
    # Ensure output directory exists for prepared ligand
    Path(ligand_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if ligand is SMILES file - MGLTools needs MOL2, so convert if needed
    ligand_ext = Path(ligand_path).suffix.lower()
    if ligand_ext in ['.smi', '.smiles']:
        # Convert SMILES to MOL2 first (MGLTools requires MOL2)
        mol2_path = str(Path(ligand_dir) / "ligand.mol2")
        mol2_file = Path(mol2_path)
        if not mol2_file.exists() or mol2_file.stat().st_size == 0:
            print("  - Converting SMILES to MOL2 for MGLTools...")
            result_conv = occonversion.convert_mols(ligand_path, mol2_path, overwrite=True)
            if isinstance(result_conv, tuple) and result_conv[0] != 0:
                raise RuntimeError(f"Failed to convert SMILES to MOL2: {result_conv}")
            elif isinstance(result_conv, int) and result_conv != 0:
                raise RuntimeError(f"Failed to convert SMILES to MOL2: return code {result_conv}")
            if not mol2_file.exists():
                raise RuntimeError(f"MOL2 file was not created at {mol2_path}")
            print("  - Conversion completed")
        else:
            print("  - MOL2 file already exists, using it")
            # Verify the file is actually readable
            if not mol2_file.is_file():
                raise RuntimeError(f"MOL2 path exists but is not a file: {mol2_path}")
    
    # Use MGLTools for preparation (standard for Vina)
    result = vina.run_prepare_ligand()
    check_result(result, "Ligand preparation")
    print("  - Ligand prepared")
    
    print("  - Running docking...")
    result = vina.run_docking()
    check_result(result, "Docking")
    print("  - Docking completed")
    
    print("  - Splitting poses...")
    result = vina.split_poses(f"{ligand_dir}/vinaFiles")
    check_result(result, "Pose splitting")
    print("  - Poses split")
    
    print("\n- Docking workflow completed successfully!")
    
except FileNotFoundError as e:
    print(f"\n! File not found: {e}")
    print("\nTo fix this:")
    print("  1. Ensure test_files directory exists with the expected structure")
    print("  2. Or update the paths in the script to point to your own files")
    print("\nExpected structure:")
    print("  test_files/test_ptn1/")
    print("    ├── receptor.pdb")
    print("    └── compounds/ligands/ligand/")
    print("        ├── ligand.smi")
    print("        └── boxes/box0.pdb")
    print("\nNote: If your test_files has a different structure, update the")
    print("      test_files_dir variable in the script to match your setup.")
except RuntimeError as e:
    print(f"\n! Workflow step failed: {e}")
    print("\nCommon issues:")
    print("  - Ligand preparation: Ensure MGLTools or OpenBabel is properly configured")
    print("  - Docking: Check that Vina binary is accessible and config file is valid")
    print("  - Pose splitting: Ensure docking completed successfully first")
    print("\nCheck the log files in the vinaFiles directory for detailed error messages.")
except Exception as e:
    print(f"\n! Error during workflow execution: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 50)
print("Script completed successfully!")
print("\nTo use this script:")
print("  ocdocker script --conf OCDocker.cfg examples/12_cli_script_example.py [args...]")

