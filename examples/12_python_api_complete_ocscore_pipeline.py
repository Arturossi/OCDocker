#!/usr/bin/env python3
"""
Example: Complete OCScore Pipeline

This example demonstrates the complete pipeline to obtain OCScore results from scratch:
1. Receptor and ligand preparation
2. Docking with multiple engines (Vina, PLANTS)
3. Pose clustering to find representative poses
4. Rescoring with multiple scoring functions (ODDT, PLANTS, Vina, SMINA)
5. Feature extraction (receptor and ligand descriptors)
6. Model inference using trained OCScore model

The rescoring results are automatically mapped to database column names for consistency.

Usage:
    # Update paths in the script to match your system
    python examples/12_python_api_complete_ocscore_pipeline.py
"""

###############################################################################
# USER CONFIGURATION - Update these variables to match your system
###############################################################################

# OCDocker configuration file path
# Set this to the absolute path of your OCDocker.cfg file
# If None, will use OCDOCKER_CONFIG environment variable or search for OCDocker.cfg
OCDOCKER_CONFIG_FILE = "/data/hd4tb/OCDocker/OCDocker/OCDocker.cfg"  # Update this path

# Receptor configuration
RECEPTOR_PATH = "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/receptor.pdb"
RECEPTOR_NAME = "Receptor"
PREPARED_RECEPTOR_PDBQT = "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/prepared_receptor.pdbqt"
PREPARED_RECEPTOR_MOL2 = "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/prepared_receptor.mol2"

# Ligand configuration - List of ligand paths (base directories)
# Each path should contain: {ligand_name}.smi, boxes/box0.pdb, and subdirectories for outputs
# Ligand names are automatically extracted from the last folder name in each path
LIGAND_PATHS = [
    "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/compounds/ligands/ligand",
    "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/compounds/decoys/ZINC000000000015",
    "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/compounds/decoys/ZINC000000000024",
    "/data/hd4tb/OCDocker/OCDocker/test_files/test_ptn1/compounds/decoys/ZINC000000000030"
    # Add more ligand paths here (you can use glob to get all ligand folders):
    # "/path/to/ligand2",
    # "/path/to/ligand3",
]

# Model configuration
MODEL_NAME = "OCScore"  # Name of your trained model (without extension)
MODELS_DIR = "OCScore_models"  # Directory containing models
PCA_MODEL_PATH = None  # Path to PCA model if used, e.g., f"{MODELS_DIR}/{MODEL_NAME}_pca.pkl"

# Preprocessing configuration (should match training settings)
SCALER = "standard"  # "standard" or "minmax"
INVERT_CONDITIONALLY = True
NORMALIZE = True
SCORE_COLUMNS_LIST = ["SMINA", "VINA", "ODDT", "PLANTS"]

###############################################################################
# !!! CRITICAL WARNING: SCORING FUNCTION COLUMN ORDER !!!
###############################################################################
# 
# THE FOLLOWING ORDER MUST BE STRICTLY RESPECTED WHEN APPLYING MASKS:
# 
#   1. SMINA_VINA
#   2. SMINA_SCORING_DKOES
#   3. SMINA_VINARDO
#   4. SMINA_OLD_SCORING_DKOES
#   5. SMINA_FAST_DKOES
#   6. SMINA_SCORING_AD4
#   7. VINA_VINA
#   8. VINA_VINARDO
#   9. PLANTS_CHEMPLP
#  10. PLANTS_PLP
#  11. PLANTS_PLP95
#  12. ODDT_RFSCORE_V1
#  13. ODDT_RFSCORE_V2
#  14. ODDT_RFSCORE_V3
#  15. ODDT_PLECRF_P5_L1_S65536
#  16. ODDT_NNSCORE
# 
# !!! WARNING: If you change the order of scoring function columns in the output,
#   the mask will be applied incorrectly, leading to wrong predictions!
# 
# The mask is a 16-element array where each position corresponds to one of the
# scoring functions above in the exact order listed. Position 0 = SMINA_VINA,
# position 1 = SMINA_SCORING_DKOES, etc.
# 
# DO NOT MODIFY THE ORDER OF SCORING FUNCTION COLUMNS WITHOUT UPDATING THE MASK!
# 
###############################################################################

# Define the correct order of scoring function columns (for mask application)
SCORING_FUNCTION_ORDER = [
    'SMINA_VINA',
    'SMINA_SCORING_DKOES',
    'SMINA_VINARDO',
    'SMINA_OLD_SCORING_DKOES',
    'SMINA_FAST_DKOES',
    'SMINA_SCORING_AD4',
    'VINA_VINA',
    'VINA_VINARDO',
    'PLANTS_CHEMPLP',
    'PLANTS_PLP',
    'PLANTS_PLP95',
    'ODDT_RFSCORE_V1',
    'ODDT_RFSCORE_V2',
    'ODDT_RFSCORE_V3',
    'ODDT_PLECRF_P5_L1_S65536',
    'ODDT_NNSCORE',
]

# GPU configuration
USE_GPU = True  # Set to False to force CPU usage (useful if CUDA is not available or to avoid GPU memory issues)

# Output configuration
OUTPUT_FILE = "ocscore_results.csv"  # CSV file to save results (None to skip saving)
SAVE_TO_FILE = True  # Set to False to only store results in memory

# Multiprocessing configuration
N_JOBS = 1  # Number of parallel jobs (cores) to use. Set to -1 for all available cores
USE_MULTIPROCESSING = True  # Set to False to process ligands sequentially

###############################################################################
# END USER CONFIGURATION
###############################################################################

# Imports
import os
import numpy as np
import pandas as pd
import argparse
import OCDocker.Initialise as init
import OCDocker.Error as ocerror

# Explicitly bootstrap OCDocker with the specified config file BEFORE other imports
# This ensures the config is loaded correctly regardless of working directory
# Set OCDOCKER_NO_AUTO_BOOTSTRAP to prevent auto-bootstrap from running first
os.environ['OCDOCKER_NO_AUTO_BOOTSTRAP'] = '1'

if OCDOCKER_CONFIG_FILE and os.path.isfile(OCDOCKER_CONFIG_FILE):
    print(f"Loading OCDocker configuration from: {OCDOCKER_CONFIG_FILE}")
    bootstrap_ns = argparse.Namespace(
        multiprocess=USE_MULTIPROCESSING,
        update=False,
        config_file=OCDOCKER_CONFIG_FILE,
        output_level=ocerror.ReportLevel.WARNING,
        overwrite=False
    )
    init.bootstrap(bootstrap_ns)
    print("OCDocker configuration loaded successfully.\n")
else:
    # Fall back to auto-bootstrap if config file not specified or not found
    if OCDOCKER_CONFIG_FILE:
        print(f"Warning: Config file not found at {OCDOCKER_CONFIG_FILE}, using auto-bootstrap...")
    # Re-enable auto-bootstrap
    os.environ.pop('OCDOCKER_NO_AUTO_BOOTSTRAP', None)

# Now import other OCDocker modules (they won't trigger auto-bootstrap since we already bootstrapped)
import OCDocker.Receptor as ocr
import OCDocker.Ligand as ocl
import OCDocker.Docking.Vina as ocvina
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Processing.Preprocessing.RmsdClustering as ocrmsdclust
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.OCScore.Scoring as ocscoring
import OCDocker.OCScore.Utils.IO as ocscoreio

# Configure sklearn/joblib to use threading backend for parallel execution
# This allows sklearn models to use multiple threads while main process uses multiprocessing
# The threading backend avoids the "Loky-backed parallel loops cannot be called in multiprocessing" issue
import warnings
warnings.filterwarnings('ignore', message='.*Loky-backed parallel loops cannot be called in a multiprocessing.*')

try:
    import joblib
    from joblib import parallel_backend, Parallel, delayed
    
    # Set default backend to threading so sklearn can parallelize within multiprocessing workers
    # Threading backend works inside multiprocessing contexts (unlike Loky)
    joblib.parallel.DEFAULT_BACKEND = 'threading'
    JOBLIB_AVAILABLE = True
except (ImportError, AttributeError):
    # If joblib not available, try setting environment variable
    os.environ['JOBLIB_BACKEND'] = 'threading'
    JOBLIB_AVAILABLE = False
    USE_MULTIPROCESSING = False
    print("Warning: joblib not available. Multiprocessing disabled.")

# Mapping function to convert rescoring keys to database column names
def map_rescoring_key_to_db_column(key: str) -> str:
    '''Map rescoring result keys to database column names.
    
    Parameters
    ----------
    key : str
        The key from rescoring results (e.g., 'vina_vina_rescoring', 'smina_vinardo_rescoring', etc.)
    
    Returns
    -------
    str
        The database column name (e.g., 'VINA_VINA', 'SMINA_VINARDO', etc.)
    '''

    key_lower = key.lower()
    
    # Mapping dictionary for rescoring keys to database column names
    mapping = {
        # VINA mappings
        'vina_vina_rescoring': 'VINA_VINA',
        'vina_vinardo_rescoring': 'VINA_VINARDO',
        # SMINA mappings
        'smina_vina_rescoring': 'SMINA_VINA',
        'smina_vinardo_rescoring': 'SMINA_VINARDO',
        'smina_dkoes_scoring_rescoring': 'SMINA_SCORING_DKOES',
        'smina_old_scoring_dkoes_rescoring': 'SMINA_OLD_SCORING_DKOES',
        'smina_fast_dkoes_rescoring': 'SMINA_FAST_DKOES',
        'smina_ad4_scoring_rescoring': 'SMINA_SCORING_AD4',
        # PLANTS mappings
        'plants_chemplp': 'PLANTS_CHEMPLP',
        'plants_plp': 'PLANTS_PLP',
        'plants_plp95': 'PLANTS_PLP95',
        # ODDT mappings (these come from the dataframe columns, already prefixed with oddt_)
        'oddt_rfscore_v1': 'ODDT_RFSCORE_V1',
        'oddt_rfscore_v2': 'ODDT_RFSCORE_V2',
        'oddt_rfscore_v3': 'ODDT_RFSCORE_V3',
        'oddt_plecrf_p5_l1_s65536': 'ODDT_PLECRF_P5_L1_S65536',
        'oddt_plec_p5_l1_s65536': 'ODDT_PLECRF_P5_L1_S65536',  # Alternative naming
        'oddt_nnscore': 'ODDT_NNSCORE',
    }
    
    # Check if exact match exists
    if key_lower in mapping:
        return mapping[key_lower]
    
    # Handle ODDT keys that are already prefixed with oddt_
    if key_lower.startswith('oddt_'):
        # Remove oddt_ prefix
        inner_key = key_lower[5:]  # Remove 'oddt_'
        # Try to match with known ODDT patterns
        if inner_key.startswith('rfscore_v'):
            # Extract version number (handle both 'rfscore_v1' and 'rfscorev1' formats)
            version = inner_key.replace('rfscore_v', '').replace('rfscorev', '')
            return f'ODDT_RFSCORE_V{version.upper()}'
        elif 'plec' in inner_key.lower():
            # Handle PLEC variations (case-insensitive)
            inner_key_lower = inner_key.lower()
            if 'p5_l1_s65536' in inner_key_lower or 'p5l1s65536' in inner_key_lower:
                return 'ODDT_PLECRF_P5_L1_S65536'
        elif inner_key.lower() == 'nnscore':
            return 'ODDT_NNSCORE'
    
    # Handle VINA/SMINA rescoring keys (remove _rescoring suffix if present)
    if key_lower.endswith('_rescoring'):
        key_without_suffix = key_lower[:-10]  # Remove '_rescoring'
        if key_without_suffix in mapping:
            return mapping[key_without_suffix]
        # Try to construct the mapping
        if key_without_suffix.startswith('vina_'):
            sf = key_without_suffix.replace('vina_', '')
            return f'VINA_{sf.upper()}'
        elif key_without_suffix.startswith('smina_'):
            sf = key_without_suffix.replace('smina_', '')
            # Handle special SMINA scoring function names
            sf_mapping = {
                'dkoes_scoring': 'SCORING_DKOES',
                'old_scoring_dkoes': 'OLD_SCORING_DKOES',
                'fast_dkoes': 'FAST_DKOES',
                'ad4_scoring': 'SCORING_AD4',
                'vina': 'VINA',
                'vinardo': 'VINARDO',
            }
            if sf in sf_mapping:
                return f'SMINA_{sf_mapping[sf]}'
            else:
                return f'SMINA_{sf.upper()}'
    
    # If no mapping found, return uppercase version of the key
    return key.upper()


def process_single_ligand(ligand_path: str, ligand_name: str, receptor: ocr.Receptor) -> dict:
    ''' Process a single ligand through the complete OCScore pipeline.
    
    Parameters
    ----------
    ligand_path : str
        Base path to the ligand directory
    ligand_name : str
        Name of the ligand (without extension)
    receptor : ocr.Receptor
        Receptor object
    
    Returns
    -------
    dict
        Dictionary containing all features and OCScore prediction
    '''

    try:
        # Ligand creation
        ligand = ocl.Ligand(f"{ligand_path}/{ligand_name}.smi", name=ligand_name)
        
        ####################### VINA #########################
        
        # Create object
        vina_ligand = ocvina.Vina(
            f"{ligand_path}/vinaFiles/conf_vina.txt", 
            f"{ligand_path}/boxes/box0.pdb", 
            receptor, PREPARED_RECEPTOR_PDBQT, 
            ligand, f"{ligand_path}/prepared_ligand.pdbqt", 
            f"{ligand_path}/vinaFiles/vina.log", f"{ligand_path}/vinaFiles/vina.pdbqt", 
            name=f"Vina {receptor.name}-{ligand_name}"
        )
        
        # Prepare receptor
        vina_ligand.run_prepare_receptor()
        
        # Prepare ligand
        vina_ligand.run_prepare_ligand()
        
        # Run docking
        vina_ligand.run_docking()
        
        # Get the docked poses for vina
        vina_ligand.split_poses()
        vinaPoses = vina_ligand.get_docked_poses()
        
        ####################### PLANTS #########################
        
        # Create object
        plants_ligand = ocplants.PLANTS(
            f"{ligand_path}/plantsFiles/conf_plants.txt", 
            f"{ligand_path}/boxes/box0.pdb", 
            receptor, PREPARED_RECEPTOR_MOL2, 
            ligand, f"{ligand_path}/prepared_ligand.pdbqt", 
            f"{ligand_path}/plantsFiles/plants.log", f"{ligand_path}/plantsFiles", 
            name=f"Plants {receptor.name}-{ligand_name}"
        )
        
        # Prepare receptor
        plants_ligand.run_prepare_receptor()
        
        # Prepare ligand
        plants_ligand.run_prepare_ligand()
        
        # Run docking
        plants_ligand.run_docking(overwrite=True)
        
        # Get the docked poses for plants
        plantsPoses = plants_ligand.get_docked_poses()
        
        ####################### SMINA #########################
        
        # Create object
        smina_ligand = ocsmina.Smina(
            f"{ligand_path}/sminaFiles/conf_smina.txt", 
            f"{ligand_path}/boxes/box0.pdb", 
            receptor, PREPARED_RECEPTOR_PDBQT, 
            ligand, f"{ligand_path}/prepared_ligand.pdbqt", 
            f"{ligand_path}/sminaFiles/smina.log", f"{ligand_path}/sminaFiles/smina.pdbqt", 
            name=f"Smina {receptor.name}-{ligand_name}"
        )
        
        #################### Clustering #######################
        
        # Make them one single list
        poses_list = vinaPoses + plantsPoses
        
        # Get the rmsd matrix from the poses list
        rmsdMatrix = ocmolproc.get_rmsd_matrix(poses_list)
        
        # Get the clusters
        clusters = ocrmsdclust.cluster_rmsd(
            rmsdMatrix, 
            algorithm='agglomerativeClustering', 
            outputPlot=f"{ligand_path}/medoids.png"
        )
        
        # Get the medoids (The plot is just for visualization, it is not required)
        medoids = ocrmsdclust.get_medoids(
            rmsdMatrix, 
            clusters, 
            onlyBiggest=True
        )
        
        # Dictionary with the medoids and its docking method (to be correctly parsed by the next function)
        medoidsDict = {}
        
        ## Find which medoid has the lowest energy
        # For each medoid
        for medoid in medoids:
            # Check if it is contained in vinaPoses list
            if medoid in vinaPoses:
                # Add it to the medoidsDict as a list with vina as the key
                medoidsDict[medoid] = vina_ligand.read_log(onlyBest=False)[ocvina.get_pose_index_from_file_path(medoid)]
            # Check if it is contained in plantsPoses list
            elif medoid in plantsPoses:
                # Add it to the medoidsDict as a list with plants as the key
                medoidsDict[medoid] = plants_ligand.read_log(onlyBest=False)[ocplants.get_pose_index_from_file_path(medoid)]
        
        ################ ODDT RESCORING ################
        
        # Initialize rescoring
        rescoringResult = {}
        
        # Run ODDT and get the result as a dataframe
        # Use threading backend for sklearn models to allow parallelization in multiprocessing context
        try:
            from joblib import parallel_backend
            with parallel_backend('threading'):
                df = ocoddt.run_oddt(
                    vina_ligand.prepared_receptor, 
                    list(medoidsDict.keys())[0], 
                    ligand.name, 
                    f"{vina_ligand.get_input_ligand_path()}/oddt",
                    overwrite=True
                )
        except ImportError:
            # Fallback if joblib not available
            df = ocoddt.run_oddt(
                vina_ligand.prepared_receptor, 
                list(medoidsDict.keys())[0], 
                ligand.name, 
                f"{vina_ligand.get_input_ligand_path()}/oddt",
                overwrite=True
            )
        
        # If you want a dict, you can convert with this function
        dt = ocoddt.df_to_dict(df)
        
        # Add ODDT results to rescoring dictionary
        for key in dt[list(dt.keys())[0]].keys():
            db_column_name = map_rescoring_key_to_db_column(f"oddt_{key}")
            rescoringResult[db_column_name] = dt[list(dt.keys())[0]][key]
        
        # If needed, convert the medoid to the proper format for vina/smina
        medoid = list(medoidsDict.keys())[0]
        medoid_extension = os.path.splitext(medoid)[1].lower()
        
        if medoid_extension != ".mol2":
            # Change the output file extension to mol2
            outfile = medoid.replace(medoid_extension, ".mol2")
            # Convert the medoid to the proper format (any to mol2)
            occonversion.convert_mols(medoid, outfile, overwrite=True)
        else:
            outfile = medoid
        
        ocplants.write_pose_list(outfile, f"{ligand_path}/plantsFiles/plants_pose_list.txt")
        
        # Run the rescoring (will create the config file and the output folder)
        plants_ligand.run_rescore(
            f"{ligand_path}/plantsFiles/plants_pose_list.txt", 
            logFile="", 
            overwrite=True
        )
        
        # Get PLANTS rescoring results and map to database column names
        # PLANTS read_rescore_logs returns Dict[str, Dict[str, float]] where:
        # - Outer key: "plants_{scoring_function}" (e.g., "plants_chemplp")
        # - Inner dict: Contains PLANTS score keys (e.g., "PLANTS_TOTAL_SCORE", "PLANTS_SCORE_RB_PEN", etc.)
        # For each PLANTS scoring function, we extract the PLANTS_TOTAL_SCORE value from the inner dict
        plants_rescoring = plants_ligand.read_rescore_logs(f"{ligand_path}/plantsFiles")
        for outer_key, inner_dict in plants_rescoring.items():
            # Map the outer key (e.g., "plants_chemplp") to database column name (e.g., "PLANTS_CHEMPLP")
            db_column_name = map_rescoring_key_to_db_column(outer_key)
            
            if isinstance(inner_dict, dict):
                # Extract PLANTS_TOTAL_SCORE from the inner dict
                if "PLANTS_TOTAL_SCORE" in inner_dict:
                    total_score = inner_dict["PLANTS_TOTAL_SCORE"]
                    # Extract numeric value if it's in a list
                    if isinstance(total_score, list) and len(total_score) > 0:
                        rescoringResult[db_column_name] = total_score[0]
                    elif isinstance(total_score, (int, float)):
                        rescoringResult[db_column_name] = total_score
                    else:
                        print(f"Warning: PLANTS_TOTAL_SCORE for {outer_key} has non-numeric value: {total_score} (type: {type(total_score)})")
                else:
                    print(f"Warning: PLANTS_TOTAL_SCORE not found in inner dict for {outer_key}. Available keys: {list(inner_dict.keys())}")
            else:
                # Fallback: if inner_dict is not a dict, try to use it directly
                if isinstance(inner_dict, list) and len(inner_dict) > 0:
                    rescoringResult[db_column_name] = inner_dict[0]
                else:
                    rescoringResult[db_column_name] = inner_dict
        
        if medoid_extension != ".pdbqt":
            # Change the output file extension to pdbqt
            outfile = medoid.replace(medoid_extension, ".pdbqt")
            # Convert the medoid to the proper format (any to pdbqt)
            occonversion.convert_mols(medoid, outfile)
        else:
            outfile = medoid
        
        # Run the rescoring with vina
        vina_ligand.run_rescore(
            f'{ligand_path}/vinaFiles/rescoring',
            outfile,
            overwrite=True,
            splitLigand=False
        )
        
        # Get VINA rescoring results and map to database column names
        vina_rescoring = vina_ligand.read_rescore_logs(f"{ligand_path}/vinaFiles/rescoring")
        for key, value in vina_rescoring.items():
            db_column_name = map_rescoring_key_to_db_column(key)
            rescoringResult[db_column_name] = value
        
        # Run the rescoring with smina
        smina_ligand.run_rescore(
            f"{ligand_path}/sminaFiles/rescoring", 
            outfile,
            overwrite=True,
            splitLigand=False
        )
        
        # Get SMINA rescoring results and map to database column names
        smina_rescoring = smina_ligand.read_rescore_logs(f"{ligand_path}/sminaFiles/rescoring")
        for key, value in smina_rescoring.items():
            db_column_name = map_rescoring_key_to_db_column(key)
            rescoringResult[db_column_name] = value

        ####################### FEATURE EXTRACTION #########################
        
        # Get receptor descriptors
        receptorDescriptors = receptor.get_descriptors()
        
        # Get ligand descriptors
        ligandDescriptors = ligand.get_descriptors()
        
        # Combine all features into a single dictionary
        all_features = {}
        all_features.update(rescoringResult)  # Add rescoring results
        all_features.update(receptorDescriptors)  # Add receptor descriptors
        all_features.update(ligandDescriptors)  # Add ligand descriptors
        
        # Add metadata
        all_features['name'] = f"{receptor.name}_{ligand.name}"
        all_features['receptor'] = receptor.name
        all_features['ligand'] = ligand.name
        
        # Store features for batch prediction (don't call get_score here)
        # We'll batch all ligands together for proper normalization
        return all_features
        
    except Exception as e:
        print(f"Error processing ligand {ligand_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    '''Main function to process all ligands.'''
    
    # OCDocker auto-bootstraps on import, so configuration is already loaded
    # If you need to verify bootstrap or use custom settings, you can:
    # 1. Set OCDOCKER_NO_AUTO_BOOTSTRAP=1 environment variable
    # 2. Import OCDocker.Initialise and call bootstrap() explicitly
    
    # Automatically derive ligand names from paths (last folder name)
    ligand_names = [os.path.basename(os.path.normpath(path)) for path in LIGAND_PATHS]
    
    # Create receptor object
    receptor = ocr.Receptor(RECEPTOR_PATH, name=RECEPTOR_NAME)
    
    # Prepare list of (ligand_path, ligand_name) tuples
    ligand_tasks = list(zip(LIGAND_PATHS, ligand_names))
    
    print(f"\n{'='*60}")
    print(f"OCSCORE PIPELINE")
    print(f"{'='*60}")
    print(f"Receptor: {RECEPTOR_NAME}")
    print(f"Number of ligands: {len(ligand_tasks)}")
    print(f"Multiprocessing: {USE_MULTIPROCESSING}")
    if USE_MULTIPROCESSING:
        print(f"Number of jobs: {N_JOBS}")
    print(f"{'='*60}\n")
    
    # Process ligands
    if USE_MULTIPROCESSING and JOBLIB_AVAILABLE and len(ligand_tasks) > 1:
        # Use joblib for parallel processing
        print(f"Processing {len(ligand_tasks)} ligands in parallel using {N_JOBS} cores...")
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_single_ligand)(ligand_path, ligand_name, receptor)
            for ligand_path, ligand_name in ligand_tasks
        )
    else:
        # Process sequentially
        print(f"Processing {len(ligand_tasks)} ligands sequentially...")
        results = []
        for ligand_path, ligand_name in ligand_tasks:
            print(f"Processing ligand: {ligand_name}")
            result = process_single_ligand(ligand_path, ligand_name, receptor)
            results.append(result)
    
    # Filter out None results (failed processing)
    results = [r for r in results if r is not None]
    
    if not results:
        print("No ligands were successfully processed.")
        return
    
    # Batch all ligands together for model inference
    # This ensures proper normalization (scaler fit on all data, not single rows)
    print(f"\n{'='*60}")
    print(f"BATCH MODEL INFERENCE")
    print(f"{'='*60}")
    
    # Convert all results to a single DataFrame
    if results:
        # Get all unique keys from all dictionaries
        all_keys = set()
        for result in results:
            if result is not None:
                all_keys.update(result.keys())
        
        # Ensure all dictionaries have all keys (fill missing with None)
        normalized_results = []
        for result in results:
            if result is not None:
                normalized_result = {key: result.get(key, None) for key in all_keys}
                normalized_results.append(normalized_result)
        
        # Create DataFrame from normalized dictionaries
        feature_df = pd.DataFrame(normalized_results)
    else:
        feature_df = pd.DataFrame()
    
    if feature_df.empty:
        print("No features to process for model inference.")
        return
    
    # Path to your trained model
    model_path = f"{MODELS_DIR}/{MODEL_NAME}.pt"
    mask_path = f"{MODELS_DIR}/{MODEL_NAME}_mask.pkl"
    
    # Load the mask if it exists
    mask = None
    if os.path.isfile(mask_path):
        try:
            mask = ocscoreio.load_mask(MODEL_NAME, models_dir=MODELS_DIR)
        except Exception as e:
            print(f"Warning: Could not load mask: {e}")
            mask = None
    
    # Get OCScore predictions for all ligands at once
    try:
        print(f"Running model inference on {len(feature_df)} ligands...")
        print(f"Feature DataFrame shape: {feature_df.shape}")
        print(f"Feature DataFrame columns (first 10): {list(feature_df.columns[:10])}")
        
        # Check if features differ between ligands
        if len(feature_df) > 1:
            sample_cols = [col for col in feature_df.columns if any(col.startswith(prefix) for prefix in ['VINA_', 'SMINA_', 'PLANTS_', 'ODDT_'])][:5]
            if sample_cols:
                print(f"\nSample SF values for first 2 ligands:")
                for idx in range(min(2, len(feature_df))):
                    print(f"  Ligand {idx} ({feature_df.iloc[idx].get('ligand', 'unknown')}): {feature_df[sample_cols].iloc[idx].to_dict()}")
        
        ocscore_predictions = ocscoring.get_score(
            model_path=model_path,
            data=feature_df,
            pca_model=PCA_MODEL_PATH,
            mask=mask,
            score_columns_list=SCORE_COLUMNS_LIST,
            scaler=SCALER,
            invert_conditionally=INVERT_CONDITIONALLY,
            normalize=NORMALIZE,
            serialization_method="auto",  # Auto-detect model format
            use_gpu=USE_GPU  # Use GPU if available and USE_GPU=True
        )
        
        # Debug: Print prediction details
        print(f"\nPrediction type: {type(ocscore_predictions)}")
        if isinstance(ocscore_predictions, pd.DataFrame):
            print(f"Prediction DataFrame shape: {ocscore_predictions.shape}")
            print(f"Prediction DataFrame columns: {list(ocscore_predictions.columns)}")
            if 'predicted_score' in ocscore_predictions.columns:
                print(f"All predicted_score values: {ocscore_predictions['predicted_score'].tolist()}")
                print(f"Unique predicted_score values: {ocscore_predictions['predicted_score'].nunique()}")
        elif isinstance(ocscore_predictions, (pd.Series, np.ndarray)):
            predictions_array = np.asarray(ocscore_predictions)
            print(f"Prediction array shape: {predictions_array.shape}")
            print(f"All prediction values: {predictions_array.tolist()}")
            print(f"Unique prediction values: {len(np.unique(predictions_array))}")
        
        # Add OCScore predictions to results
        if isinstance(ocscore_predictions, pd.DataFrame):
            if 'predicted_score' in ocscore_predictions.columns:
                # Map predictions back to results by index
                for idx, result in enumerate(results):
                    if result is not None and idx < len(ocscore_predictions):
                        result['OCSCORE'] = ocscore_predictions['predicted_score'].iloc[idx]
                        print(f"  Mapped prediction {idx} to ligand {result.get('ligand', 'unknown')}: {result['OCSCORE']}")
            elif len(ocscore_predictions.columns) == 1:
                # Single prediction column
                for idx, result in enumerate(results):
                    if result is not None and idx < len(ocscore_predictions):
                        result['OCSCORE'] = ocscore_predictions.iloc[idx, 0]
            else:
                # Try to find a numeric column
                numeric_cols = ocscore_predictions.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    for idx, result in enumerate(results):
                        if result is not None and idx < len(ocscore_predictions):
                            result['OCSCORE'] = ocscore_predictions[numeric_cols[0]].iloc[idx]
        elif isinstance(ocscore_predictions, (pd.Series, np.ndarray)):
            # Array/Series of predictions
            predictions_array = np.asarray(ocscore_predictions)
            for idx, result in enumerate(results):
                if result is not None and idx < len(predictions_array):
                    result['OCSCORE'] = float(predictions_array[idx])
        
        print(f"Model inference completed for {len(results)} ligands.")
        
    except FileNotFoundError as e:
        print(f"Warning: Model file not found: {e}")
        for result in results:
            if result is not None:
                result['OCSCORE'] = None
    except Exception as e:
        print(f"Error during model inference: {e}")
        import traceback
        traceback.print_exc()
        for result in results:
            if result is not None:
                result['OCSCORE'] = None
    
    # Convert results to DataFrame
    # Use orient='index' and transpose to preserve order, then convert properly
    # First, ensure all dictionaries have the same keys (fill missing with None)
    if results:
        # Get all unique keys from all dictionaries
        all_keys = set()
        for result in results:
            if result is not None:
                all_keys.update(result.keys())
        
        # Ensure all dictionaries have all keys (fill missing with None)
        normalized_results = []
        for result in results:
            if result is not None:
                normalized_result = {key: result.get(key, None) for key in all_keys}
                normalized_results.append(normalized_result)
            else:
                normalized_results.append({key: None for key in all_keys})
        
        # Create DataFrame from normalized dictionaries
        results_df = pd.DataFrame(normalized_results)
    else:
        results_df = pd.DataFrame()
    
    # Reorder columns: name, receptor, ligand, OCSCORE, then SFs, then other features
    if not results_df.empty:
        # Get all columns from the DataFrame
        all_cols = list(results_df.columns)
        ordered_cols = []
        
        # 1. Start with name, receptor, ligand (if they exist) - MUST be first
        metadata_cols = ['name', 'receptor', 'ligand']
        for col in metadata_cols:
            if col in all_cols:
                ordered_cols.append(col)
        
        # 2. Add OCSCORE right after ligand
        if 'OCSCORE' in all_cols:
            ordered_cols.append('OCSCORE')
        
        # 3. Add all scoring function columns (SFs) in the CORRECT ORDER
        # ⚠️ CRITICAL: The order must match SCORING_FUNCTION_ORDER for mask application!
        # See the warning comment at the top of the file for the required order.
        sf_prefixes = ['VINA_', 'SMINA_', 'PLANTS_', 'ODDT_']
        
        # First, collect all SF columns
        all_sf_cols = []
        for col in all_cols:
            # Exclude metadata columns and OCSCORE from SFs
            if col not in ordered_cols and any(col.startswith(prefix) for prefix in sf_prefixes):
                all_sf_cols.append(col)
        
        # Order SF columns according to SCORING_FUNCTION_ORDER
        # First, add columns in the specified order (if they exist)
        ordered_sf_cols = []
        for sf_name in SCORING_FUNCTION_ORDER:
            if sf_name in all_sf_cols:
                ordered_sf_cols.append(sf_name)
        
        # Then, add any remaining SF columns that weren't in the predefined order
        # (in case there are additional SFs not in the standard list)
        remaining_sf_cols = [col for col in all_sf_cols if col not in ordered_sf_cols]
        remaining_sf_cols.sort()  # Sort alphabetically for any extras
        ordered_sf_cols.extend(remaining_sf_cols)
        
        ordered_cols.extend(ordered_sf_cols)
        
        # 4. Add all remaining columns (descriptors, etc.) - exclude already added columns
        remaining_cols = [col for col in all_cols if col not in ordered_cols]
        remaining_cols.sort()  # Sort alphabetically for consistency
        ordered_cols.extend(remaining_cols)
        
        # Reorder the DataFrame - ensure all columns are included and in the correct order
        # Only include columns that actually exist in the DataFrame
        existing_ordered_cols = [col for col in ordered_cols if col in results_df.columns]
        results_df = results_df[existing_ordered_cols]
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(results)}/{len(ligand_tasks)} ligands")
    if 'OCSCORE' in results_df.columns:
        print(f"OCScore predictions: {results_df['OCSCORE'].notna().sum()}/{len(results_df)}")
        if results_df['OCSCORE'].notna().any():
            print(f"OCScore range: {results_df['OCSCORE'].min():.4f} - {results_df['OCSCORE'].max():.4f}")
    print(f"{'='*60}\n")
    
    # Save to file if requested
    if SAVE_TO_FILE and OUTPUT_FILE:
        output_path = OUTPUT_FILE
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    return results_df


if __name__ == "__main__":
    results = main()
