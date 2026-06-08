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
    python examples/11_python_api_complete_ocscore_pipeline.py
"""

###############################################################################
# USER CONFIGURATION - Update these variables to match your system
###############################################################################

# OCDocker configuration file path
# Set this to the absolute path of your OCDocker.cfg file
# If None, will use OCDOCKER_CONFIG environment variable or search for OCDocker.cfg
OCDOCKER_CONFIG_FILE = "/path/to/OCDocker.cfg"  # Update this path

# Receptor configuration
RECEPTOR_PATH = "/path/to/receptor_directory/receptor.pdb"
RECEPTOR_NAME = "Receptor"
PREPARED_RECEPTOR_PDBQT = "/path/to/receptor_directory/prepared_receptor.pdbqt"
PREPARED_RECEPTOR_MOL2 = "/path/to/receptor_directory/prepared_receptor.mol2"

BOX_CENTER = (0.0, 0.0, 0.0)  # (x, y, z) coordinates of the box center

# Ligand configuration - List of ligand paths (base directories)
# Each path should contain: {ligand_name}.smi, boxes/box0.pdb, and subdirectories for outputs
# Ligand names are automatically extracted from the last folder name in each path
LIGAND_PATHS = [
    ]

# Add more ligand paths here (you can use glob to get all ligand folders):
# "/path/to/ligand2",
# "/path/to/ligand3",

# Model configuration
MODEL_NAME = "OCScore"  # Name of your trained model (without extension)
MODELS_DIR = "OCScore_models"  # Directory containing models
PCA_MODEL_PATH = None  # Path to PCA model if used, e.g., f"{MODELS_DIR}/{MODEL_NAME}_pca.pkl"

# Preprocessing configuration (should match training settings)
SCALER = "standard"  # "standard" or "minmax"
INVERT_CONDITIONALLY = True
NORMALIZE = True
USE_MASK = True
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

# GPU configuration
USE_GPU = True  # Set to False to force CPU usage (useful if CUDA is not available or to avoid GPU memory issues)

# Output configuration
OUTPUT_FILE = "ocscore_results.csv"  # CSV file to save results (None to skip saving)
SAVE_TO_FILE = True  # Set to False to only store results in memory

# Multiprocessing configuration
N_JOBS = 4                  # Number of parallel jobs (cores) to use. Set to -1 for all available cores
USE_MULTIPROCESSING = True  # Set to False to process ligands sequentially

# Checkpoint/resume configuration
# When enabled, each ligand writes a per-ligand checkpoint and can resume from the
# last completed stage. This is backward-compatible with runs done before this code:
# if no checkpoint file exists, completion is inferred from existing output artifacts.
ENABLE_LIGAND_CHECKPOINT = True
LIGAND_CHECKPOINT_FILE = ".ocscore_pipeline_checkpoint.json"
LIGAND_FEATURES_CACHE_FILE = ".ocscore_pipeline_features.json"

###############################################################################
# END USER CONFIGURATION
###############################################################################

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from glob import glob
from typing import Any, Dict, Optional, Tuple

# Configure sklearn/joblib to use threading backend for parallel execution
# This allows sklearn models to use multiple threads while main process uses multiprocessing
# The threading backend avoids the "Loky-backed parallel loops cannot be called in multiprocessing" issue
warnings.filterwarnings('ignore', message='.*Loky-backed parallel loops cannot be called in a multiprocessing.*')

# Note: We keep the default multiprocessing start method ('fork' on Linux)
# which is faster and works well with proper tmp directory isolation

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

# Explicitly bootstrap OCDocker with the specified config file BEFORE other imports
# This ensures the config is loaded correctly regardless of working directory
# Set OCDOCKER_NO_AUTO_BOOTSTRAP to prevent auto-bootstrap from running first
os.environ['OCDOCKER_NO_AUTO_BOOTSTRAP'] = '1'

if OCDOCKER_CONFIG_FILE and os.path.isfile(OCDOCKER_CONFIG_FILE):
    import OCDocker.Error as ocerror
    import OCDocker.Initialise as init

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
import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina
import OCDocker.Ligand as ocl
import OCDocker.OCScore.Scoring as ocscoring
import OCDocker.OCScore.Utils.Data as ocscoredata
import OCDocker.OCScore.Utils.IO as ocscoreio
import OCDocker.Processing.Preprocessing.RMSDClustering as ocrmsdclust
import OCDocker.Receptor as ocr
import OCDocker.Rescoring.ODDT as ocoddt
import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
import OCDocker.Toolbox.Security as ocsec

# Allow unsafe deserialization
ocsec.allow_unsafe_runtime(deserialization=True, script_exec=False)

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
    print(f"Use mask: {USE_MASK}")
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
    scaler_path = f"{MODELS_DIR}/{MODEL_NAME}_scaler.pkl"  # Path to saved scaler
    
    # Load the mask if it exists
    mask = None
    if os.path.isfile(mask_path) and USE_MASK:
        try:
            mask = ocscoreio.load_mask(MODEL_NAME, models_dir=MODELS_DIR)
        except Exception as e:
            print(f"Warning: Could not load mask: {e}")
            mask = None
    
    # Check if scaler exists (required for proper normalization)
    if NORMALIZE and not os.path.isfile(scaler_path):
        print(f"WARNING: Scaler file not found at {scaler_path}")
        print("  This means normalization will use a NEW scaler fitted on prediction data,")
        print("  which is INCORRECT. The scaler should be saved during training.")
        print("  Predictions may be inaccurate!")
        scaler_path = None  # Will create new scaler (incorrect but won't crash)
    elif NORMALIZE:
        print(f"Using saved scaler from: {scaler_path}")
    
    # Get OCScore predictions for all ligands at once
    try:
        print(f"Running model inference on {len(feature_df)} ligands...")
        print(f"Feature DataFrame shape: {feature_df.shape}")
        
        ocscore_predictions = ocscoring.get_score(
            model_path=model_path,
            data=feature_df,
            pca_model=PCA_MODEL_PATH,
            mask=mask,
            score_columns_list=SCORE_COLUMNS_LIST,
            scaler=SCALER,
            scaler_path=scaler_path if NORMALIZE else None,  # Use saved scaler if normalization is enabled
            invert_conditionally=INVERT_CONDITIONALLY,
            normalize=NORMALIZE,
            serialization_method="auto",  # Auto-detect model format
            use_gpu=USE_GPU  # Use GPU if available and USE_GPU=True
        )
        
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
    
    # Reorder columns to match the data source order (from training data file)
    # !!! CRITICAL: This ensures all columns (especially SFs) are in the exact same order
    # as the training data, which is essential for proper mask application and model inference!
    if not results_df.empty:
        # Get the column order from config (no file path needed)
        # Uses reference_column_order from OCDocker.cfg
        source_order = ocscoredata.get_column_order()  # Uses config by default
        
        # Use the reorder function to match the config column order
        # This handles OCSCORE insertion and extra columns automatically
        results_df = ocscoredata.reorder_columns_to_match_data_order(
            df=results_df,
            data_source=None,  # Uses config.reference_column_order by default
            keep_extra_columns=True,  # Keep OCSCORE and any other extra columns
            fill_missing_columns=False  # Don't add missing columns as NaN
        )
        
        # Manually insert OCSCORE right after 'ligand' if it exists
        if 'OCSCORE' in results_df.columns:
            cols = list(results_df.columns)
            if 'ligand' in cols:
                # Remove OCSCORE from current position
                cols.remove('OCSCORE')
                # Insert after 'ligand'
                ligand_idx = cols.index('ligand')
                cols.insert(ligand_idx + 1, 'OCSCORE')
                results_df = results_df[cols]
    
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


# Mapping function to convert rescoring keys to database column names
def map_rescoring_key_to_db_column(key: str, engine: Optional[str] = None) -> str:
    '''Map rescoring result keys to database column names.
    
    Parameters
    ----------
    key : str
        The key from rescoring results (e.g., 'vina_vina_rescoring', 'smina_vinardo_rescoring', etc.)
    engine : str
        The engine name (e.g., 'vina', 'smina')
        If None, the engine will be inferred from the key.
        If provided, the engine will be used to determine the database column name.
        If not provided, the engine will be inferred from the key.

    Returns
    -------
    str
        The database column name (e.g., 'VINA_VINA', 'SMINA_VINARDO', etc.)
    '''

    key_lower = key.lower()
    
    # Mapping dictionary for rescoring keys to database column names
    # !!! CRITICAL: These must match SCORING_FUNCTION_ORDER for mask application !!!
    mapping = {
        # VINA mappings
        'vina_vina_rescoring': 'VINA_VINA',
        'vina_vinardo_rescoring': 'VINA_VINARDO',
        # SMINA mappings - MUST match SCORING_FUNCTION_ORDER
        'smina_vina_rescoring': 'SMINA_VINA',
        'smina_vinardo_rescoring': 'SMINA_VINARDO',
        'smina_dkoes_scoring_rescoring': 'SMINA_SCORING_DKOES',
        'smina_scoring_dkoes_rescoring': 'SMINA_SCORING_DKOES',  # Alternative format
        'smina_old_scoring_dkoes_rescoring': 'SMINA_OLD_SCORING_DKOES',
        'smina_fast_dkoes_rescoring': 'SMINA_FAST_DKOES',
        'smina_ad4_scoring_rescoring': 'SMINA_SCORING_AD4',
        # Handle alternative SMINA key formats (from actual rescoring output)
        'smina_dkoes_fast': 'SMINA_FAST_DKOES',
        'smina_dkoes_scoring_old': 'SMINA_OLD_SCORING_DKOES',
        'smina_dkoes_fast_rescoring': 'SMINA_FAST_DKOES',
        'smina_dkoes_scoring_old_rescoring': 'SMINA_OLD_SCORING_DKOES',
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
    
    # Handle new format: rescoring_{scoring_function}_{pose_number} or rescoring_{pose_number}
    if key_lower.startswith('rescoring_'):
        # Extract scoring function and pose number
        # Format: rescoring_{scoring_function}_{pose_number} or rescoring_{pose_number}
        parts = key_lower.split('_')
        if len(parts) >= 2:
            # Check if last part is a number (pose number)
            if parts[-1].isdigit():
                pose_number = parts[-1]
                # Remove 'rescoring' and pose number, rest is scoring function
                scoring_function_parts = parts[1:-1]
            else:
                # No pose number, just scoring function after 'rescoring'
                scoring_function_parts = parts[1:]
            
            if scoring_function_parts:
                # Reconstruct scoring function name
                scoring_function = '_'.join(scoring_function_parts)
                
                # Use provided engine if available, otherwise try to detect
                engines_to_try = [engine] if engine else ['vina', 'smina']
                
                # Try to match with known formats
                for eng in engines_to_try:
                    test_key_old = f'{eng}_{scoring_function}_rescoring'
                    test_key_new = f'{eng}_{scoring_function}'
                    if test_key_old in mapping:
                        return mapping[test_key_old]
                    if test_key_new in mapping:
                        return mapping[test_key_new]
                
                # If not found in mapping, construct based on engine or scoring function
                if engine:
                    # Use provided engine
                    if engine == 'vina':
                        return f'VINA_{scoring_function.upper()}'
                    elif engine == 'smina':
                        sf_mapping = {
                            'dkoes_scoring': 'SCORING_DKOES',
                            'scoring_dkoes': 'SCORING_DKOES',
                            'old_scoring_dkoes': 'OLD_SCORING_DKOES',
                            'dkoes_scoring_old': 'OLD_SCORING_DKOES',
                            'fast_dkoes': 'FAST_DKOES',
                            'ad4_scoring': 'SCORING_AD4',
                        }
                        if scoring_function in sf_mapping:
                            return f'SMINA_{sf_mapping[scoring_function]}'
                        return f'SMINA_{scoring_function.upper()}'
                else:
                    # Try to detect engine from scoring function
                    if scoring_function in ['vina', 'vinardo']:
                        return f'VINA_{scoring_function.upper()}'
                    else:
                        # Assume smina for other scoring functions
                        sf_mapping = {
                            'dkoes_scoring': 'SCORING_DKOES',
                            'scoring_dkoes': 'SCORING_DKOES',
                            'old_scoring_dkoes': 'OLD_SCORING_DKOES',
                            'dkoes_scoring_old': 'OLD_SCORING_DKOES',
                            'fast_dkoes': 'FAST_DKOES',
                            'ad4_scoring': 'SCORING_AD4',
                        }
                        if scoring_function in sf_mapping:
                            return f'SMINA_{sf_mapping[scoring_function]}'
                        return f'SMINA_{scoring_function.upper()}'
            else:
                # Just 'rescoring_{pose_number}' - no scoring function specified
                # Use engine if provided, otherwise default to vina_vina
                if engine == 'smina':
                    return 'SMINA_VINARDO'  # Default SMINA scoring function
                return 'VINA_VINA'  # Default VINA scoring function
    
    # Handle old format: VINA/SMINA rescoring keys (remove _rescoring suffix if present)
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
            # !!! CRITICAL: These must match SCORING_FUNCTION_ORDER !!!
            sf_mapping = {
                'dkoes_scoring': 'SCORING_DKOES',
                'scoring_dkoes': 'SCORING_DKOES',  # Alternative format
                'old_scoring_dkoes': 'OLD_SCORING_DKOES',
                'dkoes_scoring_old': 'OLD_SCORING_DKOES',  # Alternative format
                'fast_dkoes': 'FAST_DKOES',
                'dkoes_fast': 'FAST_DKOES',  # Alternative format
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


CHECKPOINT_STAGE_DOCKING = "docking"
CHECKPOINT_STAGE_RESCORING = "rescoring"
CHECKPOINT_STAGE_FEATURES = "features"


def _checkpoint_path(ligand_path: str) -> str:
    return os.path.join(ligand_path, LIGAND_CHECKPOINT_FILE)


def _features_cache_path(ligand_path: str) -> str:
    return os.path.join(ligand_path, LIGAND_FEATURES_CACHE_FILE)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp_path, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _default_checkpoint(ligand_name: str) -> Dict[str, Any]:
    return {
        "version": 1,
        "ligand": ligand_name,
        "status": "new",
        "completed_stages": [],
        "stage_data": {},
        "failed_stage": "",
        "last_error": "",
        "updated_at": time.time(),
    }


def _load_checkpoint(ligand_path: str, ligand_name: str) -> Dict[str, Any]:
    checkpoint = _default_checkpoint(ligand_name)
    if not ENABLE_LIGAND_CHECKPOINT:
        return checkpoint

    path = _checkpoint_path(ligand_path)
    if not os.path.isfile(path):
        return checkpoint

    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        return checkpoint

    if not isinstance(loaded, dict):
        return checkpoint

    checkpoint["status"] = str(loaded.get("status", checkpoint["status"]))
    checkpoint["failed_stage"] = str(loaded.get("failed_stage", ""))
    checkpoint["last_error"] = str(loaded.get("last_error", ""))
    checkpoint["updated_at"] = loaded.get("updated_at", checkpoint["updated_at"])

    completed = loaded.get("completed_stages", [])
    if isinstance(completed, list):
        checkpoint["completed_stages"] = sorted({str(stage) for stage in completed})

    stage_data = loaded.get("stage_data", {})
    if isinstance(stage_data, dict):
        checkpoint["stage_data"] = stage_data

    return checkpoint


def _save_checkpoint(ligand_path: str, checkpoint: Dict[str, Any]) -> None:
    if not ENABLE_LIGAND_CHECKPOINT:
        return
    checkpoint["updated_at"] = time.time()
    _write_json_atomic(_checkpoint_path(ligand_path), checkpoint)


def _is_stage_complete(checkpoint: Dict[str, Any], stage: str) -> bool:
    completed = checkpoint.get("completed_stages", [])
    return stage in completed if isinstance(completed, list) else False


def _clear_stage(checkpoint: Dict[str, Any], stage: str) -> None:
    completed = checkpoint.get("completed_stages", [])
    if isinstance(completed, list):
        checkpoint["completed_stages"] = sorted([st for st in completed if st != stage])
    stage_data = checkpoint.get("stage_data", {})
    if isinstance(stage_data, dict):
        stage_data.pop(stage, None)
    checkpoint["status"] = "in_progress"


def _mark_stage_complete(checkpoint: Dict[str, Any], stage: str, stage_data: Optional[Dict[str, Any]] = None) -> None:
    completed = checkpoint.get("completed_stages", [])
    completed_set = set(completed if isinstance(completed, list) else [])
    completed_set.add(stage)
    checkpoint["completed_stages"] = sorted(completed_set)
    checkpoint["status"] = "in_progress" if stage != CHECKPOINT_STAGE_FEATURES else "completed"
    checkpoint["failed_stage"] = ""
    checkpoint["last_error"] = ""
    if stage_data:
        checkpoint.setdefault("stage_data", {})[stage] = stage_data


def _mark_stage_failed(checkpoint: Dict[str, Any], stage: str, message: str) -> None:
    checkpoint["status"] = "failed"
    checkpoint["failed_stage"] = stage
    checkpoint["last_error"] = message


def _load_cached_features(ligand_path: str) -> Optional[Dict[str, Any]]:
    path = _features_cache_path(ligand_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _save_cached_features(ligand_path: str, features: Dict[str, Any]) -> None:
    if not ENABLE_LIGAND_CHECKPOINT:
        return
    _write_json_atomic(_features_cache_path(ligand_path), features)


def _normalize_sf_names(rescoring_result: Dict[str, Any]) -> Dict[str, Any]:
    sf_name_corrections = {
        "SMINA_DKOES_FAST": "SMINA_FAST_DKOES",
        "SMINA_DKOES_SCORING_OLD": "SMINA_OLD_SCORING_DKOES",
        "SMINA_SCORING_DKOES_OLD": "SMINA_OLD_SCORING_DKOES",
        "SMINA_FAST_DKOES_RESCORING": "SMINA_FAST_DKOES",
        "SMINA_OLD_SCORING_DKOES_RESCORING": "SMINA_OLD_SCORING_DKOES",
    }
    for old_name, correct_name in sf_name_corrections.items():
        if old_name in rescoring_result and correct_name not in rescoring_result:
            rescoring_result[correct_name] = rescoring_result.pop(old_name)
    return rescoring_result


def _value_from_rescore_entry(value: Any) -> Any:
    if isinstance(value, list) and len(value) > 0:
        return value[0]
    return value


def _add_plants_scores_to_rescoring_result(plants_rescoring: Dict[str, Any], rescoring_result: Dict[str, Any]) -> None:
    for outer_key, inner_dict in plants_rescoring.items():
        db_column_name = map_rescoring_key_to_db_column(outer_key)

        if isinstance(inner_dict, dict):
            if "PLANTS_TOTAL_SCORE" in inner_dict:
                total_score = inner_dict["PLANTS_TOTAL_SCORE"]
                rescoring_result[db_column_name] = _value_from_rescore_entry(total_score)
            else:
                print(
                    f"Warning: PLANTS_TOTAL_SCORE not found in inner dict for {outer_key}. "
                    f"Available keys: {list(inner_dict.keys())}"
                )
        else:
            rescoring_result[db_column_name] = _value_from_rescore_entry(inner_dict)


def _read_rescoring_outputs(
    ligand_path: str,
    ligand_name: str,
    vina_ligand: ocvina.Vina,
    plants_ligand: ocplants.PLANTS,
    smina_ligand: ocsmina.Smina,
) -> Optional[Dict[str, Any]]:
    rescoring_result: Dict[str, Any] = {}

    oddt_csv = os.path.join(ligand_path, "oddt", f"{ligand_name}.csv")
    if not os.path.isfile(oddt_csv):
        return None
    try:
        oddt_df = pd.read_csv(oddt_csv)
        oddt_dict = ocoddt.df_to_dict(oddt_df)
        if not oddt_dict:
            return None
        first_key = list(oddt_dict.keys())[0]
        oddt_scores = oddt_dict[first_key]
        if not isinstance(oddt_scores, dict):
            return None
        for key, value in oddt_scores.items():
            db_column_name = map_rescoring_key_to_db_column(f"oddt_{key}")
            rescoring_result[db_column_name] = value
    except Exception:
        return None

    try:
        plants_rescoring = plants_ligand.read_rescore_logs(f"{ligand_path}/plantsFiles")
    except Exception:
        return None
    if not plants_rescoring:
        return None
    _add_plants_scores_to_rescoring_result(plants_rescoring, rescoring_result)

    try:
        vina_rescoring = vina_ligand.read_rescore_logs(f"{ligand_path}/vinaFiles/rescoring")
    except Exception:
        return None
    if not vina_rescoring:
        return None
    for key, value in vina_rescoring.items():
        db_column_name = map_rescoring_key_to_db_column(key, engine="vina")
        rescoring_result[db_column_name] = _value_from_rescore_entry(value)

    try:
        smina_rescoring = smina_ligand.read_rescore_logs(f"{ligand_path}/sminaFiles/rescoring")
    except Exception:
        return None
    if not smina_rescoring:
        return None
    for key, value in smina_rescoring.items():
        db_column_name = map_rescoring_key_to_db_column(key, engine="smina")
        rescoring_result[db_column_name] = _value_from_rescore_entry(value)

    return _normalize_sf_names(rescoring_result)


def _run_docking_stage(ligand_path: str, vina_ligand: ocvina.Vina, plants_ligand: ocplants.PLANTS) -> Tuple[list, list]:
    # Vina
    vina_ligand.run_prepare_receptor(overwrite=True)
    vina_ligand.run_prepare_ligand(overwrite=True)
    vina_ligand.run_docking(overwrite=True)
    vina_ligand.split_poses()

    vina_poses_dir = os.path.dirname(vina_ligand.output_vina) if hasattr(vina_ligand, "output_vina") else f"{ligand_path}/vinaFiles"
    vina_pattern = f"{vina_poses_dir}/*_split_*.pdbqt"
    pose_files_found = False
    for _ in range(50):
        found_files = glob(vina_pattern)
        if found_files and wait_for_files_ready(found_files, max_wait=2.0):
            pose_files_found = True
            break
        time.sleep(0.2)
    if not pose_files_found:
        print("Warning: No stable pose files found for Vina after waiting, proceeding anyway...")
    time.sleep(0.5)
    vina_poses = vina_ligand.get_docked_poses()
    if vina_poses and not wait_for_files_ready(vina_poses, max_wait=5.0):
        print("Warning: Some Vina pose files may not be fully ready, but proceeding...")

    # PLANTS
    plants_ligand.run_prepare_receptor(overwrite=True)
    plants_ligand.run_prepare_ligand(overwrite=True)
    plants_ligand.run_docking(overwrite=True)

    plants_output_dir = plants_ligand.output_plants if hasattr(plants_ligand, "output_plants") else f"{ligand_path}/plantsFiles"
    plants_run_dir = os.path.join(plants_output_dir, "run")
    plants_pattern = f"{plants_run_dir}/*.mol2"
    plants_files_found = False
    for _ in range(100):
        found_files = [
            f for f in glob(plants_pattern)
            if not f.endswith("_protein.mol2") and not f.endswith("_fixed.mol2")
        ]
        if found_files and wait_for_files_ready(found_files, max_wait=2.0):
            plants_files_found = True
            break
        time.sleep(0.2)
    if not plants_files_found:
        print("Warning: No stable PLANTS output files found after waiting, proceeding anyway...")

    time.sleep(0.5)
    time.sleep(0.5)
    plants_poses = plants_ligand.get_docked_poses()
    if plants_poses and not wait_for_files_ready(plants_poses, max_wait=3.0):
        print("Warning: Some PLANTS pose files may not be fully ready, but proceeding...")

    if not vina_poses or not plants_poses:
        raise ValueError("Docking did not generate valid pose files for both Vina and PLANTS.")

    return vina_poses, plants_poses


def _load_existing_docking_poses(vina_ligand: ocvina.Vina, plants_ligand: ocplants.PLANTS) -> Tuple[list, list]:
    vina_poses = vina_ligand.get_docked_poses()
    plants_poses = plants_ligand.get_docked_poses()
    if vina_poses:
        _ = wait_for_files_ready(vina_poses, max_wait=2.0)
    if plants_poses:
        _ = wait_for_files_ready(plants_poses, max_wait=2.0)
    return vina_poses, plants_poses


def _select_representative_medoid(
    ligand_path: str,
    ligand_name: str,
    vina_poses: list,
    plants_poses: list,
    vina_ligand: ocvina.Vina,
    plants_ligand: ocplants.PLANTS,
) -> str:
    poses_list = vina_poses + plants_poses

    valid_poses = []
    max_retries = 5
    retry_delay = 0.3
    for pose_file in poses_list:
        validated = False
        for attempt in range(max_retries):
            if validate_molecule_file(pose_file):
                valid_poses.append(pose_file)
                validated = True
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
        if not validated:
            print(f"Warning: Could not validate pose file {pose_file} after {max_retries} attempts, skipping.")

    if not valid_poses:
        raise ValueError(f"No valid pose files found for ligand {ligand_name} after validation")
    if len(valid_poses) < 2:
        raise ValueError(f"Need at least 2 valid poses for RMSD calculation, found {len(valid_poses)} for ligand {ligand_name}")

    mol2_poses_dir = f"{ligand_path}/poses_mol2"
    os.makedirs(mol2_poses_dir, exist_ok=True)
    mol2_poses = []
    mol2_to_original_map = {}

    for pose_file in valid_poses:
        pose_ext = os.path.splitext(pose_file)[1].lower()
        pose_basename = os.path.basename(pose_file)

        if pose_ext == ".mol2":
            mol2_poses.append(pose_file)
            mol2_to_original_map[pose_file] = pose_file
        else:
            mol2_path = os.path.join(mol2_poses_dir, f"{os.path.splitext(pose_basename)[0]}.mol2")
            occonversion.convert_mols(pose_file, mol2_path, overwrite=True)
            if wait_for_file_stable(mol2_path, max_wait=3.0):
                mol2_poses.append(mol2_path)
                mol2_to_original_map[mol2_path] = pose_file
            else:
                print(f"Warning: Could not convert {pose_file} to MOL2 format, skipping for RMSD calculation")

    if len(mol2_poses) < 2:
        raise ValueError(f"Need at least 2 valid MOL2 poses for RMSD calculation, found {len(mol2_poses)} for ligand {ligand_name}")
    if not wait_for_files_ready(mol2_poses, max_wait=5.0):
        print("Warning: Some MOL2 pose files may not be fully ready for RMSD calculation, but proceeding...")
    time.sleep(0.5)

    rmsd_matrix = ocmolproc.get_rmsd_matrix(mol2_poses)

    pose_engine_map = {}
    for mol2_pose in mol2_poses:
        original_pose = mol2_to_original_map.get(mol2_pose, mol2_pose)
        if original_pose in vina_poses:
            pose_engine_map[mol2_pose] = "vina"
        elif original_pose in plants_poses:
            pose_engine_map[mol2_pose] = "plants"

    clusters = ocrmsdclust.cluster_rmsd(
        rmsd_matrix,
        algorithm="agglomerativeClustering",
        outputPlot=f"{ligand_path}/medoids.png",
        pose_engine_map=pose_engine_map,
    )
    medoids_mol2 = ocrmsdclust.get_medoids(rmsd_matrix, clusters, onlyBiggest=True)
    medoids = [mol2_to_original_map.get(medoid_mol2, medoid_mol2) for medoid_mol2 in medoids_mol2]

    medoids_dict = {}
    for medoid in medoids:
        if medoid in vina_poses:
            medoids_dict[medoid] = vina_ligand.read_log(onlyBest=False)[ocvina.get_pose_index_from_file_path(medoid)]
        elif medoid in plants_poses:
            medoids_dict[medoid] = plants_ligand.read_log(onlyBest=False)[ocplants.get_pose_index_from_file_path(medoid)]

    if not medoids_dict:
        raise ValueError(f"Could not determine medoid for ligand {ligand_name}")

    return list(medoids_dict.keys())[0]


def _run_rescoring_stage(
    ligand_path: str,
    ligand_name: str,
    ligand: ocl.Ligand,
    medoid: str,
    vina_ligand: ocvina.Vina,
    plants_ligand: ocplants.PLANTS,
    smina_ligand: ocsmina.Smina,
) -> Dict[str, Any]:
    oddt_outdir = os.path.join(ligand_path, "oddt")
    try:
        from joblib import parallel_backend
        with parallel_backend("threading"):
            _ = ocoddt.run_oddt(
                vina_ligand.prepared_receptor,
                medoid,
                ligand.name,
                oddt_outdir,
                overwrite=True,
            )
    except ImportError:
        _ = ocoddt.run_oddt(
            vina_ligand.prepared_receptor,
            medoid,
            ligand.name,
            oddt_outdir,
            overwrite=True,
        )

    medoid_extension = os.path.splitext(medoid)[1].lower()
    if medoid_extension != ".mol2":
        plants_input = medoid.replace(medoid_extension, ".mol2")
        occonversion.convert_mols(medoid, plants_input, overwrite=True)
    else:
        plants_input = medoid

    plants_pose_list = f"{ligand_path}/plantsFiles/plants_pose_list.txt"
    ocplants.write_pose_list(plants_input, plants_pose_list)
    plants_ligand.run_rescore(plants_pose_list, logFile="", overwrite=True)
    time.sleep(0.3)

    if medoid_extension != ".pdbqt":
        vina_smina_input = medoid.replace(medoid_extension, ".pdbqt")
        occonversion.convert_mols(medoid, vina_smina_input, overwrite=True)
    else:
        vina_smina_input = medoid

    vina_ligand.run_rescore(
        f"{ligand_path}/vinaFiles/rescoring",
        vina_smina_input,
        overwrite=True,
        splitLigand=False,
    )
    smina_ligand.run_rescore(
        f"{ligand_path}/sminaFiles/rescoring",
        vina_smina_input,
        overwrite=True,
        splitLigand=False,
    )

    rescoring_result = _read_rescoring_outputs(ligand_path, ligand_name, vina_ligand, plants_ligand, smina_ligand)
    if rescoring_result is None:
        raise ValueError(f"Rescoring outputs are incomplete or invalid for ligand {ligand_name}")
    return rescoring_result


def _infer_completed_stages(
    ligand_path: str,
    ligand_name: str,
    checkpoint: Dict[str, Any],
    vina_ligand: ocvina.Vina,
    plants_ligand: ocplants.PLANTS,
    smina_ligand: ocsmina.Smina,
) -> None:
    if _is_stage_complete(checkpoint, CHECKPOINT_STAGE_FEATURES) and _is_stage_complete(checkpoint, CHECKPOINT_STAGE_RESCORING):
        return

    cached_features = _load_cached_features(ligand_path)
    if isinstance(cached_features, dict) and not _is_stage_complete(checkpoint, CHECKPOINT_STAGE_FEATURES):
        _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_FEATURES)

    rescoring_result = _read_rescoring_outputs(ligand_path, ligand_name, vina_ligand, plants_ligand, smina_ligand)
    if rescoring_result is not None:
        if not _is_stage_complete(checkpoint, CHECKPOINT_STAGE_RESCORING):
            _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_RESCORING)
        if not _is_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING):
            _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING)
        return

    vina_poses, plants_poses = _load_existing_docking_poses(vina_ligand, plants_ligand)
    if vina_poses and plants_poses and not _is_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING):
        _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING)


def process_single_ligand(ligand_path: str, ligand_name: str, receptor: ocr.Receptor) -> Optional[dict]:
    ''' Process a single ligand through the complete OCScore pipeline with checkpoint/resume support.
    
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
    dict | None
        Dictionary containing all features and OCScore prediction. None if processing fails.
    '''

    checkpoint: Optional[Dict[str, Any]] = None
    current_stage = "initialization"

    try:
        ligand = ocl.Ligand(f"{ligand_path}/{ligand_name}.smi", name=ligand_name)
        ligand.create_box(centroid=BOX_CENTER, save_path=f"{ligand_path}/boxes/")

        vina_ligand = ocvina.Vina(
            f"{ligand_path}/vinaFiles/conf_vina.txt",
            f"{ligand_path}/boxes/box0.pdb",
            receptor, PREPARED_RECEPTOR_PDBQT,
            ligand, f"{ligand_path}/prepared_ligand.pdbqt",
            f"{ligand_path}/vinaFiles/vina.log", f"{ligand_path}/vinaFiles/vina.pdbqt",
            name=f"Vina {receptor.name}-{ligand_name}",
        )
        plants_ligand = ocplants.PLANTS(
            f"{ligand_path}/plantsFiles/conf_plants.txt",
            f"{ligand_path}/boxes/box0.pdb",
            receptor, PREPARED_RECEPTOR_MOL2,
            ligand, f"{ligand_path}/prepared_ligand.mol2",
            f"{ligand_path}/plantsFiles/plants.log", f"{ligand_path}/plantsFiles",
            name=f"Plants {receptor.name}-{ligand_name}",
        )
        smina_ligand = ocsmina.Smina(
            f"{ligand_path}/sminaFiles/conf_smina.txt",
            f"{ligand_path}/boxes/box0.pdb",
            receptor, PREPARED_RECEPTOR_PDBQT,
            ligand, f"{ligand_path}/prepared_ligand.pdbqt",
            f"{ligand_path}/sminaFiles/smina.log", f"{ligand_path}/sminaFiles/smina.pdbqt",
            name=f"Smina {receptor.name}-{ligand_name}",
        )

        current_stage = "checkpoint"
        checkpoint = _load_checkpoint(ligand_path, ligand_name)
        resume_enabled = ENABLE_LIGAND_CHECKPOINT
        if resume_enabled:
            _infer_completed_stages(ligand_path, ligand_name, checkpoint, vina_ligand, plants_ligand, smina_ligand)
            _save_checkpoint(ligand_path, checkpoint)

        if resume_enabled and _is_stage_complete(checkpoint, CHECKPOINT_STAGE_FEATURES):
            cached_features = _load_cached_features(ligand_path)
            if isinstance(cached_features, dict):
                print(f"[{ligand_name}] Resume: using cached features.")
                return cached_features
            _clear_stage(checkpoint, CHECKPOINT_STAGE_FEATURES)
            _save_checkpoint(ligand_path, checkpoint)

        rescoring_result: Optional[Dict[str, Any]] = None
        if resume_enabled and _is_stage_complete(checkpoint, CHECKPOINT_STAGE_RESCORING):
            rescoring_result = _read_rescoring_outputs(ligand_path, ligand_name, vina_ligand, plants_ligand, smina_ligand)
            if rescoring_result is None:
                print(f"[{ligand_name}] Checkpoint says rescoring complete, but outputs are missing/corrupted. Recomputing.")
                _clear_stage(checkpoint, CHECKPOINT_STAGE_RESCORING)
                _clear_stage(checkpoint, CHECKPOINT_STAGE_FEATURES)
                _save_checkpoint(ligand_path, checkpoint)
            else:
                print(f"[{ligand_name}] Resume: using existing rescoring outputs.")

        if rescoring_result is None:
            vina_poses: list = []
            plants_poses: list = []

            if resume_enabled and _is_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING):
                vina_poses, plants_poses = _load_existing_docking_poses(vina_ligand, plants_ligand)
                if not vina_poses or not plants_poses:
                    print(f"[{ligand_name}] Checkpoint says docking complete, but poses are missing. Re-running docking.")
                    _clear_stage(checkpoint, CHECKPOINT_STAGE_DOCKING)
                    _save_checkpoint(ligand_path, checkpoint)
                else:
                    print(f"[{ligand_name}] Resume: using existing docking outputs.")

            if not vina_poses or not plants_poses:
                current_stage = CHECKPOINT_STAGE_DOCKING
                vina_poses, plants_poses = _run_docking_stage(ligand_path, vina_ligand, plants_ligand)
                _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_DOCKING)
                _save_checkpoint(ligand_path, checkpoint)

            current_stage = CHECKPOINT_STAGE_RESCORING
            medoid = _select_representative_medoid(
                ligand_path,
                ligand_name,
                vina_poses,
                plants_poses,
                vina_ligand,
                plants_ligand,
            )
            rescoring_result = _run_rescoring_stage(
                ligand_path,
                ligand_name,
                ligand,
                medoid,
                vina_ligand,
                plants_ligand,
                smina_ligand,
            )
            _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_RESCORING, stage_data={"medoid": medoid})
            _save_checkpoint(ligand_path, checkpoint)

        current_stage = CHECKPOINT_STAGE_FEATURES
        receptor_descriptors = receptor.get_descriptors()
        ligand_descriptors = ligand.get_descriptors()

        all_features: Dict[str, Any] = {}
        all_features.update(rescoring_result if rescoring_result is not None else {})
        all_features.update(receptor_descriptors)
        all_features.update(ligand_descriptors)
        all_features["name"] = f"{receptor.name}_{ligand.name}"
        all_features["receptor"] = receptor.name
        all_features["ligand"] = ligand.name

        _save_cached_features(ligand_path, all_features)
        _mark_stage_complete(checkpoint, CHECKPOINT_STAGE_FEATURES, stage_data={"feature_count": len(all_features)})
        _save_checkpoint(ligand_path, checkpoint)
        return all_features

    except Exception as e:
        if checkpoint is not None:
            try:
                _mark_stage_failed(checkpoint, current_stage, str(e))
                _save_checkpoint(ligand_path, checkpoint)
            except Exception:
                pass
        print(f"Error processing ligand {ligand_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_molecule_file(file_path: str) -> bool:
    '''Validate that a molecule file can be loaded and is complete.
    
    Parameters
    ----------
    file_path : str
        Path to the molecule file
    
    Returns
    -------
    bool
        True if file is valid and can be loaded
    '''

    from OCDocker.Toolbox import Validation as ocvalidation
    
    # First check if file is stable (not being written)
    if not wait_for_file_stable(file_path, max_wait=2.0):
        return False
    
    # Then validate the molecule structure
    try:
        return ocvalidation.is_molecule_valid(file_path)
    except Exception:
        return False


def wait_for_file_stable(file_path: str, max_wait: float = 5.0, check_interval: float = 0.1) -> bool:
    '''Wait for a file to stabilize (size stops changing).
    
    Parameters
    ----------
    file_path : str
        Path to the file to check
    max_wait : float
        Maximum time to wait in seconds
    check_interval : float
        Time between checks in seconds
    
    Returns
    -------
    bool
        True if file stabilized, False if timeout
    '''

    if not os.path.isfile(file_path):
        return False
    
    start_time = time.time()
    last_size = -1
    stable_count = 0
    required_stable_checks = 3  # File must be stable for 3 consecutive checks
    
    while time.time() - start_time < max_wait:
        try:
            current_size = os.path.getsize(file_path)
            
            if current_size == last_size:
                stable_count += 1
                if stable_count >= required_stable_checks:
                    return True
            else:
                stable_count = 0
                last_size = current_size
            
            time.sleep(check_interval)
        except (OSError, IOError):
            # File might be locked or deleted
            time.sleep(check_interval)
            continue
    
    return False


def wait_for_files_ready(file_paths: list, max_wait: float = 8.0, check_interval: float = 0.2) -> bool:
    '''Wait for all files in a list to exist, be stable, and be readable.
    
    Parameters
    ----------
    file_paths : list
        List of file paths to wait for
    max_wait : float
        Maximum time to wait in seconds
    check_interval : float
        Time between checks in seconds
    
    Returns
    -------
    bool
        True if all files are ready, False if timeout
    '''
    
    if not file_paths:
        return True
    
    start_time = time.time()
    ready_files = set()
    
    while time.time() - start_time < max_wait:
        all_ready = True
        for file_path in file_paths:
            if file_path in ready_files:
                continue
                
            if wait_for_file_stable(file_path, max_wait=check_interval * 2, check_interval=check_interval / 2):
                ready_files.add(file_path)
            else:
                all_ready = False
        
        if all_ready and len(ready_files) == len(file_paths):
            return True
        
        time.sleep(check_interval)
    
    return len(ready_files) == len(file_paths)


if __name__ == "__main__":
    results = main()
