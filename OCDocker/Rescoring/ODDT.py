#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions to perform rescoring of docking results using the ODDT.

They are imported as:

import OCDocker.Rescoring.ODDT as ocoddt
'''

# Imports
###############################################################################
import os
import six
import traceback

import oddt as od
import pandas as pd

from glob import glob
from typing import Dict, List, Tuple, Union, Optional

from oddt.scoring import scorer
from oddt.virtualscreening import virtualscreening as vs

from OCDocker.Config import get_config
import OCDocker.Error as ocerror

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Toolbox.Running as ocrun

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##
def __build_cmd(receptorPath: str, ligandPath: str, outputFile: str) -> Union[List[str], int]:
    '''Builds the command to run ODDT.

    Parameters
    ----------
    receptorPath : str
        The path to the receptor file.
    ligandPath : str
        The path to the ligand file.
    outputFile : str
        The path to the output file.

    Returns
    -------
    List[str] | int
        The command to run ODDT or an error code (based on the Error.py code table).
    '''

    # Check if the output file is a csv
    if not outputFile.endswith(".csv"):
        return ocerror.Error.unsupported_extension("The output file must be a csv file.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Extract ligand file format from extension
    ligand_ext = os.path.splitext(ligandPath)[1]
    if ligand_ext.startswith('.'):
        ligand_format = ligand_ext[1:]  # Remove leading dot
    else:
        ligand_format = ligand_ext

    config = get_config()

    # Start building the command
    # Use configured ODDT CLI program from Initialise
    cmd = [config.oddt.executable, ligandPath, "-O", outputFile, "--receptor", receptorPath, "-i", ligand_format, "-n", "1"]

    # Check if there are scoring functions to be used
    if isinstance(config.oddt.scoring_functions, list) and len(config.oddt.scoring_functions) > 0:
        # Add the scoring functions
        for score in config.oddt.scoring_functions:
            cmd.append("--score")
            cmd.append(score)
    else:
        ocprint.print_error("No scoring functions were provided to ODDT. Please check your configuration file.")

    return cmd


## Public ##
def get_models(outputPath: str) -> List[str]:
    '''Get the models from the output path.

    Parameters
    ----------
    outputPath : str
        The path to the output folder.

    Returns
    -------
    List[str]
        A list with the paths to the models.
    '''

    # Get the models
    models = glob(f"{outputPath}/*.pickle")

    return models


def run_oddt_from_cli(receptor: Union[ocr.Receptor, str], ligand: Union[ocl.Ligand, str], outputPath: str, overwrite: bool = False, logFile: str = "", cleanModels: bool = False) -> Union[int, Tuple[int, str]]:
    '''Run ODDT using the oddt_cli command. UNSTABLE FUNCTION DO NOT USE.

    Parameters
    ----------
    receptor : ocr.Receptor | str
        The receptor to be used in the docking.
    ligand : ocl.Ligand | str
        The ligand to be used in the docking.
    outputPath : str
        The path where the output file will be saved.
    overwrite : bool, optional
        If True, the output file will be overwritten. The default is False.
    logFile : str, optional
        The path to the log file. The default is "" (no log file).
    cleanModels : bool, optional
        If True, the models will be deleted after the rescoring. The default is False. If set to False, this can speed up the rescoring process for multiple ligands.
    
    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table).   
    '''

    # Check if the output dir exists
    if not os.path.isdir(outputPath):
        return ocerror.Error.dir_not_exist(f"The output directory '{outputPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor is an ocr.Receptor object
    if isinstance(receptor, ocr.Receptor):
        # Get the receptor path
        receptorPath = receptor.path
    # Check if the receptor is a string
    elif isinstance(receptor, str):
        # Get the receptor path
        receptorPath = receptor
    else:
        return ocerror.Error.wrong_type(f"The receptor must be a string or an ocr.Receptor object. The type {type(receptor)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the ligand is an ocl.Ligand object
    if isinstance(ligand, ocl.Ligand):
        # Get the ligand path
        ligandPath = ligand.path
        # Output file name
        outputFile = f"{outputPath}/{ligand.name}.csv"
    # Check if the ligand is a string
    elif isinstance(ligand, str):
        # Get the ligand path
        ligandPath = ligand
        # Get the ligand name from the path
        ligandName = ".".join(os.path.basename(ligandPath).split(".")[:-1])
        # Output file name
        outputFile = f"{outputPath}/{ligandName}.csv"
    else:
        return ocerror.Error.wrong_type(f"The ligand must be a string or an ocl.Ligand object. The type {type(ligand)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the output file exists
    if os.path.isfile(outputFile) and not overwrite:
        return ocerror.Error.file_exists(f"The output file '{outputFile}' already exists. Please use the overwrite option if you want to overwrite it.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor exists
    if not os.path.isfile(receptorPath):
        return ocerror.Error.file_not_exist(f"The receptor file '{receptorPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Check if the ligand exists
    if not os.path.isfile(ligandPath):
        return ocerror.Error.file_not_exist(f"The ligand file '{ligandPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the output file path
    
    # Get the command
    cmd = __build_cmd(receptorPath, ligandPath, outputFile)

    # If the command is an int, it is an error code
    if isinstance(cmd, int):
        return cmd
    
    # Run the command
    config = get_config()
    exitCode = ocrun.run(cmd, logFile = logFile, cwd = config.oddt_models_dir)

    # If the models should be deleted
    if cleanModels:
        # Get the models
        models = get_models(outputPath)

        # For each model
        for model in models:
            # Delete it
            ocff.safe_remove_file(model)
    
    return exitCode


def run_oddt(preparedReceptorPath: str, preparedLigandPath: Union[str, List[str]], ligandName: str, outputPath: str, returnData: bool = True, overwrite: bool = False, cleanModels: bool = False, n_cpu: int = -1, verbose: bool = False, chunksize: int = 100) -> Union[int, pd.DataFrame]:
    '''Run ODDT programatically.

    Parameters
    ----------
    preparedReceptorPath : str
        The receptor to be used in the rescoring.
    preparedLigandPath : str | List[str]
        The ligand to be used in the rescoring. If a list is given, the rescoring will be performed for each ligand in the list.
    ligandName : str
        The name of the ligand.
    outputPath : str
        The path where the output file will be saved.
    returnData : bool, optional
        If True, the data will be returned. The default is True.
    overwrite : bool, optional
        If True, the output file will be overwritten. The default is False.
    cleanModels : bool, optional
        If True, the models will be deleted after the rescoring. The default is False. If set to False, this can speed up the rescoring process for multiple ligands (you probably will not want to set this to True).
    n_cpu : int, optional
        The number of CPUs to be used. The default is -1 (all available CPUs).
    verbose : bool, optional
        If True, the output will be verbose. The default is False.
    chunksize : int, optional
        The chunksize to be used. The default is 100.
    
    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).   
    '''

    # Check if the output dir exists
    if not os.path.isdir(outputPath):
        # Try to create it
        try:
            _ = ocff.safe_create_dir(outputPath)
        except Exception as e:
            return ocerror.Error.dir_not_exist(f"The output directory '{outputPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
        
    # If the ligand path is a string
    if isinstance(preparedLigandPath, str):
        # Transform it into a list
        preparedLigandPath = [preparedLigandPath]

    # Get the models (only files)
    config = get_config()
    models = [model for model in glob(f"{config.oddt_models_dir}/*.pickle") if os.path.isfile(model)]

    # Check if are there any model
    if len(models) <= 0:
        return ocerror.Error.missing_oddt_models("There are no models in the models folder. Please run the initialise_oddt() function (with proper arguments) to download the models.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor is a string
    if not isinstance(preparedReceptorPath, str):
        return ocerror.Error.wrong_type(f"The receptor must be a string. The type {type(preparedReceptorPath)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the receptor exists
    if not os.path.isfile(preparedReceptorPath):
        return ocerror.Error.file_not_exist(f"The receptor file '{preparedReceptorPath}' does not exist.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check if the ligand is not a string
    if not isinstance(preparedLigandPath, list):
        return ocerror.Error.wrong_type(f"The ligand must be a string or a list. The type {type(preparedLigandPath)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Set the output file name
    outputFile = f"{outputPath}/{ligandName}.csv"

    # Check if the output file exists and if it should be overwritten
    if os.path.isfile(outputFile) and not overwrite:
        # Check if the returnData is True
        if returnData:
            try:
                # Read the output file
                return pd.read_csv(outputFile, sep = ",")
            except Exception as e:
                return ocerror.Error.corrputed_file(f"Failed to read output file '{outputFile}'.", level=ocerror.ReportLevel.ERROR) # type: ignore
        else:
            return ocerror.Error.file_exists(f"The output file '{outputFile}' already exists. Please use the overwrite option if you want to overwrite it.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the vs object
    pipeline = vs(n_cpu=n_cpu, verbose=verbose, chunksize=chunksize)

    # Load the receptor - extract format using os.path.splitext for robustness
    receptor_ext = os.path.splitext(preparedReceptorPath)[1]
    if receptor_ext.startswith('.'):
        receptor_format = receptor_ext[1:]  # Remove leading dot
    else:
        receptor_format = receptor_ext
    
    receptorObj = six.next(od.toolkit.readfile(receptor_format, preparedReceptorPath))

    # Check if the receptor is None
    if receptorObj is None:
        return ocerror.Error.empty(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    receptorObj.protein = True

    # Find missing ligands
    missing = [ligand for ligand in preparedLigandPath if not os.path.isfile(ligand)]

    # Check if there are missing ligands
    if missing:
        return ocerror.Error.file_not_exist(f"Missing ligands: {missing}", level=ocerror.ReportLevel.ERROR)  # type: ignore

    # Check if all the ligands exist and load them
    loaded_ligands = []
    for ligand in preparedLigandPath:
        # Extract format using os.path.splitext for robustness
        ligand_ext = os.path.splitext(ligand)[1]
        if ligand_ext.startswith('.'):
            ligand_format = ligand_ext[1:]  # Remove leading dot
        else:
            ligand_format = ligand_ext
        
        # Try to validate the ligand can be loaded by ODDT before adding to pipeline
        try:
            # Test if ODDT can read the ligand file
            test_mol = six.next(od.toolkit.readfile(ligand_format, ligand))
            if test_mol is None:
                return ocerror.Error.rescoring_failed(f"ODDT could not read ligand file '{ligand}'. The file may be empty or invalid.", level = ocerror.ReportLevel.ERROR) # type: ignore
            
            # Check if molecule has atoms
            if not hasattr(test_mol, 'atoms') or len(test_mol.atoms) == 0:
                return ocerror.Error.rescoring_failed(f"Ligand file '{ligand}' contains no atoms. The file may be corrupted.", level = ocerror.ReportLevel.ERROR) # type: ignore
            
            # Load the ligand into pipeline
            pipeline.load_ligands(ligand_format, ligand)
            loaded_ligands.append(ligand)
        except StopIteration:
            return ocerror.Error.rescoring_failed(f"ODDT could not read ligand file '{ligand}'. The file appears to be empty.", level = ocerror.ReportLevel.ERROR) # type: ignore
        except Exception as e:
            return ocerror.Error.rescoring_failed(f"Failed to load ligand file '{ligand}' into ODDT. Error: {e}", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    if len(loaded_ligands) == 0:
        return ocerror.Error.rescoring_failed(f"No ligands were successfully loaded for '{ligandName}'.", level = ocerror.ReportLevel.ERROR) # type: ignore

    # Determine which scoring families should be loaded.
    requested_scores = [score.lower() for score in getattr(config.oddt, 'scoring_functions', []) if isinstance(score, str)]
    if requested_scores:
        sf_set: set[str] = set()
        for score in requested_scores:
            if 'nnscore' in score:
                sf_set.add('nnscore')
            if 'rfscore' in score:
                sf_set.add('rfscore')
            if 'plec' in score:
                sf_set.add('plec')
    else:
        # Fall back to all supported scoring families if nothing was configured.
        sf_set = {'nnscore', 'rfscore', 'plec'}

    # Process each scoring function separately to handle failures gracefully
    # This allows other scoring functions to succeed even if one fails
    
    # Patch ODDT's descriptor generator to handle 0-d arrays (ODDT bug workaround for PLEC)
    # This needs to be done before any scoring functions are used
    # NOTE: universal_descriptor imports sparse_to_csr_matrix from oddt.fingerprints, not oddt.scoring.descriptors
    _patch_oddt_descriptors_for_plec = False
    try:
        from oddt.fingerprints import sparse_to_csr_matrix as original_sparse_to_csr_matrix
        import numpy as np
        from scipy.sparse import csr_matrix
        
        def _patched_sparse_to_csr_matrix(fp, size, count_bits=True):
            """Patched version that handles 0-d arrays (ODDT bug workaround)"""
            fp_arr = np.asarray(fp, dtype=np.uint64)
            
            # Fix 0-d arrays by converting to empty array
            if fp_arr.ndim == 0:
                fp_arr = np.array([], dtype=np.uint64)
            elif fp_arr.ndim == 1 and fp_arr.size == 0:
                fp_arr = np.array([], dtype=np.uint64)
            elif fp_arr.ndim > 1:
                raise ValueError("Input fingerprint must be a vector (1D)")
            elif fp_arr.ndim == 1:
                fp_arr = fp_arr.astype(np.uint64)
            
            if fp_arr.size == 0:
                return csr_matrix((1, size), dtype=np.uint8 if count_bits else bool)
            
            try:
                return original_sparse_to_csr_matrix(fp_arr, size=size, count_bits=count_bits)
            except Exception:
                return csr_matrix((1, size), dtype=np.uint8 if count_bits else bool)
        
        # Patch oddt.fingerprints module (this is where universal_descriptor imports it from)
        import oddt.fingerprints as oddt_fp
        oddt_fp.sparse_to_csr_matrix = _patched_sparse_to_csr_matrix
        
        # Patch universal_descriptor.build to normalize arrays before processing
        from oddt.scoring.descriptors import universal_descriptor
        from scipy.sparse import vstack as sparse_vstack
        from oddt.utils import is_molecule
        
        _original_universal_build = universal_descriptor.build
        
        def _patched_universal_build(self, ligands, protein=None):
            """Patched version that normalizes arrays before they reach sparse_to_csr_matrix"""
            from oddt.fingerprints import sparse_to_csr_matrix as patched_stcsr
            
            if protein:
                self.protein = protein
            if is_molecule(ligands):
                ligands = [ligands]
            
            out = []
            for mol in ligands:
                try:
                    if self.protein is None:
                        result = self.func(mol)
                    else:
                        result = self.func(mol, protein=self.protein)
                    
                    result_arr = np.asarray(result)
                    
                    # Handle 0-d arrays (scalars from PLEC when no contacts)
                    if result_arr.ndim == 0:
                        result_arr = np.array([], dtype=np.uint64)
                    elif result_arr.ndim == 1 and result_arr.size == 0:
                        result_arr = np.array([], dtype=np.uint64)
                    elif result_arr.ndim == 1:
                        result_arr = result_arr.astype(np.uint64)
                    else:
                        result_arr = result_arr.flatten().astype(np.uint64)
                    
                    out.append(result_arr)
                    
                except Exception as e:
                    mol_title = getattr(mol, 'title', 'unknown')
                    ocprint.print_warning(f"Descriptor generation failed for '{mol_title}': {e}. Using empty descriptor.")
                    out.append(np.array([], dtype=np.uint64))
            
            if self.sparse:
                csr_matrices = []
                for arr in out:
                    try:
                        if arr.size == 0:
                            csr_mat = csr_matrix((1, self.shape), dtype=np.uint8)
                        else:
                            csr_mat = patched_stcsr(arr, size=self.shape, count_bits=True)
                        csr_matrices.append(csr_mat)
                    except Exception as e:
                        ocprint.print_warning(f"CSR conversion failed: {e}. Using empty matrix.")
                        csr_mat = csr_matrix((1, self.shape), dtype=np.uint8)
                        csr_matrices.append(csr_mat)
                
                if csr_matrices:
                    try:
                        return sparse_vstack(csr_matrices, format='csr')
                    except Exception as e:
                        ocprint.print_warning(f"sparse_vstack failed: {e}. Fixing matrices...")
                        fixed = []
                        for mat in csr_matrices:
                            if not isinstance(mat, csr_matrix):
                                mat = csr_matrix(mat)
                            if mat.shape[1] != self.shape: # type: ignore
                                mat = csr_matrix((1, self.shape), dtype=np.uint8)
                            fixed.append(mat)
                        return sparse_vstack(fixed, format='csr')
                else:
                    return csr_matrix((0, self.shape), dtype=np.uint8)
            else:
                normalized = []
                for arr in out:
                    if arr.size == 0:
                        shape = self.shape if self.shape else 1
                        normalized.append(np.zeros(shape, dtype=np.float32))
                    else:
                        normalized.append(arr)
                
                if normalized:
                    return np.vstack(normalized)
                else:
                    shape = self.shape if self.shape else 1
                    return np.array([]).reshape(0, shape)
        
        universal_descriptor.build = _patched_universal_build
        _patch_oddt_descriptors_for_plec = True
    except (ImportError, AttributeError) as patch_err:
        ocprint.print_warning(f"Could not patch ODDT descriptor functions: {patch_err}")
    
    scoring_functions_loaded = []
    model_sf_map = {}  # Map model to scoring function for identification
    
    for model in models:
        # Extract the model name and convert it to lower case
        model_name = os.path.basename(model).lower()

        # Check if the model name is in the set of scoring functions
        if any(sf in model_name for sf in sf_set):
            try:
                # Load the model
                sf = scorer.load(model)
                scoring_functions_loaded.append((model, sf))
                # Store mapping for error reporting
                for sf_name in sf_set:
                    if sf_name in model_name:
                        model_sf_map[model] = sf_name
                        break
            except Exception as e:
                ocprint.print_warning(f"Failed to load scoring function model '{model}': {e}")
                continue

    if len(scoring_functions_loaded) == 0:
        return ocerror.Error.rescoring_failed(f"No scoring functions could be loaded for ligand '{ligandName}'. Please check your ODDT models configuration.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Try processing all scoring functions together first
    # If that fails, process them individually
    all_datas = []
    failed_scoring_functions = []
    
    # Check if multiprocess is enabled (via config or n_cpu > 1)
    # If so, use threading backend to avoid loky nested process issues
    use_threading_backend = False
    try:
        config = get_config()
        use_threading_backend = (config.multiprocess or n_cpu > 1)
    except (ImportError, AttributeError):
        # Fallback: check n_cpu if config not available
        use_threading_backend = (n_cpu > 1)
    
    # Use threading backend context manager if multiprocess is enabled
    # This prevents loky from trying to spawn new processes in nested multiprocessing contexts
    if use_threading_backend:
        try:
            from joblib import parallel_backend
            parallel_ctx = parallel_backend('threading')
        except ImportError:
            # joblib not available, continue without threading backend
            parallel_ctx = None
            ocprint.print_warning("joblib not available. Cannot use threading backend for ODDT scoring.")
    else:
        parallel_ctx = None
    
    # Use context manager to ensure proper cleanup
    if parallel_ctx is not None:
        parallel_ctx.__enter__()
    
    try:
        # Add all scoring functions to pipeline
        for model, sf in scoring_functions_loaded:
            pipeline.score(sf, receptorObj)
        
        # Try to fetch results from all at once
        for mol in pipeline.fetch():
            # Transform the results into a dict
            data = mol.data.to_dict()
            # Add the ligand name
            data["ligand_name"] = ".".join(os.path.basename(mol.title).split(".")[:-1])
            # Set the blacklist keys
            blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges', 'TORSDO', 'REMARK']
            
            # For each key in the blacklist
            for b in blacklist_keys:
                # Check if the key is in the data
                if b in data:
                    # Delete it
                    del data[b]
            
            # Check if there is anything in the data dict
            if len(data) > 0:
                all_datas.append(data)
        
        # If group processing failed, try processing each scoring function individually
        # Note: parallel_context is still active from above if use_threading_backend is True
        if len(all_datas) == 0 and len(scoring_functions_loaded) > 0:
            ocprint.print_warning("Processing scoring functions individually due to group processing failure...")
            for model, sf in scoring_functions_loaded:
                sf_name = model_sf_map.get(model, os.path.basename(model))
                try:
                    # Create a new pipeline for this scoring function
                    individual_pipeline = vs(n_cpu=n_cpu, verbose=verbose, chunksize=chunksize)
                    for ligand in preparedLigandPath:
                        # Extract format using os.path.splitext for robustness
                        ligand_ext = os.path.splitext(ligand)[1]
                        if ligand_ext.startswith('.'):
                            ligand_format = ligand_ext[1:]  # Remove leading dot
                        else:
                            ligand_format = ligand_ext
                        individual_pipeline.load_ligands(ligand_format, ligand)
                    
                    # Add only this scoring function
                    individual_pipeline.score(sf, receptorObj)
                    
                    # Fetch results
                    for mol in individual_pipeline.fetch():
                        data = mol.data.to_dict()
                        data["ligand_name"] = ".".join(os.path.basename(mol.title).split(".")[:-1])
                        
                        blacklist_keys = ['OpenBabel Symmetry Classes', 'MOL Chiral Flag', 'PartialCharges', 'TORSDO', 'REMARK']
                        
                        for b in blacklist_keys:
                            if b in data:
                                del data[b]
                        
                        if len(data) > 0:
                            all_datas.append(data)
                        else:
                            ocprint.print_warning(f"No data collected from '{sf_name}' for ligand '{ligandName}'")
                            
                except AttributeError as e2:
                    # Handle scikit-learn version incompatibility
                    if 'monotonic_cst' in str(e2) or 'DecisionTreeRegressor' in str(e2):
                        error_msg = f"scikit-learn version incompatibility: {e2}"
                        ocprint.print_error(f"Scoring function '{sf_name}' failed due to scikit-learn version mismatch")
                        ocprint.print_error(f"Model was pickled with different scikit-learn version than current installation")
                        full_traceback = traceback.format_exc()
                        ocprint.print_error(f"Full traceback for '{sf_name}':\n{full_traceback}")
                    else:
                        error_msg = str(e2)
                        full_traceback = traceback.format_exc()
                        ocprint.print_error(f"Scoring function '{sf_name}' failed with AttributeError: {error_msg}")
                        ocprint.print_error(f"Full traceback for '{sf_name}':\n{full_traceback}")
                    failed_scoring_functions.append(sf_name)
                    continue
                except (TypeError, ValueError) as e2:
                    error_msg = str(e2)
                    full_traceback = traceback.format_exc()
                    ocprint.print_error(f"Scoring function '{sf_name}' failed for ligand '{ligandName}': {error_msg}")
                    ocprint.print_error(f"Full traceback for '{sf_name}':\n{full_traceback}")
                    failed_scoring_functions.append(sf_name)
                    continue
                except Exception as e2:
                    failed_scoring_functions.append(sf_name)
                    full_traceback = traceback.format_exc()
                    ocprint.print_error(f"Scoring function '{sf_name}' failed for ligand '{ligandName}': {e2}")
                    ocprint.print_error(f"Error type: {type(e2).__name__}")
                    ocprint.print_error(f"Full traceback for '{sf_name}':\n{full_traceback}")
                    continue
                
    except AttributeError as e:
        # Handle scikit-learn version incompatibility
        if 'monotonic_cst' in str(e) or 'DecisionTreeRegressor' in str(e):
            ocprint.print_warning(f"Group processing failed due to scikit-learn version incompatibility: {e}")
            ocprint.print_warning("Processing scoring functions individually...")
            # Fall through to individual processing below
        else:
            # Other AttributeError, log and continue
            full_traceback = traceback.format_exc()
            ocprint.print_error(f"Group processing failed with AttributeError: {e}")
            ocprint.print_error(f"Full traceback:\n{full_traceback}")
    except (TypeError, ValueError) as e:
        # If all-together fails (likely due to descriptor generation error or version incompatibility),
        # try each scoring function individually
        if "0-d array" in str(e) or "iteration" in str(e).lower() or "monotonic_cst" in str(e):
            ocprint.print_warning(f"Processing failed with all scoring functions together: {e}")
            ocprint.print_warning("Trying each scoring function individually...")
            # Fall through to individual processing below
        else:
            # Other TypeError/ValueError, log and continue
            full_traceback = traceback.format_exc()
            ocprint.print_error(f"Group processing failed with {type(e).__name__}: {e}")
            ocprint.print_error(f"Full traceback:\n{full_traceback}")
    except Exception as e:
        # Catch-all for other exceptions
        full_traceback = traceback.format_exc()
        ocprint.print_error(f"Group processing failed with unexpected error: {e}")
        ocprint.print_error(f"Error type: {type(e).__name__}")
        ocprint.print_error(f"Full traceback:\n{full_traceback}")
    
    finally:
        # Clean up parallel context if it was opened
        if parallel_ctx is not None:
            try:
                parallel_ctx.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Check if we got any results
    if len(all_datas) == 0:
        error_msg = f"All scoring functions failed for ligand '{ligandName}'."
        if failed_scoring_functions:
            error_msg += f" Failed scoring functions: {', '.join(failed_scoring_functions)}"
        return ocerror.Error.rescoring_failed(error_msg, level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Check which scoring functions succeeded and which failed
    successful_sf = set()
    for data in all_datas:
        for key in data.keys():
            if key != "ligand_name":
                # Extract scoring function name from column name
                for sf_type in ['rfscore', 'nnscore', 'plec']:
                    if sf_type in key.lower():
                        successful_sf.add(sf_type)
                        break
    
    # Check if all expected scoring functions are present
    expected_sf = {sf.lower() for sf in sf_set}
    missing_sf = expected_sf - successful_sf
    
    # Report on failed scoring functions
    if failed_scoring_functions or missing_sf:
        failed_msg = f"Some scoring functions failed for ligand '{ligandName}': {', '.join(failed_scoring_functions) if failed_scoring_functions else 'None explicitly reported'}"
        if missing_sf:
            failed_msg += f". Missing scoring functions in results: {', '.join(missing_sf)}"
        ocprint.print_error(failed_msg)
    
    # If we processed individually, we might have multiple data dicts for the same ligand
    # Merge them by ligand_name
    merged_datas = {}
    for data in all_datas:
        lig_name = data.get("ligand_name", ligandName)
        if lig_name in merged_datas:
            # Merge dictionaries, keeping all keys
            merged_datas[lig_name].update(data)
        else:
            merged_datas[lig_name] = data.copy()
    
    datas = list(merged_datas.values())

    # Check if datas is empty
    if len(datas) <= 0:
        return ocerror.Error.rescoring_failed(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Create the dataframe
    df = pd.DataFrame(datas)

    # Set the ligand_name as the first column and remove all columns with vina in the name (maybe there is a better way to fix this)
    df = df[["ligand_name"] + [col for col in df.columns if col != "ligand_name" and "vina" not in col]]

    # Set the index to the ligand name
    df = df.set_index("ligand_name")

    # Write the output csv file
    df.to_csv(outputFile, sep = ",", index = False)

    # If the models should be deleted
    if cleanModels:
        # Get the models
        models = get_models(outputPath)

        # For each model
        for model in models:
            # Delete it
            ocff.safe_remove_file(model)
    
    # Check if the returnData is True
    if returnData:
        # Return the dataframe
        return df
    
    # Just return an ok code
    return ocerror.Error.ok() # type: ignore


def df_to_dict(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    '''Convert the data from a pandas dataframe to a dict.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be converted.

    Returns
    -------
    Dict[str, Dict[str, float]]
        The converted data.
    '''

    # Check if the data is a dataframe
    if not isinstance(data, pd.DataFrame):
        return ocerror.Error.wrong_type(f"The data must be a pandas dataframe. The type {type(data)} was given.", level = ocerror.ReportLevel.ERROR) # type: ignore
    
    # Convert the dataframe to dict, one row per index
    return data.to_dict(orient = "index") # type: ignore


def read_log(path: str) -> Optional[pd.DataFrame]:
    '''Read the oddt log path, returning the data from complexes.

    Parameters
    ----------
    path : str
        The path to the oddt csv log file.

    Returns
    -------
    pd.DataFrame | None
        A pd.DataFrame with the data from the vina log file. If the file does not exist, None is returned.
    '''

    # Check if file exists
    if os.path.isfile(path):
        # Read the dataframe
        data = pd.read_csv(path, sep = ",")

        # Return the dataframe
        return data

    # Throw an error
    _ = ocerror.Error.file_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.") # type: ignore

    # Return None
    return None
