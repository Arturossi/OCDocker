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
import threading
import time
import traceback

import oddt as od
import pandas as pd

from glob import glob
from oddt.scoring import scorer
from oddt.virtualscreening import virtualscreening as vs
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import OCDocker.Error as ocerror
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Printing as ocprint

from OCDocker.Config import get_config

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

# Functions
###############################################################################
def __read_receptor_with_retry(receptor_format: str, receptor_path: str, retries: int = 5, delay: float = 1.0) -> Tuple[Optional[object], Optional[Exception]]:
    '''Read a prepared receptor with retries to avoid transient empty-file reads.

    Parameters
    ----------
    receptor_format : str
        The receptor format (e.g., pdbqt, mol2).
    receptor_path : str
        Path to the prepared receptor file.
    retries : int, optional
        Number of read attempts before giving up. Default is 5.
    delay : float, optional
        Delay in seconds between attempts. Default is 1.0.

    Returns
    -------
    Tuple[Optional[object], Optional[Exception]]
        The receptor object or None, plus the last error if any.
    '''

    attempts = max(1, retries)
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            receptor_obj = six.next(od.toolkit.readfile(receptor_format, receptor_path))
            if receptor_obj is not None:
                return receptor_obj, None
            last_error = ValueError("ODDT returned None for receptor object.")
        except StopIteration as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < attempts - 1 and delay > 0:
            time.sleep(delay)
    return None, last_error


## Public ##

def df_to_dict(data: pd.DataFrame) -> Union[int, Dict[str, Dict[str, float]]]:
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
        return ocerror.Error.wrong_type(
            f"The data must be a pandas dataframe. The type {type(data)} was given.",
            level = ocerror.ReportLevel.ERROR
        )

    # Convert the dataframe to dict, one row per index
    return cast(Dict[str, Dict[str, float]], data.to_dict(orient = "index"))

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

_scorer_cache_local = threading.local()

def _load_scorer_cached(model_path: str):
    '''Load and cache an ODDT scorer model, keyed by its file path.

    Deserializing the gzipped RF/NN/PLEC pickles is a fixed ~1-3s cost per
    model (measured up to ~6s total for all five combined); with one
    `run_oddt` invocation per ligand and no caching, that cost was paid
    fresh for every single ligand.

    Caching must be per-thread, not a shared/global cache: ODDT's
    `virtualscreening.score()` calls `sf.set_protein(protein)` on the
    scorer object, mutating its `.protein` attribute in place. Under
    Snakemake's `--force-use-threads` local execution, concurrently
    running ligand jobs share the same process and memory, so a single
    shared cached instance would race across threads -- two ligands
    scoring concurrently could clobber each other's `.protein` and
    silently score against the wrong receptor. `threading.local()` gives
    each worker thread its own private copy, so repeat loads within that
    thread's lifetime (across the many ligands it processes over the run)
    are instant, with no cross-thread shared mutable state.

    Parameters
    ----------
    model_path : str
        Path to the pickled scorer model.

    Returns
    -------
    Any
        The loaded ODDT scorer object (private to the calling thread).
    '''

    cache = getattr(_scorer_cache_local, "cache", None)
    if cache is None:
        cache = {}
        _scorer_cache_local.cache = cache
    if model_path not in cache:
        cache[model_path] = scorer.load(model_path)
    return cache[model_path]

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
    _ = ocerror.Error.file_not_exist(f"The file '{path}' does not exists. Please ensure its existance before calling this function.")

    # Return None
    return None

def run_oddt(preparedReceptorPath: str, preparedLigandPath: Union[str, List[str]], ligandName: str, outputPath: str, returnData: bool = True, overwrite: bool = False, cleanModels: bool = False, n_cpu: int = -1, verbose: bool = False, chunksize: int = 100, read_receptor_retries: int = 5, read_receptor_delay: float = 1.0) -> Union[int, pd.DataFrame]:
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
    read_receptor_retries : int, optional
        Number of attempts to read the prepared receptor. The default is 5.
    read_receptor_delay : float, optional
        Delay in seconds between attempts to read the prepared receptor. The default is 1.0.

    Returns
    -------
    int
        The exit code of the command (based on the Error.py code table).
    '''

    # Check if the output dir exists
    if not os.path.isdir(outputPath):
        # Try to create it (parents included)
        _ = ocff.safe_create_dir(Path(outputPath))
        if not os.path.isdir(outputPath):
            return ocerror.Error.dir_not_exist(f"The output directory '{outputPath}' does not exist.", level = ocerror.ReportLevel.ERROR)

    # If the ligand path is a string
    if isinstance(preparedLigandPath, str):
        # Transform it into a list
        preparedLigandPath = [preparedLigandPath]

    # ODDT's multiprocessing path provides no benefit for a single ligand and
    # can deadlock/leave zombies in nested workflow execution contexts.
    effective_n_cpu = n_cpu
    if len(preparedLigandPath) <= 1 and (effective_n_cpu is None or int(effective_n_cpu) != 1):
        effective_n_cpu = 1

    # Get configuration and requested scoring functions
    config = get_config()

    # Determine which scoring families should be loaded.
    # If specific model names are requested (e.g., rfscore_v2_pdbbind2016),
    # load only those exact models. Family-only requests (e.g., rfscore) load all.
    requested_scores = [
        score.lower().strip()
        for score in getattr(config.oddt, 'scoring_functions', [])
        if isinstance(score, str) and score.strip()
    ]

    family_only: set[str] = set()
    exact_requested: set[str] = set()
    sf_set: set[str] = set()

    if requested_scores:
        for score in requested_scores:
            if score.startswith('rfscore'):
                sf_set.add('rfscore')
                if score == 'rfscore':
                    family_only.add('rfscore')
                else:
                    exact_requested.add(score)
            elif score.startswith('nnscore'):
                sf_set.add('nnscore')
                if score == 'nnscore':
                    family_only.add('nnscore')
                else:
                    exact_requested.add(score)
            elif score.startswith('plec') or score.startswith('plecrf'):
                sf_set.add('plec')
                if score in ('plec', 'plecrf'):
                    family_only.add('plec')
                else:
                    exact_requested.add(score)
    else:
        # Fall back to all supported scoring families if nothing was configured.
        sf_set = {'nnscore', 'rfscore', 'plec'}

    # Get the models (only files)
    models = [model for model in glob(f"{config.oddt_models_dir}/*.pickle") if os.path.isfile(model)]

    def _is_exact_model_available(requested_name: str, available_stems: set[str]) -> bool:
        '''Check if the exact model name is available in the available stems, with backward-compatible alias support for plecrf_pdbbind2016.
        
        For example, if requested_name is "plecrf_pdbbind2016", it will be considered available if any model stem starts with "plecrf_" and contains "pdbbind2016" (e.g., "plecrf_p5_l1_pdbbind2016_s65536"). This allows users to request the general plecrf_pdbbind2016 model without needing to specify the exact variant, while still supporting specific model requests.

        Parameters
        ----------
        requested_name : str
            The exact model name being requested (e.g., "rfscore_v2_pdbbind2016").
        available_stems : set[str]
            A set of available model stems (filenames without extension, in lower case).

        Returns
        -------
        bool
            True if the requested model is available, False otherwise.
        '''
        
        if requested_name in available_stems:
            return True

        # Backward-compatible alias support:
        # treat plecrf_pdbbind2016 as satisfied when any concrete plecrf model
        # for that pdbbind version exists (e.g., plecrf_p5_l1_pdbbind2016_s65536).
        if requested_name.startswith("plecrf_"):
            _, _, suffix = requested_name.partition("_")
            for stem in available_stems:
                if not stem.startswith("plecrf_"):
                    continue
                if not suffix or suffix in stem:
                    return True
        return False

    # Attempt to generate missing exact models if requested
    if exact_requested:
        existing_stems = {os.path.splitext(os.path.basename(m))[0].lower() for m in models}
        missing_exact = sorted(
            [name for name in exact_requested if not _is_exact_model_available(name, existing_stems)]
        )
        if missing_exact:
            if config.oddt_models_dir and os.path.isdir(config.oddt_models_dir):
                try:
                    from OCDocker.Initialise import initialise_oddt_models
                    ocprint.print_warning(
                        "Missing ODDT models: " + ", ".join(missing_exact) + ". Attempting to generate them."
                    )
                    initialise_oddt_models(config.oddt_models_dir, missing_exact)
                    models = [model for model in glob(f"{config.oddt_models_dir}/*.pickle") if os.path.isfile(model)]
                except Exception as e:
                    ocprint.print_warning(
                        "Failed to initialize missing ODDT models (" + ", ".join(missing_exact) + f"): {e}"
                    )
            else:
                ocprint.print_warning("ODDT models directory is not set or does not exist; cannot initialize missing models.")

    # Check if are there any model
    if len(models) <= 0:
        return ocerror.Error.missing_oddt_models("There are no models in the models folder. Please run the initialise_oddt() function (with proper arguments) to download the models.", level = ocerror.ReportLevel.ERROR)

    # Check if the receptor is a string
    if not isinstance(preparedReceptorPath, str):
        return ocerror.Error.wrong_type(f"The receptor must be a string. The type {type(preparedReceptorPath)} was given.", level = ocerror.ReportLevel.ERROR)

    # Check if the receptor exists
    if not os.path.isfile(preparedReceptorPath):
        return ocerror.Error.file_not_exist(f"The receptor file '{preparedReceptorPath}' does not exist.", level = ocerror.ReportLevel.ERROR)

    # Check if the ligand is not a string
    if not isinstance(preparedLigandPath, list):
        return ocerror.Error.wrong_type(f"The ligand must be a string or a list. The type {type(preparedLigandPath)} was given.", level = ocerror.ReportLevel.ERROR)

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
                return ocerror.Error.corrupted_file(f"Failed to read output file '{outputFile}'.", level=ocerror.ReportLevel.ERROR)
        else:
            return ocerror.Error.file_exists(f"The output file '{outputFile}' already exists. Please use the overwrite option if you want to overwrite it.", level = ocerror.ReportLevel.ERROR)

    # Create the vs object
    pipeline = vs(n_cpu=effective_n_cpu, verbose=verbose, chunksize=chunksize)

    # Load the receptor - extract format using os.path.splitext for robustness
    receptor_ext = os.path.splitext(preparedReceptorPath)[1]
    if receptor_ext.startswith('.'):
        receptor_format = receptor_ext[1:]  # Remove leading dot
    else:
        receptor_format = receptor_ext

    receptorObj, receptor_err = __read_receptor_with_retry(
        receptor_format,
        preparedReceptorPath,
        retries=read_receptor_retries,
        delay=read_receptor_delay
    )

    # Check if the receptor is None
    if receptorObj is None:
        err_note = f" Last error: {receptor_err}" if receptor_err else ""
        return ocerror.Error.rescoring_failed(
            f"ODDT could not read receptor file '{preparedReceptorPath}' after {max(1, read_receptor_retries)} attempts.{err_note}",
            level = ocerror.ReportLevel.ERROR
        )

    setattr(receptorObj, "protein", True)

    # Find missing ligands
    missing = [ligand for ligand in preparedLigandPath if not os.path.isfile(ligand)]

    # Check if there are missing ligands
    if missing:
        return ocerror.Error.file_not_exist(f"Missing ligands: {missing}", level=ocerror.ReportLevel.ERROR)

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
                return ocerror.Error.rescoring_failed(f"ODDT could not read ligand file '{ligand}'. The file may be empty or invalid.", level = ocerror.ReportLevel.ERROR)

            # Check if molecule has atoms
            if not hasattr(test_mol, 'atoms') or len(test_mol.atoms) == 0:
                return ocerror.Error.rescoring_failed(f"Ligand file '{ligand}' contains no atoms. The file may be corrupted.", level = ocerror.ReportLevel.ERROR)

            # Load the ligand into pipeline
            pipeline.load_ligands(ligand_format, ligand)
            loaded_ligands.append(ligand)
        except StopIteration:
            return ocerror.Error.rescoring_failed(f"ODDT could not read ligand file '{ligand}'. The file appears to be empty.", level = ocerror.ReportLevel.ERROR)
        except Exception as e:
            return ocerror.Error.rescoring_failed(f"Failed to load ligand file '{ligand}' into ODDT. Error: {e}", level = ocerror.ReportLevel.ERROR)

    if len(loaded_ligands) == 0:
        return ocerror.Error.rescoring_failed(f"No ligands were successfully loaded for '{ligandName}'.", level = ocerror.ReportLevel.ERROR)

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
                            if mat.shape[1] != self.shape:
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
        model_stem = os.path.splitext(model_name)[0]
        model_family = None
        if 'rfscore' in model_stem:
            model_family = 'rfscore'
        elif 'nnscore' in model_stem:
            model_family = 'nnscore'
        elif 'plec' in model_stem:
            model_family = 'plec'

        # Decide if this model should be loaded
        if requested_scores:
            if model_family not in sf_set:
                continue
            if model_family in family_only:
                match = True
            else:
                match = model_stem in exact_requested
                if (not match) and model_family == "plec":
                    for req in exact_requested:
                        if not req.startswith("plecrf_"):
                            continue
                        _, _, suffix = req.partition("_")
                        if not suffix or suffix in model_stem:
                            match = True
                            break
        else:
            match = any(sf in model_name for sf in sf_set)

        if match:
            try:
                # Load the model (cached by path across ligands in this worker)
                sf = _load_scorer_cached(model)
                scoring_functions_loaded.append((model, sf))
                # Store mapping for error reporting
                if requested_scores and model_family and model_family not in family_only:
                    model_sf_map[model] = model_stem
                else:
                    model_sf_map[model] = model_family or os.path.basename(model)
            except Exception as e:
                ocprint.print_warning(f"Failed to load scoring function model '{model}': {e}")
                continue

    if len(scoring_functions_loaded) == 0:
        return ocerror.Error.rescoring_failed(f"No scoring functions could be loaded for ligand '{ligandName}'. Please check your ODDT models configuration.", level = ocerror.ReportLevel.ERROR)

    # Try processing all scoring functions together first
    # If that fails, process them individually
    all_datas = []
    failed_scoring_functions = []

    # Decide whether the caller asked for parallel work.
    # Keep this tied to requested n_cpu (not effective_n_cpu) so we still
    # guard nested joblib/loky behavior even when single-ligand runs are
    # coerced to n_cpu=1 internally for stability.
    try:
        use_threading_backend = (n_cpu is not None) and (int(n_cpu) != 1)
    except (TypeError, ValueError):
        use_threading_backend = int(effective_n_cpu) != 1

    # Use threading backend context manager when parallel execution was requested.
    # This prevents loky from trying to spawn new processes in nested multiprocessing contexts.
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
                    individual_pipeline = vs(n_cpu=effective_n_cpu, verbose=verbose, chunksize=chunksize)
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
            except Exception as exc:
                ocprint.print_warning(
                    f"Failed to close ODDT parallel context cleanly: {type(exc).__name__}: {exc}"
                )

    # Check if we got any results
    if len(all_datas) == 0:
        error_msg = f"All scoring functions failed for ligand '{ligandName}'."
        if failed_scoring_functions:
            error_msg += f" Failed scoring functions: {', '.join(failed_scoring_functions)}"
        return ocerror.Error.rescoring_failed(error_msg, level = ocerror.ReportLevel.ERROR)

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
    merged_datas: Dict[str, Dict[str, float]] = {}
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
        return ocerror.Error.rescoring_failed(f"The rescoring of the ligand '{ligandName}' failed.", level = ocerror.ReportLevel.ERROR)

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
    return ocerror.Error.ok()
