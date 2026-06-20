#!/usr/bin/env python3

# Description
###############################################################################
'''
Set of functions to manage I/O operations in OCDocker in the context of scoring
functions.

Usage:

import OCDocker.OCScore.Utils.IO as ocscoreio
'''

# Imports
###############################################################################

import joblib
import os
import pickle
import tarfile

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging
import OCDocker.Toolbox.Security as ocsec

LOGGER = oclogging.get_logger("ocscore.utils.io")

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##

def get_models_dir() -> str:
    ''' Get the path to the OCScore models directory.

    This directory is used to store models and masks that are shipped with the code.
    The directory is located at the project root level (same level as ODDT_models),
    separate from the code folder. The directory is created if it doesn't exist.

    Returns
    -------
    str
        Path to the models directory.
    '''

    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to OCScore directory
    ocscore_dir = os.path.dirname(current_dir)
    # Go up to OCDocker package directory
    ocdocker_dir = os.path.dirname(ocscore_dir)
    # Go up to project root (where ODDT_models is located)
    project_root = os.path.dirname(ocdocker_dir)
    # Create models directory path at project root
    models_dir = os.path.join(project_root, "OCScore_models")

    # Create directory if it doesn't exist
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)

    return models_dir


def load_data(file_name : str, exclude_column : str = 'experimental') -> pd.DataFrame:
    ''' Loads a CSV file into a DataFrame, removes rows with NaNs (except in a specified column), and notifies the user.

    Parameters
    ----------
    file_name: str
        Name of the CSV file to load.
    exclude_column: str
        Column to exclude from the NaN removal process.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.
    '''

    # Read the csv file into a DataFrame
    df = pd.read_csv(file_name)

    # Identify columns to check for NaNs (excluding the specified column)
    columns_to_check = [col for col in df.columns if col != exclude_column]

    if df[columns_to_check].isnull().values.any():
        # Count the number of rows with NaN values in the columns to check
        original_size = len(df)
        rows_with_nan = df[columns_to_check].isnull().any(axis=1).sum()

        # Calculate the percentage of rows that will be removed
        percentage_lost = (rows_with_nan / original_size) * 100

        LOGGER.warning(
            'Removing %d rows with NaNs outside "%s" (%.2f%% of dataset).',
            rows_with_nan,
            exclude_column,
            percentage_lost,
        )

        # Remove rows with NaN values (except in the specified column)
        df = df.dropna(subset=columns_to_check)

    return df


def load_mask(name: str, models_dir: Optional[str] = None) -> np.ndarray:
    ''' Load a mask from a file in the models directory.

    Parameters
    ----------
    name : str
        Name of the mask file (without extension). The function will look for
        '{name}_mask.pkl' in the models directory.
    models_dir : str, optional
        Custom directory to load the mask from. If None, uses the default OCScore
        models directory. Default is None.

    Returns
    -------
    np.ndarray
        The loaded mask array.

    Raises
    ------
    FileNotFoundError
        If the mask file is not found.
    '''

    # Get models directory
    if models_dir is None:
        models_dir = get_models_dir()

    # Create filename
    filename = os.path.join(models_dir, f"{name}_mask.pkl")

    # Check if file exists
    if not os.path.isfile(filename):
        ocerror.Error.file_not_exist(f"Mask file not found: {filename}")
        raise FileNotFoundError(f"Mask file not found: {filename}")

    # Load the mask - try different serialization methods
    try:
        # First try joblib (most common for masks)
        mask = load_object(filename, serialization_method="joblib", trusted=True)
    except (ValueError, EOFError, pickle.UnpicklingError) as e:
        # If joblib fails, try pickle
        try:
            mask = load_object(filename, serialization_method="pickle", trusted=True)
        except (ValueError, EOFError, pickle.UnpicklingError) as e2:
            ocerror.Error.value_error(f"Failed to load mask from {filename}: {e}. Tried both joblib and pickle.")
            raise ValueError(f"Failed to load mask from {filename}. The file may be corrupted or in an unsupported format. Error: {e}")

    # Ensure it's a numpy array
    # Handle different mask formats
    if isinstance(mask, dict):
        # If mask is a dict, try to extract the array
        if 'mask' in mask:
            mask = mask['mask']
        elif 'array' in mask:
            mask = mask['array']
        else:
            # Try to get the first value that looks like an array
            for key, value in mask.items():
                if isinstance(value, (list, np.ndarray)):
                    mask = value
                    break
            else:
                ocerror.Error.value_error(f"Mask loaded as dict but no array found. Keys: {list(mask.keys())}")
                raise ValueError(f"Mask loaded as dict but no array found. Keys: {list(mask.keys())}")

    mask_array = np.asarray(mask, dtype=int)

    # Validate mask contains only 0s and 1s
    if not np.all((mask_array == 0) | (mask_array == 1)):
        ocerror.Error.value_error("Loaded mask must contain only 0s and 1s.")
        raise ValueError("Loaded mask must contain only 0s and 1s.")

    return mask_array


def load_object(file_name : str, serialization_method : str = "auto", trusted: bool = False) -> Any:
    ''' Load an object from a file using pickle, joblib, or torch.

    Security
    --------
    Only load serialized files from trusted sources. Pickle/joblib deserialization
    can execute arbitrary code if the file is malicious or untrusted.

    Parameters
    ----------
    file_name : str
        The name of the file from which to load the object.
    serialization_method : str
        The serialization method used to save the object. Options are:
        - "auto": Automatically detect from file extension (.pt/.pth -> torch, .pkl -> joblib/pickle)
        - "joblib": Use joblib to load
        - "pickle": Use pickle to load
        - "torch": Use torch.load to load (for PyTorch models)
    trusted : bool, optional
        Explicit opt-in that the serialized input is trusted.
        If False, loading is blocked unless
        ``OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION=1`` is set.
        Default is False.

    Returns
    -------
    Any
        The loaded object.

    Raises
    ------
    ValueError
        If the serialization method is not recognized.
    '''

    # Auto-detect format from file extension if "auto" is specified
    if serialization_method == "auto":
        if file_name.endswith('.pt') or file_name.endswith('.pth'):
            serialization_method = "torch"
        elif file_name.endswith('.pkl'):
            serialization_method = "joblib"  # Default to joblib for .pkl
        else:
            # Default to joblib for unknown extensions
            serialization_method = "joblib"

    # Load based on method
    if serialization_method == "torch":
        # Serialized model loading is a security boundary (pickle/joblib/torch can
        # execute arbitrary code when loading crafted inputs).
        ocsec.require_trusted_input(
            trusted=trusted,
            operation="torch deserialization",
            env_var="OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION",
            source=file_name,
        )
        try:
            import torch
            # Explicitly set weights_only=False to suppress FutureWarning
            # This is safe for trusted model files
            return torch.load(file_name, map_location='cpu', weights_only=False)
        except ImportError:
            ocerror.Error.value_error("PyTorch is not installed. Cannot load .pt/.pth files.")
            raise ValueError("PyTorch is not installed. Cannot load .pt/.pth files.")
    elif serialization_method == "joblib":
        ocsec.require_trusted_input(
            trusted=trusted,
            operation="joblib deserialization",
            env_var="OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION",
            source=file_name,
        )
        return joblib.load(file_name)
    elif serialization_method == "pickle":
        ocsec.require_trusted_input(
            trusted=trusted,
            operation="pickle deserialization",
            env_var="OCDOCKER_ALLOW_UNSAFE_DESERIALIZATION",
            source=file_name,
        )
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    else:
        ocerror.Error.value_error(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")
        raise ValueError(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")


def save_mask(mask: Union[list, np.ndarray], name: str, models_dir: Optional[str] = None) -> str:
    ''' Save a mask to a file in the models directory.

    Parameters
    ----------
    mask : list | np.ndarray
        The mask array of 0s and 1s to save.
    name : str
        Name for the mask file (without extension). The file will be saved as
        '{name}_mask.pkl' in the models directory.
    models_dir : str, optional
        Custom directory to save the mask. If None, uses the default OCScore
        models directory. Default is None.

    Returns
    -------
    str
        Path to the saved mask file.

    Raises
    ------
    ValueError
        If the mask is not a valid array of 0s and 1s.
    '''

    # Convert mask to numpy array
    mask_array = np.asarray(mask, dtype=int)

    # Validate mask contains only 0s and 1s
    if not np.all((mask_array == 0) | (mask_array == 1)):
        ocerror.Error.value_error("Mask must contain only 0s and 1s.")
        raise ValueError("Mask must contain only 0s and 1s.")

    # Get models directory
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        # Ensure the custom directory exists
        if not os.path.isdir(models_dir):
            os.makedirs(models_dir, exist_ok=True)

    # Create filename
    filename = os.path.join(models_dir, f"{name}_mask.pkl")

    # Save the mask
    save_object(mask_array, filename)

    return filename


def save_object(obj : Any, filename : str, serialization_method : str = "auto") -> None:
    ''' Save an object to a file using pickle, joblib, or torch.

    Parameters
    ----------
    obj : Any
        The object to be saved.
    filename : str
        The name of the file where the object will be stored.
    serialization_method : str
        The serialization method to use. Options are:
        - "auto": Automatically detect from file extension (.pt/.pth -> torch, .pkl -> joblib)
        - "joblib": Use joblib to save (recommended for sklearn models, XGBoost)
        - "pickle": Use pickle to save
        - "torch": Use torch.save to save (for PyTorch models)
    '''

    # Auto-detect format from file extension if "auto" is specified
    if serialization_method == "auto":
        if filename.endswith('.pt') or filename.endswith('.pth'):
            serialization_method = "torch"
        elif filename.endswith('.pkl'):
            serialization_method = "joblib"  # Default to joblib for .pkl
        else:
            # Default to joblib for unknown extensions
            serialization_method = "joblib"

    # Save based on method
    if serialization_method == "torch":
        try:
            import torch
            torch.save(obj, filename)
        except ImportError:
            ocerror.Error.value_error("PyTorch is not installed. Cannot save .pt/.pth files.")
            raise ValueError("PyTorch is not installed. Cannot save .pt/.pth files.")
    elif serialization_method == "joblib":
        joblib.dump(obj, filename)
    elif serialization_method == "pickle":
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)
    else:
        ocerror.Error.value_error(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")
        raise ValueError(f"Invalid serialization method: '{serialization_method}'. Must be 'auto', 'joblib', 'pickle', or 'torch'.")

    return None


PIPELINE_RESULTS_NAME = "pipeline_results.csv"
PIPELINE_CSV_BASENAMES = (
    PIPELINE_RESULTS_NAME,
    "PDBbind.csv",
    "DUDEz.csv",
)
DATASET_COLUMN = "dataset"
TARGET_COLUMN = "experimental"
DUDEZ_KIND_COLUMN = "kind"
LABEL_COLUMN = "label"


def _read_pipeline_csv(csv_path: Path) -> pd.DataFrame:
    '''Read a pipeline results CSV and raise on empty files.'''

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{csv_path.name!r} at {csv_path} is empty.") from exc
    cleaned, _ = drop_empty_input_rows(df, label=str(csv_path))
    return cleaned


def read_csv_column_names(csv_path: str | Path) -> list[str]:
    '''Read CSV column names without loading table rows.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Path to a CSV file.

    Returns
    -------
    list[str]
        Column names from the header row.

    Raises
    ------
    ValueError
        If the CSV is empty or has no columns.
    FileNotFoundError
        If ``csv_path`` does not exist.
    '''

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        frame = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path.name!r} at {path} is empty.") from exc
    columns = [str(column) for column in frame.columns.tolist()]
    if not columns:
        raise ValueError(f"{path.name!r} at {path} has no columns.")
    return columns


def pdbbind_columns_from_header(columns: Sequence[str]) -> list[str]:
    '''Return PDBbind column names after ``prepare_pdbbind_dataframe`` additions.

    Parameters
    ----------
    columns : Sequence[str]
        Raw PDBbind CSV header columns.

    Returns
    -------
    list[str]
        Header columns plus any workflow columns added during preparation.
    '''

    output = [str(column) for column in columns]
    for name in (DATASET_COLUMN, LABEL_COLUMN):
        if name not in output:
            output.append(name)
    if DUDEZ_KIND_COLUMN not in output:
        output.append(DUDEZ_KIND_COLUMN)
    return output


def dudez_columns_from_header(columns: Sequence[str]) -> list[str]:
    '''Return DUDEz column names after ``prepare_dudez_dataframe`` additions.

    Parameters
    ----------
    columns : Sequence[str]
        Raw DUDEz CSV header columns.

    Returns
    -------
    list[str]
        Header columns plus any workflow columns added during preparation.
    '''

    output = [str(column) for column in columns]
    for name in (DATASET_COLUMN, LABEL_COLUMN):
        if name not in output:
            output.append(name)
    return output


def read_pipeline_csv_columns(
        archive_path: str | Path,
        member_name: str | None = None,
    ) -> list[str]:
    '''Read pipeline CSV column names without loading table rows.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Path to a pipeline CSV file, extracted directory, or tar archive.
    member_name : str, optional
        Explicit tar member path when multiple pipeline CSV files exist.

    Returns
    -------
    list[str]
        Column names from the selected pipeline CSV header.

    Raises
    ------
    FileNotFoundError
        If the path or a canonical pipeline CSV is missing.
    ValueError
        If the archive cannot be read or the CSV header is empty.
    '''

    path = Path(archive_path)

    if path.suffix.lower() == ".csv" and path.is_file():
        return read_csv_column_names(path)

    if path.is_dir():
        csv_path = _find_directory_pipeline_csv(path)
        return read_csv_column_names(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Pipeline input not found: {path}")

    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = _collect_tar_pipeline_members(archive.getmembers())

            if not members:
                expected = ", ".join(PIPELINE_CSV_BASENAMES)
                raise FileNotFoundError(
                    f"Could not find a pipeline results CSV inside archive {path}. "
                    f"Expected one of: {expected}"
                )

            if member_name is not None:
                selected = next((member for member in members if member.name == member_name), None)
                if selected is None:
                    raise FileNotFoundError(
                        f"Could not find tar member {member_name!r} in archive: {path}"
                    )
                members = [selected]
            elif len(members) > 1:
                names = ", ".join(member.name for member in members[:5])
                suffix = "..." if len(members) > 5 else ""
                raise ValueError(
                    f"Found {len(members)} pipeline CSV files inside archive {path}: "
                    f"{names}{suffix}. Pass member_name to select one."
                )

            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError(f"Could not open {members[0].name!r} from archive: {path}")

            try:
                frame = pd.read_csv(handle, nrows=0)
            except pd.errors.EmptyDataError as exc:
                raise ValueError(f"Pipeline CSV in archive {path} is empty.") from exc
            columns = [str(column) for column in frame.columns.tolist()]
            if not columns:
                raise ValueError(f"Pipeline CSV in archive {path} has no columns.")
            return columns

    except tarfile.TarError as exc:
        raise ValueError(f"Could not read tar archive {path}: {exc}") from exc


def _find_directory_pipeline_csv(directory: Path) -> Path:
    '''Return the first canonical pipeline CSV in ``directory`` (KTD3 order).'''

    for basename in PIPELINE_CSV_BASENAMES:
        candidate = directory / basename
        if candidate.is_file():
            return candidate

    lower_index = {
        entry.name.lower(): entry
        for entry in directory.iterdir()
        if entry.is_file()
    }
    for basename in PIPELINE_CSV_BASENAMES:
        match = lower_index.get(basename.lower())
        if match is not None:
            return match

    expected = ", ".join(PIPELINE_CSV_BASENAMES)
    raise FileNotFoundError(
        f"Could not find a pipeline results CSV in directory {directory}. "
        f"Expected one of: {expected}"
    )


def _collect_tar_pipeline_members(members: Sequence[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    '''Return tar members whose basename matches a canonical pipeline CSV name.'''

    canonical = set(PIPELINE_CSV_BASENAMES)
    canonical_lower = {name.lower() for name in PIPELINE_CSV_BASENAMES}
    matched: list[tarfile.TarInfo] = []
    for member in members:
        if not member.isfile():
            continue
        basename = Path(member.name).name
        if basename in canonical or basename.lower() in canonical_lower:
            matched.append(member)
    return matched


def load_pipeline_results_from_archive(
        archive_path: str | Path,
        member_name: str | None = None,
    ) -> pd.DataFrame:
    '''Load pipeline results from a CSV file, directory, or tar archive.

    Accepts a bare ``.csv`` path, a directory, or a tar archive containing one of:
    ``pipeline_results.csv``, ``PDBbind.csv``, or ``DUDEz.csv`` (classic and ocdb2
    layouts). When multiple matching members exist inside a tar archive, pass
    ``member_name`` to select one.

    Parameters
    ----------
    archive_path : str or pathlib.Path
        Path to a pipeline CSV file, extracted directory, or tar archive.
    member_name : str, optional
        Explicit tar member path when multiple pipeline CSV files exist.

    Returns
    -------
    pd.DataFrame
        Loaded pipeline results table.

    Raises
    ------
    FileNotFoundError
        If the path or a canonical pipeline CSV is missing.
    ValueError
        If the archive cannot be read, the CSV is empty, or multiple members exist
        without an explicit ``member_name``.
    '''

    path = Path(archive_path)

    if path.suffix.lower() == ".csv" and path.is_file():
        return _read_pipeline_csv(path)

    if path.is_dir():
        csv_path = _find_directory_pipeline_csv(path)
        return _read_pipeline_csv(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Pipeline input not found: {path}")

    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = _collect_tar_pipeline_members(archive.getmembers())

            if not members:
                expected = ", ".join(PIPELINE_CSV_BASENAMES)
                raise FileNotFoundError(
                    f"Could not find a pipeline results CSV inside archive {path}. "
                    f"Expected one of: {expected}"
                )

            if member_name is not None:
                selected = next((member for member in members if member.name == member_name), None)
                if selected is None:
                    raise FileNotFoundError(
                        f"Could not find tar member {member_name!r} in archive: {path}"
                    )
                members = [selected]
            elif len(members) > 1:
                names = ", ".join(member.name for member in members[:5])
                suffix = "..." if len(members) > 5 else ""
                raise ValueError(
                    f"Found {len(members)} pipeline CSV files inside archive {path}: "
                    f"{names}{suffix}. Pass member_name to select one."
                )

            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError(f"Could not open {members[0].name!r} from archive: {path}")

            df = pd.read_csv(handle, low_memory=False)
            cleaned, _ = drop_empty_input_rows(df, label=members[0].name)
            return cleaned

    except tarfile.TarError as exc:
        raise ValueError(f"Could not read tar archive {path}: {exc}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Pipeline CSV in archive {path} is empty.") from exc


load_pipeline_results = load_pipeline_results_from_archive


def drop_empty_input_rows(df: pd.DataFrame, *, label: str = "input") -> tuple[pd.DataFrame, int]:
    '''Drop rows that are entirely empty before OCScore modeling preparation.

    CSV rows containing only blank strings are treated as empty as well as rows
    parsed as all-NaN.
    '''

    if df.empty:
        return df.copy(), 0
    normalized = df.replace(r"^\s*$", np.nan, regex=True)
    empty_mask = normalized.isna().all(axis=1)
    dropped = int(empty_mask.sum())
    if dropped:
        LOGGER.warning("Dropped %d completely empty row(s) from %s.", dropped, label)
    return df.loc[~empty_mask].reset_index(drop=True).copy(), dropped


def prepare_pdbbind_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''Prepare PDBbind pipeline rows for OCScore feature workflows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw PDBbind pipeline results.

    Returns
    -------
    pd.DataFrame
        Prepared PDBbind rows with ``dataset`` and ``label`` columns.

    Raises
    ------
    ValueError
        If the PDBbind affinity target column is missing.
    '''

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"PDBbind input must contain the target column {TARGET_COLUMN!r}.")

    prepared = df.copy()
    prepared[DATASET_COLUMN] = "pdbbind"
    prepared[LABEL_COLUMN] = np.nan

    if DUDEZ_KIND_COLUMN not in prepared.columns:
        prepared[DUDEZ_KIND_COLUMN] = np.nan

    return prepared


def prepare_dudez_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    '''Prepare DUDEz pipeline rows for OCScore feature workflows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DUDEz pipeline results.

    Returns
    -------
    pd.DataFrame
        Prepared DUDEz rows with ``dataset`` and ``label`` columns.

    Raises
    ------
    ValueError
        If the DUDEz ``kind`` column is missing.
    '''

    if DUDEZ_KIND_COLUMN not in df.columns:
        raise ValueError(
            f"DUDEz input must contain {DUDEZ_KIND_COLUMN!r} so ligands/decoys can be preserved."
        )

    prepared = df.copy()
    prepared[DATASET_COLUMN] = "dudez"
    prepared[TARGET_COLUMN] = np.nan

    kind_values = prepared[DUDEZ_KIND_COLUMN].where(prepared[DUDEZ_KIND_COLUMN].notna(), np.nan)
    normalized_kind = kind_values.astype("string").str.strip().str.lower()
    prepared[LABEL_COLUMN] = normalized_kind.map({"ligands": 1, "decoys": 0}).astype("float")

    unknown_kinds = sorted(normalized_kind[prepared[LABEL_COLUMN].isna()].dropna().unique().tolist())
    if unknown_kinds:
        LOGGER.warning(
            "Some DUDEz rows have kind values that were not mapped to label: %s",
            unknown_kinds,
        )

    return prepared
