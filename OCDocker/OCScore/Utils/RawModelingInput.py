#!/usr/bin/env python3

# Description
###############################################################################
'''
Load and validate raw unreduced OCScore modeling inputs.

Training must start from pipeline-wide tables before any data-dependent feature
cleaning, selection, or reduction. Precomputed global reduction artifacts are
rejected for training.
'''

from __future__ import annotations

# Imports
###############################################################################
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import OCDocker.OCScore.Utils.IO as ocscoreio
from OCDocker.OCScore.Utils.ContentHash import hash_file
from OCDocker.OCScore.Utils.ContentHash import hash_text

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

MERGED_INPUT_DATASET_NAME = "merged_input_dataset.csv"
RAW_PDBBIND_NAME = "raw_pdbbind.csv"
RAW_DUDEZ_NAME = "raw_dudez.csv"

FORBIDDEN_TRAINING_ARTIFACTS = (
    "reduced_pdbbind.csv",
    "reduced_dudez.csv",
    "reduced_dataset.csv",
    "selected_features.json",
    "selected_features.txt",
    "feature_reduction_protocol.json",
)

PREPARE_MANIFEST_NAME = "prepare_manifest.json"

# Classes
###############################################################################


@dataclass(frozen=True)
class RawModelingInput:
    """Validated raw wide tables for staged OCScore training.

    Parameters
    ----------
    merged : pd.DataFrame
        Aligned raw PDBbind+DUDEz wide table.
    pdbbind_source : str | None
        Source path when separate raw PDBbind input was supplied.
    dudez_source : str | None
        Source path when separate raw DUDEz input was supplied.
    merged_source : str | None
        Source path when a merged raw CSV was supplied directly.
    pdbbind_hash : str | None
        Content hash of the raw PDBbind source file, if applicable.
    dudez_hash : str | None
        Content hash of the raw DUDEz source file, if applicable.
    merged_hash : str
        Content hash of the merged raw table or composite hash for separate inputs.
    artifact_paths : dict[str, str]
        Resolved filesystem paths for provenance logging.
    """

    merged: pd.DataFrame
    pdbbind_source: Optional[str]
    dudez_source: Optional[str]
    merged_source: Optional[str]
    pdbbind_hash: Optional[str]
    dudez_hash: Optional[str]
    merged_hash: str
    artifact_paths: dict[str, str] = field(default_factory=dict)


# Functions
###############################################################################
## Private ##


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _preview_columns(columns: Sequence[str], limit: int = 20) -> str:
    shown = list(columns[:limit])
    suffix = "" if len(columns) <= limit else f", ... ({len(columns) - limit} more)"
    return ", ".join(shown) + suffix


def _is_tar_source(source: Path) -> bool:
    if not source.is_file():
        return False
    if tarfile.is_tarfile(source):
        return True
    return source.suffix.lower() in {".tgz", ".tar.gz"} and tarfile.is_tarfile(source)


def _find_file(source: Path, filename: str) -> Optional[Path]:
    if source.is_file():
        return source if source.name == filename else None
    if source.is_dir():
        direct = source / filename
        if direct.is_file():
            return direct
        matches = sorted(source.rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple {filename!r} files found under {source}.")
    return None


def _list_forbidden_artifacts(source: Path) -> list[str]:
    found: list[str] = []
    if source.is_dir():
        for name in FORBIDDEN_TRAINING_ARTIFACTS:
            if _find_file(source, name) is not None:
                found.append(name)
    return found


def _split_merged_raw(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from OCDocker.OCScore.Utils.TrainOnlyFeatureReduction import split_wide_dataset_by_column

    return split_wide_dataset_by_column(merged)


## Public ##


def align_and_concatenate_inputs(pdbbind_df: pd.DataFrame, dudez_df: pd.DataFrame) -> pd.DataFrame:
    '''Align raw PDBbind and DUDEz tables to a shared column schema and concatenate.

    Parameters
    ----------
    pdbbind_df : pd.DataFrame
        Prepared raw PDBbind table.
    dudez_df : pd.DataFrame
        Prepared raw DUDEz table.

    Returns
    -------
    pd.DataFrame
        Wide merged table with union-of-columns alignment and NaN padding.
    '''

    pdbbind = pdbbind_df.copy()
    dudez = dudez_df.copy()
    all_columns = _unique_preserve_order([*pdbbind.columns.tolist(), *dudez.columns.tolist()])

    pdbbind_missing = [column for column in all_columns if column not in pdbbind.columns]
    dudez_missing = [column for column in all_columns if column not in dudez.columns]

    if pdbbind_missing:
        print(
            f"Adding {len(pdbbind_missing)} DUDEz-only columns to PDBbind as NaN: "
            f"{_preview_columns(pdbbind_missing)}"
        )
        for column in pdbbind_missing:
            pdbbind[column] = np.nan

    if dudez_missing:
        print(
            f"Adding {len(dudez_missing)} PDBbind-only columns to DUDEz as NaN: "
            f"{_preview_columns(dudez_missing)}"
        )
        for column in dudez_missing:
            dudez[column] = np.nan

    return pd.concat(
        [pdbbind.loc[:, all_columns], dudez.loc[:, all_columns]],
        ignore_index=True,
        sort=False,
    )


def reject_precomputed_training_artifacts(source: Path) -> None:
    '''Fail when a path contains global/precomputed reduction training artifacts.

    Parameters
    ----------
    source : pathlib.Path
        Directory to inspect for forbidden training artifacts.

    Raises
    ------
    ValueError
        If any forbidden precomputed reduction artifact is present.
    '''

    forbidden = _list_forbidden_artifacts(source)
    if forbidden:
        raise ValueError(
            "Training from precomputed/global feature-reduction artifacts is not supported. "
            f"Found forbidden files under {source}: {forbidden}. "
            "Provide raw unreduced pipeline CSVs (merged or separate PDBbind+DUDEz)."
        )


def validate_raw_schema(pdbbind: pd.DataFrame, dudez: pd.DataFrame) -> None:
    '''Run schema-only checks allowed before train/validation/test splitting.

    Parameters
    ----------
    pdbbind : pd.DataFrame
        Prepared raw PDBbind table.
    dudez : pd.DataFrame
        Prepared raw DUDEz table.

    Raises
    ------
    ValueError
        If required columns are missing or duplicate column names are present.
    '''

    if ocscoreio.TARGET_COLUMN not in pdbbind.columns:
        raise ValueError(f"PDBbind input must contain target column {ocscoreio.TARGET_COLUMN!r}.")
    if ocscoreio.DUDEZ_KIND_COLUMN not in dudez.columns:
        raise ValueError(f"DUDEz input must contain kind column {ocscoreio.DUDEZ_KIND_COLUMN!r}.")
    if len(pdbbind.columns) != len(set(pdbbind.columns)):
        raise ValueError("PDBbind input contains duplicate column names.")
    if len(dudez.columns) != len(set(dudez.columns)):
        raise ValueError("DUDEz input contains duplicate column names.")


def _resolve_raw_modeling_paths(
        *,
        merged_input: Optional[str | Path] = None,
        pdbbind_input: Optional[str | Path] = None,
        dudez_input: Optional[str | Path] = None,
        raw_input_dir: Optional[str | Path] = None,
    ) -> tuple[Optional[Path], Optional[Path], Optional[Path], dict[str, str]]:
    '''Resolve raw modeling input paths without loading CSV rows.

    Parameters
    ----------
    merged_input : str or pathlib.Path, optional
        Path to a merged raw unreduced CSV.
    pdbbind_input : str or pathlib.Path, optional
        Path to a raw unreduced PDBbind pipeline CSV or archive.
    dudez_input : str or pathlib.Path, optional
        Path to a raw unreduced DUDEz pipeline CSV or archive.
    raw_input_dir : str or pathlib.Path, optional
        Directory containing ``merged_input_dataset.csv`` or separate raw tables.

    Returns
    -------
    tuple[pathlib.Path | None, pathlib.Path | None, pathlib.Path | None, dict[str, str]]
        Resolved merged, PDBbind, and DUDEz paths plus provenance paths.

    Raises
    ------
    ValueError
        If input modes are ambiguous or forbidden artifacts are present.
    FileNotFoundError
        If ``raw_input_dir`` does not contain the required raw tables.
    '''

    merged_path = Path(merged_input).expanduser() if merged_input else None
    pdb_path = Path(pdbbind_input).expanduser() if pdbbind_input else None
    dudez_path = Path(dudez_input).expanduser() if dudez_input else None
    input_dir = Path(raw_input_dir).expanduser() if raw_input_dir else None

    modes = sum(
        bool(flag)
        for flag in (
            merged_path is not None,
            pdb_path is not None or dudez_path is not None,
            input_dir is not None,
        )
    )
    if modes != 1:
        raise ValueError(
            "Specify exactly one raw input mode: --merged-input, "
            "(--pdbbind-input and --dudez-input), or --raw-input-dir."
        )

    artifact_paths: dict[str, str] = {}
    if input_dir is not None:
        reject_precomputed_training_artifacts(input_dir)
        merged_candidate = _find_file(input_dir, MERGED_INPUT_DATASET_NAME)
        if merged_candidate is not None:
            merged_path = merged_candidate
            artifact_paths["merged_input_dataset"] = str(merged_candidate.resolve())
        else:
            pdb_path = _find_file(input_dir, RAW_PDBBIND_NAME) or _find_file(input_dir, "PDBbind.csv")
            dudez_path = _find_file(input_dir, RAW_DUDEZ_NAME) or _find_file(input_dir, "DUDEz.csv")
            if pdb_path is None or dudez_path is None:
                raise FileNotFoundError(
                    f"{input_dir} must contain {MERGED_INPUT_DATASET_NAME!r} "
                    f"or both raw PDBbind and DUDEz CSV files."
                )
            artifact_paths["raw_pdbbind"] = str(pdb_path.resolve())
            artifact_paths["raw_dudez"] = str(dudez_path.resolve())

    if merged_path is not None:
        if pdb_path is not None or dudez_path is not None:
            raise ValueError("Do not combine --merged-input with separate PDBbind/DUDEz inputs.")
        if merged_path.name in FORBIDDEN_TRAINING_ARTIFACTS:
            raise ValueError(
                f"Training input {merged_path.name!r} is not a raw unreduced modeling table."
            )
        artifact_paths["merged_input_dataset"] = str(merged_path.resolve())

    if pdb_path is not None and dudez_path is not None and "raw_pdbbind" not in artifact_paths:
        reject_precomputed_training_artifacts(pdb_path.parent)
        reject_precomputed_training_artifacts(dudez_path.parent)
        artifact_paths["raw_pdbbind"] = str(pdb_path.resolve())
        artifact_paths["raw_dudez"] = str(dudez_path.resolve())

    return merged_path, pdb_path, dudez_path, artifact_paths


def _read_modeling_columns(path: Path) -> list[str]:
    '''Read raw modeling CSV header columns from a file path.'''

    if path.suffix.lower() == ".csv" and path.is_file():
        return ocscoreio.read_csv_column_names(path)
    return ocscoreio.read_pipeline_csv_columns(path)


def discover_raw_modeling_input_columns(
        *,
        merged_input: Optional[str | Path] = None,
        pdbbind_input: Optional[str | Path] = None,
        dudez_input: Optional[str | Path] = None,
        raw_input_dir: Optional[str | Path] = None,
    ) -> tuple[dict[str, Optional[list[str]]], dict[str, str]]:
    '''Discover modeling column names from CSV headers without loading rows.

    Parameters
    ----------
    merged_input : str or pathlib.Path, optional
        Path to a merged raw unreduced CSV.
    pdbbind_input : str or pathlib.Path, optional
        Path to a raw unreduced PDBbind pipeline CSV or archive.
    dudez_input : str or pathlib.Path, optional
        Path to a raw unreduced DUDEz pipeline CSV or archive.
    raw_input_dir : str or pathlib.Path, optional
        Directory containing ``merged_input_dataset.csv`` or separate raw tables.

    Returns
    -------
    tuple[dict[str, list[str] | None], dict[str, str]]
        PDBbind/DUDEz column lists and resolved provenance paths.
    '''

    merged_path, pdb_path, dudez_path, artifact_paths = _resolve_raw_modeling_paths(
        merged_input=merged_input,
        pdbbind_input=pdbbind_input,
        dudez_input=dudez_input,
        raw_input_dir=raw_input_dir,
    )

    if merged_path is not None:
        merged_columns = ocscoreio.pdbbind_columns_from_header(_read_modeling_columns(merged_path))
        return (
            {"pdbbind": merged_columns, "dudez": list(merged_columns)},
            artifact_paths,
        )

    if pdb_path is not None and dudez_path is not None:
        pdbbind_columns = ocscoreio.pdbbind_columns_from_header(_read_modeling_columns(pdb_path))
        dudez_columns = ocscoreio.dudez_columns_from_header(_read_modeling_columns(dudez_path))
        return (
            {"pdbbind": pdbbind_columns, "dudez": dudez_columns},
            artifact_paths,
        )

    if pdb_path is not None:
        return (
            {"pdbbind": ocscoreio.pdbbind_columns_from_header(_read_modeling_columns(pdb_path)), "dudez": None},
            {"pdbbind_input": str(pdb_path.resolve())},
        )

    if dudez_path is not None:
        return (
            {"pdbbind": None, "dudez": ocscoreio.dudez_columns_from_header(_read_modeling_columns(dudez_path))},
            {"dudez_input": str(dudez_path.resolve())},
        )

    raise ValueError(
        "Provide raw_input_dir, merged_input, pdbbind_input, dudez_input, or both pdbbind_input and dudez_input."
    )


def load_raw_modeling_input(
        *,
        merged_input: Optional[str | Path] = None,
        pdbbind_input: Optional[str | Path] = None,
        dudez_input: Optional[str | Path] = None,
        raw_input_dir: Optional[str | Path] = None,
    ) -> RawModelingInput:
    '''Load raw unreduced modeling inputs from merged and/or separate pipeline tables.

    Exactly one input mode must be supplied:

    - ``merged_input``
    - ``pdbbind_input`` + ``dudez_input``
    - ``raw_input_dir`` containing merged or separate raw CSVs

    Parameters
    ----------
    merged_input : str or pathlib.Path, optional
        Path to a merged raw unreduced CSV.
    pdbbind_input : str or pathlib.Path, optional
        Path to a raw unreduced PDBbind pipeline CSV or archive.
    dudez_input : str or pathlib.Path, optional
        Path to a raw unreduced DUDEz pipeline CSV or archive.
    raw_input_dir : str or pathlib.Path, optional
        Directory containing ``merged_input_dataset.csv`` or separate raw tables.

    Returns
    -------
    RawModelingInput
        Validated raw modeling input with content hashes and provenance paths.

    Raises
    ------
    ValueError
        If input modes are ambiguous, forbidden artifacts are present, or schema checks fail.
    FileNotFoundError
        If ``raw_input_dir`` does not contain the required raw tables.
    '''

    merged_path = Path(merged_input).expanduser() if merged_input else None
    pdb_path = Path(pdbbind_input).expanduser() if pdbbind_input else None
    dudez_path = Path(dudez_input).expanduser() if dudez_input else None
    input_dir = Path(raw_input_dir).expanduser() if raw_input_dir else None

    modes = sum(
        bool(flag)
        for flag in (
            merged_path is not None,
            pdb_path is not None or dudez_path is not None,
            input_dir is not None,
        )
    )
    if modes != 1:
        raise ValueError(
            "Specify exactly one raw input mode: --merged-input, "
            "(--pdbbind-input and --dudez-input), or --raw-input-dir."
        )

    if input_dir is not None:
        reject_precomputed_training_artifacts(input_dir)
        merged_candidate = _find_file(input_dir, MERGED_INPUT_DATASET_NAME)
        if merged_candidate is not None:
            merged_path = merged_candidate
        else:
            pdb_path = _find_file(input_dir, RAW_PDBBIND_NAME) or _find_file(input_dir, "PDBbind.csv")
            dudez_path = _find_file(input_dir, RAW_DUDEZ_NAME) or _find_file(input_dir, "DUDEz.csv")
            if pdb_path is None or dudez_path is None:
                raise FileNotFoundError(
                    f"{input_dir} must contain {MERGED_INPUT_DATASET_NAME!r} "
                    f"or both raw PDBbind and DUDEz CSV files."
                )

    if merged_path is not None:
        if pdb_path is not None or dudez_path is not None:
            raise ValueError("Do not combine --merged-input with separate PDBbind/DUDEz inputs.")
        if merged_path.name in FORBIDDEN_TRAINING_ARTIFACTS:
            raise ValueError(
                f"Training input {merged_path.name!r} is not a raw unreduced modeling table."
            )
        merged_raw = pd.read_csv(merged_path, low_memory=False)
        merged, _ = ocscoreio.drop_empty_input_rows(merged_raw, label=str(merged_path))
        pdbbind, dudez = _split_merged_raw(merged)
        validate_raw_schema(pdbbind, dudez)
        merged_hash = hash_file(merged_path)
        return RawModelingInput(
            merged=merged,
            pdbbind_source=None,
            dudez_source=None,
            merged_source=str(merged_path.resolve()),
            pdbbind_hash=None,
            dudez_hash=None,
            merged_hash=merged_hash,
            artifact_paths={"merged_input_dataset": str(merged_path.resolve())},
        )

    if pdb_path is None or dudez_path is None:
        raise ValueError("Both --pdbbind-input and --dudez-input are required for separate raw inputs.")

    reject_precomputed_training_artifacts(pdb_path.parent)
    reject_precomputed_training_artifacts(dudez_path.parent)

    pdbbind_raw = ocscoreio.load_pipeline_results_from_archive(pdb_path)
    dudez_raw = ocscoreio.load_pipeline_results_from_archive(dudez_path)
    pdbbind = ocscoreio.prepare_pdbbind_dataframe(pdbbind_raw)
    dudez = ocscoreio.prepare_dudez_dataframe(dudez_raw)
    validate_raw_schema(pdbbind, dudez)
    merged = align_and_concatenate_inputs(pdbbind, dudez)

    pdb_hash = hash_file(pdb_path) if pdb_path.is_file() else hash_text(str(pdb_path.resolve()))
    dudez_hash = hash_file(dudez_path) if dudez_path.is_file() else hash_text(str(dudez_path.resolve()))
    merged_hash = hash_text(json.dumps({"pdbbind": pdb_hash, "dudez": dudez_hash}, sort_keys=True))

    return RawModelingInput(
        merged=merged,
        pdbbind_source=str(pdb_path.resolve()),
        dudez_source=str(dudez_path.resolve()),
        merged_source=None,
        pdbbind_hash=pdb_hash,
        dudez_hash=dudez_hash,
        merged_hash=merged_hash,
        artifact_paths={
            "raw_pdbbind": str(pdb_path.resolve()),
            "raw_dudez": str(dudez_path.resolve()),
        },
    )


def write_prepare_manifest(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    '''Write ``prepare_manifest.json`` to ``output_dir``.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory where the manifest is written.
    payload : dict[str, Any]
        JSON-serializable prepare-stage metadata.

    Returns
    -------
    pathlib.Path
        Path to the written manifest file.
    '''

    path = Path(output_dir) / PREPARE_MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "FORBIDDEN_TRAINING_ARTIFACTS",
    "MERGED_INPUT_DATASET_NAME",
    "PREPARE_MANIFEST_NAME",
    "RAW_DUDEZ_NAME",
    "RAW_PDBBIND_NAME",
    "RawModelingInput",
    "align_and_concatenate_inputs",
    "discover_raw_modeling_input_columns",
    "load_raw_modeling_input",
    "reject_precomputed_training_artifacts",
    "validate_raw_schema",
    "write_prepare_manifest",
]
