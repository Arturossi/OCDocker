#!/usr/bin/env python3

"""CLI to merge raw PDBbind and DUDEz pipeline tables for OCScore modeling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import OCDocker.OCScore.Utils.IO as ocscoreio
from OCDocker.OCScore.Utils.ContentHash import hash_file
from OCDocker.OCScore.Utils.RawModelingInput import MERGED_INPUT_DATASET_NAME
from OCDocker.OCScore.Utils.RawModelingInput import RAW_DUDEZ_NAME
from OCDocker.OCScore.Utils.RawModelingInput import RAW_PDBBIND_NAME
from OCDocker.OCScore.Utils.RawModelingInput import align_and_concatenate_inputs
from OCDocker.OCScore.Utils.RawModelingInput import validate_raw_schema
from OCDocker.OCScore.Utils.RawModelingInput import write_prepare_manifest

load_pipeline_results_from_archive = ocscoreio.load_pipeline_results_from_archive
prepare_pdbbind_dataframe = ocscoreio.prepare_pdbbind_dataframe
prepare_dudez_dataframe = ocscoreio.prepare_dudez_dataframe

# License
###############################################################################
"""
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""


# Functions
###############################################################################
## Public ##


def add_arguments(parser: argparse.ArgumentParser) -> None:
    '''Register ``ocscore reduce`` command-line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser or subparser to extend.
    '''

    parser.add_argument(
        "--pdbbind-archive",
        required=True,
        help=(
            "Raw PDBbind pipeline input: a .csv file, directory, or tar.gz containing "
            "pipeline_results.csv or PDBbind.csv."
        ),
    )
    parser.add_argument(
        "--dudez-archive",
        required=True,
        help=(
            "Raw DUDEz pipeline input: a .csv file, directory, or tar.gz containing "
            "pipeline_results.csv or DUDEz.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged and separate raw modeling tables are written.",
    )


def build_argparser() -> argparse.ArgumentParser:
    '''Build the ``ocscore reduce`` command-line parser.'''

    parser = argparse.ArgumentParser(
        description=(
            "Merge raw unreduced PDBbind and DUDEz pipeline archives into a wide modeling table. "
            "Feature cleaning/reduction is fit only during training after splitting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    return parser


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``reduce`` subcommand on the ``ocscore`` parser.'''

    parser = subparsers.add_parser(
        "reduce",
        help="Merge raw PDBbind + DUDEz pipeline tables for modeling (no feature reduction)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    parser.set_defaults(func=cmd_reduce)


def cmd_reduce(args: argparse.Namespace) -> int:
    '''Dispatch handler for ``ocscore reduce``.'''

    return main_from_args(args)


def main(argv: Optional[list[str]] = None) -> int:
    '''Run the ``ocscore reduce`` CLI.

    Parameters
    ----------
    argv : list[str], optional
        Optional argument list for testing or programmatic execution.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    args = build_argparser().parse_args(argv)
    return main_from_args(args)


def main_from_args(args: argparse.Namespace) -> int:
    '''Merge raw PDBbind and DUDEz inputs and write prepare-stage artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw PDBbind pipeline results...")
    pdbbind_raw = load_pipeline_results_from_archive(args.pdbbind_archive)
    pdbbind = prepare_pdbbind_dataframe(pdbbind_raw)

    print("Loading raw DUDEz pipeline results...")
    dudez_raw = load_pipeline_results_from_archive(args.dudez_archive)
    dudez = prepare_dudez_dataframe(dudez_raw)

    validate_raw_schema(pdbbind, dudez)

    print("Aligning datasets by column name...")
    merged = align_and_concatenate_inputs(pdbbind, dudez)

    merged_input_path = output_dir / MERGED_INPUT_DATASET_NAME
    raw_pdbbind_path = output_dir / RAW_PDBBIND_NAME
    raw_dudez_path = output_dir / RAW_DUDEZ_NAME
    merged.to_csv(merged_input_path, index=False)
    pdbbind.to_csv(raw_pdbbind_path, index=False)
    dudez.to_csv(raw_dudez_path, index=False)

    manifest = {
        "protocol": "raw_prepare_only",
        "global_feature_reduction_used": False,
        "precomputed_features_used_for_training": False,
        "merged_input_dataset": str(merged_input_path.resolve()),
        "raw_pdbbind": str(raw_pdbbind_path.resolve()),
        "raw_dudez": str(raw_dudez_path.resolve()),
        "pdbbind_archive": str(Path(args.pdbbind_archive).expanduser().resolve()),
        "dudez_archive": str(Path(args.dudez_archive).expanduser().resolve()),
        "merged_rows": int(merged.shape[0]),
        "merged_columns": int(merged.shape[1]),
        "pdbbind_rows": int(pdbbind.shape[0]),
        "dudez_rows": int(dudez.shape[0]),
        "merged_hash": hash_file(merged_input_path),
        "pdbbind_hash": hash_file(raw_pdbbind_path),
        "dudez_hash": hash_file(raw_dudez_path),
    }
    manifest_path = write_prepare_manifest(output_dir, manifest)

    print(f"Merged input rows: {merged.shape[0]}; columns: {merged.shape[1]}")
    print(f"Saved merged raw dataset: {merged_input_path}")
    print(f"Saved raw PDBbind table: {raw_pdbbind_path}")
    print(f"Saved raw DUDEz table: {raw_dudez_path}")
    print(f"Prepare manifest: {manifest_path}")
    print(
        "Note: no data-dependent feature cleaning/reduction was applied. "
        "Run ocscore train with --raw-input-dir to fit train-only reduction after splitting."
    )
    return 0
