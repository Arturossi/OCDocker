#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI to score a merged OCDocker results table with the shipped OCScore models.

This is the hand-off path for an external evaluation: no training data, no fitting and no search,
only the frozen models applied to a table.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import json
import sys

from typing import Optional

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


# Functions
###############################################################################
## Public ##


def add_arguments(parser: argparse.ArgumentParser) -> None:
    '''Register ``ocscore predict`` command-line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser or subparser to extend.
    '''

    parser.add_argument(
        "--csv",
        required=True,
        help="Merged OCDocker results table to score.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions, metrics and the run report.",
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help=(
            "Generation bundle directory. Defaults to the bundle shipped in OCScore_models, or "
            "to the OCSCORE_MODELS_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--families",
        default="release,ablation,control,baselines",
        help=(
            "Comma-separated model families to run: release, ablation, control, baselines. The "
            "control is the randomised-label model and the baselines are the consensus rules, "
            "ridge, elastic-net, ECFP4 memorisation and random ranking; both should normally "
            "be kept, since a screening number is only readable against them."
        ),
    )
    parser.add_argument(
        "--receptor-descriptors",
        default=None,
        help=(
            "JSON with receptor and pocket descriptors for the target, written onto every row. "
            "Only needed when the table does not already carry the context columns."
        ),
    )
    parser.add_argument(
        "--group-column",
        default="receptor",
        help="Column naming the receptor each row was docked against.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Numeric label column, when the table carries one.",
    )
    parser.add_argument(
        "--kind-column",
        default="kind",
        help="Column naming each compound active or inactive.",
    )
    parser.add_argument(
        "--keep-kinds",
        default="active,inactive",
        help="Comma-separated kinds to score; pass an empty string to score every row.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Column holding each compound's SMILES; needed only for the ECFP4 baseline.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--bedroc-alpha",
        type=float,
        default=20.0,
        help="BEDROC weighting factor used when labels are available.",
    )


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``predict`` subcommand on the ``ocscore`` parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        OCScore subparser registry.
    '''

    parser = subparsers.add_parser(
        "predict",
        help="Score a results table with the shipped OCScore models (release, ablation, control)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    parser.set_defaults(func=cmd_predict)


def main_from_args(args: argparse.Namespace) -> int:
    '''Run a prediction from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    import OCDocker.OCScore.Release.Inference as ocrelinference

    families = tuple(f for f in str(args.families).split(",") if f) or None
    keep = tuple(k for k in str(args.keep_kinds).split(",") if k)

    config = ocrelinference.PredictionConfig(
        csv=args.csv,
        output_dir=args.output_dir,
        models_root=args.models_root,
        families=families,
        receptor_descriptors=args.receptor_descriptors,
        group_column=args.group_column,
        label_column=args.label_column,
        kind_column=args.kind_column,
        smiles_column=args.smiles_column,
        keep_kinds=keep,
        device=args.device,
        bedroc_alpha=args.bedroc_alpha,
        command=list(sys.argv),
    )
    report = ocrelinference.run_prediction(config)

    print(json.dumps({k: report[k] for k in (
        "models_scored", "baselines_scored", "n_rows_scored", "n_rows_dropped_incomplete", "n_receptors",
        "predictions_csv", "metrics_csv", "control_check")}, indent=2))

    check = report["control_check"]
    if check.get("status") == "FAIL":
        print(f"\nWARNING: {check['message']}", file=sys.stderr)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    '''Dispatch handler for ``ocscore predict``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process exit code.
    '''

    return main_from_args(args)


def main(argv: Optional[list[str]] = None) -> int:
    '''Run the ``ocscore predict`` CLI.

    Parameters
    ----------
    argv : list[str], optional
        Optional argument list for testing or programmatic execution.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    parser = argparse.ArgumentParser(description="Score a table with the shipped OCScore models")
    add_arguments(parser)
    return main_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
