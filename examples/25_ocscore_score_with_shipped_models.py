#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: score raw pipeline data with one of the pretrained OCScore models
shipped in ``OCScore_models/`` (configurations #03, #05, #09, and #12 from the
DUDEz ablation study).

Each shipped configuration is a DUDEz-screening ``best_model/`` export bundle
transfer-linked to its own PDBbind ``best_model/`` export bundle (see
``OCDocker.OCScore.Optimization.ModelExport.predict_from_export``). The script
reads ``OCScore_models/manifest.json`` to resolve both bundle paths for the
requested configuration, then scores a raw pipeline archive the same way
``ocdocker ocscore score`` does.

Usage:

    python examples/25_ocscore_score_with_shipped_models.py \\
        --config 03 \\
        --raw-archive /path/to/pipeline_results.csv \\
        --output-csv /path/to/predictions.csv

List the shipped configurations without scoring anything:

    python examples/25_ocscore_score_with_shipped_models.py --list
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
from typing import Sequence

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Utils.IO as ocscoreio

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
DEFAULT_MODELS_DIR = Path(_PROJECT_ROOT) / "OCScore_models"


# Functions
###############################################################################
## Private ##

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    '''Parse command-line arguments.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    '''

    parser = argparse.ArgumentParser(
        description="Score raw pipeline data with a shipped OCScore_models configuration.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing manifest.json and the numbered config folders (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="03",
        choices=("03", "05", "09", "12"),
        help="Shipped configuration to score with (default: %(default)s; #03 is the configuration recommended in the paper).",
    )
    parser.add_argument(
        "--raw-archive",
        type=str,
        default=None,
        help="Raw pipeline input: a .csv file, directory, or tar.gz containing pipeline_results.csv or DUDEz.csv.",
    )
    parser.add_argument(
        "--archive-member",
        type=str,
        default=None,
        help="Explicit tar member path when multiple pipeline CSV files exist.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path for the predictions CSV output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference (default: %(default)s).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the manifest for every shipped configuration and exit.",
    )
    return parser.parse_args(argv)


## Public ##

def main(argv: Sequence[str] | None = None) -> int:
    '''Score a raw pipeline archive with a shipped OCScore_models configuration.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command-line arguments.

    Returns
    -------
    int
        Process exit code.
    '''

    args = _parse_args(argv)
    manifest = json.loads((args.models_dir / "manifest.json").read_text(encoding="utf-8"))

    if args.list:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if not args.raw_archive or not args.output_csv:
        print("--raw-archive and --output-csv are required unless --list is given.", file=sys.stderr)
        return 2

    entry = manifest[args.config]
    dudez_export_dir = args.models_dir / entry["dudez_best_model_dir"]
    pdbbind_export_dir = args.models_dir / entry["pdbbind_best_model_dir"]

    raw = ocscoreio.load_pipeline_results_from_archive(args.raw_archive, member_name=args.archive_member)
    dataframe = ocscoreio.prepare_dudez_dataframe(raw)

    predictions = ocexport.predict_from_export(
        dudez_export_dir,
        dataframe,
        device=args.device,
        pdbbind_export_dir=pdbbind_export_dir,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)

    print(
        f"Scored {len(predictions)} rows with config #{args.config} "
        f"({entry['feature_policy_name']}) -> {output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
