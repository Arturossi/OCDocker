#!/usr/bin/env python3

# Description
###############################################################################
'''
Example: score a docked library with the pretrained OCScore models shipped in
``OCScore_models/gen3b/``.

Nothing is trained here. The frozen release model, the eight-arm block ablation, the
randomised-label control and the reference baselines are applied to a merged OCDocker results
table, and every model is reported per seed and as a three-seed rank ensemble.

Usage:

    python examples/25_ocscore_score_with_shipped_models.py \\
        --csv /path/to/merged_results_all.csv \\
        --output-dir /path/to/results

The same run is available from the command line as::

    ocdocker ocscore predict --csv <table>.csv --output-dir <dir>

List what the bundle ships without scoring anything:

    python examples/25_ocscore_score_with_shipped_models.py --list

Two properties of the feature contract constrain the input. Scores and ligand descriptors are
standardised within receptor, so a whole library is ranked against one target at a time and a
single compound cannot be scored on its own. The 55 receptor and pocket descriptors describe the
target rather than any compound, so a new target needs them computed once; pass them with
``--receptor-descriptors`` when the table does not already carry them.

Read the control before reading the ranking. The control model never saw a true label, so on a
sound evaluation set it must land at chance. When it does not, the labels are partly predictable
from the inputs without any activity supervision, and the ranking of the other models says
nothing about the models.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import json
import sys

from typing import Optional

import OCDocker.OCScore.Release.Catalog as ocrelcatalog
import OCDocker.OCScore.Release.Inference as ocrelinference

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''


# Functions
###############################################################################
## Private ##

def _list_models() -> int:
    '''Print the models shipped in the bundle.

    Returns
    -------
    int
        Process exit code.
    '''

    specs = ocrelcatalog.discover()
    print(f"{len(specs)} models in {ocrelcatalog.default_root()}\n")
    print(f"{'model':28s} {'family':10s} {'blocks':8s} seeds")
    for spec in specs:
        print(f"{spec.name:28s} {spec.family:10s} {spec.live_blocks:8s} "
              f"{', '.join(str(s) for s in spec.seeds)}")
    print("\nBaselines scored alongside them: mean and median consensus, ridge and elastic-net "
          "at three feature arms, ECFP4 memorisation (needs --smiles-column), random ranking.")
    return 0


## Public ##

def main(argv: Optional[list[str]] = None) -> int:
    '''Run the example.

    Parameters
    ----------
    argv : list[str], optional
        Optional argument list for testing or programmatic execution.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    parser = argparse.ArgumentParser(
        description="Score a docked library with the shipped OCScore models.",
    )
    parser.add_argument("--list", action="store_true",
                        help="List the shipped models and exit.")
    parser.add_argument("--csv", help="Merged OCDocker results table to score.")
    parser.add_argument("--output-dir", help="Directory for predictions, metrics and report.")
    parser.add_argument("--receptor-descriptors", default=None,
                        help="JSON with the target's receptor and pocket descriptors.")
    parser.add_argument("--smiles-column", default=None,
                        help="Column holding SMILES, needed only for the ECFP4 baseline.")
    args = parser.parse_args(argv)

    if args.list:
        return _list_models()
    if not args.csv or not args.output_dir:
        parser.error("--csv and --output-dir are required unless --list is given")

    report = ocrelinference.run_prediction(ocrelinference.PredictionConfig(
        csv=args.csv,
        output_dir=args.output_dir,
        receptor_descriptors=args.receptor_descriptors,
        smiles_column=args.smiles_column,
        command=list(sys.argv),
    ))

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["control_check"].get("status") == "FAIL":
        print(f"\nWARNING: {report['control_check']['message']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
