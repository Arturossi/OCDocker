#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI to compute the receptor and pocket descriptors a target needs before it can be scored.

The 55 context features describe the target rather than any compound, so they are computed once
per receptor from its structure and the docking centre of its binding site. The result feeds
``ocdocker ocscore predict --receptor-descriptors``.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
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
    '''Register ``ocscore descriptors`` command-line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser or subparser to extend.
    '''

    import OCDocker.OCScore.Release.Context as ocrelcontext

    parser.add_argument("--receptor", required=True,
                        help="Receptor structure, protein only.")
    site = parser.add_mutually_exclusive_group(required=True)
    site.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Docking centre of the binding site, in angstroms.")
    site.add_argument("--box", default=None,
                      help="OCDocker docking box file holding the docking centre.")
    parser.add_argument("--out", required=True,
                        help="Destination JSON, consumed by 'ocscore predict'.")
    parser.add_argument("--target-id", default=None,
                        help=("Receptor identifier as it appears in the table's receptor column. "
                              "Given, the JSON is keyed by target and merged into --out when it "
                              "already exists, which is how several targets share one file."))
    parser.add_argument("--name", default="receptor",
                        help="Name recorded on the receptor and pocket objects.")
    parser.add_argument("--radius", type=float, default=ocrelcontext.DEFAULT_RADIUS,
                        help="Angstrom radius around the docking centre defining the pocket.")
    parser.add_argument("--ph", type=float, default=ocrelcontext.DEFAULT_PH,
                        help="pH used for the pocket protonation.")
    parser.add_argument("--gravy-scale", default=ocrelcontext.DEFAULT_GRAVY_SCALE,
                        help="Hydropathy scale used for GRAVY.")


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``descriptors`` subcommand on the ``ocscore`` parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        OCScore subparser registry.
    '''

    parser = subparsers.add_parser(
        "descriptors",
        help="Compute a target's receptor and pocket descriptors for 'ocscore predict'",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_arguments(parser)
    parser.set_defaults(func=cmd_descriptors)


def main_from_args(args: argparse.Namespace) -> int:
    '''Compute and write the descriptors from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process exit code. Zero indicates success.
    '''

    import OCDocker.OCScore.Release.Context as ocrelcontext

    computed = ocrelcontext.compute_context(
        receptor=args.receptor,
        center=args.box if args.box else args.center,
        name=args.name,
        radius=args.radius,
        ph=args.ph,
        gravy_scale=args.gravy_scale,
    )

    out = Path(args.out)
    if args.target_id:
        existing = {}
        if out.is_file():
            existing = ocrelcontext.load_context(out)
            if not ocrelcontext.is_multi_target(existing):
                raise SystemExit(
                    f"{out} holds a single target's descriptors and cannot take another under "
                    "--target-id. Write to a new file, or rebuild the file with --target-id for "
                    "every target."
                )
        existing[args.target_id] = computed
        payload = existing
    else:
        payload = computed

    ocrelcontext.write_context(payload, out)
    print(json.dumps({
        "receptor_descriptors": len(computed["receptor"]),
        "pocket_descriptors": len(computed["pocket"]),
        "targets_in_file": sorted(payload) if args.target_id else [args.name],
        "written": str(out.resolve()),
    }, indent=2))
    return 0


def cmd_descriptors(args: argparse.Namespace) -> int:
    '''Dispatch handler for ``ocscore descriptors``.

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
    '''Run the ``ocscore descriptors`` CLI.

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
        description="Compute a target's receptor and pocket descriptors",
    )
    add_arguments(parser)
    return main_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
