# Description
###############################################################################
'''
Registration and dispatch for ``ocdocker ocscore``.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse


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
## Public ##

def register_ocscore_subparser(subparsers: argparse._SubParsersAction) -> None:
    '''Register the ``ocdocker ocscore`` command group.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Main CLI subparser registry receiving the OCScore command group.
    '''

    parser = subparsers.add_parser(
        "ocscore",
        description=(
            "Staged OCScore machine-learning pipeline.\n\n"
            "Subcommands mirror examples 14–16:\n"
            "  reduce           - shared PDBbind + DUDEz feature reduction\n"
            "  train            - staged Optuna from reduction output\n"
            "  validate/load/retrain/cross-validate/plot/architecture-plot/shap/score - export tools\n\n"
            'Requires optional ML dependencies: pip install "ocdocker[ml]"\n\n'
            "Note: ML subcommands do not use OCDocker docking config (--conf); "
            "pass global flags before ``ocscore`` if needed for other commands."
        ),
        help="Staged OCScore pipeline (reduce, train, export tools)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=cmd_ocscore)
    ocscore_sub = parser.add_subparsers(dest="ocscore_command", required=True)
    from OCDocker.OCScore.CLI import register_subparsers

    register_subparsers(ocscore_sub)


def cmd_ocscore(args: argparse.Namespace) -> int:
    '''Dispatch an ``ocscore`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    int
        Process-style exit code returned by the selected subcommand.
    '''

    import OCDocker.CLI.common as cli_common

    handler = getattr(args, "func", None)
    if handler is None:
        return 2
    try:
        return int(handler(args))
    except ModuleNotFoundError as exc:
        extra = cli_common._suggest_extra_for_missing_module(getattr(exc, "name", ""))
        return cli_common._print_optional_dependency_hint(
            feature="OCScore pipeline",
            extra=extra,
            exc=exc,
        )
