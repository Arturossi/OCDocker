#!/usr/bin/env python3

# Description
###############################################################################
'''
OCScore staged-pipeline CLI subcommands.
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse

from OCDocker.OCScore.CLI import export_tools
from OCDocker.OCScore.CLI import reduce
from OCDocker.OCScore.CLI import train

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def register_subparsers(subparsers: argparse._SubParsersAction) -> None:
    '''Register reduce, train, and export-tool subcommands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        OCScore command subparser registry from the main CLI.
    '''

    reduce.register_subparser(subparsers)
    train.register_subparser(subparsers)
    export_tools.register_subparsers(subparsers)
