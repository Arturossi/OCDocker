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
