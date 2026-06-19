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
'''
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
