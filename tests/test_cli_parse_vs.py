#!/usr/bin/env python3

# Description
###############################################################################
'''
CLI parser smoke tests for virtual screening commands.

Usage:

pytest tests/test_cli_parse_vs.py
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.CLI.__init__ import build_parser

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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(24)
def test_cli_vs_parse_smoke():
    parser = build_parser()
    argv = [
        'vs',
        '--engine', 'vina',
        '--receptor', 'rec.pdb',
        '--ligand', 'ligand.mol2',
        '--box', 'box.pdb',
        '--name', 'job',
        '--outdir', 'out',
        '--skip-rescore', '--skip-split',
        '--timeout', '60',
        '--store-db',
        '--overwrite', '--no-stdout-log',
    ]
    ns = parser.parse_args(argv)
    # basic assertions on parsed args
    assert ns.engine == 'vina'
    assert ns.receptor and ns.ligand and ns.box
    assert ns.name == 'job'
    assert ns.outdir == 'out'
    assert ns.skip_rescore and ns.skip_split
    assert ns.timeout == 60 and ns.store_db
    assert ns.overwrite and ns.no_stdout_log
