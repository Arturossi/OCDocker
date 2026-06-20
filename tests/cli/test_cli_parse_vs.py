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

from OCDocker.CLI import build_parser

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
        '--overwrite', '--no-stdout-log', '--no-splash',
    ]
    ns = parser.parse_args(argv)
    # basic assertions on parsed args
    assert ns.engine == 'vina'
    assert ns.receptor and ns.ligand and ns.box
    assert ns.name == 'job'
    assert ns.outdir == 'out'
    assert ns.skip_rescore and ns.skip_split
    assert ns.timeout == 60 and ns.store_db
    assert ns.overwrite and ns.no_stdout_log and ns.no_splash


@pytest.mark.order(25)
def test_cli_vs_parse_gnina_engine():
    '''Ensure VS parser accepts gnina as docking engine.'''

    parser = build_parser()
    ns = parser.parse_args([
        'vs',
        '--engine', 'gnina',
        '--receptor', 'rec.pdb',
        '--ligand', 'ligand.mol2',
        '--box', 'box.pdb',
    ])
    assert ns.engine == 'gnina'
