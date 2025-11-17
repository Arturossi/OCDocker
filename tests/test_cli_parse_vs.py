#!/usr/bin/env python3

from __future__ import annotations

import pytest

from OCDocker.CLI.__init__ import build_parser


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

