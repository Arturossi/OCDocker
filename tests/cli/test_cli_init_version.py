#!/usr/bin/env python3

# Description
###############################################################################
'''
Lightweight CLI coverage: ensure init-config and version commands run.

Usage:

pytest tests/test_cli_init_version.py
'''

# Imports
###############################################################################
import pytest

from pathlib import Path

from OCDocker.CLI.init_config import cmd_init_config
from OCDocker.CLI.manifest import cmd_version

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

# Classes
###############################################################################


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)

# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(22)
def test_cli_init_config(tmp_path):
    # Point target to tmp; read example from repo root CWD
    target = tmp_path / "OCDocker.cfg"
    args = _Args(config_file=str(target))
    rc = cmd_init_config(args)
    assert rc == 0
    assert target.exists()


@pytest.mark.order(23)
def test_cli_version():
    # Should not raise; returns 0 and prints
    args = _Args()
    rc = cmd_version(args)
    assert rc == 0
