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

from OCDocker.CLI.__init__ import cmd_init_config, cmd_version

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
