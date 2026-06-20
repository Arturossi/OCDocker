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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
