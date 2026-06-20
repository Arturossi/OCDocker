#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for running utilities.

Usage:

pytest tests/test_running.py
'''

# Imports
###############################################################################
import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Running as ocrun

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

@pytest.mark.order(3)
def test_run_echo(tmp_path):
    log = tmp_path / "run.log"
    code = ocrun.run(['echo', 'hello'], logFile=str(log))
    assert code == ocerror.ErrorCode.OK
    assert log.exists()


@pytest.mark.order(1)
def test_run_empty_cmd():
    result = ocrun.run([])
    assert result == ocerror.ErrorCode.NOT_SET


@pytest.mark.order(2)
def test_run_wrong_type():
    result = ocrun.run('notalist') # type: ignore
    assert result == ocerror.ErrorCode.WRONG_TYPE
