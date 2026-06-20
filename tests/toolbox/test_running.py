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
