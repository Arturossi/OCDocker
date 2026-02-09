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
