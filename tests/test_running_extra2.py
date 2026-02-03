#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Running helpers.
'''

# Imports
###############################################################################
import os
import sys

import pytest

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


@pytest.mark.order(87)
def test_missing_executable_returns_error(tmp_path):
    # Nonexistent executable should return a subprocess error code (non-zero)
    res = ocrun.run(["definitely_not_here_exe_xyz"], logFile="")
    code = res[0] if isinstance(res, tuple) else res
    assert isinstance(code, int) and code != 0


@pytest.mark.order(88)
def test_timeout_env_enforced(tmp_path, monkeypatch):
    # Force a 1s timeout and run a 2s python sleep
    monkeypatch.setenv("OCDOCKER_TIMEOUT", "1")
    log = tmp_path / "run.log"
    cmd = [sys.executable, "-c", "import time; time.sleep(2)"]
    res = ocrun.run(cmd, logFile=str(log))
    code = res[0] if isinstance(res, tuple) else res
    assert isinstance(code, int) and code != 0
