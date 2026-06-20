#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Running helpers.
'''

# Imports
###############################################################################
import os
import pytest
import sys

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
