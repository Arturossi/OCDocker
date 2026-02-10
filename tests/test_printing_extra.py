#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for Toolbox.Printing helpers.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Printing as ocprint

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

@pytest.mark.order(84)
def test_print_to_log_files(tmp_path):
    log = tmp_path / "out.log"
    ocprint.print_info_log("alpha", str(log))
    ocprint.print_warning_log("beta", str(log))
    ocprint.print_error_log("gamma", str(log))
    txt = log.read_text()
    assert "INFO: alpha" in txt
    assert "WARNING: beta" in txt
    assert "ERROR: gamma" in txt


@pytest.mark.order(83)
def test_printv_gated_by_level(capsys):
    # Ensure DEBUG prints
    ocerror.Error.set_output_level(ocerror.ReportLevel.DEBUG)
    ocprint.printv("visible")
    out = capsys.readouterr().out
    assert "visible" in out

    # Ensure non-DEBUG suppresses
    ocerror.Error.set_output_level(ocerror.ReportLevel.INFO)
    ocprint.printv("hidden")
    out2 = capsys.readouterr().out
    assert "hidden" not in out2
