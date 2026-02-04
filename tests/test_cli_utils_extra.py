#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra CLI utility tests for argument handling.

Usage:

pytest tests/test_cli_utils_extra.py
'''

# Imports
###############################################################################
import os

import pytest

from pathlib import Path

import OCDocker.CLI.__init__ as cli

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

@pytest.mark.order(25)
def test_preparse_global_args_and_require(tmp_path):
    ns = cli._preparse_global_args([
        "vs", "--output-level", "5", "--conf", "cfg.ini", "--overwrite", "--no-stdout-log",
    ])
    assert ns.output_level == 5
    assert ns.config_file == "cfg.ini"
    assert ns.overwrite is True
    assert ns.no_stdout_log is True

    p = tmp_path / "x.pdb"
    p.write_text("ATOM\n")
    got = cli._require_file(str(p), "--receptor")
    assert isinstance(got, Path) and got.exists()

    with pytest.raises(SystemExit):
        _ = cli._require_file("…/fake.pdb", "--receptor")
