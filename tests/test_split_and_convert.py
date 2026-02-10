#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for split_and_convert error handling.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Conversion as occonversion

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

@pytest.mark.order(1)
def test_split_and_convert_invalid_input(tmp_path):
    invalid_path = tmp_path / "molecule.bad"
    invalid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(invalid_path), str(tmp_path), "sdf")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(2)
def test_split_and_convert_invalid_output(tmp_path):
    valid_path = tmp_path / "molecule.smi"
    valid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(valid_path), str(tmp_path), "bad")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION
