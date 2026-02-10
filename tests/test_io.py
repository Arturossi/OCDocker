#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for lazy file readers in Toolbox.IO.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Toolbox.IO as ocio

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

@pytest.mark.order(67)
def test_lazyread(tmp_path):
    lines = ["line1\n", "line2\n", "line3"]
    file_path = tmp_path / "example.txt"
    file_path.write_text("".join(lines))

    expected_in_order = lines
    assert list(ocio.lazyread(str(file_path))) == expected_in_order
    assert list(ocio.lazyread_mmap(str(file_path))) == expected_in_order

    expected_reverse = [line.rstrip("\n") for line in reversed(lines)]
    assert list(ocio.lazyread_reverse_order(str(file_path))) == expected_reverse
    assert list(ocio.lazyread_reverse_order_mmap(str(file_path))) == expected_reverse
