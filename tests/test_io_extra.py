#!/usr/bin/env python3

# Description
###############################################################################
'''
Extra tests for lazy read helpers in Toolbox.IO.
'''

# Imports
###############################################################################
import os

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

@pytest.mark.order(68)
def test_lazyread_and_reverse(tmp_path):
    p = tmp_path / "sample.txt"
    lines = [f"line-{i}" for i in range(5)]
    p.write_text("\n".join(lines) + "\n")

    # Forward mmap reader
    fwd = list(ocio.lazyread_mmap(str(p)))
    assert [s.strip() for s in fwd] == lines

    # Reverse mmap reader
    rev = list(ocio.lazyread_reverse_order_mmap(str(p)))
    assert [s.strip() for s in rev] == list(reversed(lines))

    # Non-mmap variants
    fwd2 = list(ocio.lazyread(str(p)))
    assert [s.strip() for s in fwd2] == lines
    rev2 = list(ocio.lazyread_reverse_order(str(p)))
    assert [s.strip() for s in rev2] == list(reversed(lines))
