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
