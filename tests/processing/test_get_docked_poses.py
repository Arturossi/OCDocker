#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for get_docked_poses error handling across dockers.
'''

# Imports
###############################################################################
import pytest

import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Docking.Smina as ocsmina
import OCDocker.Docking.Vina as ocvina

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

@pytest.mark.parametrize(
    "getter",
    [
        ocvina.get_docked_poses,
        ocplants.get_docked_poses,
        ocsmina.get_docked_poses,
    ],
)
@pytest.mark.order(66)
def test_get_docked_poses_missing_dir(getter, capsys, tmp_path):
    missing_dir = tmp_path / "non_existent"
    poses = getter(str(missing_dir))
    captured = capsys.readouterr()
    assert poses == []
    assert "does not exist" in captured.out
