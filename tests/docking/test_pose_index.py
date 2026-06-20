#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests pose index extraction helpers for Vina-like docking outputs.
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
    "func,file_name,expected",
    [
        (ocvina.get_pose_index_from_file_path, "pose_split_5.pdbqt", 5),
        (ocsmina.get_pose_index_from_file_path, "pose_split_5.pdbqt", 5),
        (ocplants.get_pose_index_from_file_path, "ligand_pose_3.mol2", 3),
    ],
)
@pytest.mark.order(72)
def test_get_pose_index_from_file_path(func, file_name, expected):
    assert func(file_name) == expected
