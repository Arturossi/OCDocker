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
