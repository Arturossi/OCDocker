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
