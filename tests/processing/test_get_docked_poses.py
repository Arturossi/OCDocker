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
