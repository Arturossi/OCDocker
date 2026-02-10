#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for physical constants and conversion helpers.

Usage:

pytest tests/test_constants.py
'''

# Imports
###############################################################################
import math
import pytest

import OCDocker.Toolbox.Constants as occ

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



@pytest.mark.order(2)
def test_c_to_k_and_back():
    celsius = 25.0
    kelvin = occ.C_to_K(celsius)
    assert pytest.approx(298.15, rel=1e-12) == kelvin
    assert pytest.approx(celsius, rel=1e-12) == occ.K_to_C(kelvin)


# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##


@pytest.mark.order(1)
def test_cal_to_j_round_trip():
    cal = 123.4
    assert pytest.approx(cal) == occ.J_to_cal(occ.cal_to_J(cal))



@pytest.mark.order(5)
def test_convert_dG_to_Ki_Kd_numeric():
    dG = 5.0
    expected = math.exp(-dG / (occ.R * 298.15))
    assert math.isclose(occ.convert_dG_to_Ki_Kd(dG), expected, rel_tol=1e-7)



@pytest.mark.order(4)
def test_convert_Ki_Kd_to_dG_numeric():
    K = 2.0
    expected = occ.R * 298.15 * math.log(K)
    assert math.isclose(occ.convert_Ki_Kd_to_dG(K), expected, rel_tol=1e-7)



@pytest.mark.order(3)
def test_negative_kelvin_error():
    with pytest.raises(ValueError):
        occ.K_to_C(-1.0)
