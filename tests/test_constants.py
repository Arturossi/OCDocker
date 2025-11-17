import math
import pytest

import OCDocker.Toolbox.Constants as occ


@pytest.mark.order(1)
def test_cal_to_j_round_trip():
    cal = 123.4
    assert pytest.approx(cal) == occ.J_to_cal(occ.cal_to_J(cal))


@pytest.mark.order(2)
def test_c_to_k_and_back():
    celsius = 25.0
    kelvin = occ.C_to_K(celsius)
    assert pytest.approx(298.15, rel=1e-12) == kelvin
    assert pytest.approx(celsius, rel=1e-12) == occ.K_to_C(kelvin)


@pytest.mark.order(3)
def test_negative_kelvin_error():
    with pytest.raises(ValueError):
        occ.K_to_C(-1.0)


@pytest.mark.order(4)
def test_convert_Ki_Kd_to_dG_numeric():
    K = 2.0
    expected = occ.R * 298.15 * math.log(K)
    assert math.isclose(occ.convert_Ki_Kd_to_dG(K), expected, rel_tol=1e-7)


@pytest.mark.order(5)
def test_convert_dG_to_Ki_Kd_numeric():
    dG = 5.0
    expected = math.exp(-dG / (occ.R * 298.15))
    assert math.isclose(occ.convert_dG_to_Ki_Kd(dG), expected, rel_tol=1e-7)
