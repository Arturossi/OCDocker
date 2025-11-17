import math
import pytest

import OCDocker.Toolbox.Conversion as occonversion
import OCDocker.Toolbox.Constants as occ


@pytest.mark.order(1)
@pytest.mark.parametrize("kikd, order, factor", [
    (1.0, "nM", 1e9),
    (2.0, "uM", 1e6),
])
def test_kikd_to_deltag_various_orders(kikd, order, factor):
    expected = -occ.RJ * occ.ZERO_C_IN_K * math.log(kikd * factor)
    result = occonversion.kikd_to_deltag(kikd, T = occ.ZERO_C_IN_K, kikd_order = order)
    assert math.isclose(result, expected, rel_tol = 1e-5)


@pytest.mark.order(2)
def test_convert_from_string_and_file(tmp_path):
    smiles = "CCO"
    out_from_str = tmp_path / "string_out.sdf"
    out_from_file = tmp_path / "file_out.sdf"

    # convert from SMILES string
    res_str = occonversion.convert_mols_from_string(smiles, str(out_from_str))
    assert res_str == 0 or res_str is True
    assert out_from_str.exists()

    # write smiles to file then convert
    smi_path = tmp_path / "mol.smi"
    smi_path.write_text(smiles)
    res_file = occonversion.convert_mols(str(smi_path), str(out_from_file))
    assert res_file == 0 or res_file is True
    assert out_from_file.exists()
