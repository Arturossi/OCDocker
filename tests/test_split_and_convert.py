import pytest
import OCDocker.Error as ocerror

import OCDocker.Toolbox.Conversion as occonversion


@pytest.mark.order(1)
def test_split_and_convert_invalid_input(tmp_path):
    invalid_path = tmp_path / "molecule.bad"
    invalid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(invalid_path), str(tmp_path), "sdf")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(2)
def test_split_and_convert_invalid_output(tmp_path):
    valid_path = tmp_path / "molecule.smi"
    valid_path.write_text("CCO")
    result = occonversion.split_and_convert(str(valid_path), str(tmp_path), "bad")
    assert result == ocerror.ErrorCode.UNSUPPORTED_EXTENSION
