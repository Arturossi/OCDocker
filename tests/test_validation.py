import pytest

import OCDocker.Toolbox.Validation as ocvalidation
import OCDocker.Error as ocerror

# Tests for is_algorithm_allowed
@pytest.mark.parametrize("path,expected", [
    ("/tmp/ap", True),
    ("/tmp/not_allowed", False),
])
def test_is_algorithm_allowed(path, expected):
    assert ocvalidation.is_algorithm_allowed(path) is expected

def test_validate_digest_extension():
    # valid extension
    assert ocvalidation.validate_digest_extension("results.json", "json")
    # invalid extension should return False after warning
    assert not ocvalidation.validate_digest_extension("results.hdf5", "hdf5")

# Tests for validate_obabel_extension
@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("molecule.smi", "smi"),
        ("molecule.bad", ocerror.ErrorCode.UNSUPPORTED_EXTENSION),
    ],
)
def test_validate_obabel_extension(file_path, expected):
    result = ocvalidation.validate_obabel_extension(file_path)
    if isinstance(expected, str):
        assert result == expected
    else:
        assert result == expected

def test_is_molecule_valid_pdb():
    pytest.importorskip("Bio.PDB")
    path = (
        "test_files/test_ptn1/receptor.pdb"
    )
    assert ocvalidation.is_molecule_valid(path)
