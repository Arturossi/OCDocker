import pytest

import OCDocker.Toolbox.Validation as ocvalidation
import OCDocker.Error as ocerror

# Tests for is_algorithm_allowed
@pytest.mark.parametrize("path,expected", [
    ("/tmp/ap", True),
    ("/tmp/not_allowed", False),
])
@pytest.mark.order(1)
def test_is_algorithm_allowed(path, expected):
    assert ocvalidation.is_algorithm_allowed(path) is expected


@pytest.mark.order(2)
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
@pytest.mark.order(3)
def test_validate_obabel_extension(file_path, expected):
    result = ocvalidation.validate_obabel_extension(file_path)
    if isinstance(expected, str):
        assert result == expected
    else:
        assert result == expected


@pytest.mark.order(4)
def test_is_molecule_valid_pdb():
    pytest.importorskip("Bio.PDB")
    from pathlib import Path
    # Get absolute path to receptor file
    test_dir = Path(__file__).resolve().parent.parent
    path = test_dir / "test_files/test_ptn1/receptor.pdb"
    assert ocvalidation.is_molecule_valid(str(path))


@pytest.mark.order(5)
def test_is_molecule_valid_missing_file(tmp_path):
    missing = tmp_path / "missing.pdb"
    assert not ocvalidation.is_molecule_valid(str(missing))


@pytest.mark.order(6)
def test_is_molecule_valid_bad_extension(tmp_path):
    bad = tmp_path / "dummy.xyz"
    bad.write_text("dummy")
    assert not ocvalidation.is_molecule_valid(str(bad))
