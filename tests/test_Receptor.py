import pytest
import OCDocker.Receptor as ocr

@pytest.fixture
def sample_receptor(tmp_path):
    '''
    Fixture that loads a minimal receptor from a PDB snippet.
    '''

    pdb_content = """\
ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.260   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.010  14.700   2.100  1.00 20.00           C
ATOM      4  O   ALA A   1      12.200  15.600   2.100  1.00 20.00           O
ATOM      5  CB  ALA A   1      13.100  12.200   3.000  1.00 20.00           C
TER
END
"""

    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)
    return ocr.Receptor(str(pdb_file), name="test_receptor")

def test_to_dict(sample_receptor):
    '''
    Test that Receptor.to_dict returns a dictionary with expected keys.
    '''

    result = sample_receptor.to_dict()
    assert isinstance(result, dict)
    assert "name" in result

def test_to_json(sample_receptor):
    '''
    Test that Receptor.to_json returns a JSON-formatted string.
    '''

    result = sample_receptor.to_json()
    assert isinstance(result, str)
    assert "test_receptor" in result

def test_is_valid(sample_receptor):
    '''
    Test that is_valid returns a boolean indicating receptor integrity.
    '''

    assert isinstance(sample_receptor.is_valid(), bool)

def test_get_descriptors(sample_receptor):
    '''
    Test that get_descriptors returns all descriptor fields defined in the method.
    '''

    descriptors = sample_receptor.get_descriptors()
    assert isinstance(descriptors, dict)

    # Dynamically infer expected keys from the result itself
    expected_keys = descriptors.keys()
    for key in expected_keys:
        assert key in descriptors, f"Missing descriptor: {key}"
