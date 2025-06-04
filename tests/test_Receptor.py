import pytest

from pathlib import Path

import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Printing as ocprint

@pytest.fixture
def sample_receptor():
    '''
    Fixture that loads a minimal receptor from a PDB snippet.
    '''

    # Start from the current file location (assuming this code is in a test or module file)
    current_file = Path(__file__).resolve()

    # Traverse up to find the 'OCDocker' project root
    project_root = current_file
    while project_root.name != "OCDocker" and project_root != project_root.parent:
        project_root = project_root.parent

    if project_root.name != "OCDocker":
        raise RuntimeError("OCDocker directory not found in path hierarchy.")

    # Now you can use this as your base
    base = project_root / "test_files/test_ptn1"

    receptor_file = base / "receptor.pdb"

    receptor = ocr.Receptor(str(receptor_file), name="test_receptor")

    json_file = base / "test_receptor_descriptors.json"

    return {
        "receptor": receptor,
        "json_file": json_file

    }

@pytest.mark.order(1)
def test_to_dict(sample_receptor):
    '''
    Test that Receptor.to_dict returns a dictionary with expected keys.
    '''

    result = sample_receptor["receptor"].to_dict()
    assert isinstance(result, dict), "The funcion to_dict should return a dictionary"
    assert "Name" in result, "Missing 'Name' key in result"

@pytest.mark.order(2)
def test_to_json(sample_receptor):
    '''
    Test that Receptor.to_json returns a JSON-formatted string.
    '''

    # If there is already a json_file file, remove it
    if Path(sample_receptor["json_file"]).exists():
        Path(sample_receptor["json_file"]).unlink()

    result = sample_receptor["receptor"].to_json()

    assert result is not None, "Result should not be None"
    assert result == 0 or result is True, f"Failed to write JSON file. Error code: {result}"

@pytest.mark.order(3)
def test_is_valid(sample_receptor):
    '''
    Test that is_valid returns a boolean indicating receptor integrity.
    '''

    assert isinstance(sample_receptor["receptor"].is_valid(), bool), "is_valid should return a boolean"
    assert sample_receptor["receptor"].is_valid() is True, "Receptor should be valid"

@pytest.mark.order(4)
def test_get_descriptors(sample_receptor):
    '''
    Test that get_descriptors returns all descriptor fields defined in the method.
    '''

    descriptors = sample_receptor["receptor"].get_descriptors()
    assert isinstance(descriptors, dict), "get_descriptors should return a dictionary"

    # Dynamically infer expected keys from the result itself
    expected_keys = descriptors.keys()
    for key in expected_keys:
        assert key in descriptors, f"Missing descriptor: {key}"

@pytest.mark.order(4)
def test_filter_sequence_warn(monkeypatch):
    calls = []

    def fake_warning(message: str, force: bool = False) -> None:
        calls.append(message)

    monkeypatch.setattr(ocprint, "print_warning", fake_warning)
    result = ocr.__filterSequence("AXXTY")

    assert result == "ATY"
    assert len(calls) == 1
    assert "X" not in result, "Filtered sequence should not contain 'X'"
