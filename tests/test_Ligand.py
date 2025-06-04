import pytest

from pathlib import Path
from rdkit import Chem

import OCDocker.Ligand as ocl

@pytest.fixture
def sample_ligand():
    '''
    Fixture to create a sample Ligand instance using an RDKit molecule
    parsed from the SMILES of aspirin.
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

    # Now you can use this as your base
    mol = base / "compounds/ligands/ligand/ligand.smi"

    ligand = ocl.Ligand(molecule=str(mol), name="ligand_test")

    json_file = base / "compounds/ligands/ligand/test_ligand_descriptors.json"

    boxes_dir = base / "boxes"

    return {
        "ligand": ligand,
        "json_file": json_file,
        "mol": mol,
        "boxes_dir": boxes_dir,
        "box_path": boxes_dir / "box0.pdb"
    }

@pytest.mark.order(1)
def test_to_dict(sample_ligand):
    '''
    Test that Ligand.to_dict returns a dictionary containing key attributes.
    '''
    
    result = sample_ligand["ligand"].to_dict()
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "Name" in result, "Name should be in the dictionary"

@pytest.mark.order(2)
def test_to_json(sample_ligand):
    '''
    Test that Ligand.to_json returns a JSON string representation of the object.
    '''

    # If there is already a json_file file, remove it
    if sample_ligand["json_file"].exists():
        sample_ligand["json_file"].unlink()

    result = sample_ligand["ligand"].to_json(overwrite=True)

    assert result is not None, "Result should not be None"
    assert result == 0 or result is True, f"Result should be 0 or True. Error code: {result}"

@pytest.mark.order(3)
def test_is_valid(sample_ligand):
    '''
    Test that Ligand.is_valid returns a boolean and is True for valid input.
    '''

    assert isinstance(sample_ligand["ligand"].is_valid(), bool), "is_valid should return a boolean"
    assert sample_ligand["ligand"].is_valid(), "Ligand should be valid"

@pytest.mark.order(4)
def test_get_descriptors(sample_ligand):
    '''
    Test that get_descriptors returns all expected descriptor keys
    defined in Ligand.allDescriptors.
    '''
    
    desc = sample_ligand["ligand"].get_descriptors()
    expected_keys = ocl.Ligand.allDescriptors

    assert isinstance(desc, dict), "get_descriptors should return a dictionary"
    assert "RadiusOfGyration" in desc, "RadiusOfGyration should be in descriptors"
    assert isinstance(desc["RadiusOfGyration"], float), "RadiusOfGyration should be an float"
    assert desc["RadiusOfGyration"] > 0, "RadiusOfGyration should be positive"

    for key in expected_keys:
        assert key in desc, f"Missing descriptor: {key}"

@pytest.mark.order(5)
def test_create_box_overwrite(sample_ligand):
    box_path = sample_ligand["box_path"]
    boxes_dir = sample_ligand["boxes_dir"]

    # remove existing box if present
    if box_path.exists():
        box_path.unlink()

    result = sample_ligand["ligand"].create_box(savePath = str(boxes_dir))
    assert result is None
    assert box_path.exists(), "box0.pdb should be created"
    # calling again without overwrite should return an int error code
    result_again = sample_ligand["ligand"].create_box(savePath = str(boxes_dir), overwrite = False)
    assert isinstance(result_again, int)
    # with overwrite should succeed
    result_over = sample_ligand["ligand"].create_box(savePath = str(boxes_dir), overwrite = True)
    assert result_over is None

@pytest.mark.order(6)
def test_same_molecule_checks(sample_ligand):
    lig = sample_ligand["ligand"]
    
    # Compare ligand with itself
    assert lig.is_same_molecule(lig) is True
    assert lig.is_same_molecule_SMILES(lig) is True

    # Compare with a different molecule
    other = Chem.MolFromSmiles("CC")
    assert lig.is_same_molecule(other) is False
    assert lig.is_same_molecule_SMILES(other) is False
