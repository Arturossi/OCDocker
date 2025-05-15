import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
import OCDocker.Ligand as ocl

@pytest.fixture
def sample_ligand():
    '''

    Fixture to create a sample Ligand instance using an RDKit molecule
    parsed from the SMILES of aspirin.
    '''

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles) # type: ignore
    mol = Chem.AddHs(mol) # type: ignore
    AllChem.EmbedMolecule(mol) # type: ignore

    return ocl.Ligand(molecule=mol, name="aspirin")

def test_to_dict(sample_ligand):
    '''
    Test that Ligand.to_dict returns a dictionary containing key attributes.
    '''
    
    result = sample_ligand.to_dict()
    assert isinstance(result, dict)
    assert "name" in result

def test_to_json(sample_ligand):
    '''
    Test that Ligand.to_json returns a JSON string representation of the object.
    '''

    result = sample_ligand.to_json()
    assert isinstance(result, str)
    assert "aspirin" in result

def test_is_valid(sample_ligand):
    '''
    Test that Ligand.is_valid returns a boolean and is True for valid input.
    '''

    assert isinstance(sample_ligand.is_valid(), bool)
    assert sample_ligand.is_valid()

def test_get_descriptors(sample_ligand):
    '''
    Test that get_descriptors returns all expected descriptor keys
    defined in Ligand.allDescriptors.
    '''
    
    desc = sample_ligand.get_descriptors()
    expected_keys = ocl.Ligand.allDescriptors
    assert isinstance(desc, dict)
    for key in expected_keys:
        assert key in desc, f"Missing descriptor: {key}"
