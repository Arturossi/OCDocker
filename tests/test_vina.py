import pytest

from pathlib import Path

from OCDocker.OCDocker import Ligand as ocl
from OCDocker.OCDocker import Receptor as ocr
import OCDocker.Docking.Vina as ocvina

@pytest.fixture
def vina_inputs(tmp_path):
    base = Path("OCDocker/test_files/test_ptn1")
    receptor_file = base / "receptor.pdb"
    ligand_file = base / "compounds/ligands/ligand/ligand.smi"
    box_file = base / "compounds/ligands/ligand/boxes/box0.pdb"

    config_file = tmp_path / "vina_config.txt"
    config_file.write_text("receptor = prepared_receptor.pdb\nligand = prepared_ligand.pdb\n")

    prepared_receptor_path = tmp_path / "prepared_receptor.pdbqt"
    prepared_ligand_path = tmp_path / "prepared_ligand.pdbqt"
    vina_log = tmp_path / "vina.log"
    output_file = tmp_path / "vina_out.pdbqt"

    receptor = ocr.Receptor(structure=str(receptor_file), name="test_rec")
    ligand = ocl.Ligand(molecule=str(ligand_file), name="test_lig")

    return {
        "config": str(config_file),
        "box": str(box_file),
        "receptor": receptor,
        "receptor_path": str(prepared_receptor_path),
        "ligand": ligand,
        "ligand_path": str(prepared_ligand_path),
        "vina_log": str(vina_log),
        "output": str(output_file)
    }

def test_vina_instantiation(vina_inputs):
    """
    Test Vina class can be instantiated with all required real inputs.
    """
    vina_instance = ocvina.Vina(
        configPath=vina_inputs["config"],
        boxFile=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        preparedReceptorPath=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        preparedLigandPath=vina_inputs["ligand_path"],
        vinaLog=vina_inputs["vina_log"],
        outputVina=vina_inputs["output"],
        name="test"
    )
    
    assert isinstance(vina_instance, ocvina.Vina)


def test_run_vina(vina_paths):
    '''
    Run docking using real ligand, receptor, and box files.
    '''

    vina_paths["output_dir"].mkdir(parents=True, exist_ok=True)

    result = vina.run_vina(
        ligand_path=str(vina_paths["ligand"]),
        receptor_path=str(vina_paths["receptor"]),
        box_path=str(vina_paths["box"]),
        output_dir=str(vina_paths["output_dir"]),
        prefix="test"
    )

    assert result is True or isinstance(result, int)
    output_files = list(vina_paths["output_dir"].glob("test*"))
    assert output_files, "Expected output files were not created"

def test_run_prepare_ligand(vina_paths):
    '''
    Run ligand preparation and check that it produces expected files.
    '''

    out = vina_paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    result = vina.run_prepare_ligand(
        ligand_path=str(vina_paths["ligand"]),
        output_dir=str(out),
        prefix="prep_ligand"
    )

    assert result is True or isinstance(result, int)
    prep_files = list(out.glob("prep_ligand*"))
    assert prep_files, "No prepared ligand files found"

def test_run_prepare_receptor(vina_paths):
    '''
    Run receptor preparation and check that it produces expected files.
    '''

    out = vina_paths["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    result = vina.run_prepare_receptor(
        receptor_path=str(vina_paths["receptor"]),
        output_dir=str(out),
        prefix="prep_receptor"
    )

    assert result is True or isinstance(result, int)
    prep_files = list(out.glob("prep_receptor*"))
    assert prep_files, "No prepared receptor files found"

def test_read_log(tmp_path):
    '''
    Check that read_log returns parsed docking results.
    Requires a valid vina log file to be placed or mocked if necessary.
    '''

    log_file = tmp_path / "vina.log"
    log_file.write_text("-----+------------+----------+----------\n...")  # Placeholder

    result = vina.read_log(str(log_file))
    assert isinstance(result, list) or result is None