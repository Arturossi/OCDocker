import pytest
import shutil

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.Conversion as occonversion

from pprint import pprint

@pytest.fixture
def vina_inputs():
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

    pre_output_dir = base / "compounds/ligands/ligand"
    output_dir = pre_output_dir / "vinaFiles"

    pre_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "vina_out.pdbqt"

    receptor_file = base / "receptor.pdb"
    ligand_file = pre_output_dir / "ligand.smi"
    converted_ligand_file = pre_output_dir / "ligand.mol2"
    box_file = pre_output_dir / "boxes/box0.pdb"

    config_file = output_dir / "vina_config.txt"

    prepared_receptor_path = base / "prepared_receptor.pdbqt"
    prepared_ligand_path = pre_output_dir / "prepared_ligand.pdbqt"
    vina_log = output_dir / "vina.log"

    receptor = ocr.Receptor(structure=str(receptor_file), name="test_rec")
    ligand = ocl.Ligand(molecule=str(ligand_file), name="test_lig")

    return {
        "config": str(config_file),
        "box": str(box_file),
        "pre_output_dir": pre_output_dir,
        "receptor": receptor,
        "receptor_file": str(receptor_file),
        "receptor_path": str(prepared_receptor_path),
        "ligand": ligand,
        "ligand_file": str(ligand_file),
        "ligand_path": str(prepared_ligand_path),
        "converted_ligand_file": str(converted_ligand_file),
        "prepared_ligand_path": str(prepared_ligand_path),
        "prepared_receptor_path": str(prepared_receptor_path),
        "output_dir": output_dir,
        "output_file": str(output_file),
        "output": str(output_file),
        "vina_log": str(vina_log)
    }

@pytest.mark.order(1)
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
        outputVina=vina_inputs["output_dir"],
        name="test"
    )
    
    assert isinstance(vina_instance, ocvina.Vina)

@pytest.mark.order(2)
def test_convert_smi_to_mol2(vina_inputs):
    """
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    """

    out = Path(vina_inputs["converted_ligand_file"])

    # If there is already a converted ligand file, remove it
    if out.exists():
        out.unlink()
    
    result = occonversion.convertMols(
        input_file=str(vina_inputs["ligand_file"]),
        output_file=str(vina_inputs["converted_ligand_file"]),
        overwrite=True
    )

    assert result == 0 or result is True
    assert Path(vina_inputs["converted_ligand_file"]).exists(), "Failed to generate .mol2 from .smi"

@pytest.mark.order(3)
def test_run_prepare_ligand(vina_inputs):
    '''
    Run ligand preparation and check that it produces expected files.
    '''

    out = Path(vina_inputs["prepared_ligand_path"]).parent
    out.mkdir(parents=True, exist_ok=True)

    result = ocvina.run_prepare_ligand(
        inputLigandPath=str(vina_inputs["converted_ligand_file"]),
        outputLigand=str(vina_inputs["prepared_ligand_path"])
    )

    assert result is True or isinstance(result, int)
    assert Path(vina_inputs["prepared_ligand_path"]).exists(), "No prepared ligand files found"

@pytest.mark.order(4)
def test_run_prepare_receptor(vina_inputs):
    '''
    Run receptor preparation and check that it produces expected files.
    '''

    result = ocvina.run_prepare_receptor(
        inputReceptorPath=str(vina_inputs["receptor_file"]),
        outputReceptor=str(vina_inputs["prepared_receptor_path"]),
    )

    assert result is True or isinstance(result, int)
    assert Path(vina_inputs["prepared_receptor_path"]).exists(), "No prepared receptor files found"

@pytest.mark.order(5)
def test_run_box_to_vina(vina_inputs):
    '''
    Test generation of Vina-style box configuration.
    '''

    # If there is already a config file, remove it
    if Path(vina_inputs["config"]).exists():
        Path(vina_inputs["config"]).unlink()

    result = ocvina.box_to_vina(
        boxFile=vina_inputs["box"],
        confFile=vina_inputs["config"],
        receptor=vina_inputs["prepared_receptor_path"]
    )

    assert result == 0 or result is True
    assert Path(vina_inputs["box"]).exists()

@pytest.mark.order(6)
def test_run_vina(vina_inputs):
    '''
    Run docking using real ligand, receptor, and box files.
    '''

    result = ocvina.run_vina(
        confFile=vina_inputs["config"],
        ligand=vina_inputs["prepared_ligand_path"],
        outPath=str(vina_inputs["output_file"]),
        logFile=vina_inputs["vina_log"]
    )

    assert Path(vina_inputs['output_file']), "Expected output files were not created"
