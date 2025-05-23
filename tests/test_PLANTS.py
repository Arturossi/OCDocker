import pytest
import shutil

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Toolbox.Conversion as occonversion

@pytest.fixture
def plants_inputs():
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
    plants_files_dir = pre_output_dir / "plantsFiles"
    plants_files_dir.mkdir(parents=True, exist_ok=True)

    receptor_file = base / "receptor.pdb"
    ligand_file = pre_output_dir / "ligand.smi"
    converted_ligand_file = pre_output_dir / "ligand.mol2"
    box_file = pre_output_dir / "boxes/box0.pdb"

    config_file = plants_files_dir / "plants_config.txt"

    prepared_receptor_path = base / "prepared_receptor.mol2"
    prepared_ligand_path = pre_output_dir / "prepared_ligand.mol2"
    plants_log = plants_files_dir / "plants.log"

    receptor = ocr.Receptor(structure=str(receptor_file), name="test_rec")
    ligand = ocl.Ligand(molecule=str(ligand_file), name="test_lig")

    return {
        "config": str(config_file),
        "box": str(box_file),
        "receptor": receptor,
        "receptor_file": str(receptor_file),
        "receptor_path": str(prepared_receptor_path),
        "plants_files_dir": str(plants_files_dir),
        "ligand": ligand,
        "ligand_file": str(ligand_file),
        "ligand_path": str(prepared_ligand_path),
        "converted_ligand_file": str(converted_ligand_file),
        "plants_log": str(plants_log)
    }

@pytest.mark.order(1)
def test_plants_instantiation(plants_inputs):
    '''
    Test PLANTS class can be instantiated with all required real inputs.
    '''
    plants_instance = ocplants.PLANTS(
        configPath=plants_inputs["config"],
        boxFile=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        preparedReceptorPath=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        preparedLigandPath=plants_inputs["ligand_path"],
        plantsLog=plants_inputs["plants_log"],
        outputPlants=plants_inputs["plants_files_dir"],
        name="test",
        boxSpacing=1.0,
        overwriteConfig=True
    )
    assert isinstance(plants_instance, ocplants.PLANTS)

@pytest.mark.order(2)
def test_convert_smi_to_mol2(plants_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    result = occonversion.convertMols(
        input_file=str(plants_inputs["ligand_file"]),
        output_file=str(plants_inputs["converted_ligand_file"]),
        overwrite=True
    )

    assert result == 0 or result is True
    assert Path(plants_inputs["converted_ligand_file"]).exists(), "Failed to generate .mol2 from .smi"

@pytest.mark.order(3)
def test_box_to_plants(plants_inputs):
    '''
    Test generation of PLANTS-style box configuration.
    '''

    result = ocplants.box_to_plants(
        boxFile=plants_inputs["box"],
        confFile=str(plants_inputs["config"]),
        receptor=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand_file"],
        outputPlants=plants_inputs["plants_files_dir"],
        center=None,
        bindingSiteRadius=None,
        spacing=2.9
    )

    assert result == 0 or result is True
    assert Path(plants_inputs["config"]).exists()

@pytest.mark.order(4)
def test_run_prepare_ligand(plants_inputs):
    '''
    Run ligand preparation for PLANTS and verify output files.
    '''

    # If there are already prepared ligand files, remove them
    if Path(plants_inputs["ligand_path"]).exists():
        Path(plants_inputs["ligand_path"]).unlink()

    result = ocplants.run_prepare_ligand(
        inputLigandPath=plants_inputs["converted_ligand_file"],
        outputLigand=plants_inputs["ligand_path"]
    )
    assert result is True or isinstance(result, int)
    assert Path(plants_inputs["ligand_path"]).exists()

@pytest.mark.order(5)
def test_run_prepare_receptor(plants_inputs):
    '''
    Run receptor preparation for PLANTS and verify output files.
    '''

    result = ocplants.run_prepare_receptor(
        inputReceptorPath=str(plants_inputs["receptor_file"]),
        outputReceptor=str(plants_inputs["receptor_path"])
    )
    assert result is True or isinstance(result, int)
    assert Path(plants_inputs["receptor_path"]).exists()

@pytest.mark.order(6)
def test_run_plants(plants_inputs):
    '''
    Run the full PLANTS docking routine and verify expected output.
    '''

    # If there is a run directory inside the plants_files_dir, remove it
    plants_run_dir = Path(plants_inputs["plants_files_dir"]) / "run"
    if plants_run_dir.exists():
        if plants_run_dir.is_dir():
            shutil.rmtree(plants_run_dir)
        else:
            plants_run_dir.unlink()

    # Make sure these are already prepared in previous tests
    assert Path(plants_inputs["receptor_path"]).exists(), "Prepared receptor file missing"
    assert Path(plants_inputs["ligand_path"]).exists(), "Prepared ligand file missing"
    assert Path(plants_inputs["config"]).exists(), "PLANTS config file missing"

    result = ocplants.run_plants(
        confFile=plants_inputs["config"],
        outputPlants=plants_inputs["plants_files_dir"],
        overwrite=False,
        logFile=plants_inputs["plants_log"]
    )

    assert result is True or isinstance(result, int)
    assert Path(plants_inputs["plants_log"]).exists(), "PLANTS log file not generated"
