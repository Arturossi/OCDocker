import pytest
import shutil

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Docking.PLANTS as ocplants
import OCDocker.Toolbox.Conversion as occonversion

@pytest.fixture
def plants_inputs():
    base = Path("test_files/test_ptn1")

    pre_output_dir = base / "compounds/ligands/ligand"
    plants_files_dir = pre_output_dir / "plants_files"
    output_dir = plants_files_dir / "plants_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "plants_out.mol2"

    receptor_file = base / "receptor.pdb"
    ligand_file = pre_output_dir / "ligand.smi"
    converted_ligand_file = pre_output_dir / "ligand.mol2"
    box_file = pre_output_dir / "boxes/box0.pdb"

    config_file = plants_files_dir / "plants_config.txt"
    config_file.write_text("receptor = prepared_receptor.pdb\nligand = prepared_ligand.pdb\n")

    prepared_receptor_path = base / "prepared_receptor.mol2"
    prepared_ligand_path = pre_output_dir / "prepared_ligand.mol2"
    plants_log = plants_files_dir / "plants.log"

    # If there are already plants folders, remove them
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    receptor = ocr.Receptor(structure=str(receptor_file), name="test_rec")
    ligand = ocl.Ligand(molecule=str(ligand_file), name="test_lig")

    return {
        "config": str(config_file),
        "box": str(box_file),
        "receptor": receptor,
        "receptor_file": str(receptor_file),
        "receptor_path": str(prepared_receptor_path),
        "ligand": ligand,
        "ligand_file": str(ligand_file),
        "ligand_path": str(prepared_ligand_path),
        "converted_ligand_file": str(converted_ligand_file),
        "output_dir": output_dir,
        "output": str(output_file),
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
        outputPlants=plants_inputs["output"],
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

    config_out = Path(plants_inputs["config"]).parent / "box_config.txt"

    result = ocplants.box_to_plants(
        boxFile=plants_inputs["box"],
        confFile=str(config_out),
        receptor=plants_inputs["receptor_file"],
        ligand=plants_inputs["ligand_file"],
        outputPlants=plants_inputs["output"],
        center=None,
        bindingSiteRadius=None,
        spacing=2.9
    )

    assert result == 0 or result is True
    assert config_out.exists()

@pytest.mark.order(4)
def test_run_prepare_ligand(plants_inputs):
    '''
    Run ligand preparation for PLANTS and verify output files.
    '''
    out = plants_inputs["output_dir"]
    out.mkdir(parents=True, exist_ok=True)

    output_ligand = out / "prep_ligand.mol2"
    result = ocplants.run_prepare_ligand(
        inputLigandPath=plants_inputs["converted_ligand_file"],
        outputLigand=str(output_ligand)
    )
    assert result is True or isinstance(result, int)
    assert output_ligand.exists()

@pytest.mark.order(5)
def test_run_prepare_receptor(plants_inputs):
    '''
    Run receptor preparation for PLANTS and verify output files.
    '''

    output_receptor = plants_inputs["base"] / "prep_receptor.mol2"
    result = ocplants.run_prepare_receptor(
        inputReceptorPath=plants_inputs["receptor_file"],
        outputReceptor=str(output_receptor)
    )
    assert result is True or isinstance(result, int)
    assert output_receptor.exists()

@pytest.mark.order(6)
def test_run_plants(plants_inputs):
    '''
    Run the full PLANTS docking routine and verify expected output.
    '''

    # Paths to the prepared receptor and ligand
    prepared_receptor = plants_inputs["prepared_receptor_path"]
    prepared_ligand = plants_inputs["prepared_ligand_path"]
    config_path = plants_inputs["config"]

    # Make sure these are already prepared in previous tests
    assert prepared_receptor.exists(), "Prepared receptor file missing"
    assert prepared_ligand.exists(), "Prepared ligand file missing"
    assert config_path.exists(), "PLANTS config file missing"

    result = ocplants.run_plants(
        confFile=str(config_path),
        outputPlants=str(plants_inputs["output"]),
        overwrite=True,
        logFile=plants_inputs["plants_log"]
    )

    assert result is True or isinstance(result, int)
    assert Path(plants_inputs["plants_log"]).exists(), "PLANTS log file not generated"
