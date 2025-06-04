import pytest

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr

import OCDocker.Docking.Smina as ocsmina
import OCDocker.Toolbox.Conversion as occonversion

@pytest.fixture
def smina_inputs():
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

    output_dir = pre_output_dir / "sminaFiles"

    pre_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "smina_out.pdbqt"

    receptor_file = base / "receptor.pdb"
    ligand_file = pre_output_dir / "ligand.smi"
    converted_ligand_file = pre_output_dir / "ligand.mol2"
    box_file = pre_output_dir / "boxes/box0.pdb"

    config_file = output_dir / "smina_config.txt"

    prepared_receptor_path = base / "prepared_receptor.pdbqt"
    prepared_ligand_path = output_dir / "prepared_ligand.pdbqt"
    smina_log = output_dir / "smina.log"
    smina_exec_log = output_dir / "smina_exec.log"
    
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
        "smina_log": str(smina_log),
        "smina_exec_log": str(smina_exec_log),
    }

@pytest.mark.order(1)
def test_smina_instantiation(smina_inputs):
    '''
    Test Smina class can be instantiated with all required real inputs.
    '''

    smina_instance = ocsmina.Smina(
        configPath=smina_inputs["config"],
        boxFile=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        preparedReceptorPath=smina_inputs["prepared_receptor_path"],
        ligand=smina_inputs["ligand"],
        preparedLigandPath=smina_inputs["prepared_ligand_path"],
        sminaLog=smina_inputs["smina_log"],
        outputSmina=smina_inputs["output_file"],
        name="test",
        overwriteConfig=True
    )
    assert isinstance(smina_instance, ocsmina.Smina), "Smina instance not created correctly"

@pytest.mark.order(2)
def test_convert_smi_to_mol2(smina_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    result = occonversion.convertMols(
        input_file=str(smina_inputs["ligand_file"]),
        output_file=str(smina_inputs["converted_ligand_file"]),
        overwrite=True
    )

    assert result == 0 or result is True, f"Conversion from .smi to .mol2 failed. Error code: {result}"
    assert Path(smina_inputs["converted_ligand_file"]).exists(), "Failed to generate .mol2 from .smi"

@pytest.mark.order(3)
def test_run_prepare_ligand(smina_inputs):
    '''
    Run ligand preparation and check that it produces expected files.
    '''

    result = ocsmina.run_prepare_ligand(
        inputLigandPath=smina_inputs["converted_ligand_file"],
        preparedLigand=str(smina_inputs["prepared_ligand_path"]),
    )

    assert result is True or isinstance(result, int), "Ligand preparation failed"
    assert Path(smina_inputs["prepared_ligand_path"]).exists(), "Prepared ligand file not found"

@pytest.mark.order(4)
def test_run_prepare_receptor(smina_inputs):
    '''
    Run receptor preparation and check that it produces expected files.
    '''

    result = ocsmina.run_prepare_receptor(
        inputReceptorPath=str(smina_inputs["receptor_file"]),
        preparedReceptor=str(smina_inputs["prepared_receptor_path"]),
    )

    assert result is True or isinstance(result, int), "Receptor preparation failed"
    assert Path(smina_inputs["prepared_receptor_path"]).exists(), "No prepared receptor files found"

@pytest.mark.order(5)
def test_run_gen_smina_conf(smina_inputs):
    '''
    Test generation of Vina-style box configuration.
    '''

    # If there is already a config file, remove it
    if Path(smina_inputs["config"]).exists():
        Path(smina_inputs["config"]).unlink()

    assert Path(smina_inputs["box"]).exists(), "Box file not found"

    result = ocsmina.gen_smina_conf(
        boxFile=smina_inputs["box"],
        confFile=smina_inputs["config"],
        receptor=smina_inputs["prepared_receptor_path"]
    )

    assert result == 0 or result is True, f"Configuration generation failed. Error code: {result}"

@pytest.mark.order(6)
def test_run_smina(tmp_path, smina_inputs):
    '''
    Run Smina docking using real prepared ligand and receptor.
    '''
    
    smina_inputs["output_dir"].mkdir(parents=True, exist_ok=True)

    result = ocsmina.run_smina(
        config=smina_inputs["config"],
        preparedLigand=smina_inputs["prepared_ligand_path"],
        outputSmina=smina_inputs["output_file"],
        sminaLog=smina_inputs["smina_log"],
        logPath=smina_inputs["smina_exec_log"]
    )

    assert result is True or isinstance(result, int), f"Smina docking failed. Error code: {result}"
    assert Path(smina_inputs["output_file"]).exists(), "Docking output file not created"
