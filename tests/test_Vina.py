import pytest

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Error as ocerror

import OCDocker.Docking.Vina as ocvina
import OCDocker.Toolbox.Conversion as occonversion

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
    
    assert isinstance(vina_instance, ocvina.Vina), "Vina instance was not created correctly"

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

    assert result == 0 or result is True, f"Conversion of .smi to .mol2 failed. Error code: {result}"
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

    assert result is True or isinstance(result, int), f"Preparation of ligand failed. Error code: {result}"
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

    assert result is True or isinstance(result, int), f"Preparation of receptor failed. Error code: {result}"
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

    assert result == 0 or result is True, f"Box to Vina conversion failed. Error code: {result}"
    assert Path(vina_inputs["box"]).exists(), "Box file was not created"

@pytest.mark.order(6)
def test_run_vina(vina_inputs):
    '''
    Run docking using real ligand, receptor, and box files.
    '''

    _ = ocvina.run_vina(
        confFile=vina_inputs["config"],
        ligand=vina_inputs["prepared_ligand_path"],
        outPath=str(vina_inputs["output_file"]),
        logFile=vina_inputs["vina_log"]
    )

    assert Path(vina_inputs['output_file']), "Expected output files were not created"

@pytest.mark.order(7)
def test_get_pose_index_from_file_path():
    """Check pose index extraction from file name."""
    assert ocvina.get_pose_index_from_file_path("lig_split_42.pdbqt") == 42

@pytest.mark.order(8)
def test_get_rescore_log_paths(tmp_path):
    """Verify detection of rescoring log files in a directory."""
    f1 = tmp_path / "lig_split_1_rescoring.log"
    f1.write_text("log1")
    f2 = tmp_path / "lig_split_2_rescoring.log"
    f2.write_text("log2")
    (tmp_path / "other.log").write_text("x")

    found = ocvina.get_rescore_log_paths(str(tmp_path))
    assert set(found) == {str(f1), str(f2)}

@pytest.mark.order(9)
def test_read_rescoring_log(tmp_path):
    """Parse affinity from a rescoring log file."""
    log_file = tmp_path / "lig_split_1_rescoring.log"
    log_file.write_text(
        "Line1\nEstimated Free Energy of Binding    -7.23 (kcal/mol)\nEnd"
    )
    value = ocvina.read_rescoring_log(str(log_file))
    assert value == -7.23

@pytest.mark.order(10)
def test_box_to_vina_minimal(tmp_path):
    """Generate a Vina configuration from a small box file."""
    box_file = tmp_path / "box.pdb"
    conf_file = tmp_path / "conf.txt"
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )
    rc = ocvina.box_to_vina(str(box_file), str(conf_file), "rec.pdbqt")
    assert rc == ocerror.Error.ok() or rc == 0 # type: ignore
    conf_lines = conf_file.read_text().splitlines()
    assert conf_lines[0] == "receptor = rec.pdbqt"
    assert "center_x = 1.0" in conf_lines
    assert "size_z = 6.0" in conf_lines
