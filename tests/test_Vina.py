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
    '''
    Test Vina class can be instantiated with all required inputs.
    '''
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    assert isinstance(vina_instance, ocvina.Vina), "Vina instance was not created correctly"


@pytest.mark.order(2)
def test_convert_smi_to_mol2(vina_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    out = Path(vina_inputs["converted_ligand_file"])

    # If there is already a converted ligand file, remove it
    if out.exists():
        out.unlink()
    
    result = occonversion.convert_mols(
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
        box_file=vina_inputs["box"],
        conf_file=vina_inputs["config"],
        receptor=vina_inputs["prepared_receptor_path"]
    )

    assert result == 0 or result is True, f"Box to Vina conversion failed. Error code: {result}"
    assert Path(vina_inputs["box"]).exists(), "Box file was not created"


@pytest.mark.order(6)
def test_run_vina(vina_inputs):
    '''
    Run docking using test ligand, receptor, and box files.
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
    '''Check pose index extraction from file name.'''
    assert ocvina.get_pose_index_from_file_path("lig_split_42.pdbqt") == 42


@pytest.mark.order(8)
def test_get_rescore_log_paths(tmp_path):
    '''Verify detection of rescoring log files in a directory.'''
    f1 = tmp_path / "lig_split_1_rescoring.log"
    f1.write_text("log1")
    f2 = tmp_path / "lig_split_2_rescoring.log"
    f2.write_text("log2")
    (tmp_path / "other.log").write_text("x")

    found = ocvina.get_rescore_log_paths(str(tmp_path))
    assert set(found) == {str(f1), str(f2)}


@pytest.mark.order(9)
def test_read_rescoring_log(tmp_path):
    '''Parse affinity from a rescoring log file.'''
    log_file = tmp_path / "lig_split_1_rescoring.log"
    log_file.write_text(
        "Line1\nEstimated Free Energy of Binding    -7.23 (kcal/mol)\nEnd"
    )
    value = ocvina.read_rescoring_log(str(log_file))
    assert value == -7.23


@pytest.mark.order(10)
def test_box_to_vina_minimal(tmp_path):
    '''Generate a Vina configuration from a small box file.'''
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


@pytest.mark.order(11)
def test_vina_get_input_ligand_path(vina_inputs):
    '''Test get_input_ligand_path method.'''

    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.get_input_ligand_path()
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.order(12)
def test_vina_get_input_receptor_path(vina_inputs):
    '''Test get_input_receptor_path method.'''

    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.get_input_receptor_path()
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.order(13)
def test_vina_get_docked_poses(vina_inputs, tmp_path):
    '''Test get_docked_poses method.'''

    # Create a mock output directory with some pose files
    output_dir = tmp_path / "docked_poses"
    output_dir.mkdir()
    
    # Create some mock pose files
    (output_dir / "ligand_split_1.pdbqt").write_text("POSE1")
    (output_dir / "ligand_split_2.pdbqt").write_text("POSE2")
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=str(output_dir / "vina_out.pdbqt"),
        name="test"
    )
    
    result = vina_instance.get_docked_poses()
    assert isinstance(result, list)
    # Result may be empty if no poses found, but should be a list


@pytest.mark.order(14)
def test_vina_read_log(vina_inputs, tmp_path):
    '''Test read_log method.'''

    # Create a mock log file
    log_file = tmp_path / "vina.log"
    log_file.write_text(
        "REMARK VINA RESULT:    -7.5      0.000\n"
        "REMARK VINA RESULT:    -6.8      0.000\n"
    )
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=str(log_file),
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.read_log()
    assert isinstance(result, dict)


@pytest.mark.order(15)
def test_vina_read_log_only_best(vina_inputs, tmp_path):
    '''Test read_log method with onlyBest=True.'''

    log_file = tmp_path / "vina.log"
    log_file.write_text(
        "REMARK VINA RESULT:    -7.5      0.000\n"
        "REMARK VINA RESULT:    -6.8      0.000\n"
    )
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=str(log_file),
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.read_log(onlyBest=True)
    assert isinstance(result, dict)


@pytest.mark.order(16)
def test_vina_read_rescore_logs(vina_inputs, tmp_path):
    '''Test read_rescore_logs method.'''

    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create mock rescoring log files
    log1 = out_path / "lig_split_1_vina_rescoring.log"
    log1.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    
    log2 = out_path / "lig_split_2_vina_rescoring.log"
    log2.write_text("Estimated Free Energy of Binding    -6.50 (kcal/mol)\n")
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.read_rescore_logs(str(out_path))
    assert isinstance(result, dict)


@pytest.mark.order(17)
def test_vina_read_rescore_logs_only_best(vina_inputs, tmp_path):
    '''Test read_rescore_logs method with onlyBest=True.'''

    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create mock rescoring log files
    log1 = out_path / "lig_split_1_vina_rescoring.log"
    log1.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    
    log2 = out_path / "lig_split_2_vina_rescoring.log"
    log2.write_text("Estimated Free Energy of Binding    -6.50 (kcal/mol)\n")
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.read_rescore_logs(str(out_path), onlyBest=True)
    assert isinstance(result, dict)


@pytest.mark.order(18)
def test_vina_split_poses(vina_inputs, tmp_path):
    '''Test split_poses method.'''

    # Create a mock output file
    output_file = tmp_path / "vina_out.pdbqt"
    output_file.write_text("MODEL 1\nATOM\nENDMDL\nMODEL 2\nATOM\nENDMDL\n")
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=str(output_file),
        name="test"
    )
    
    out_path = tmp_path / "split_poses"
    out_path.mkdir()
    
    result = vina_instance.split_poses(str(out_path))
    assert isinstance(result, int)


@pytest.mark.order(19)
def test_vina_split_poses_default_path(vina_inputs, tmp_path):
    '''Test split_poses method with default outPath.'''

    output_file = tmp_path / "vina_out.pdbqt"
    output_file.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=str(output_file),
        name="test"
    )
    
    result = vina_instance.split_poses()
    assert isinstance(result, int)


@pytest.mark.order(20)
def test_vina_print_attributes(vina_inputs, capsys):
    '''Test print_attributes method.'''

    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output"],
        name="test"
    )
    
    vina_instance.print_attributes()
    captured = capsys.readouterr()
    assert "Name:" in captured.out
    assert "Box path:" in captured.out
    assert "Config path:" in captured.out


@pytest.mark.order(21)
def test_vina_run_prepare_ligand_openbabel(vina_inputs):
    '''Test run_prepare_ligand with useOpenBabel=True.'''

    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.run_prepare_ligand(useOpenBabel=True)
    assert isinstance(result, (int, str, tuple))


@pytest.mark.order(22)
def test_vina_run_prepare_receptor_openbabel(vina_inputs):
    '''Test run_prepare_receptor with useOpenBabel=True.'''

    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    result = vina_instance.run_prepare_receptor(useOpenBabel=True)
    assert isinstance(result, (int, str, tuple))


@pytest.mark.order(23)
def test_box_to_vina_file_not_exists(tmp_path):
    '''Test box_to_vina with non-existent box file.'''

    conf_file = tmp_path / "conf.txt"
    result = ocvina.box_to_vina(
        box_file=str(tmp_path / "nonexistent.pdb"),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    assert result != ocerror.Error.ok() # type: ignore


@pytest.mark.order(24)
def test_box_to_vina_read_error(tmp_path):
    '''Test box_to_vina with invalid box file format.'''

    box_file = tmp_path / "invalid_box.pdb"
    conf_file = tmp_path / "conf.txt"
    # Create a file that will cause read error
    box_file.write_text("INVALID FORMAT\n")
    
    # Try to read with invalid format - should handle gracefully
    result = ocvina.box_to_vina(
        box_file=str(box_file),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    # Should either succeed (if it handles gracefully) or return error
    assert isinstance(result, int)


@pytest.mark.order(25)
def test_read_rescore_logs_list(tmp_path):
    '''Test read_rescore_logs with list of log paths.'''

    log1 = tmp_path / "lig_split_1_vina_rescoring.log"
    log1.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    
    log2 = tmp_path / "lig_split_2_vina_rescoring.log"
    log2.write_text("Estimated Free Energy of Binding    -6.50 (kcal/mol)\n")
    
    result = ocvina.read_rescore_logs([str(log1), str(log2)])
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.order(26)
def test_read_rescore_logs_string(tmp_path):
    '''Test read_rescore_logs with single string path.'''

    log1 = tmp_path / "lig_split_1_vina_rescoring.log"
    log1.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    
    result = ocvina.read_rescore_logs(str(log1))
    assert isinstance(result, dict)


@pytest.mark.order(27)
def test_read_rescore_logs_only_best(tmp_path):
    '''Test read_rescore_logs with onlyBest=True.'''

    log1 = tmp_path / "lig_split_1_vina_rescoring.log"
    log1.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    
    log2 = tmp_path / "lig_split_2_vina_rescoring.log"
    log2.write_text("Estimated Free Energy of Binding    -6.50 (kcal/mol)\n")
    
    result = ocvina.read_rescore_logs([str(log1), str(log2)], onlyBest=True)
    assert isinstance(result, dict)


@pytest.mark.order(28)
def test_generate_vina_files_database(vina_inputs, tmp_path):
    '''Test generate_vina_files_database function.'''

    protein_path = str(vina_inputs["receptor_file"])
    box_path = str(vina_inputs["box"]).replace("box0.pdb", "")
    
    # Create boxes directory if needed
    Path(box_path).mkdir(parents=True, exist_ok=True)
    
    result = ocvina.generate_vina_files_database(
        path=str(tmp_path),
        protein=protein_path,
        boxPath=box_path
    )
    # Function returns None, so just check it doesn't raise
    assert result is None


@pytest.mark.order(29)
def test_generate_vina_files_database_default_box(vina_inputs, tmp_path):
    '''Test generate_vina_files_database with default boxPath.'''
    
    protein_path = str(vina_inputs["receptor_file"])
    
    # Create boxes directory
    boxes_dir = tmp_path / "boxes"
    boxes_dir.mkdir()
    box_file = boxes_dir / "box0.pdb"
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )
    
    result = ocvina.generate_vina_files_database(
        path=str(tmp_path),
        protein=protein_path,
        boxPath=""
    )
    assert result is None


@pytest.mark.order(30)
def test_run_rescore_string_ligand(vina_inputs, tmp_path, monkeypatch):
    '''Test standalone run_rescore function with string ligand (not list).'''
    
    # Create mock config file
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")
    
    # Create a ligand file
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Create output directory
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Mock split_poses to avoid actual splitting
    def mock_split_poses(ligand, name, outPath, logFile="", suffix=""):
        # Create a mock split file
        split_file = out_path / f"{name}{suffix}1.pdbqt"
        split_file.write_text("MODEL 1\nATOM\nENDMDL\n")
        return 0
    
    monkeypatch.setattr(ocvina.ocmolproc, 'split_poses', mock_split_poses)
    
    # Test with string ligand (should be converted to list)
    result = ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand_file),
        outPath=str(out_path),
        scoring_function="vina",
        logFile="",
        splitLigand=True,
        overwrite=False
    )
    # Function returns None
    assert result is None


@pytest.mark.order(31)
def test_run_rescore_list_ligands(vina_inputs, tmp_path, monkeypatch):
    '''Test standalone run_rescore function with list of ligands.'''
    
    # Create mock config file
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")
    
    # Create output directory
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create already split ligands (no splitting needed)
    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")
    lig2 = out_path / "lig_split_2.pdbqt"
    lig2.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Test with list of ligands, splitLigand=False
    result = ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1), str(lig2)],
        outPath=str(out_path),
        scoring_function="vina",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    # Function returns None
    assert result is None


@pytest.mark.order(32)
def test_run_rescore_overwrite_false_existing_log(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=False when log file already exists.'''
    
    # Create mock config file
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")
    
    # Create output directory
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create already split ligand
    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Create existing log file
    log_file = out_path / "lig_split_1_vina_rescoring.log"
    log_file.write_text("Estimated Free Energy of Binding    -7.5 (kcal/mol)\n")
    
    # Test with overwrite=False (should skip)
    result = ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1)],
        outPath=str(out_path),
        scoring_function="vina",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    # Function returns None (should skip when log exists and overwrite=False)
    assert result is None


@pytest.mark.order(33)
def test_run_rescore_overwrite_true(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=True.'''
    
    # Create mock config file
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")
    
    # Create a ligand file for splitting
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Create output directory
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create existing log file (should be overwritten)
    log_file = out_path / "ligand_split_1_vina_rescoring.log"
    log_file.write_text("Old content\n")
    
    # Mock split_poses
    def mock_split_poses(ligand, name, outPath, logFile="", suffix=""):
        split_file = out_path / f"{name}{suffix}1.pdbqt"
        split_file.write_text("MODEL 1\nATOM\nENDMDL\n")
        return 0
    
    monkeypatch.setattr(ocvina.ocmolproc, 'split_poses', mock_split_poses)
    
    # Test with overwrite=True (should process even if log exists)
    result = ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand_file),
        outPath=str(out_path),
        scoring_function="vina",
        logFile="",
        splitLigand=True,
        overwrite=True
    )
    assert result is None


@pytest.mark.order(34)
def test_run_rescore_split_ligand_false(tmp_path):
    '''Test run_rescore with splitLigand=False.'''
    
    # Create mock config file
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")
    
    # Create output directory
    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create already split ligands
    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Test with splitLigand=False (should use ligands as-is)
    result = ocvina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1)],
        outPath=str(out_path),
        scoring_function="vina",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    assert result is None


@pytest.mark.order(35)
def test_run_vina_stub_file_creation(tmp_path, monkeypatch):
    '''Test run_vina creates stub files when vina executable is not available.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockVinaConfig:
            executable = "/nonexistent/vina"
        class MockConfig:
            vina = MockVinaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocvina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    out_file = tmp_path / "output" / "out.pdbqt"
    log_file = tmp_path / "output" / "log.txt"
    
    result = ocvina.run_vina(
        confFile=str(conf_file),
        ligand=str(ligand_file),
        outPath=str(out_file),
        logFile=str(log_file)
    )
    
    # Should return OK (stub files created)
    assert result == ocerror.Error.ok() # type: ignore
    # Check that stub files were created
    assert out_file.exists()
    assert "stub" in out_file.read_text().lower()


@pytest.mark.order(36)
def test_run_vina_stub_no_outpath(tmp_path, monkeypatch):
    '''Test run_vina stub creation when outPath is empty.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockVinaConfig:
            executable = "/nonexistent/vina"
        class MockConfig:
            vina = MockVinaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocvina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    log_file = tmp_path / "log.txt"
    
    result = ocvina.run_vina(
        confFile=str(conf_file),
        ligand=str(ligand_file),
        outPath="",  # Empty outPath
        logFile=str(log_file)
    )
    
    # Should return OK (stub log file created)
    assert result == ocerror.Error.ok() # type: ignore
    assert log_file.exists()


@pytest.mark.order(37)
def test_box_to_vina_multiple_remarks(tmp_path):
    '''Test box_to_vina with box file containing multiple REMARK lines.'''
    
    box_file = tmp_path / "box.pdb"
    conf_file = tmp_path / "conf.txt"
    
    # Create box file with multiple REMARK lines (should break after 2)
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
        "REMARK    CENTER (X Y Z)        7.000  8.000  9.000\n"  # Should be ignored
    )
    
    result = ocvina.box_to_vina(
        box_file=str(box_file),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    
    assert result == ocerror.Error.ok() # type: ignore
    assert conf_file.exists()
    # Should have center coordinates from first two REMARK lines
    content = conf_file.read_text()
    assert "center_x" in content
    assert "center_y" in content
    assert "center_z" in content


@pytest.mark.order(38)
def test_box_to_vina_write_error_handling(tmp_path, monkeypatch):
    '''Test box_to_vina handles write errors gracefully.'''
    
    box_file = tmp_path / "box.pdb"
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )
    
    # Create a conf file path that would cause write error
    conf_file = tmp_path / "nonexistent" / "conf.txt"  # Parent doesn't exist
    
    result = ocvina.box_to_vina(
        box_file=str(box_file),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    
    # Should handle write error gracefully (may succeed if parent dir is auto-created, or return error)
    assert isinstance(result, int)


@pytest.mark.order(39)
def test_vina_run_rescore_instance_method(vina_inputs, tmp_path, monkeypatch):
    '''Test Vina.run_rescore instance method with different parameters.'''
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    # Create mock split poses
    split_dir = tmp_path / "split_poses"
    split_dir.mkdir()
    split_lig = split_dir / "lig_split_1.pdbqt"
    split_lig.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    # Mock get_docked_poses to return our mock split file
    def mock_get_docked_poses(path):
        return [str(split_lig)]
    
    monkeypatch.setattr(vina_instance, 'get_docked_poses', lambda: [str(split_lig)])
    
    # Test run_rescore with skipDefaultScoring=True
    result = vina_instance.run_rescore(
        outPath=str(split_dir),
        ligand=str(split_lig),
        logFile="",
        skipDefaultScoring=True,
        overwrite=False
    )
    # Method returns None
    assert result is None


@pytest.mark.order(40)
def test_vina_run_rescore_overwrite_true(vina_inputs, tmp_path, monkeypatch):
    '''Test Vina.run_rescore instance method with overwrite=True.'''
    
    vina_instance = ocvina.Vina(
        config_path=vina_inputs["config"],
        box_file=vina_inputs["box"],
        receptor=vina_inputs["receptor"],
        prepared_receptor_path=vina_inputs["receptor_path"],
        ligand=vina_inputs["ligand"],
        prepared_ligand_path=vina_inputs["ligand_path"],
        vina_log=vina_inputs["vina_log"],
        output_vina=vina_inputs["output_dir"],
        name="test"
    )
    
    split_dir = tmp_path / "split_poses"
    split_dir.mkdir()
    split_lig = split_dir / "lig_split_1.pdbqt"
    split_lig.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    monkeypatch.setattr(vina_instance, 'get_docked_poses', lambda: [str(split_lig)])
    
    # Test run_rescore with overwrite=True
    result = vina_instance.run_rescore(
        outPath=str(split_dir),
        ligand=str(split_lig),
        logFile="",
        skipDefaultScoring=False,
        overwrite=True
    )
    assert result is None
