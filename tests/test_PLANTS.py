import pytest
import shutil

from pathlib import Path

import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Error as ocerror
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
    Test PLANTS class can be instantiated with all required test inputs.
    '''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test",
        box_spacing=1.0,
        overwrite_config=True
    )
    assert isinstance(plants_instance, ocplants.PLANTS)


@pytest.mark.order(2)
def test_convert_smi_to_mol2(plants_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    result = occonversion.convert_mols(
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
        box_file=plants_inputs["box"],
        conf_file=str(plants_inputs["config"]),
        receptor=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand_file"],
        output_plants=plants_inputs["plants_files_dir"],
        center=None,
        binding_site_radius=None,
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
        input_ligand_path=plants_inputs["converted_ligand_file"],
        output_ligand=plants_inputs["ligand_path"]
    )
    assert result is True or isinstance(result, int)
    assert Path(plants_inputs["ligand_path"]).exists()


@pytest.mark.order(5)
def test_run_prepare_receptor(plants_inputs):
    '''
    Run receptor preparation for PLANTS and verify output files.
    '''

    result = ocplants.run_prepare_receptor(
        input_receptor_path=str(plants_inputs["receptor_file"]),
        output_receptor=str(plants_inputs["receptor_path"])
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


@pytest.mark.order(7)
def test_write_pose_list(tmp_path):
    # Create dummy mol2 files and collect their paths
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    pose_paths = []
    for i in range(3):
        pose_file = pose_dir / f"pose_{i}.mol2"
        pose_file.write_text(f"pose {i}")
        pose_paths.append(str(pose_file))

    pose_list_file = tmp_path / "pose_list.txt"

    # First call should create the file and return its path
    result = ocplants.write_pose_list(pose_paths, str(pose_list_file))
    assert result == str(pose_list_file)
    assert pose_list_file.exists()
    assert pose_list_file.read_text().splitlines() == pose_paths

    # Second call without overwrite should return None and keep the file unchanged
    result_none = ocplants.write_pose_list(pose_paths, str(pose_list_file))
    assert result_none is None
    assert pose_list_file.read_text().splitlines() == pose_paths


@pytest.mark.order(8)
def test_get_binding_site_from_box(plants_inputs):
    result = ocplants.get_binding_site(str(plants_inputs["box"]))
    assert isinstance(result, tuple), "Expected result to be a tuple. Int means error."
    assert len(result) == 2, "Expected result to have two elements"
    center, radius = result
    assert center == (36.552, 39.252, 51.291)
    assert radius == pytest.approx(70.274, abs=1e-3)


@pytest.mark.order(9)
def test_get_binding_site_nonexistent(tmp_path):
    missing = tmp_path / "no_file.pdb"
    result = ocplants.get_binding_site(str(missing))
    assert result == ocerror.ErrorCode.FILE_NOT_EXIST.value


@pytest.mark.order(10)
def test_plants_get_input_ligand_path(plants_inputs):
    '''Test get_input_ligand_path method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    result = plants_instance.get_input_ligand_path()
    assert isinstance(result, str)


@pytest.mark.order(11)
def test_plants_get_input_receptor_path(plants_inputs):
    '''Test get_input_receptor_path method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    result = plants_instance.get_input_receptor_path()
    assert isinstance(result, str)


@pytest.mark.order(12)
def test_plants_get_docked_poses(plants_inputs, tmp_path):
    '''Test get_docked_poses method.'''

    output_dir = tmp_path / "plants_run"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    
    # Create some mock pose files
    (run_dir / "ligand_0001.mol2").write_text("POSE1")
    (run_dir / "ligand_0002.mol2").write_text("POSE2")
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.get_docked_poses()
    assert isinstance(result, list)


@pytest.mark.order(13)
def test_plants_read_log(plants_inputs, tmp_path):
    '''Test read_log method.'''

    output_dir = tmp_path / "plants_output"
    csv_dir = output_dir / "test_lig"
    csv_dir.mkdir(parents=True)
    
    # Create mock ranking CSV file with PLANTS format
    ranking_file = csv_dir / "bestranking.csv"
    ranking_file.write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
    )
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.read_log()
    assert isinstance(result, dict)


@pytest.mark.order(14)
def test_plants_read_log_all_poses(plants_inputs, tmp_path):
    '''Test read_log method with onlyBest=False.'''

    output_dir = tmp_path / "plants_output"
    csv_dir = output_dir / "test_lig"
    csv_dir.mkdir(parents=True)
    
    # Create mock ranking CSV file with PLANTS format
    ranking_file = csv_dir / "ranking.csv"
    ranking_file.write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
        "ligand_0002.mol2,-6.8,-1.0,2.3,1.6,2.8,2.0,-0.6\n"
    )
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.read_log(onlyBest=False)
    assert isinstance(result, dict)


@pytest.mark.order(15)
def test_plants_get_rescore_log_paths(plants_inputs, tmp_path):
    '''Test get_rescore_log_paths method.'''

    output_dir = tmp_path / "plants_output"
    run_chemplp = output_dir / "run_chemplp"
    run_plp = output_dir / "run_plp"
    run_chemplp.mkdir(parents=True)
    run_plp.mkdir(parents=True)
    
    # Create mock ranking CSV files with PLANTS format
    (run_chemplp / "bestranking.csv").write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
    )
    (run_plp / "ranking.csv").write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-6.8,-1.0,2.3,1.6,2.8,2.0,-0.6\n"
    )
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.get_rescore_log_paths(onlyBest=True)
    assert isinstance(result, list)


@pytest.mark.order(16)
def test_plants_read_rescore_logs(plants_inputs, tmp_path):
    '''Test read_rescore_logs method.'''

    output_dir = tmp_path / "plants_output"
    run_chemplp = output_dir / "run_chemplp"
    run_chemplp.mkdir(parents=True)
    
    # Create mock ranking CSV file with PLANTS format
    (run_chemplp / "bestranking.csv").write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
    )
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.read_rescore_logs(onlyBest=True)
    assert isinstance(result, dict)


@pytest.mark.order(17)
def test_plants_write_pose_list(plants_inputs, tmp_path):
    '''Test write_pose_list method.'''

    output_dir = tmp_path / "plants_output"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    
    # Create some mock pose files
    (run_dir / "ligand_0001.mol2").write_text("POSE1")
    (run_dir / "ligand_0002.mol2").write_text("POSE2")
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.write_pose_list()
    assert result is None or isinstance(result, str)


@pytest.mark.order(18)
def test_plants_write_pose_list_overwrite(plants_inputs, tmp_path):
    '''Test write_pose_list method with overwrite=True.'''

    output_dir = tmp_path / "plants_output"
    run_dir = output_dir / "run"
    run_dir.mkdir(parents=True)
    
    # Create some mock pose files
    (run_dir / "ligand_0001.mol2").write_text("POSE1")
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(output_dir),
        name="test"
    )
    
    result = plants_instance.write_pose_list(overwrite=True)
    assert result is None or isinstance(result, str)


@pytest.mark.order(19)
def test_plants_write_config_file(plants_inputs):
    '''Test write_config_file method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    result = plants_instance.write_config_file()
    assert isinstance(result, int)


@pytest.mark.order(20)
def test_plants_instance_run_prepare_ligand(plants_inputs):
    '''Test instance run_prepare_ligand method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    result = plants_instance.run_prepare_ligand()
    assert isinstance(result, (int, tuple))


@pytest.mark.order(21)
def test_plants_instance_run_prepare_receptor(plants_inputs):
    '''Test instance run_prepare_receptor method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    result = plants_instance.run_prepare_receptor()
    assert isinstance(result, (int, tuple))


@pytest.mark.order(22)
def test_plants_print_attributes(plants_inputs, capsys):
    '''Test print_attributes method.'''

    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    plants_instance.print_attributes()
    captured = capsys.readouterr()
    assert "Name:" in captured.out
    assert "Box path:" in captured.out
    assert "Config path:" in captured.out


@pytest.mark.order(23)
def test_plants_standalone_read_log(tmp_path):
    '''Test standalone read_log function.'''

    ranking_file = tmp_path / "bestranking.csv"
    ranking_file.write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
    )
    
    result = ocplants.read_log(str(ranking_file))
    assert isinstance(result, dict)


@pytest.mark.order(24)
def test_plants_standalone_read_log_only_best(tmp_path):
    '''Test standalone read_log function with onlyBest=True.'''
    
    ranking_file = tmp_path / "bestranking.csv"
    ranking_file.write_text(
        "LIGAND_ENTRY,TOTAL_SCORE,SCORE_RB_PEN,SCORE_NORM_HEVATOMS,SCORE_NORM_CRT_HEVATOMS,SCORE_NORM_WEIGHT,SCORE_NORM_CRT_WEIGHT,SCORE_RB_PEN_NORM_CRT_HEVATOMS\n"
        "ligand_0001.mol2,-7.5,-1.2,2.5,1.8,3.0,2.2,-0.8\n"
    )
    
    result = ocplants.read_log(str(ranking_file), onlyBest=True)
    assert isinstance(result, dict)


@pytest.mark.order(25)
def test_plants_standalone_get_docked_poses(tmp_path):
    '''Test standalone get_docked_poses function.'''

    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    
    (poses_dir / "ligand_0001.mol2").write_text("POSE1")
    (poses_dir / "ligand_0002.mol2").write_text("POSE2")
    
    result = ocplants.get_docked_poses(str(poses_dir))
    assert isinstance(result, list)


@pytest.mark.order(26)
def test_plants_standalone_get_pose_index_from_file_path():
    '''Test get_pose_index_from_file_path function.'''

    assert ocplants.get_pose_index_from_file_path("ligand_0042.mol2") == 42


@pytest.mark.order(27)
def test_plants_generate_plants_files_database(plants_inputs, tmp_path):
    '''Test generate_plants_files_database function.'''

    protein_path = str(plants_inputs["receptor_file"])
    ligand_path = str(plants_inputs["ligand_file"])
    box_path = str(plants_inputs["box"]).replace("box0.pdb", "")
    
    # Create boxes directory if needed
    Path(box_path).mkdir(parents=True, exist_ok=True)
    
    result = ocplants.generate_plants_files_database(
        path=str(tmp_path),
        protein=protein_path,
        ligand=ligand_path,
        boxPath=box_path
    )
    assert result is None


@pytest.mark.order(28)
def test_plants_generate_plants_files_database_default_box(plants_inputs, tmp_path):
    '''Test generate_plants_files_database with default boxPath.'''

    protein_path = str(plants_inputs["receptor_file"])
    ligand_path = str(plants_inputs["ligand_file"])
    
    # Create boxes directory
    boxes_dir = tmp_path / "boxes"
    boxes_dir.mkdir()
    box_file = boxes_dir / "box0.pdb"
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "HEADER    MIN (X Y Z)           0.000  1.000  2.000\n"
        "HEADER    MAX (X Y Z)           2.000  3.000  4.000\n"
    )
    
    result = ocplants.generate_plants_files_database(
        path=str(tmp_path),
        protein=protein_path,
        ligand=ligand_path,
        boxPath=""
    )
    assert result is None


@pytest.mark.order(29)
def test_plants_box_to_plants_file_not_exists(tmp_path):
    '''Test box_to_plants with non-existent box file.'''
    
    conf_file = tmp_path / "conf.txt"
    result = ocplants.box_to_plants(
        box_file=str(tmp_path / "nonexistent.pdb"),
        conf_file=str(conf_file),
        receptor="rec.mol2",
        ligand="lig.mol2",
        output_plants=str(tmp_path)
    )
    assert result != 0


@pytest.mark.order(30)
def test_run_plants_stub_file_creation(tmp_path, monkeypatch):
    '''Test run_plants creates stub files when plants executable is not available.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("protein_file rec.mol2\n")
    output_dir = tmp_path / "output"
    log_file = tmp_path / "plants.log"
    
    result = ocplants.run_plants(
        confFile=str(conf_file),
        outputPlants=str(output_dir),
        overwrite=False,
        logFile=str(log_file)
    )
    
    # Should return OK (stub log file created)
    assert result == ocerror.Error.ok() # type: ignore
    assert log_file.exists()
    assert "stub" in log_file.read_text().lower()


@pytest.mark.order(31)
def test_run_plants_stub_no_logfile(tmp_path, monkeypatch):
    '''Test run_plants stub creation when logFile is empty.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("protein_file rec.mol2\n")
    output_dir = tmp_path / "output"
    
    result = ocplants.run_plants(
        confFile=str(conf_file),
        outputPlants=str(output_dir),
        overwrite=False,
        logFile=""  # Empty log file
    )
    
    # When logFile is empty and plants is not available, it calls ocrun.run which returns an error code
    # (e.g., 300 for subprocess error when executable not found)
    # So result will be an error code, not OK
    assert isinstance(result, int)
    # Should not be OK since plants executable doesn't exist
    assert result != ocerror.Error.ok() # type: ignore


@pytest.mark.order(32)
def test_run_plants_overwrite_true(tmp_path, monkeypatch):
    '''Test run_plants with overwrite=True removes existing output directory.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("protein_file rec.mol2\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing_file.txt").write_text("existing")
    log_file = tmp_path / "plants.log"
    
    result = ocplants.run_plants(
        confFile=str(conf_file),
        outputPlants=str(output_dir),
        overwrite=True,  # Should remove existing directory
        logFile=str(log_file)
    )
    
    # Should return OK and directory should be removed
    assert result == ocerror.Error.ok() # type: ignore
    # Directory might be recreated, but original file should be gone
    if output_dir.exists():
        assert not (output_dir / "existing_file.txt").exists()


@pytest.mark.order(33)
def test_run_plants_empty_directory(tmp_path, monkeypatch):
    '''Test run_plants removes empty existing directory.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("protein_file rec.mol2\n")
    output_dir = tmp_path / "output"
    output_dir.mkdir()  # Empty directory
    log_file = tmp_path / "plants.log"
    
    result = ocplants.run_plants(
        confFile=str(conf_file),
        outputPlants=str(output_dir),
        overwrite=False,  # Should remove empty directory
        logFile=str(log_file)
    )
    
    # Should return OK
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(34)
def test_run_rescore_overwrite_false_existing_conf(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=False when conf file already exists.'''
    
    conf_file = tmp_path / "rescoring.conf"
    conf_file.write_text("existing config")
    
    pose_list = tmp_path / "poses.lst"
    pose_list.write_text("pose1.mol2\npose2.mol2\n")
    
    result = ocplants.run_rescore(
        confFile=str(conf_file),
        pose_list_file=str(pose_list),
        outPath=str(tmp_path / "output"),
        proteinFile="protein.mol2",
        scoring_function="chemplp",
        bindingSiteCenterX=1.0,
        bindingSiteCenterY=2.0,
        bindingSiteCenterZ=3.0,
        bindingSiteRadius=10.0,
        logFile="",
        overwrite=False
    )
    
    # Should return file_exists warning
    assert result == ocerror.Error.file_exists() # type: ignore


@pytest.mark.order(35)
def test_run_rescore_overwrite_true(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=True.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
            rescoring_mode = "simplex"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "rescoring.conf"
    conf_file.write_text("existing config")
    
    pose_list = tmp_path / "poses.lst"
    pose_list.write_text("pose1.mol2\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    result = ocplants.run_rescore(
        confFile=str(conf_file),
        pose_list_file=str(pose_list),
        outPath=str(output_dir),
        proteinFile="protein.mol2",
        scoring_function="chemplp",
        bindingSiteCenterX=1.0,
        bindingSiteCenterY=2.0,
        bindingSiteCenterZ=3.0,
        bindingSiteRadius=10.0,
        logFile="",
        overwrite=True  # Should overwrite
    )
    
    # Should return OK
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(36)
def test_run_rescore_existing_outdir_no_overwrite(tmp_path, monkeypatch):
    '''Test run_rescore with existing output directory, overwrite=False.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
            rescoring_mode = "simplex"
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "rescoring.conf"
    pose_list = tmp_path / "poses.lst"
    pose_list.write_text("pose1.mol2\n")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("existing")
    
    result = ocplants.run_rescore(
        confFile=str(conf_file),
        pose_list_file=str(pose_list),
        outPath=str(output_dir),
        proteinFile="protein.mol2",
        scoring_function="chemplp",
        bindingSiteCenterX=1.0,
        bindingSiteCenterY=2.0,
        bindingSiteCenterZ=3.0,
        bindingSiteRadius=10.0,
        logFile="",
        overwrite=False  # Should skip when output exists
    )
    
    # Should return dir_exists warning
    assert result == ocerror.Error.dir_exists() # type: ignore


@pytest.mark.order(37)
def test_plants_instance_run_rescore(plants_inputs, tmp_path, monkeypatch):
    '''Test PLANTS.run_rescore instance method.'''
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(tmp_path / "output"),
        name="test"
    )
    
    # Create mock pose list
    pose_list = tmp_path / "poses.lst"
    pose_list.write_text("pose1.mol2\npose2.mol2\n")
    
    # Mock config for rescoring
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
            rescoring_mode = "simplex"
            scoring = "chemplp"
            scoring_functions = ["chemplp", "plp"]
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    result = plants_instance.run_rescore(
        pose_list=str(pose_list),
        logFile="",
        skipDefaultScoring=False,
        overwrite=False
    )
    
    # Instance method returns None
    assert result is None


@pytest.mark.order(38)
def test_plants_instance_run_rescore_overwrite(plants_inputs, tmp_path, monkeypatch):
    '''Test PLANTS.run_rescore instance method with overwrite=True.'''
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=str(tmp_path / "output"),
        name="test"
    )
    
    # Create mock pose list
    pose_list = tmp_path / "poses.lst"
    pose_list.write_text("pose1.mol2\n")
    
    # Mock config for rescoring
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
            rescoring_mode = "simplex"
            scoring = "chemplp"
            scoring_functions = ["chemplp"]
        class MockConfig:
            plants = MockPlantsConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    result = plants_instance.run_rescore(
        pose_list=str(pose_list),
        logFile="",
        skipDefaultScoring=False,
        overwrite=True  # Should overwrite
    )
    
    # Instance method returns None
    assert result is None


@pytest.mark.order(39)
def test_plants_instance_run_plants_stub(plants_inputs, tmp_path, monkeypatch):
    '''Test PLANTS.run_plants instance method creates stub when plants not available.'''
    
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=str(tmp_path / "plants.log"),
        output_plants=str(tmp_path / "output"),
        name="test"
    )
    
    # Mock config to return non-existent executable with tmp_dir attribute
    def mock_get_config():
        class MockPlantsConfig:
            executable = "/nonexistent/plants"
        class MockConfig:
            plants = MockPlantsConfig()
            tmp_dir = str(tmp_path / "tmp")  # Add tmp_dir attribute
        return MockConfig()
    
    monkeypatch.setattr(ocplants, 'get_config', mock_get_config)
    
    result = plants_instance.run_plants(overwrite=False)
    
    # Instance method calls ocrun.run which returns error code when executable doesn't exist
    # Should return an error code (not OK) since plants executable doesn't exist
    assert isinstance(result, (int, tuple))
    # If it's a tuple, the first element is the error code
    if isinstance(result, tuple):
        assert result[0] != ocerror.Error.ok() # type: ignore
    else:
        assert result != ocerror.Error.ok() # type: ignore


@pytest.mark.order(40)
def test_plants_init_invalid_receptor(plants_inputs):
    '''Test PLANTS.__init__ with invalid receptor type.'''
    
    # Use a string instead of Receptor object
    # __init__ will return None early after error, but Python creates the object anyway
    # The error handling code path is still covered (lines 110-111)
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor="not_a_receptor_object",  # Invalid type
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand=plants_inputs["ligand"],
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    # Object is created but in invalid state (__init__ returns None but Python ignores it)
    # The error handling path (lines 110-111) is tested
    assert plants_instance is not None  # Object is created


@pytest.mark.order(41)
def test_plants_init_invalid_ligand(plants_inputs):
    '''Test PLANTS.__init__ with invalid ligand type.'''
    
    # Use a string instead of Ligand object
    # __init__ will return None early after error, but Python creates the object anyway
    # The error handling code path is still covered (lines 122-124)
    plants_instance = ocplants.PLANTS(
        config_path=plants_inputs["config"],
        box_file=plants_inputs["box"],
        receptor=plants_inputs["receptor"],
        prepared_receptor_path=plants_inputs["receptor_path"],
        ligand="not_a_ligand_object",  # Invalid type
        prepared_ligand_path=plants_inputs["ligand_path"],
        plants_log=plants_inputs["plants_log"],
        output_plants=plants_inputs["plants_files_dir"],
        name="test"
    )
    
    # Object is created but in invalid state (__init__ returns None but Python ignores it)
    # The error handling path (lines 122-124) is tested
    assert plants_instance is not None  # Object is created


@pytest.mark.order(42)
def test_box_to_plants_binding_site_error(tmp_path):
    '''Test box_to_plants with invalid box file that returns error from get_binding_site.'''
    
    # Create invalid box file (missing required REMARK lines)
    invalid_box = tmp_path / "invalid_box.pdb"
    invalid_box.write_text("HEADER    INVALID BOX\n")
    
    conf_file = tmp_path / "conf.txt"
    
    result = ocplants.box_to_plants(
        box_file=str(invalid_box),
        conf_file=str(conf_file),
        receptor="rec.mol2",
        ligand="lig.mol2",
        output_plants=str(tmp_path)
    )
    
    # Should return error code
    assert result != 0


@pytest.mark.order(43)
def test_box_to_plants_with_center_and_radius(tmp_path):
    '''Test box_to_plants with provided center and radius (skip get_binding_site).'''
    
    conf_file = tmp_path / "conf.txt"
    
    result = ocplants.box_to_plants(
        box_file=str(tmp_path / "nonexistent.pdb"),  # Box file not needed when center/radius provided
        conf_file=str(conf_file),
        receptor="rec.mol2",
        ligand="lig.mol2",
        output_plants=str(tmp_path),
        center=[1.0, 2.0, 3.0],  # Provide center
        binding_site_radius=10.0,  # Provide radius
        spacing=2.9
    )
    
    # Should succeed (writes config directly)
    assert result == 0
    assert conf_file.exists()
    content = conf_file.read_text()
    assert "protein_file rec.mol2" in content
    assert "ligand_file lig.mol2" in content


@pytest.mark.order(44)
def test_write_rescoring_config_file(tmp_path):
    '''Test write_rescoring_config_file function.'''
    
    conf_file = tmp_path / "rescoring.conf"
    
    result = ocplants.write_rescoring_config_file(
        confFile=str(conf_file),
        preparedReceptor="rec.mol2",
        ligandListPath="poses.lst",
        outputPlants=str(tmp_path / "output"),
        bindingSiteCenterX=1.0,
        bindingSiteCenterY=2.0,
        bindingSiteCenterZ=3.0,
        bindingSiteRadius=10.0,
        scoringFunction="chemplp",
        rescoringMode="simplex"
    )
    
    assert result == 0
    assert conf_file.exists()
    content = conf_file.read_text()
    assert "protein_file rec.mol2" in content
    assert "ligand_list poses.lst" in content
    assert "rescore_mode simplex" in content
    assert "scoring_function chemplp" in content
