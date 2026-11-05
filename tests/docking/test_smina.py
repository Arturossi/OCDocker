#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Smina docking integration.

Usage:

pytest tests/test_Smina.py
'''

# Imports
###############################################################################
import pytest

from pathlib import Path
from types import SimpleNamespace

import OCDocker.Docking.Smina as ocsmina
import OCDocker.Error as ocerror
import OCDocker.Ligand as ocl
import OCDocker.Receptor as ocr
import OCDocker.Toolbox.Conversion as occonversion

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

def _record_smina_runs(monkeypatch, log_text: str = "Affinity    -7.00 (kcal/mol)\n") -> list:
    '''Replace the smina run by a stub that records the command and writes the given log.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The pytest monkeypatch fixture.
    log_text : str, optional
        The text written to the log file (--log) of each run.

    Returns
    -------
    list
        The commands run, filled as the runs happen.
    '''

    commands = []

    def fake_run(cmd, logFile = "", **_kwargs):
        commands.append(cmd)
        Path(_option(cmd, "--log")).write_text(log_text)
        return ocerror.Error.ok()

    monkeypatch.setattr(ocsmina.ocrun, "run", fake_run)
    return commands


def _option(cmd: list, name: str) -> str:
    '''Return the value following an option in a command list.'''

    return cmd[cmd.index(name) + 1]

## Public ##

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

@pytest.mark.order(2)
def test_convert_smi_to_mol2(smina_inputs):
    '''
    Test explicit call to convert .smi to .mol2 using Conversion.py routine.
    '''

    result = occonversion.convert_mols(
        input_file=str(smina_inputs["ligand_file"]),
        output_file=str(smina_inputs["converted_ligand_file"]),
        overwrite=True
    )

    assert result == 0 or result is True, f"Conversion from .smi to .mol2 failed. Error code: {result}"
    assert Path(smina_inputs["converted_ligand_file"]).exists(), "Failed to generate .mol2 from .smi"

@pytest.mark.order(36)
def test_gen_smina_conf_custom_attributes(tmp_path, monkeypatch):
    '''Test gen_smina_conf with custom attributes from Config.'''
    
    # Mock Config with custom attributes
    def mock_get_config():
        class MockSminaConfig:
            custom_scoring = "custom.score"
            custom_atoms = "custom.atoms"
            minimize_iters = "100"
            user_grid = "grid.txt"
            user_grid_lambda = "0.5"
            approximation = "spline"
            factor = "32"
            force_cap = "10"
            energy_range = "10"
            exhaustiveness = "5"
            num_modes = "3"
        class MockConfig:
            smina = MockSminaConfig()
            logdir = str(tmp_path)
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    box_file = tmp_path / "box.pdb"
    conf_file = tmp_path / "conf.txt"
    
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )
    
    result = ocsmina.gen_smina_conf(
        box_file=str(box_file),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    
    assert result == ocerror.Error.ok() # type: ignore
    assert conf_file.exists()
    content = conf_file.read_text()
    # Check that custom attributes are written
    assert "custom_scoring = custom.score" in content
    assert "custom_atoms = custom.atoms" in content
    assert "minimize_iters = 100" in content
    assert "user_grid = grid.txt" in content
    assert "user_grid_lambda = 0.5" in content

@pytest.mark.order(25)
def test_gen_smina_conf_file_not_exists(tmp_path):
    '''Test gen_smina_conf with non-existent box file.'''
    
    conf_file = tmp_path / "conf.txt"
    result = ocsmina.gen_smina_conf(
        box_file=str(tmp_path / "nonexistent.pdb"),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    assert result != 0

@pytest.mark.order(39)
def test_gen_smina_conf_multiple_remarks(tmp_path):
    '''Test gen_smina_conf with box file containing multiple REMARK lines.'''
    
    box_file = tmp_path / "box.pdb"
    conf_file = tmp_path / "conf.txt"
    
    # Create box file with multiple REMARK lines (should break after 2)
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
        "REMARK    CENTER (X Y Z)        7.000  8.000  9.000\n"  # Should be ignored
    )
    
    result = ocsmina.gen_smina_conf(
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

@pytest.mark.order(40)
def test_gen_smina_conf_write_error_handling(tmp_path):
    '''Test gen_smina_conf handles write errors gracefully.'''
    
    box_file = tmp_path / "box.pdb"
    box_file.write_text(
        "REMARK    CENTER (X Y Z)        1.000  2.000  3.000\n"
        "REMARK    DIMENSIONS (X Y Z)    4.000  5.000  6.000\n"
    )
    
    # Create a conf file path that would cause write error (parent doesn't exist)
    # But this should succeed as parent dir is auto-created
    conf_file = tmp_path / "nonexistent" / "conf.txt"
    
    result = ocsmina.gen_smina_conf(
        box_file=str(box_file),
        conf_file=str(conf_file),
        receptor="rec.pdbqt"
    )
    
    # The conf file is not written when its directory does not exist
    assert result == ocerror.ErrorCode.WRITE_FILE
    assert not conf_file.exists()

@pytest.mark.order(20)
def test_get_pose_index_from_file_path():
    '''Check pose index extraction from file name.'''

    assert ocsmina.get_pose_index_from_file_path("lig_split_42.pdbqt") == 42

@pytest.mark.order(21)
def test_get_rescore_log_paths(tmp_path):
    '''Verify detection of rescoring log files in a directory.'''

    f1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    f1.write_text("log1")
    f2 = tmp_path / "lig_split_2_vinardo_rescoring.log"
    f2.write_text("log2")
    f3 = tmp_path / "other.log"
    f3.write_text("x")
    
    found = ocsmina.get_rescore_log_paths(str(tmp_path))
    # Function returns only rescoring log files (matching *_rescoring.log pattern)
    assert set(found) == {str(f1), str(f2)}

@pytest.mark.order(22)
def test_read_rescore_logs_list(tmp_path):
    '''Test read_rescore_logs with list of log paths.'''

    log1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    log1.write_text("Affinity    -7.23 (kcal/mol)\n")
    
    log2 = tmp_path / "lig_split_2_vinardo_rescoring.log"
    log2.write_text("Affinity    -6.50 (kcal/mol)\n")
    
    result = ocsmina.read_rescore_logs([str(log1), str(log2)])
    assert result == {"rescoring_vinardo_1": -7.23, "rescoring_vinardo_2": -6.5}

@pytest.mark.order(24)
def test_read_rescore_logs_only_best(tmp_path):
    '''Test read_rescore_logs with onlyBest=True.'''

    log1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    log1.write_text("Affinity    -7.23 (kcal/mol)\n")
    
    log2 = tmp_path / "lig_split_2_vinardo_rescoring.log"
    log2.write_text("Affinity    -6.50 (kcal/mol)\n")
    
    result = ocsmina.read_rescore_logs([str(log1), str(log2)], onlyBest=True)
    assert result == {"rescoring_vinardo_1": -7.23}

@pytest.mark.order(23)
def test_read_rescore_logs_string(tmp_path):
    '''Test read_rescore_logs with single string path.'''

    log1 = tmp_path / "lig_split_1_vinardo_rescoring.log"
    log1.write_text("Affinity    -7.23 (kcal/mol)\n")
    
    result = ocsmina.read_rescore_logs(str(log1))
    assert result == {"rescoring_vinardo_1": -7.23}

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
        box_file=smina_inputs["box"],
        conf_file=smina_inputs["config"],
        receptor=smina_inputs["prepared_receptor_path"]
    )

    assert result == 0 or result is True, f"Configuration generation failed. Error code: {result}"

@pytest.mark.order(3)
def test_run_prepare_ligand(smina_inputs):
    '''
    Run ligand preparation and check that it produces expected files.
    '''

    result = ocsmina.run_prepare_ligand(
        input_ligand_path=smina_inputs["converted_ligand_file"],
        prepared_ligand=str(smina_inputs["prepared_ligand_path"]),
    )

    assert result is True or isinstance(result, int), "Ligand preparation failed"
    assert Path(smina_inputs["prepared_ligand_path"]).exists(), "Prepared ligand file not found"

@pytest.mark.order(4)
def test_run_prepare_receptor(smina_inputs):
    '''
    Run receptor preparation and check that it produces expected files.
    '''

    result = ocsmina.run_prepare_receptor(
        input_receptor_path=str(smina_inputs["receptor_file"]),
        prepared_receptor=str(smina_inputs["prepared_receptor_path"]),
    )

    assert result is True or isinstance(result, int), "Receptor preparation failed"
    assert Path(smina_inputs["prepared_receptor_path"]).exists(), "No prepared receptor files found"

@pytest.mark.order(27)
def test_run_rescore_list_ligands(smina_inputs, tmp_path, monkeypatch):
    '''Test standalone run_rescore scores every ligand of a list with the given scoring function.'''

    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")

    out_path = tmp_path / "rescoring"
    out_path.mkdir()

    # Already split ligands (no splitting needed)
    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")
    lig2 = out_path / "lig_split_2.pdbqt"
    lig2.write_text("MODEL 1\nATOM\nENDMDL\n")

    commands = _record_smina_runs(monkeypatch)

    result = ocsmina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1), str(lig2)],
        outPath=str(out_path),
        scoring_function="vinardo",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    assert result is None
    assert [_option(cmd, "--ligand") for cmd in commands] == [str(lig1), str(lig2)]
    assert all("--score_only" in cmd and _option(cmd, "--scoring") == "vinardo" and _option(cmd, "--config") == str(conf_file) for cmd in commands)
    assert ocsmina.read_rescore_logs(ocsmina.get_rescore_log_paths(str(out_path))) == {"rescoring_vinardo_1": -7.0, "rescoring_vinardo_2": -7.0}

@pytest.mark.order(28)
def test_run_rescore_overwrite_false_existing_log(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=False skips a ligand whose log file already exists.'''

    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")

    out_path = tmp_path / "rescoring"
    out_path.mkdir()

    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")

    log_file = out_path / "lig_split_1_vinardo_rescoring.log"
    log_file.write_text("Affinity    -7.5 (kcal/mol)\n")

    commands = _record_smina_runs(monkeypatch)

    result = ocsmina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1)],
        outPath=str(out_path),
        scoring_function="vinardo",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    assert result is None
    assert commands == []
    assert ocsmina.read_rescoring_log(str(log_file)) == -7.5

@pytest.mark.order(29)
def test_run_rescore_overwrite_true(tmp_path, monkeypatch):
    '''Test run_rescore with overwrite=True rescores a ligand whose log file already exists.'''

    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")

    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\nATOM\nENDMDL\n")

    out_path = tmp_path / "rescoring"
    out_path.mkdir()

    # Existing log file (should be overwritten)
    log_file = out_path / "ligand_split_1_vinardo_rescoring.log"
    log_file.write_text("Old content\n")

    def mock_split_poses(ligand, name, outPath, logFile="", suffix=""):
        split_file = out_path / f"{name}{suffix}1.pdbqt"
        split_file.write_text("MODEL 1\nATOM\nENDMDL\n")
        return 0

    monkeypatch.setattr(ocsmina.ocmolproc, 'split_poses', mock_split_poses)
    commands = _record_smina_runs(monkeypatch)

    result = ocsmina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand_file),
        outPath=str(out_path),
        scoring_function="vinardo",
        logFile="",
        splitLigand=True,
        overwrite=True
    )
    assert result is None
    assert [_option(cmd, "--ligand") for cmd in commands] == [str(out_path / "ligand_split_1.pdbqt")]
    assert ocsmina.read_rescoring_log(str(log_file)) == -7.0

@pytest.mark.order(30)
def test_run_rescore_removes_failed_log(tmp_path, monkeypatch):
    '''Test run_rescore removes the log file of a run that did not give an affinity.'''

    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")

    out_path = tmp_path / "rescoring"
    out_path.mkdir()

    lig1 = out_path / "lig_split_1.pdbqt"
    lig1.write_text("MODEL 1\nATOM\nENDMDL\n")

    commands = _record_smina_runs(monkeypatch, log_text = "Error: parse error\n")

    result = ocsmina.run_rescore(
        confFile=str(conf_file),
        ligands=[str(lig1)],
        outPath=str(out_path),
        scoring_function="vinardo",
        logFile="",
        splitLigand=False,
        overwrite=False
    )
    assert result is None
    assert len(commands) == 1
    assert not (out_path / "lig_split_1_vinardo_rescoring.log").exists()

@pytest.mark.order(26)
def test_run_rescore_string_ligand(smina_inputs, tmp_path, monkeypatch):
    '''Test standalone run_rescore splits a single ligand file and scores its poses.'''

    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 10\nsize_y = 10\nsize_z = 10\n")

    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\nATOM\nENDMDL\n")

    out_path = tmp_path / "rescoring"
    out_path.mkdir()

    def mock_split_poses(ligand, name, outPath, logFile="", suffix=""):
        split_file = out_path / f"{name}{suffix}1.pdbqt"
        split_file.write_text("MODEL 1\nATOM\nENDMDL\n")
        return 0

    monkeypatch.setattr(ocsmina.ocmolproc, 'split_poses', mock_split_poses)
    commands = _record_smina_runs(monkeypatch)

    # Test with string ligand (should be converted to list)
    result = ocsmina.run_rescore(
        confFile=str(conf_file),
        ligands=str(ligand_file),
        outPath=str(out_path),
        scoring_function="vinardo",
        logFile="",
        splitLigand=True,
        overwrite=False
    )
    assert result is None
    assert [_option(cmd, "--ligand") for cmd in commands] == [str(out_path / "ligand_split_1.pdbqt")]

@pytest.mark.order(6)
def test_run_smina(tmp_path, smina_inputs):
    '''
    Run Smina docking using prepared ligand and receptor.
    '''
    
    smina_inputs["output_dir"].mkdir(parents=True, exist_ok=True)

    result = ocsmina.run_smina(
        config=smina_inputs["config"],
        prepared_ligand=smina_inputs["prepared_ligand_path"],
        output_smina=smina_inputs["output_file"],
        smina_log=smina_inputs["smina_log"],
        log_path=smina_inputs["smina_exec_log"]
    )

    assert result is True or isinstance(result, int), f"Smina docking failed. Error code: {result}"
    assert Path(smina_inputs["output_file"]).exists(), "Docking output file not created"

@pytest.mark.order(37)
def test_run_smina_local_only_flag(tmp_path, monkeypatch):
    '''Test run_smina with local_only flag enabled.'''
    
    # Mock config with local_only = "yes"
    def mock_get_config():
        class MockSminaConfig:
            executable = "/nonexistent/smina"
            local_only = "yes"
            minimize = "no"
            randomize_only = "no"
            accurate_line = "no"
            minimize_early_term = "no"
        class MockConfig:
            smina = MockSminaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    out_file = tmp_path / "output" / "out.pdbqt"
    smina_log = tmp_path / "output" / "smina.log"
    
    result = ocsmina.run_smina(
        config=str(conf_file),
        prepared_ligand=str(ligand_file),
        output_smina=str(out_file),
        smina_log=str(smina_log),
        log_path=""
    )
    
    # Should return OK (stub files created)
    assert result == ocerror.Error.ok() # type: ignore

@pytest.mark.order(38)
def test_run_smina_minimize_flag(tmp_path, monkeypatch):
    '''Test run_smina with minimize flag enabled.'''
    
    # Mock config with minimize = "yes"
    def mock_get_config():
        class MockSminaConfig:
            executable = "/nonexistent/smina"
            local_only = "no"
            minimize = "yes"
            randomize_only = "no"
            accurate_line = "no"
            minimize_early_term = "no"
        class MockConfig:
            smina = MockSminaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    out_file = tmp_path / "output" / "out.pdbqt"
    smina_log = tmp_path / "output" / "smina.log"
    
    result = ocsmina.run_smina(
        config=str(conf_file),
        prepared_ligand=str(ligand_file),
        output_smina=str(out_file),
        smina_log=str(smina_log),
        log_path=""
    )
    
    assert result == ocerror.Error.ok() # type: ignore

@pytest.mark.order(31)
def test_run_smina_stub_file_creation(tmp_path, monkeypatch):
    '''Test run_smina creates stub files when smina executable is not available.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockSminaConfig:
            executable = "/nonexistent/smina"
            local_only = "no"
            minimize = "no"
            randomize_only = "no"
            accurate_line = "no"
            minimize_early_term = "no"
        class MockConfig:
            smina = MockSminaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    out_file = tmp_path / "output" / "out.pdbqt"
    smina_log = tmp_path / "output" / "smina.log"
    log_file = tmp_path / "output" / "log.txt"
    
    result = ocsmina.run_smina(
        config=str(conf_file),
        prepared_ligand=str(ligand_file),
        output_smina=str(out_file),
        smina_log=str(smina_log),
        log_path=str(log_file)
    )
    
    # Should return OK (stub files created)
    assert result == ocerror.Error.ok() # type: ignore
    # Check that stub files were created
    assert out_file.exists()
    assert "stub" in out_file.read_text().lower()

@pytest.mark.order(32)
def test_run_smina_stub_no_output(tmp_path, monkeypatch):
    '''Test run_smina stub creation when output_smina is empty.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockSminaConfig:
            executable = "/nonexistent/smina"
            local_only = "no"
            minimize = "no"
            randomize_only = "no"
            accurate_line = "no"
            minimize_early_term = "no"
        class MockConfig:
            smina = MockSminaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    conf_file = tmp_path / "conf.txt"
    conf_file.write_text("receptor = rec.pdbqt\n")
    ligand_file = tmp_path / "ligand.pdbqt"
    ligand_file.write_text("MODEL 1\n")
    smina_log = tmp_path / "smina.log"
    
    result = ocsmina.run_smina(
        config=str(conf_file),
        prepared_ligand=str(ligand_file),
        output_smina="",  # Empty output
        smina_log=str(smina_log),
        log_path=""
    )
    
    # Should return OK (stub log file created)
    assert result == ocerror.Error.ok() # type: ignore
    assert smina_log.exists()

@pytest.mark.order(9)
def test_smina_get_docked_poses(smina_inputs, tmp_path):
    '''Test get_docked_poses method.'''

    # Create a mock output directory with some pose files
    output_dir = tmp_path / "docked_poses"
    output_dir.mkdir()
    
    # Create some mock pose files
    (output_dir / "ligand_split_1.pdbqt").write_text("POSE1")
    (output_dir / "ligand_split_2.pdbqt").write_text("POSE2")
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=str(output_dir / "smina_out.pdbqt"),
        name="test"
    )
    
    result = smina_instance.get_docked_poses()
    assert sorted(result) == [str(output_dir / "ligand_split_1.pdbqt"), str(output_dir / "ligand_split_2.pdbqt")]

@pytest.mark.order(7)
def test_smina_get_input_ligand_path(smina_inputs):
    '''Test get_input_ligand_path method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.get_input_ligand_path()
    assert result == str(Path(smina_inputs["converted_ligand_file"]).parent)

@pytest.mark.order(8)
def test_smina_get_input_receptor_path(smina_inputs):
    '''Test get_input_receptor_path method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.get_input_receptor_path()
    assert result == str(Path(smina_inputs["receptor_file"]).parent)

@pytest.mark.order(17)
def test_smina_instance_run_prepare_ligand(smina_inputs):
    '''Test instance run_prepare_ligand method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.run_prepare_ligand()
    assert result == ocerror.Error.ok(), f"Preparation of ligand failed. Error code: {result}" # type: ignore
    assert Path(smina_inputs["ligand_path"]).is_file()

@pytest.mark.order(18)
def test_smina_instance_run_prepare_receptor(smina_inputs):
    '''Test instance run_prepare_receptor method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.run_prepare_receptor()
    assert result == ocerror.Error.ok(), f"Preparation of receptor failed. Error code: {result}" # type: ignore
    assert Path(smina_inputs["receptor_path"]).is_file()

@pytest.mark.order(19)
def test_smina_instance_run_smina(smina_inputs, tmp_path):
    '''Test instance run_smina method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=str(tmp_path / "smina_out.pdbqt"),
        name="test"
    )
    
    log_file = tmp_path / "smina_exec.log"
    result = smina_instance.run_smina(str(log_file))
    assert result == ocerror.Error.ok(), f"Smina docking failed. Error code: {result}" # type: ignore
    assert (tmp_path / "smina_out.pdbqt").is_file()

@pytest.mark.order(35)
def test_smina_instance_run_smina_stub(smina_inputs, tmp_path, monkeypatch):
    '''Test Smina.run_smina creates stub files when smina not available.'''
    
    # Mock config to return non-existent executable
    def mock_get_config():
        class MockSminaConfig:
            executable = "/nonexistent/smina"
            local_only = "no"
            minimize = "no"
            randomize_only = "no"
            accurate_line = "no"
            minimize_early_term = "no"
        class MockConfig:
            smina = MockSminaConfig()
        return MockConfig()
    
    monkeypatch.setattr(ocsmina, 'get_config', mock_get_config)
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=str(tmp_path / "output" / "smina_out.pdbqt"),
        name="test"
    )
    
    result = smina_instance.run_smina(logFile=str(tmp_path / "log.txt"))
    
    # Should return OK (stub files created)
    assert result == ocerror.Error.ok() # type: ignore
    assert smina_instance.output_smina and Path(smina_instance.output_smina).exists()

@pytest.mark.order(1)
def test_smina_instantiation(smina_inputs):
    '''
    Test Smina class can be instantiated with all required inputs.
    '''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["prepared_receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["prepared_ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test",
        overwrite_config=True
    )
    assert isinstance(smina_instance, ocsmina.Smina), "Smina instance not created correctly"

@pytest.mark.order(16)
def test_smina_print_attributes(smina_inputs, capsys):
    '''Test print_attributes method.'''

    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    smina_instance.print_attributes()
    captured = capsys.readouterr()
    assert "Name:" in captured.out
    assert "Config path:" in captured.out

@pytest.mark.order(10)
def test_smina_read_log(smina_inputs, tmp_path):
    '''Test read_log method.'''

    # Create a mock log file
    log_file = tmp_path / "smina.log"
    log_file.write_text(
        "mode |   affinity | dist from best mode\n"
        "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
        "-----+------------+----------+----------\n"
        "1       -7.5       0.000      0.000\n"
        "2       -6.8       1.234      2.345\n"
    )
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=str(log_file),
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    from OCDocker.Config import get_config
    smina_scoring = get_config().smina.scoring

    result = smina_instance.read_log()
    assert result == {1: {smina_scoring: -7.5}, 2: {smina_scoring: -6.8}}

@pytest.mark.order(11)
def test_smina_read_log_only_best(smina_inputs, tmp_path):
    '''Test read_log method with onlyBest=True.'''

    log_file = tmp_path / "smina.log"
    log_file.write_text(
        "mode |   affinity | dist from best mode\n"
        "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
        "-----+------------+----------+----------\n"
        "1       -7.5       0.000      0.000\n"
        "2       -6.8       1.234      2.345\n"
    )
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=str(log_file),
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    from OCDocker.Config import get_config
    smina_scoring = get_config().smina.scoring

    result = smina_instance.read_log(onlyBest=True)
    assert result == {1: {smina_scoring: -7.5}}

@pytest.mark.order(12)
def test_smina_read_rescore_logs(smina_inputs, tmp_path):
    '''Test read_rescore_logs method.'''

    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create mock rescoring log files
    log1 = out_path / "lig_split_1_vinardo_rescoring.log"
    log1.write_text("Affinity    -7.23 (kcal/mol)\n")
    
    log2 = out_path / "lig_split_2_vinardo_rescoring.log"
    log2.write_text("Affinity    -6.50 (kcal/mol)\n")
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.read_rescore_logs(str(out_path))
    assert result == {"rescoring_vinardo_1": -7.23, "rescoring_vinardo_2": -6.5}

@pytest.mark.order(13)
def test_smina_read_rescore_logs_only_best(smina_inputs, tmp_path):
    '''Test read_rescore_logs method with onlyBest=True.'''

    out_path = tmp_path / "rescoring"
    out_path.mkdir()
    
    # Create mock rescoring log files
    log1 = out_path / "lig_split_1_vinardo_rescoring.log"
    log1.write_text("Affinity    -7.23 (kcal/mol)\n")
    
    log2 = out_path / "lig_split_2_vinardo_rescoring.log"
    log2.write_text("Affinity    -6.50 (kcal/mol)\n")
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    result = smina_instance.read_rescore_logs(str(out_path), onlyBest=True)
    assert result == {"rescoring_vinardo_1": -7.23}

@pytest.mark.order(33)
def test_smina_run_rescore_instance_method(smina_inputs, tmp_path, monkeypatch):
    '''Test Smina.run_rescore instance method with different parameters.'''
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    # Create mock split poses
    split_dir = tmp_path / "split_poses"
    split_dir.mkdir()
    split_lig = split_dir / "lig_split_1.pdbqt"
    split_lig.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    calls = []
    monkeypatch.setattr(ocsmina, 'get_config', lambda: SimpleNamespace(smina=SimpleNamespace(scoring="vinardo", scoring_functions=["vinardo", "dkoes_scoring"])))
    monkeypatch.setattr(ocsmina, 'run_rescore', lambda confFile, ligand, outPath, scoring_function, **kwargs: calls.append(scoring_function))
    
    # Test run_rescore with skipDefaultScoring=True (the default scoring function was used in the docking)
    result = smina_instance.run_rescore(
        outPath=str(split_dir),
        ligand=str(split_lig),
        logFile="",
        skipDefaultScoring=True,
        overwrite=False
    )
    assert calls == ["dkoes_scoring"]
    # Method returns None
    assert result is None

@pytest.mark.order(34)
def test_smina_run_rescore_overwrite_true(smina_inputs, tmp_path, monkeypatch):
    '''Test Smina.run_rescore instance method with overwrite=True.'''
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=smina_inputs["output_file"],
        name="test"
    )
    
    split_dir = tmp_path / "split_poses"
    split_dir.mkdir()
    split_lig = split_dir / "lig_split_1.pdbqt"
    split_lig.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    calls = []
    monkeypatch.setattr(ocsmina, 'get_config', lambda: SimpleNamespace(smina=SimpleNamespace(scoring="vinardo", scoring_functions=["vinardo", "dkoes_scoring"])))
    monkeypatch.setattr(ocsmina, 'run_rescore', lambda confFile, ligand, outPath, scoring_function, splitLigand, overwrite, **kwargs: calls.append((scoring_function, splitLigand, overwrite)))
    
    # Test run_rescore with overwrite=True (the ligand is split only for the first scoring function)
    result = smina_instance.run_rescore(
        outPath=str(split_dir),
        ligand=str(split_lig),
        logFile="",
        skipDefaultScoring=False,
        splitLigand=True,
        overwrite=True
    )
    assert calls == [("vinardo", True, True), ("dkoes_scoring", False, True)]
    assert result is None

@pytest.mark.order(14)
def test_smina_split_poses(smina_inputs, tmp_path):
    '''Test split_poses method.'''

    # Create a mock output file
    output_file = tmp_path / "smina_out.pdbqt"
    output_file.write_text("MODEL 1\nATOM\nENDMDL\nMODEL 2\nATOM\nENDMDL\n")
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=str(output_file),
        name="test"
    )
    
    out_path = tmp_path / "split_poses"
    out_path.mkdir()
    
    result = smina_instance.split_poses(str(out_path))
    assert result == ocerror.Error.ok() # type: ignore
    name = smina_instance.input_ligand.name
    assert sorted(f.name for f in out_path.iterdir()) == [f"{name}_split_1.pdbqt", f"{name}_split_2.pdbqt"]

@pytest.mark.order(15)
def test_smina_split_poses_default_path(smina_inputs, tmp_path):
    '''Test split_poses method with default outPath.'''

    output_file = tmp_path / "smina_out.pdbqt"
    output_file.write_text("MODEL 1\nATOM\nENDMDL\n")
    
    smina_instance = ocsmina.Smina(
        config_path=smina_inputs["config"],
        box_file=smina_inputs["box"],
        receptor=smina_inputs["receptor"],
        prepared_receptor_path=smina_inputs["receptor_path"],
        ligand=smina_inputs["ligand"],
        prepared_ligand_path=smina_inputs["ligand_path"],
        smina_log=smina_inputs["smina_log"],
        output_smina=str(output_file),
        name="test"
    )
    
    result = smina_instance.split_poses()
    assert result == ocerror.Error.ok() # type: ignore
    # The poses are written next to the smina output
    assert (tmp_path / f"{smina_instance.input_ligand.name}_split_1.pdbqt").is_file()
