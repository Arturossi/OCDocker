import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import OCDocker.DB.baseDB as ocbdb
import OCDocker.Error as ocerror


@pytest.fixture
def mock_config(monkeypatch):
    '''Fixture to mock get_config.'''

    def mock_get_config():
        class MockPaths:
            dudez_archive = "/mock/dudez"
            pdbbind_archive = "/mock/pdbbind"
        class MockConfig:
            dudez_archive = "/mock/dudez"
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    return mock_get_config


@pytest.mark.order(1)
def test_prepare_dudez(mock_config, monkeypatch, tmp_path):
    '''Test prepare function with dudez archive.'''
    
    # Create mock directories
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    (dudez_dir / "protein1").mkdir()
    (dudez_dir / "protein2").mkdir()
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocprepare.prepare
    prepare_called = []
    def mock_prepare(paths, overwrite, archive, sanitize, spacing):
        prepare_called.append((paths, overwrite, archive, sanitize, spacing))
        return None
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocprepare.prepare", mock_prepare)
    
    # Call prepare
    result = ocbdb.prepare("dudez", overwrite=True, spacing=0.5, sanitize=False)
    
    # Verify prepare was called
    assert len(prepare_called) == 1
    assert prepare_called[0][1] is True  # overwrite
    assert prepare_called[0][2] == "dudez"  # archive
    assert prepare_called[0][3] is False  # sanitize
    assert prepare_called[0][4] == 0.5  # spacing
    assert result is None


@pytest.mark.order(2)
def test_prepare_pdbbind(mock_config, monkeypatch, tmp_path):
    '''Test prepare function with pdbbind archive.'''
    
    # Create mock directories
    pdbbind_dir = tmp_path / "pdbbind"
    pdbbind_dir.mkdir()
    (pdbbind_dir / "protein1").mkdir()
    (pdbbind_dir / "index").mkdir()  # Should be excluded
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = "/mock/dudez"
            pdbbind_archive = str(pdbbind_dir)
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocprepare.prepare
    prepare_called = []
    def mock_prepare(paths, overwrite, archive, sanitize, spacing):
        prepare_called.append((paths, overwrite, archive, sanitize, spacing))
        return None
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocprepare.prepare", mock_prepare)
    
    # Call prepare
    result = ocbdb.prepare("pdbbind", overwrite=False)
    
    # Verify prepare was called
    assert len(prepare_called) == 1
    assert prepare_called[0][1] is False  # overwrite
    assert prepare_called[0][2] == "pdbbind"  # archive
    assert result is None


@pytest.mark.order(3)
def test_prepare_invalid_archive(mock_config, monkeypatch):
    '''Test prepare function with invalid archive type.'''
    
    # Mock ocprint.print_error
    error_called = []
    def mock_print_error(message):
        error_called.append(message)
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocprint.print_error", mock_print_error)
    
    # Call prepare with invalid archive
    result = ocbdb.prepare("invalid_archive")
    
    # Verify error was printed
    assert len(error_called) == 1
    assert "Not valid archive type" in error_called[0]
    assert result is None


@pytest.mark.order(4)
def test_run_docking_dudez(mock_config, monkeypatch, tmp_path):
    '''Test run_docking function with dudez archive.'''
    
    # Create mock directories
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    ptn1 = dudez_dir / "protein1"
    ptn1.mkdir()
    (ptn1 / "compounds" / "ligands").mkdir(parents=True)
    (ptn1 / "compounds" / "decoys").mkdir(parents=True)
    (ptn1 / "compounds" / "candidates").mkdir(parents=True)
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocdock.run_docking
    docking_called = []
    def mock_run_docking(complexList, archive, dockingAlgorithm, overwrite, digestFormat):
        docking_called.append((complexList, archive, dockingAlgorithm, overwrite, digestFormat))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocdock.run_docking", mock_run_docking)
    
    # Call run_docking
    result = ocbdb.run_docking("dudez", "vina", digestFormat="json", overwrite=True)
    
    # Verify run_docking was called
    assert len(docking_called) == 1
    assert docking_called[0][1] == "dudez"  # archive
    assert docking_called[0][2] == "vina"  # dockingAlgorithm
    assert docking_called[0][3] is True  # overwrite
    assert docking_called[0][4] == "json"  # digestFormat
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(5)
def test_run_docking_pdbbind(mock_config, monkeypatch, tmp_path):
    '''Test run_docking function with pdbbind archive.'''
    
    # Create mock directories
    pdbbind_dir = tmp_path / "pdbbind"
    pdbbind_dir.mkdir()
    ptn1 = pdbbind_dir / "protein1"
    ptn1.mkdir()
    (ptn1 / "compounds" / "ligands").mkdir(parents=True)
    (ptn1 / "compounds" / "decoys").mkdir(parents=True)
    (ptn1 / "compounds" / "candidates").mkdir(parents=True)
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = "/mock/dudez"
            pdbbind_archive = str(pdbbind_dir)
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocdock.run_docking
    docking_called = []
    def mock_run_docking(complexList, archive, dockingAlgorithm, overwrite, digestFormat):
        docking_called.append((complexList, archive, dockingAlgorithm, overwrite, digestFormat))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocdock.run_docking", mock_run_docking)
    
    # Call run_docking
    result = ocbdb.run_docking("pdbbind", "smina", overwrite=False)
    
    # Verify run_docking was called
    assert len(docking_called) == 1
    assert docking_called[0][1] == "pdbbind"  # archive
    assert docking_called[0][2] == "smina"  # dockingAlgorithm
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(6)
def test_run_docking_invalid_archive(mock_config, monkeypatch):
    '''Test run_docking function with invalid archive type.'''
    
    # Call run_docking with invalid archive
    result = ocbdb.run_docking("invalid_archive", "vina")
    
    # Verify error was returned
    assert result != ocerror.Error.ok() # type: ignore
    assert isinstance(result, int)


@pytest.mark.order(7)
def test_run_docking_invalid_algorithm(mock_config, monkeypatch, tmp_path):
    '''Test run_docking function with invalid docking algorithm.'''
    
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Call run_docking with invalid algorithm
    result = ocbdb.run_docking("dudez", "invalid_algorithm")
    
    # Verify error was returned
    assert result != ocerror.Error.ok() # type: ignore
    assert isinstance(result, int)


@pytest.mark.order(8)
def test_run_docking_gnina(mock_config, monkeypatch, tmp_path):
    '''Test run_docking function with gnina algorithm.'''
    
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocdock.run_docking
    monkeypatch.setattr("OCDocker.DB.baseDB.ocdock.run_docking", lambda *args, **kwargs: ocerror.Error.ok()) # type: ignore
    
    # Call run_docking with gnina
    result = ocbdb.run_docking("dudez", "gnina")
    
    # Verify ok was returned
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(9)
def test_run_docking_plants(mock_config, monkeypatch, tmp_path):
    '''Test run_docking function with plants algorithm.'''
    
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocdock.run_docking
    monkeypatch.setattr("OCDocker.DB.baseDB.ocdock.run_docking", lambda *args, **kwargs: ocerror.Error.ok()) # type: ignore
    
    # Call run_docking with plants
    result = ocbdb.run_docking("dudez", "plants")
    
    # Verify ok was returned
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(10)
def test_run_docking_excludes_index_directory(mock_config, monkeypatch, tmp_path):
    '''Test run_docking excludes index directory from complex list.'''
    
    dudez_dir = tmp_path / "dudez"
    dudez_dir.mkdir()
    (dudez_dir / "protein1").mkdir()
    (dudez_dir / "index").mkdir()  # Should be excluded
    
    def mock_get_config():
        class MockConfig:
            dudez_archive = str(dudez_dir)
            pdbbind_archive = "/mock/pdbbind"
        return MockConfig()
    
    monkeypatch.setattr("OCDocker.DB.baseDB.get_config", mock_get_config)
    
    # Mock ocdock.run_docking
    docking_complex_list = []
    def mock_run_docking(complexList, archive, dockingAlgorithm, overwrite, digestFormat):
        docking_complex_list.extend(complexList)
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.baseDB.ocdock.run_docking", mock_run_docking)
    
    # Call run_docking
    ocbdb.run_docking("dudez", "vina")
    
    # Verify index directory was excluded
    complex_paths = [str(c[0]) for c in docking_complex_list]
    assert str(dudez_dir / "index") not in complex_paths
    assert str(dudez_dir / "protein1") in complex_paths

