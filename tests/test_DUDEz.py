import pytest
from unittest.mock import Mock, patch, MagicMock

import OCDocker.DB.DUDEz as ocdudez
import OCDocker.DB.baseDB as ocbdb
import OCDocker.Error as ocerror


@pytest.mark.order(1)
def test_prepare(monkeypatch):
    '''Test DUDEz.prepare function.'''
    
    # Mock ocbdb.prepare
    prepare_called = []
    def mock_prepare(archive, overwrite, spacing, sanitize):
        prepare_called.append((archive, overwrite, spacing, sanitize))
        return None
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.prepare", mock_prepare)
    
    # Call prepare
    result = ocdudez.prepare(overwrite=True, spacing=0.5, sanitize=False)
    
    # Verify prepare was called with correct arguments
    assert len(prepare_called) == 1
    assert prepare_called[0][0] == "dudez"  # archive
    assert prepare_called[0][1] is True  # overwrite
    assert prepare_called[0][2] == 0.5  # spacing
    assert prepare_called[0][3] is False  # sanitize
    assert result is None


@pytest.mark.order(2)
def test_prepare_defaults(monkeypatch):
    '''Test DUDEz.prepare function with default parameters.'''
    
    # Mock ocbdb.prepare
    prepare_called = []
    def mock_prepare(archive, overwrite, spacing, sanitize):
        prepare_called.append((archive, overwrite, spacing, sanitize))
        return None
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.prepare", mock_prepare)
    
    # Call prepare with defaults
    result = ocdudez.prepare()
    
    # Verify prepare was called with default arguments
    assert len(prepare_called) == 1
    assert prepare_called[0][0] == "dudez"  # archive
    assert prepare_called[0][1] is False  # overwrite (default)
    assert prepare_called[0][2] == 0.33  # spacing (default)
    assert prepare_called[0][3] is True  # sanitize (default)
    assert result is None


@pytest.mark.order(3)
def test_run_gnina(monkeypatch):
    '''Test DUDEz.run_gnina function.'''
    
    # Mock ocbdb.run_docking
    docking_called = []
    def mock_run_docking(archive, algorithm, overwrite):
        docking_called.append((archive, algorithm, overwrite))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.run_docking", mock_run_docking)
    
    # Call run_gnina
    result = ocdudez.run_gnina(overwrite=True)
    
    # Verify run_docking was called with correct arguments
    assert len(docking_called) == 1
    assert docking_called[0][0] == "dudez"  # archive
    assert docking_called[0][1] == "gnina"  # algorithm
    assert docking_called[0][2] is True  # overwrite
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(4)
def test_run_gnina_default(monkeypatch):
    '''Test DUDEz.run_gnina function with default parameters.'''
    
    # Mock ocbdb.run_docking
    docking_called = []
    def mock_run_docking(archive, algorithm, overwrite):
        docking_called.append((archive, algorithm, overwrite))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.run_docking", mock_run_docking)
    
    # Call run_gnina with defaults
    result = ocdudez.run_gnina()
    
    # Verify run_docking was called with default overwrite=False
    assert len(docking_called) == 1
    assert docking_called[0][2] is False  # overwrite (default)
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(5)
def test_run_vina(monkeypatch):
    '''Test DUDEz.run_vina function.'''
    
    # Mock ocbdb.run_docking
    docking_called = []
    def mock_run_docking(archive, algorithm, overwrite):
        docking_called.append((archive, algorithm, overwrite))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.run_docking", mock_run_docking)
    
    # Call run_vina
    result = ocdudez.run_vina(overwrite=False)
    
    # Verify run_docking was called with correct arguments
    assert len(docking_called) == 1
    assert docking_called[0][0] == "dudez"  # archive
    assert docking_called[0][1] == "vina"  # algorithm
    assert docking_called[0][2] is False  # overwrite
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(6)
def test_run_smina(monkeypatch):
    '''Test DUDEz.run_smina function.'''
    
    # Mock ocbdb.run_docking
    docking_called = []
    def mock_run_docking(archive, algorithm, overwrite):
        docking_called.append((archive, algorithm, overwrite))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.run_docking", mock_run_docking)
    
    # Call run_smina
    result = ocdudez.run_smina(overwrite=True)
    
    # Verify run_docking was called with correct arguments
    assert len(docking_called) == 1
    assert docking_called[0][0] == "dudez"  # archive
    assert docking_called[0][1] == "smina"  # algorithm
    assert docking_called[0][2] is True  # overwrite
    assert result == ocerror.Error.ok() # type: ignore


@pytest.mark.order(7)
def test_run_plants(monkeypatch):
    '''Test DUDEz.run_plants function.'''
    
    # Mock ocbdb.run_docking
    docking_called = []
    def mock_run_docking(archive, algorithm, overwrite):
        docking_called.append((archive, algorithm, overwrite))
        return ocerror.Error.ok() # type: ignore
    
    monkeypatch.setattr("OCDocker.DB.DUDEz.ocbdb.run_docking", mock_run_docking)
    
    # Call run_plants
    result = ocdudez.run_plants(overwrite=False)
    
    # Verify run_docking was called with correct arguments
    assert len(docking_called) == 1
    assert docking_called[0][0] == "dudez"  # archive
    assert docking_called[0][1] == "plants"  # algorithm
    assert docking_called[0][2] is False  # overwrite
    assert result == ocerror.Error.ok() # type: ignore

