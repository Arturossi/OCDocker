import json
import os
from pathlib import Path
import types
import sys

import pytest


def _setup_basevina_stubs(monkeypatch, tmp_path):
    """Set up minimal stubs needed for all BaseVinaLike tests.
    
    All tests need:
    1. numpy - must be ModuleType (not SimpleNamespace) with nan/isnan
    2. FilesFolders - must be stubbed to prevent it from importing real numpy
    
    Additional stubs based on what functions are tested:
    - IO, Printing, Config: used by read_log and read_rescoring_log
    - Validation, FilesFolders functions: used by generate_digest
    """
    # CRITICAL: Clear and set up numpy FIRST, before any other imports
    if 'numpy' in sys.modules:
        monkeypatch.delitem(sys.modules, 'numpy', raising=False)
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.nan = float('nan')  # type: ignore
    numpy_stub.isnan = lambda x: x != x  # type: ignore
    monkeypatch.setitem(sys.modules, 'numpy', numpy_stub)
    
    # Clear modules that BaseVinaLike imports (to prevent cached imports)
    for mod_name in ['OCDocker.Toolbox.FilesFolders', 'OCDocker.Toolbox.IO', 
                     'OCDocker.Toolbox.Printing', 'OCDocker.Toolbox.Validation',
                     'OCDocker.Config', 'OCDocker.Docking.BaseVinaLike']:
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)
    
    # Stub Toolbox package (needed for submodule imports)
    toolbox_pkg = types.ModuleType('OCDocker.Toolbox')
    toolbox_pkg.__path__ = []  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox', toolbox_pkg)
    
    # Stub FilesFolders - CRITICAL: prevents real module from loading and importing numpy
    # Also provide functions used by generate_digest
    filesfolders_mod = types.ModuleType('OCDocker.Toolbox.FilesFolders')
    def empty_docking_digest(path, overwrite=False):
        # Minimal stub - just return empty dict
        return {}
    filesfolders_mod.empty_docking_digest = empty_docking_digest  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.FilesFolders', filesfolders_mod)
    
    # Stub IO (used by read_log and read_rescoring_log)
    io_mod = types.ModuleType('OCDocker.Toolbox.IO')
    def lazyread_reverse_order_mmap(path):
        return []  # Empty iterator - tests will provide actual file content
    io_mod.lazyread_reverse_order_mmap = lazyread_reverse_order_mmap  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.IO', io_mod)
    
    # Stub Printing (used by read_rescoring_log and generate_digest)
    printing_mod = types.ModuleType('OCDocker.Toolbox.Printing')
    printing_mod.print_error = lambda *a, **k: None  # type: ignore
    printing_mod.print_error_log = lambda *a, **k: None  # type: ignore
    printing_mod.printv = lambda *a, **k: None  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Printing', printing_mod)
    
    # Stub Validation (used by generate_digest)
    validation_mod = types.ModuleType('OCDocker.Toolbox.Validation')
    def validate_digest_extension(path, format_str):
        # Accept json, reject others
        return format_str.lower() == 'json'
    validation_mod.validate_digest_extension = validate_digest_extension  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Validation', validation_mod)
    
    # Stub Config (used by read_rescoring_log and generate_digest)
    config_mod = types.ModuleType('OCDocker.Config')
    def get_config():
        cfg = types.SimpleNamespace()
        cfg.vina = types.SimpleNamespace(scoring='vina', scoring_functions=['vina'])
        cfg.smina = types.SimpleNamespace(scoring='vinardo', scoring_functions=['vinardo'])
        cfg.logdir = str(tmp_path)  # type: ignore
        return cfg
    config_mod.get_config = get_config  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Config', config_mod)
    
    return numpy_stub


@pytest.mark.order(99)
def test_vina_log_and_rescoring_parsing(tmp_path, monkeypatch):
    # Avoid heavy Initialise auto-bootstrap on import
    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    log = tmp_path / "vina.log"
    # Header first, then rows (reverse reader stops at header when it reaches it)
    log.write_text('''-----+ header
1 -7.50 0 0
2 -6.20 0 0
''')
    from OCDocker.Config import get_config
    config = get_config()
    vina_scoring = config.vina.scoring
    
    data = basevina.read_vina_log(str(log))
    assert set(data.keys()) == {1, 2}
    assert data[1][vina_scoring] == "-7.50"  # type: ignore

    best_only = basevina.read_vina_log(str(log), onlyBest=True)
    assert set(best_only.keys()) == {1}

    resc = tmp_path / "rescore.log"
    resc.write_text("Estimated Free Energy of Binding    -7.23 (kcal/mol)\n")
    val = basevina.read_vina_rescoring_log(str(resc))
    assert val == -7.23

    out_json = tmp_path / "digest.json"
    rc = basevina.generate_vina_digest(str(out_json), str(log), overwrite=True, digestFormat="json")
    assert rc == 0 and out_json.exists()
    from OCDocker.Config import get_config
    config = get_config()
    vina_scoring = config.vina.scoring
    
    j = json.loads(out_json.read_text())
    # Top level contains base keys and pose keys as strings
    assert "vina_affinity" in j
    assert "1" in j and j["1"][vina_scoring] == "-7.50"  # type: ignore


@pytest.mark.order(100)
def test_smina_log_and_rescoring_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    log = tmp_path / "smina.log"
    log.write_text('''-----+ header
1 -8.00 0 0
3 -6.75 0 0
''')
    from OCDocker.Config import get_config
    config = get_config()
    smina_scoring = config.smina.scoring
    
    data = basevina.read_smina_log(str(log))
    assert set(data.keys()) == {1, 3}
    assert data[3][smina_scoring] == "-6.75"  # type: ignore

    resc = tmp_path / "rescore_smina.log"
    resc.write_text("Affinity: -6.71 (kcal/mol)\n")
    val = basevina.read_smina_rescoring_log(str(resc))
    assert val == -6.71

    out_json = tmp_path / "digest_smina.json"
    rc = basevina.generate_smina_digest(str(out_json), str(log), overwrite=True, digestFormat="json")
    assert rc == 0 and out_json.exists()
    j = json.loads(out_json.read_text())
    assert "smina_affinity" in j
    assert "1" in j


@pytest.mark.order(101)
def test_read_log_empty_file(tmp_path, monkeypatch):
    '''Test reading an empty log file.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("")  # Create empty file
    
    result = basevina.read_vina_log(str(empty_log))
    assert result == {}  # Should return empty dict


@pytest.mark.order(102)
def test_read_log_nonexistent_file(tmp_path, monkeypatch):
    '''Test reading a non-existent log file.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    nonexistent = tmp_path / "nonexistent.log"
    result = basevina.read_vina_log(str(nonexistent))
    assert result == {}  # Should return empty dict


@pytest.mark.order(103)
def test_read_rescoring_log_empty_file(tmp_path, monkeypatch):
    '''Test reading an empty rescoring log file.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import numpy as np
    
    empty_resc = tmp_path / "empty_rescore.log"
    empty_resc.write_text("")
    
    result = basevina.read_vina_rescoring_log(str(empty_resc))
    assert np.isnan(result)  # Should return NaN


@pytest.mark.order(104)
def test_read_rescoring_log_nonexistent_file(tmp_path, monkeypatch):
    '''Test reading a non-existent rescoring log file.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import numpy as np
    
    nonexistent = tmp_path / "nonexistent_rescore.log"
    result = basevina.read_vina_rescoring_log(str(nonexistent))
    assert np.isnan(result)  # Should return NaN


@pytest.mark.order(105)
def test_read_rescoring_log_no_affinity_line(tmp_path, monkeypatch):
    '''Test reading rescoring log without affinity line.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import numpy as np
    
    resc = tmp_path / "no_affinity.log"
    resc.write_text("Some random text\nNo affinity here\n")
    
    result = basevina.read_vina_rescoring_log(str(resc))
    assert np.isnan(result)  # Should return NaN when no affinity line found


@pytest.mark.order(106)
def test_read_smina_rescoring_log(tmp_path, monkeypatch):
    '''Test reading smina rescoring log.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    resc = tmp_path / "smina_rescore.log"
    resc.write_text("Affinity: -8.50 (kcal/mol)\n")
    
    result = basevina.read_smina_rescoring_log(str(resc))
    assert result == -8.50


@pytest.mark.order(107)
def test_generate_digest_file_exists_no_overwrite(tmp_path, monkeypatch):
    '''Test generate_digest when path is a directory and overwrite=False.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import OCDocker.Error as ocerror
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    # Create a directory (not a file) - this triggers file_exists error
    digest_dir = tmp_path / "digest_dir"
    digest_dir.mkdir()
    
    rc = basevina.generate_vina_digest(str(digest_dir), str(log), overwrite=False)
    assert rc == ocerror.Error.file_exists()  # type: ignore


@pytest.mark.order(108)
def test_generate_digest_invalid_existing_digest(tmp_path, monkeypatch):
    '''Test generate_digest with invalid existing digest file (not a dict).'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import OCDocker.Error as ocerror
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    digest = tmp_path / "digest.json"
    digest.write_text('["not", "a", "dict"]')  # Invalid digest (array, not dict)
    
    rc = basevina.generate_vina_digest(str(digest), str(log), overwrite=True)
    assert rc == ocerror.Error.wrong_type()  # type: ignore


@pytest.mark.order(109)
def test_generate_digest_read_error(tmp_path, monkeypatch):
    '''Test generate_digest with unreadable digest file.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import OCDocker.Error as ocerror
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    digest = tmp_path / "digest.json"
    digest.write_text('{"invalid": json}')  # Invalid JSON
    
    rc = basevina.generate_vina_digest(str(digest), str(log), overwrite=True)
    assert rc == ocerror.Error.file_not_exist()  # type: ignore


@pytest.mark.order(110)
def test_generate_digest_unsupported_format(tmp_path, monkeypatch):
    '''Test generate_digest with unsupported format.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import OCDocker.Error as ocerror
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    digest = tmp_path / "digest.hdf5"
    
    rc = basevina.generate_vina_digest(str(digest), str(log), overwrite=True, digestFormat="hdf5")
    assert rc == ocerror.Error.unsupported_extension()  # type: ignore


@pytest.mark.order(111)
def test_get_docked_poses_nonexistent_directory(tmp_path, monkeypatch):
    '''Test get_docked_poses with non-existent directory.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    nonexistent = tmp_path / "nonexistent_poses"
    result = basevina.get_vina_docked_poses(str(nonexistent))
    assert result == []  # Should return empty list


@pytest.mark.order(112)
def test_get_docked_poses_empty_directory(tmp_path, monkeypatch):
    '''Test get_docked_poses with empty directory.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    
    result = basevina.get_vina_docked_poses(str(poses_dir))
    assert result == []  # Should return empty list when no matching files


@pytest.mark.order(113)
def test_get_docked_poses_with_files(tmp_path, monkeypatch):
    '''Test get_docked_poses with matching pose files.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    poses_dir = tmp_path / "poses"
    poses_dir.mkdir()
    
    # Create matching pose files
    pose1 = poses_dir / "lig_split_1.pdbqt"
    pose2 = poses_dir / "lig_split_2.pdbqt"
    non_match = poses_dir / "lig_regular.pdbqt"  # Doesn't match pattern
    
    pose1.write_text("POSE 1")
    pose2.write_text("POSE 2")
    non_match.write_text("NON-MATCH")
    
    result = basevina.get_vina_docked_poses(str(poses_dir))
    assert len(result) == 2
    assert any("lig_split_1.pdbqt" in p for p in result)
    assert any("lig_split_2.pdbqt" in p for p in result)


@pytest.mark.order(114)
def test_read_log_with_exception(tmp_path, monkeypatch):
    '''Test read_log handling of general exceptions during file reading.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    # Mock lazyread_reverse_order_mmap to raise an exception
    def mock_raise(*args, **kwargs):
        raise RuntimeError("Mock error")
    
    import OCDocker.Toolbox.IO as ocio
    monkeypatch.setattr(ocio, 'lazyread_reverse_order_mmap', mock_raise)
    
    result = basevina.read_vina_log(str(log))
    assert result == {}  # Should return empty dict on exception


@pytest.mark.order(115)
def test_read_log_invalid_line_format(tmp_path, monkeypatch):
    '''Test read_log with lines that don't have exactly 4 elements.'''

    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    from OCDocker.Config import get_config
    config = get_config()
    vina_scoring = config.vina.scoring
    
    log = tmp_path / "vina.log"
    # Write without comments - split() will process whitespace-separated values
    log.write_text('''-----+ header
1 -7.50 0 0 5 6
2 -6.50 0
3 -5.50 0 0
''')
    
    result = basevina.read_vina_log(str(log))
    # Only line with exactly 4 elements should be parsed
    # Line 1 has 6 elements, line 2 has 3 elements, line 3 has 4 elements
    assert len(result) == 1
    assert 3 in result
    assert result[3][vina_scoring] == "-5.50"


@pytest.mark.order(116)
def test_generate_digest_write_error(tmp_path, monkeypatch):
    '''Test generate_digest with write error.'''
    
    monkeypatch.setenv("OCDOCKER_NO_AUTO_BOOTSTRAP", "1")
    _setup_basevina_stubs(monkeypatch, tmp_path)
    import OCDocker.Docking.BaseVinaLike as basevina  # type: ignore
    import OCDocker.Error as ocerror
    
    log = tmp_path / "vina.log"
    log.write_text('''-----+ header
1 -7.50 0 0
''')
    
    digest = tmp_path / "digest.json"
    
    # Mock open to raise PermissionError on write
    original_open = open
    call_count = [0]
    
    def mock_open(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2 and 'w' in kwargs.get('mode', '') or (len(args) > 1 and 'w' in args[1]):
            raise PermissionError("Permission denied")
        return original_open(*args, **kwargs)
    
    monkeypatch.setattr('builtins.open', mock_open)
    
    rc = basevina.generate_vina_digest(str(digest), str(log), overwrite=True)
    assert rc == ocerror.Error.write_file()  # type: ignore
