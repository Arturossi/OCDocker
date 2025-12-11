import sys
import types
import importlib.util as util
from pathlib import Path
import OCDocker.Error as ocerror
import pytest

'''Load ``OCDocker.Docking.Vina`` with heavy dependencies stubbed.'''
@pytest.fixture
def vina_mod(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "OCDocker"
    # Fake Initialise module with basic config values
    fake_init = types.ModuleType("OCDocker.Initialise")
    fake_init.clrs = {k: "" for k in ["r", "g", "y", "b", "p", "c", "n"]} # type: ignore
    fake_init.ocerror = ocerror # type: ignore
    fake_init.logdir = str(tmp_path) # type: ignore
    fake_init.vina_scoring = "vina" # type: ignore
    fake_init.vina_scoring_functions = ["vina"] # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", fake_init)

    # Stub heavy modules required for import
    heavy_modules = [
        "OCDocker.Ligand",
        "OCDocker.Receptor",
        "OCDocker.Toolbox.Conversion",
        "OCDocker.Toolbox.MoleculeProcessing",
        "OCDocker.Toolbox.FilesFolders",
        "OCDocker.Toolbox.Running",
        "OCDocker.Toolbox.Validation",
    ]

    for name in heavy_modules:
        mod = types.ModuleType(name)
        mod.Ligand = type("Ligand", (), {}) # type: ignore
        mod.Receptor = type("Receptor", (), {}) # type: ignore
        monkeypatch.setitem(sys.modules, name, mod)

    # Minimal Printing implementation
    printing = types.ModuleType("OCDocker.Toolbox.Printing")
    printing.print_error = lambda *a, **k: None # type: ignore
    printing.print_error_log = lambda *a, **k: None # type: ignore
    printing.printv = lambda *a, **k: None # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing)

    # IO module
    spec_io = util.spec_from_file_location(
        "OCDocker.Toolbox.IO", root / "Toolbox" / "IO.py"
    )
    io_mod = util.module_from_spec(spec_io) # type: ignore
    assert spec_io.loader is not None # type: ignore
    spec_io.loader.exec_module(io_mod) # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.IO", io_mod)

    # Toolbox package to satisfy imports
    tb_pkg = types.ModuleType("OCDocker.Toolbox")
    tb_pkg.Printing = printing # type: ignore
    tb_pkg.IO = io_mod # type: ignore
    for sub in [
        "Conversion",
        "MoleculeProcessing",
        "FilesFolders",
        "Running",
        "Validation",
    ]:
        tb_pkg.__dict__[sub] = sys.modules[f"OCDocker.Toolbox.{sub}"]
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox", tb_pkg)

    # Provide a tiny numpy substitute
    fake_np = types.ModuleType("numpy")
    fake_np.nan = float("nan") # type: ignore
    fake_np.isnan = lambda x: x != x  # type: ignore
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    # Load Vina module from file
    spec = util.spec_from_file_location(
        "OCDocker.Docking.Vina", root / "Docking" / "Vina.py"
    )
    vina = util.module_from_spec(spec) # type: ignore
    assert spec.loader is not None # type: ignore
    spec.loader.exec_module(vina) # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Vina", vina)
    return vina


@pytest.mark.order(16)
def test_read_log(vina_mod, tmp_path):
    log = tmp_path / "vina.log"
    log.write_text(
        "header\n"
        "-----+------------+----------+----------\n"
        "    1          -7.0      0.0      0.0\n"
        "    2          -6.5      0.1      0.2\n"
    )
    from OCDocker.Config import get_config
    config = get_config()
    vina_scoring = config.vina.scoring
    
    data = vina_mod.read_log(str(log))
    assert data[1][vina_scoring] == "-7.0"
    assert data[2][vina_scoring] == "-6.5"
    best = vina_mod.read_log(str(log), onlyBest=True)
    assert best == {1: {vina_scoring: "-7.0"}}


@pytest.mark.order(17)
def test_rescore_logs(vina_mod, tmp_path):
    r1 = tmp_path / "lig_split_1_rescoring.log"
    r1.write_text("Estimated Free Energy of Binding    -7.1 (kcal/mol)\n")
    r2 = tmp_path / "lig_split_2_rescoring.log"
    r2.write_text("Estimated Free Energy of Binding    -6.0 (kcal/mol)\n")

    assert vina_mod.read_rescoring_log(str(r1)) == -7.1

    found = vina_mod.get_rescore_log_paths(str(tmp_path))
    assert set(found) == {str(r1), str(r2)}

    data = vina_mod.read_rescore_logs(found)
    assert data == {"rescoring_1": -7.1, "rescoring_2": -6.0}

    best = vina_mod.read_rescore_logs(found, onlyBest=True)
    assert best == {"rescoring_1": -7.1}


@pytest.mark.order(18)
def test_get_pose_index(vina_mod):
    '''Ensure pose index is parsed correctly from file name.'''
    assert vina_mod.get_pose_index_from_file_path("lig_split_3.pdbqt") == 3
