import importlib
import types
import sys
from pathlib import Path
import pytest


@pytest.fixture
def docking_modules(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import OCDocker
    # stub packages requiring heavy deps
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.nan = float('nan')  # type: ignore
    numpy_stub.isnan = lambda x: x != x  # type: ignore
    monkeypatch.setitem(sys.modules, 'numpy', numpy_stub)
    monkeypatch.setitem(sys.modules, 'pandas', types.ModuleType('pandas'))

    ligand_mod = types.ModuleType('OCDocker.Ligand')
    class Ligand: pass


    ligand_mod.Ligand = Ligand # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Ligand', ligand_mod)

    receptor_mod = types.ModuleType('OCDocker.Receptor')
    class Receptor: pass


    receptor_mod.Receptor = Receptor # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Receptor', receptor_mod)

    toolbox_pkg = types.ModuleType('OCDocker.Toolbox')
    toolbox_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox', toolbox_pkg)

    init_mod = types.ModuleType('OCDocker.Initialise')
    init_mod.vina_scoring = 'vina' # type: ignore
    init_mod.smina_scoring = 'vinardo' # type: ignore
    init_mod.logdir = '/tmp' # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init_mod)

    for name in [
        'OCDocker.Toolbox.Conversion',
        'OCDocker.Toolbox.FilesFolders',
        'OCDocker.Toolbox.MoleculeProcessing',
        'OCDocker.Toolbox.Running',
        'OCDocker.Toolbox.Validation',
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    printing_mod = types.ModuleType('OCDocker.Toolbox.Printing')
    printing_mod.print_error = lambda *a, **k: None # type: ignore
    printing_mod.print_error_log = lambda *a, **k: None # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Printing', printing_mod)

    io_mod = types.ModuleType('OCDocker.Toolbox.IO')
    def lazyread_reverse_order_mmap(file_name: str, decode: str = 'utf-8'):
        with open(file_name, 'r', encoding=decode) as f:
            for line in reversed(f.read().splitlines()):
                yield line


    io_mod.lazyread_reverse_order_mmap = lazyread_reverse_order_mmap # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.IO', io_mod)

    vina_path = Path(__file__).resolve().parents[1] / 'OCDocker' / 'Docking' / 'Vina.py'
    smina_path = Path(__file__).resolve().parents[1] / 'OCDocker' / 'Docking' / 'Smina.py'
    spec_vina = importlib.util.spec_from_file_location('ocvina', vina_path) # type: ignore
    ocvina = importlib.util.module_from_spec(spec_vina) # type: ignore
    sys.modules['ocvina'] = ocvina
    spec_vina.loader.exec_module(ocvina)
    spec_smina = importlib.util.spec_from_file_location('ocsmina', smina_path) # type: ignore
    ocsmina = importlib.util.module_from_spec(spec_smina) # type: ignore
    sys.modules['ocsmina'] = ocsmina
    spec_smina.loader.exec_module(ocsmina)
    yield ocvina, ocsmina
    for mod in ['OCDocker.Docking.Vina', 'OCDocker.Docking.Smina']:
        sys.modules.pop(mod, None)


@pytest.mark.order(85)
def test_vina_log_parsing(docking_modules, tmp_path):
    ocvina, _ = docking_modules
    log_file = tmp_path / "vina.log"
    log_file.write_text(
        "header\n-----+------------+----------+----------+\n" \
        "    1 -8.0 0.0 0.0\n    2 -7.5 1.0 2.0\n"
    )
    from OCDocker.Config import get_config
    config = get_config()
    vina_scoring = config.vina.scoring
    
    result = ocvina.read_log(str(log_file))
    expected = {
        1: {vina_scoring: "-8.0"},
        2: {vina_scoring: "-7.5"},
    }
    assert result == expected

    rescoring = tmp_path / "vina_res.log"
    rescoring.write_text(
        "Some line\nEstimated Free Energy of Binding    -8.3 (kcal/mol)\n"
    )
    affinity = ocvina.read_rescoring_log(str(rescoring))
    assert affinity == -8.3


@pytest.mark.order(86)
def test_smina_log_parsing(docking_modules, tmp_path):
    _, ocsmina = docking_modules
    log_file = tmp_path / "smina.log"
    log_file.write_text(
        "header\n-----+------------+----------+----------+\n" \
        "    1 -6.0 0.0 0.0\n"
    )
    from OCDocker.Config import get_config
    config = get_config()
    smina_scoring = config.smina.scoring
    
    result = ocsmina.read_log(str(log_file))
    expected = {1: {smina_scoring: "-6.0"}}
    assert result == expected

    rescoring = tmp_path / "smina_res.log"
    rescoring.write_text("Header\nAffinity: -6.5 (kcal/mol)\n")
    affinity = ocsmina.read_rescoring_log(str(rescoring))
    assert affinity == -6.5
