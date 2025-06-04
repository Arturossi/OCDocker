import json
import importlib
import sys
import types
from pathlib import Path
import pytest

def _setup_stubs(monkeypatch, tmp_path):
    # Minimal stubs for heavy dependencies
    bio = types.ModuleType('Bio'); bio.__path__ = []
    pdb = types.ModuleType('Bio.PDB'); pdb.MMCIFParser = lambda *a, **k: None # type: ignore
    pdb.PDBParser = lambda *a, **k: None; pdb.PDBIO = lambda *a, **k: None # type: ignore
    pdb.SASA = lambda *a, **k: None; pdb.DSSP = lambda *a, **k: None # type: ignore
    seq = types.ModuleType('Bio.SeqUtils'); seq.seq1 = lambda x: x # type: ignore
    prot = types.ModuleType('Bio.SeqUtils.ProtParam'); prot.ProteinAnalysis = lambda x: None # type: ignore
    seq.ProtParam = prot # type: ignore
    monkeypatch.setitem(sys.modules, 'Bio', bio)
    monkeypatch.setitem(sys.modules, 'Bio.PDB', pdb)
    monkeypatch.setitem(sys.modules, 'Bio.SeqUtils', seq)
    monkeypatch.setitem(sys.modules, 'Bio.SeqUtils.ProtParam', prot)

    monkeypatch.setitem(sys.modules, 'numpy', types.SimpleNamespace(NaN=float('nan')))
    monkeypatch.setitem(sys.modules, 'pandas', types.ModuleType('pandas'))
    monkeypatch.setitem(sys.modules, 'rdkit', types.ModuleType('rdkit'))
    ob = types.ModuleType('openbabel')
    ob.openbabel = types.SimpleNamespace() # type: ignore
    ob.pybel = types.SimpleNamespace(ob=types.SimpleNamespace(OBMessageHandler=lambda: None)) # type: ignore
    monkeypatch.setitem(sys.modules, 'openbabel', ob)
    monkeypatch.setitem(sys.modules, 'h5py', types.ModuleType('h5py'))
    tq = types.ModuleType('tqdm'); tq.tqdm = lambda x, **kw: x # type: ignore
    monkeypatch.setitem(sys.modules, 'tqdm', tq)
    monkeypatch.setitem(sys.modules, 'tqdm.auto', tq)
    monkeypatch.setitem(sys.modules, 'spyrmsd', types.ModuleType('spyrmsd'))

    conv_mod = types.ModuleType('OCDocker.Toolbox.Conversion')
    molproc_mod = types.ModuleType('OCDocker.Toolbox.MoleculeProcessing')
    molproc_mod.split_poses = lambda *a, **k: 0 # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Conversion', conv_mod)
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.MoleculeProcessing', molproc_mod)

    # Ensure base package is loaded so submodules resolve correctly
    import OCDocker as base_pkg
    monkeypatch.setitem(sys.modules, 'OCDocker', base_pkg)

    error_mod = types.ModuleType('OCDocker.Error')
    class ReportLevel(int):
        DEBUG = 5
        SUCCESS = 4
        INFO = 3
        WARNING = 2
        ERROR = 1
        NONE = 0
    class Error:
        output_level = ReportLevel.INFO
        @staticmethod
        def ok(*a, **k):
            return 0
        @staticmethod
        def file_not_exist(*a, **k):
            return 0
        @staticmethod
        def read_file(*a, **k):
            return 0
        @staticmethod
        def write_file(*a, **k):
            return 0
        @staticmethod
        def dir_does_not_exist(*a, **k):
            return 0
        @staticmethod
        def empty_file(*a, **k):
            return 0
        @staticmethod
        def wrong_type(*a, **k):
            return 0
        @staticmethod
        def unsupported_extension(*a, **k):
            return 0
        @staticmethod
        def file_exists(*a, **k):
            return 0
        @staticmethod
        def read_docking_log_error(*a, **k):
            return 0
        @staticmethod
        def subprocess(*a, **k):
            return 0
    error_mod.Error = Error # type: ignore
    error_mod.ReportLevel = ReportLevel # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Error', error_mod)
    init_mod = types.ModuleType('OCDocker.Initialise')
    defaults = {
        'smina_custom_scoring': 'no',
        'smina_custom_atoms': 'no',
        'smina_minimize_iters': '0',
        'smina_approximation': 'spline',
        'smina_factor': '32',
        'smina_force_cap': '10',
        'smina_user_grid': 'no',
        'smina_user_grid_lambda': 'no',
        'smina_energy_range': '10',
        'smina_exhaustiveness': '5',
        'smina_num_modes': '3',
        'smina_scoring': 'vinardo',
        'smina_scoring_functions': ['vinardo'],
        'pythonsh': 'pythonsh',
        'prepare_receptor': 'prep_rec.py',
        'prepare_ligand': 'prep_lig.py',
        'smina': 'smina',
        'smina_local_only': 'no',
        'smina_minimize': 'no',
        'smina_randomize_only': 'no',
        'smina_accurate_line': 'no',
        'smina_minimize_early_term': 'no',
        'seed': 0,
        'logdir': str(tmp_path),
        'vina_split': 'vina_split',
    }
    for k, v in defaults.items():
        setattr(init_mod, k, v)
    init_mod.ocerror = error_mod # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init_mod)

    lig_mod = types.ModuleType('OCDocker.Ligand')
    class Ligand:
        def __init__(self, molecule, name='lig'):
            self.path = str(molecule)
            self.name = name
    lig_mod.Ligand = Ligand # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Ligand', lig_mod)

    rec_mod = types.ModuleType('OCDocker.Receptor')
    class Receptor:
        def __init__(self, structure, name='rec'):
            self.path = str(structure)
            self.name = name
    rec_mod.Receptor = Receptor # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Receptor', rec_mod)

@pytest.fixture
def smina(monkeypatch, tmp_path):
    _setup_stubs(monkeypatch, tmp_path)
    import importlib
    mod_name = 'OCDocker.Docking.Smina'
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)
    monkeypatch.setattr(mod.Smina, '_Smina__smina_cmd', lambda self: ['smina'])
    return mod

@pytest.fixture
def sample_paths(tmp_path):
    root = Path(__file__).resolve()
    while root.name != 'OCDocker':
        root = root.parent
    base = root / 'test_files/test_ptn1'
    return {
        'box': base / 'compounds/ligands/ligand/boxes/box0.pdb',
        'ligand': base / 'compounds/ligands/ligand/ligand.smi',
        'receptor': base / 'receptor.pdb',
        'tmp': tmp_path
    }

def _create_log(path: Path):
    path.write_text('header\n-----+------------+----------+----------+\n 2 -7.0 0 0\n 1 -7.5 0 0\n')
    return path

def test_gen_smina_conf(smina, sample_paths):
    conf = sample_paths['tmp'] / 'conf.txt'
    smina.gen_smina_conf(str(sample_paths['box']), str(conf), str(sample_paths['receptor']))
    text = conf.read_text()
    assert f'receptor = {sample_paths["receptor"]}' in text
    assert 'center_x = 36.552' in text
    assert 'size_z = 102.582' in text

def test_read_rescoring_log(smina, sample_paths):
    path = sample_paths['tmp'] / 'rescore.log'
    path.write_text('Affinity: -8.1 (kcal/mol)\n')
    assert smina.read_rescoring_log(str(path)) == -8.1

def test_generate_digest(smina, sample_paths):
    log = _create_log(sample_paths['tmp'] / 'smina.log')
    digest = sample_paths['tmp'] / 'digest.json'
    smina.generate_digest(str(digest), str(log), overwrite=True)
    data = json.loads(digest.read_text())
    assert data['2'][smina.smina_scoring] == '-7.0'

def test_get_docked_poses_and_index(smina, sample_paths):
    poses_dir = sample_paths['tmp'] / 'poses'
    poses_dir.mkdir()
    p1 = poses_dir / 'lig_split_1.pdbqt'
    p2 = poses_dir / 'lig_split_2.pdbqt'
    p1.write_text('x'); p2.write_text('x')
    result = smina.get_docked_poses(str(poses_dir))
    assert set(map(Path, result)) == {p1, p2}
    assert smina.get_pose_index_from_file_path(str(p1)) == 1

def test_split_poses_method(smina, sample_paths, monkeypatch):
    # Ensure the ligand file exists and is a valid format
    assert sample_paths['ligand'].exists(), f"Ligand file not found: {sample_paths['ligand']}"
    ligand = smina.ocl.Ligand(sample_paths['ligand'], name="ligand")

    # Ensure the receptor file exists and is a valid format
    assert sample_paths['receptor'].exists(), f"Receptor file not found: {sample_paths['receptor']}"
    receptor = smina.ocr.Receptor(sample_paths['receptor'], name="receptor")

    conf = sample_paths['tmp'] / 'conf.txt'
    smina.gen_smina_conf(str(sample_paths['box']), str(conf), str(sample_paths['receptor']))
    out = sample_paths['tmp'] / 'dock.pdbqt'
    log = sample_paths['tmp'] / 'dock.log'
    called = {}
    def fake_split(lig, name, outPath, logFile='', suffix=''):
        called['args'] = (lig, name, outPath, logFile, suffix)
        return 0
    monkeypatch.setattr(smina.ocmolproc, 'split_poses', fake_split)
    inst = smina.Smina(str(conf), str(sample_paths['box']), receptor, str(sample_paths['receptor']), ligand, str(sample_paths['ligand']), str(log), str(out))
    inst.split_poses(outPath=str(sample_paths['tmp'] / 'poses'))
    assert called['args'][0] == str(out)
    assert called['args'][2] == str(sample_paths['tmp'] / 'poses')
