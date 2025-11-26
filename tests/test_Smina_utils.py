import json
import importlib
import sys
import types
import os
from pathlib import Path
import importlib.util as util
import pytest


def _setup_stubs(monkeypatch, tmp_path):
    """Set up minimal stubs for truly heavy external dependencies.
    
    Uses real implementations for:
    - Error: lightweight module
    - IO: only standard library deps
    - Printing: only standard library deps (with minimal Initialise stub)
    - Config: lightweight module
    
    Keeps stubs only for:
    - External heavy deps: Bio, rdkit, openbabel, h5py, tqdm, spyrmsd, pandas
    - Modules with heavy deps: FilesFolders, Validation, Conversion, MoleculeProcessing
    - numpy: minimal stub (only nan/isnan needed)
    """
    root = Path(__file__).resolve().parents[1] / "OCDocker"
    
    # Minimal stubs for heavy external dependencies
    bio = types.ModuleType('Bio'); bio.__path__ = []
    pdb = types.ModuleType('Bio.PDB'); pdb.MMCIFParser = lambda *a, **k: None # type: ignore
    pdb.PDBParser = lambda *a, **k: None; pdb.PDBIO = lambda *a, **k: None # type: ignore
    # Create SASA stub that adds sasa attribute to structure
    class ShrakeRupley:
        def __init__(self, n_points=1000):
            self.n_points = n_points
        def compute(self, structure, level="S"):
            # Add sasa attribute to structure for tests
            structure.sasa = 1000.0  # Dummy value for testing
    pdb.SASA = types.SimpleNamespace(ShrakeRupley=ShrakeRupley) # type: ignore
    pdb.DSSP = lambda *a, **k: None # type: ignore
    seq = types.ModuleType('Bio.SeqUtils'); seq.seq1 = lambda x: x # type: ignore
    prot = types.ModuleType('Bio.SeqUtils.ProtParam'); prot.ProteinAnalysis = lambda x: None # type: ignore
    seq.ProtParam = prot # type: ignore
    monkeypatch.setitem(sys.modules, 'Bio', bio)
    monkeypatch.setitem(sys.modules, 'Bio.PDB', pdb)
    monkeypatch.setitem(sys.modules, 'Bio.SeqUtils', seq)
    monkeypatch.setitem(sys.modules, 'Bio.SeqUtils.ProtParam', prot)

    # Set up numpy mock - must be ModuleType (not SimpleNamespace) with nan/isnan
    numpy_stub = types.ModuleType('numpy')
    numpy_stub.nan = float('nan')  # type: ignore
    numpy_stub.isnan = lambda x: x != x  # type: ignore
    monkeypatch.setitem(sys.modules, 'numpy', numpy_stub)
    monkeypatch.setitem(sys.modules, 'pandas', types.ModuleType('pandas'))
    # Create rdkit stub - make Chem a proper package to support submodule imports
    rdkit_mod = types.ModuleType('rdkit')
    rdkit_mod.__path__ = []  # Make rdkit a package
    rdkit_chem_mod = types.ModuleType('rdkit.Chem')
    rdkit_chem_mod.__path__ = []  # Make rdkit.Chem a package (required for submodule imports)
    # Create rdchem module with Mol class
    rdkit_chem_mod.rdchem = types.ModuleType('rdkit.Chem.rdchem')  # type: ignore
    # Add Mol class to rdchem stub (used in type hints)
    class Mol:
        pass
    rdkit_chem_mod.rdchem.Mol = Mol  # type: ignore
    rdkit_chem_mod.rdmolfiles = types.ModuleType('rdkit.Chem.rdmolfiles')  # type: ignore
    # Add MolToMolFile function to rdmolfiles stub
    rdkit_chem_mod.rdmolfiles.MolToMolFile = lambda *a, **k: None  # type: ignore
    # Add AllChem stub (used by Conversion and Ligand modules)
    allchem_mod = types.ModuleType('rdkit.Chem.AllChem')
    allchem_mod.EmbedMolecule = lambda *a, **k: 0  # type: ignore
    allchem_mod.ETKDG = lambda *a, **k: types.SimpleNamespace()  # type: ignore
    allchem_mod.UFFOptimizeMolecule = lambda *a, **k: 0  # type: ignore
    rdkit_chem_mod.AllChem = allchem_mod  # type: ignore
    # Add SaltRemover stub (used by Conversion and Ligand modules)
    saltremover_mod = types.ModuleType('rdkit.Chem.SaltRemover')
    class SaltRemover:
        def __init__(self, *a, **k):
            pass
    saltremover_mod.SaltRemover = SaltRemover  # type: ignore
    rdkit_chem_mod.SaltRemover = saltremover_mod  # type: ignore
    # Add rdMolTransforms stub (used by Ligand module)
    rdmoltransforms_mod = types.ModuleType('rdkit.Chem.rdMolTransforms')
    rdmoltransforms_mod.ComputeCentroid = lambda *a, **k: (0.0, 0.0, 0.0)  # type: ignore
    rdkit_chem_mod.rdMolTransforms = rdmoltransforms_mod  # type: ignore
    # Add other rdkit.Chem stubs (used by Ligand module)
    rdkit_chem_mod.Descriptors = types.ModuleType('rdkit.Chem.Descriptors')  # type: ignore
    rdkit_chem_mod.Descriptors3D = types.ModuleType('rdkit.Chem.Descriptors3D')  # type: ignore
    rdkit_chem_mod.MACCSkeys = types.ModuleType('rdkit.Chem.MACCSkeys')  # type: ignore
    # Add top-level rdkit imports
    rdkit_mod.Chem = rdkit_chem_mod  # type: ignore
    rdkit_mod.DataStructs = types.ModuleType('rdkit.DataStructs')  # type: ignore
    rdkit_mod.RDLogger = types.ModuleType('rdkit.RDLogger')  # type: ignore
    rdkit_mod.RDLogger.DisableLog = lambda *a, **k: None  # type: ignore
    monkeypatch.setitem(sys.modules, 'rdkit', rdkit_mod)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem', rdkit_chem_mod)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.rdchem', rdkit_chem_mod.rdchem)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.rdmolfiles', rdkit_chem_mod.rdmolfiles)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.AllChem', allchem_mod)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.SaltRemover', saltremover_mod)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.rdMolTransforms', rdmoltransforms_mod)
    monkeypatch.setitem(sys.modules, 'rdkit.DataStructs', rdkit_mod.DataStructs)
    monkeypatch.setitem(sys.modules, 'rdkit.RDLogger', rdkit_mod.RDLogger)
    monkeypatch.setitem(sys.modules, 'rdkit.Chem.AllChem', allchem_mod)
    ob = types.ModuleType('openbabel')
    # Create OBMessageHandler stub with SetOutputLevel method
    class OBMessageHandler:
        def SetOutputLevel(self, level):
            pass
    ob.openbabel = types.SimpleNamespace(OBMessageHandler=OBMessageHandler) # type: ignore
    ob.pybel = types.SimpleNamespace(ob=types.SimpleNamespace(OBMessageHandler=OBMessageHandler)) # type: ignore
    monkeypatch.setitem(sys.modules, 'openbabel', ob)
    monkeypatch.setitem(sys.modules, 'h5py', types.ModuleType('h5py'))
    tq = types.ModuleType('tqdm'); tq.tqdm = lambda x, **kw: x # type: ignore
    monkeypatch.setitem(sys.modules, 'tqdm', tq)
    monkeypatch.setitem(sys.modules, 'tqdm.auto', tq)
    # Stub spyrmsd (used by MoleculeProcessing module)
    spyrmsd_mod = types.ModuleType('spyrmsd')
    spyrmsd_mod.io = types.ModuleType('spyrmsd.io')  # type: ignore
    spyrmsd_mod.rmsd = types.ModuleType('spyrmsd.rmsd')  # type: ignore
    monkeypatch.setitem(sys.modules, 'spyrmsd', spyrmsd_mod)
    monkeypatch.setitem(sys.modules, 'spyrmsd.io', spyrmsd_mod.io)
    monkeypatch.setitem(sys.modules, 'spyrmsd.rmsd', spyrmsd_mod.rmsd)

    # Stub Toolbox modules with heavy dependencies
    conv_mod = types.ModuleType('OCDocker.Toolbox.Conversion')
    conv_mod.convert_mols = lambda *a, **k: 0 # type: ignore
    conv_mod.convert_mols_from_string = lambda *a, **k: 0 # type: ignore
    molproc_mod = types.ModuleType('OCDocker.Toolbox.MoleculeProcessing')
    molproc_mod.split_poses = lambda *a, **k: 0 # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Conversion', conv_mod)
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.MoleculeProcessing', molproc_mod)
    
    # Stub FilesFolders (has heavy deps: h5py, numpy, tqdm)
    filesfolders_mod = types.ModuleType('OCDocker.Toolbox.FilesFolders')
    filesfolders_mod.empty_docking_digest = lambda *a, **k: {}  # type: ignore
    filesfolders_mod.safe_create_dir = lambda *a, **k: None  # type: ignore
    filesfolders_mod.ensure_parent_dir = lambda *a, **k: None  # type: ignore - used by Preparation
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.FilesFolders', filesfolders_mod)
    
    # Stub Validation (has heavy deps: Bio.PDB)
    validation_mod = types.ModuleType('OCDocker.Toolbox.Validation')
    validation_mod.validate_digest_extension = lambda path, format_str: format_str.lower() == 'json'  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Validation', validation_mod)
    
    # Stub Running (used by Preparation module)
    running_mod = types.ModuleType('OCDocker.Toolbox.Running')
    running_mod.is_tool_available = lambda exe: True  # type: ignore - stub always returns True
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Running', running_mod)

    # Ensure base package is loaded so submodules resolve correctly
    import OCDocker as base_pkg
    monkeypatch.setitem(sys.modules, 'OCDocker', base_pkg)

    # Use REAL Error module (lightweight)
    import OCDocker.Error as error_mod

        # Stub Initialise for Printing module (minimal stub)
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
    init_mod.clrs = {k: "" for k in ["r", "g", "y", "b", "p", "c", "n"]}  # type: ignore
    init_mod.logdir = str(tmp_path)  # type: ignore
    init_mod.vina_scoring = 'vina'  # type: ignore
    init_mod.vina_scoring_functions = ['vina']  # type: ignore
    init_mod.smina_scoring = 'vinardo'  # type: ignore
    init_mod.smina_scoring_functions = ['vinardo']  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init_mod)
    
    # Use REAL IO module (only standard library deps: os, mmap)
    spec_io = util.spec_from_file_location(
        "OCDocker.Toolbox.IO", root / "Toolbox" / "IO.py"
    )
    io_mod = util.module_from_spec(spec_io)  # type: ignore
    assert spec_io.loader is not None  # type: ignore
    spec_io.loader.exec_module(io_mod)  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.IO', io_mod)
    
    # Use REAL Printing module (only standard library deps)
    spec_printing = util.spec_from_file_location(
        "OCDocker.Toolbox.Printing", root / "Toolbox" / "Printing.py"
    )
    printing_mod = util.module_from_spec(spec_printing)  # type: ignore
    assert spec_printing.loader is not None  # type: ignore
    spec_printing.loader.exec_module(printing_mod)  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Printing', printing_mod)
    
    # Stub Logging (used by Printing)
    logging_mod = types.ModuleType('OCDocker.Toolbox.Logging')
    monkeypatch.setitem(sys.modules, 'OCDocker.Toolbox.Logging', logging_mod)
    
    # Use REAL Config module (lightweight - only stdlib + Error)
    spec_config = util.spec_from_file_location(
        "OCDocker.Config", root / "Config.py"
    )
    config_mod = util.module_from_spec(spec_config)  # type: ignore
    assert spec_config.loader is not None  # type: ignore
    spec_config.loader.exec_module(config_mod)  # type: ignore
    # Override get_config to set logdir for tests
    original_get_config = config_mod.get_config
    def get_config():
        cfg = original_get_config()
        cfg.logdir = str(tmp_path)  # type: ignore
        return cfg
    config_mod.get_config = get_config  # type: ignore
    monkeypatch.setitem(sys.modules, 'OCDocker.Config', config_mod)

    # Stub Ligand and Receptor (used by Smina class but not by utility functions)
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
    # Make sure the stub is in place before importing
    # Delete dependent modules to force re-import with stubs
    modules_to_reload = [
        'OCDocker.Docking.Smina',
        'OCDocker.Toolbox.MoleculeProcessing',
        'OCDocker.Toolbox.Conversion',
        'OCDocker.Docking.BaseVinaLike',
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    # Now import the module (it will pick up the stubs)
    mod = importlib.import_module('OCDocker.Docking.Smina')
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


@pytest.mark.order(11)
def test_gen_smina_conf(smina, sample_paths):
    conf = sample_paths['tmp'] / 'conf.txt'
    smina.gen_smina_conf(str(sample_paths['box']), str(conf), str(sample_paths['receptor']))
    text = conf.read_text()
    assert f'receptor = {sample_paths["receptor"]}' in text
    assert 'center_x = 36.552' in text
    assert 'size_z = 102.582' in text


@pytest.mark.order(12)
def test_read_rescoring_log(smina, sample_paths):
    path = sample_paths['tmp'] / 'rescore.log'
    path.write_text('Affinity: -8.1 (kcal/mol)\n')
    assert smina.read_rescoring_log(str(path)) == -8.1


@pytest.mark.order(13)
def test_generate_digest(smina, sample_paths):
    log = _create_log(sample_paths['tmp'] / 'smina.log')
    digest = sample_paths['tmp'] / 'digest.json'
    smina.generate_digest(str(digest), str(log), overwrite=True)
    data = json.loads(digest.read_text())
    from OCDocker.Config import get_config
    config = get_config()
    smina_scoring = config.smina.scoring
    assert data['2'][smina_scoring] == '-7.0'


@pytest.mark.order(14)
def test_get_docked_poses_and_index(smina, sample_paths):
    poses_dir = sample_paths['tmp'] / 'poses'
    poses_dir.mkdir()
    p1 = poses_dir / 'lig_split_1.pdbqt'
    p2 = poses_dir / 'lig_split_2.pdbqt'
    p1.write_text('x'); p2.write_text('x')
    result = smina.get_docked_poses(str(poses_dir))
    assert set(map(Path, result)) == {p1, p2}
    assert smina.get_pose_index_from_file_path(str(p1)) == 1


@pytest.mark.order(15)
def test_split_poses_method(smina, sample_paths, monkeypatch):
    # Ensure the ligand file exists and is a valid format
    assert sample_paths['ligand'].exists(), f"Ligand file not found: {sample_paths['ligand']}"
    # Use absolute path to ensure ligand.path is absolute (not empty or relative)
    ligand_file_abs = str(Path(sample_paths['ligand']).resolve())
    ligand = smina.ocl.Ligand(ligand_file_abs, name="ligand")
    # Verify ligand.path is absolute and not empty
    assert ligand.path, "Ligand path should not be empty"
    assert os.path.isabs(ligand.path), f"Ligand path should be absolute, got: {ligand.path}"

    # Ensure the receptor file exists and is a valid format
    assert sample_paths['receptor'].exists(), f"Receptor file not found: {sample_paths['receptor']}"
    receptor = smina.ocr.Receptor(str(sample_paths['receptor']), name="receptor")

    conf = sample_paths['tmp'] / 'conf.txt'
    smina.gen_smina_conf(str(sample_paths['box']), str(conf), str(sample_paths['receptor']))
    out = sample_paths['tmp'] / 'dock.pdbqt'
    log = sample_paths['tmp'] / 'dock.log'
    called = {}
    def fake_split(lig, name, outPath, logFile='', suffix=''):
        called['args'] = (lig, name, outPath, logFile, suffix)
        return 0
    # Ensure the stub module is properly set up
    # Get the module reference directly from sys.modules to avoid stale references
    molproc_module = sys.modules.get('OCDocker.Toolbox.MoleculeProcessing')
    if molproc_module:
        monkeypatch.setattr(molproc_module, 'split_poses', fake_split)
        # Also patch the reference in the smina module
        monkeypatch.setattr(smina, 'ocmolproc', molproc_module)
    else:
        # Fallback: patch directly
        monkeypatch.setattr(smina.ocmolproc, 'split_poses', fake_split)
    inst = smina.Smina(str(conf), str(sample_paths['box']), receptor, str(sample_paths['receptor']), ligand, ligand_file_abs, str(log), str(out))
    inst.split_poses(outPath=str(sample_paths['tmp'] / 'poses'))
    assert called['args'][0] == str(out)
    assert called['args'][2] == str(sample_paths['tmp'] / 'poses')
