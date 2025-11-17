'''Tests for PLANTS utility helpers.'''

import importlib
import sys
import types
import os
import json
import csv
from pathlib import Path
import pytest

import OCDocker.Error as ocerror

# Helpers ---------------------------------------------------------------------


def _load_plants(monkeypatch, tmp_path):
    '''Import ``OCDocker.Docking.PLANTS`` with heavy dependencies stubbed.'''

    # Stub pandas
    pandas_mod = types.ModuleType("pandas")
    pandas_mod.read_csv = lambda *a, **k: None # type: ignore
    monkeypatch.setitem(sys.modules, "pandas", pandas_mod)

    # Stub Initialise with required attributes
    init_mod = types.ModuleType('OCDocker.Initialise')
    init_mod.ocerror = ocerror # type: ignore
    init_mod.clrs = {'c':'','n':'','g':'','y':'','r':'','b':'','p':''} # type: ignore
    init_mod.plants_search_speed = 'speed1' # type: ignore
    init_mod.plants_cluster_structures = 1 # type: ignore
    init_mod.plants_cluster_rmsd = 2.0 # type: ignore
    init_mod.plants_scoring = 'chemplp' # type: ignore
    init_mod.plants_scoring_functions = ['chemplp','plp'] # type: ignore
    init_mod.plants_rescoring_mode = 'simplex' # type: ignore
    init_mod.plants = 'plants' # type: ignore
    init_mod.spores = 'spores' # type: ignore
    init_mod.logdir = str(tmp_path/'logs') # type: ignore
    init_mod.tmpDir = str(tmp_path/'tmp') # type: ignore
    os.makedirs(init_mod.logdir, exist_ok=True)
    os.makedirs(init_mod.tmpDir, exist_ok=True)
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)

    # Minimal Ligand and Receptor classes
    lig_mod = types.ModuleType('OCDocker.Ligand')
    class Ligand:
        def __init__(self, path='', name=''):
            self.path = path
            self.name = name


    lig_mod.Ligand = Ligand # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Ligand", lig_mod)

    rec_mod = types.ModuleType('OCDocker.Receptor')
    
    class Receptor:
        def __init__(self, path='', name=''):
            self.path = path
            self.name = name
            self.mol2Path = ''


    rec_mod.Receptor = Receptor # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Receptor", rec_mod)

    # Toolbox package and required submodules
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox", types.ModuleType("OCDocker.Toolbox"))

    ff_mod = types.ModuleType('OCDocker.Toolbox.FilesFolders')
    ff_mod.safe_create_dir = lambda path: (os.makedirs(path, exist_ok=True) or 0) # type: ignore
    ff_mod.empty_docking_digest = lambda path, overwrite, digestFormat='': {'existing':[0]} # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.FilesFolders", ff_mod)

    val_mod = types.ModuleType('OCDocker.Toolbox.Validation')
    val_mod.validate_digest_extension = lambda path, fmt: True # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Validation", val_mod)

    run_mod = types.ModuleType('OCDocker.Toolbox.Running')
    run_mod.run = lambda *a, **k: 0 # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Running", run_mod)

    print_mod = types.ModuleType('OCDocker.Toolbox.Printing')
    print_mod.printv = lambda *a, **k: None # type: ignore
    print_mod.print_error = lambda *a, **k: None # type: ignore
    print_mod.print_error_log = lambda *a, **k: None # type: ignore
    print_mod.print_warning = lambda *a, **k: None # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", print_mod)

    conv_mod = types.ModuleType("OCDocker.Toolbox.Conversion")
    conv_mod.convert_mols = lambda *a, **k: 0 # type: ignore
    conv_mod.convert_mols_from_string = lambda *a, **k: 0 # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Conversion", conv_mod)

    ocplants = importlib.import_module("OCDocker.Docking.PLANTS")
    return ocplants


@pytest.fixture()
def ocplants(monkeypatch, tmp_path):
    return _load_plants(monkeypatch, tmp_path)


class FakeSeries(list):
    @property
    def values(self):
        return list(self)


    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeSeries(super().__getitem__(item))
        return list.__getitem__(self, item)


class FakeDF:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)


    def iterrows(self):
        for i, r in enumerate(self.rows):
            yield i, r


    def __getattr__(self, name):
        return FakeSeries([r[name] for r in self.rows])


def fake_read_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        for r in rows:
            for k,v in r.items():
                if k != 'LIGAND_ENTRY':
                    r[k] = float(v)
    return FakeDF(rows)


@pytest.mark.order(2)
def test_write_config_files(ocplants, tmp_path):
    conf = tmp_path/'conf.txt'
    outdir = str(tmp_path / 'outdir')  # Use tmp_path, not relative path
    res = ocplants.write_config_file(str(conf), 'rec.mol2', 'lig.mol2', outdir, 1.0,2.0,3.0,4.0)
    assert res == 0
    txt = conf.read_text()
    assert 'protein_file rec.mol2' in txt
    assert 'ligand_file lig.mol2' in txt
    resc = tmp_path/'resc.txt'
    res2 = ocplants.write_rescoring_config_file(str(resc), 'rec.mol2', 'poses.lst', outdir, 1,2,3,4)
    assert res2 == 0
    lines = resc.read_text()
    assert 'ligand_list poses.lst' in lines
    assert 'rescore_mode simplex' in lines


@pytest.mark.order(3)
def test_get_binding_site(ocplants):
    from pathlib import Path
    # Get absolute path to box file
    test_dir = Path(__file__).resolve().parent.parent
    box = test_dir / 'test_files/test_ptn1/compounds/ligands/ligand/boxes/box0.pdb'
    result = ocplants.get_binding_site(str(box), spacing=2.9)
    # Check if result is an error code (int)
    assert not isinstance(result, int), f"get_binding_site returned error code: {result}"
    # Unpack the tuple
    center, radius = result
    assert center == (36.552, 39.252, 51.291)
    assert round(radius,3) == 70.274


@pytest.mark.order(4)
def test_generate_plants_files_database(monkeypatch, ocplants, tmp_path):
    called = {}
    def mock_box(box, conf, prot, lig, out, spacing=None, center=None, bindingSiteRadius=None):
        called['args'] = (box, conf, prot, lig, out, spacing)


    monkeypatch.setattr(ocplants, 'box_to_plants', mock_box)
    ocplants.generate_plants_files_database(str(tmp_path), 'prot.pdb', 'lig.mol2', spacing=1.1)
    assert called['args'][0] == f"{tmp_path}/boxes/box0.pdb"
    assert called['args'][1] == f"{tmp_path}/plantsFiles/conf_plants.conf"
    assert called['args'][2] == 'prot.pdb'
    assert called['args'][3] == 'lig.mol2'
    assert called['args'][4] == f"{tmp_path}/plantsFiles/run"
    assert called['args'][5] == 1.1


@pytest.mark.order(5)
def test_read_log_and_generate_digest(monkeypatch, ocplants, tmp_path):
    csv_file = tmp_path/'ranking.csv'
    fieldnames = ['LIGAND_ENTRY','TOTAL_SCORE','SCORE_RB_PEN','SCORE_NORM_HEVATOMS','SCORE_NORM_CRT_HEVATOMS','SCORE_NORM_WEIGHT','SCORE_NORM_CRT_WEIGHT','SCORE_RB_PEN_NORM_CRT_HEVATOMS']
    with open(csv_file,'w',newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'LIGAND_ENTRY':'pose_1.mol2','TOTAL_SCORE':1,'SCORE_RB_PEN':2,'SCORE_NORM_HEVATOMS':3,'SCORE_NORM_CRT_HEVATOMS':4,'SCORE_NORM_WEIGHT':5,'SCORE_NORM_CRT_WEIGHT':6,'SCORE_RB_PEN_NORM_CRT_HEVATOMS':7})
        writer.writerow({'LIGAND_ENTRY':'pose_2.mol2','TOTAL_SCORE':8,'SCORE_RB_PEN':9,'SCORE_NORM_HEVATOMS':10,'SCORE_NORM_CRT_HEVATOMS':11,'SCORE_NORM_WEIGHT':12,'SCORE_NORM_CRT_WEIGHT':13,'SCORE_RB_PEN_NORM_CRT_HEVATOMS':14})
    monkeypatch.setattr(ocplants.pd, 'read_csv', fake_read_csv)
    data = ocplants.read_log(str(csv_file), onlyBest=False)
    assert data[1]['PLANTS_TOTAL_SCORE'] == 1
    assert data[2]['PLANTS_TOTAL_SCORE'] == 8
    data_best = ocplants.read_log(str(csv_file), onlyBest=True)
    assert data_best[1]['PLANTS_TOTAL_SCORE'][0] == 1

    digest = tmp_path/'digest.json'
    ocplants.generate_digest(str(digest), str(csv_file))
    with open(digest) as f:
        j = json.load(f)
    assert 'PLANTS_TOTAL_SCORE' in j.get('1', {})


@pytest.mark.order(6)
def test_pose_handling(ocplants, tmp_path):
    poses_dir = tmp_path/'run'
    poses_dir.mkdir()
    good1 = poses_dir/'ligand_1.mol2'
    good1.write_text('a')
    (poses_dir/'ligand_1_fixed.mol2').write_text('a')
    good2 = poses_dir/'ligand_2.mol2'
    good2.write_text('a')
    (poses_dir/'ligand_2_protein.mol2').write_text('a')

    poses = ocplants.get_docked_poses(str(poses_dir))
    assert set(map(Path, poses)) == {good1, good2}

    pose_list = tmp_path/'pose_list.txt'
    res = ocplants.write_pose_list(poses, str(pose_list))
    assert res == str(pose_list)
    lines = pose_list.read_text().splitlines()
    assert set(lines) == {str(good1), str(good2)}
    assert len(lines) == 2
    assert ocplants.write_pose_list(poses, str(pose_list)) is None
    assert ocplants.get_pose_index_from_file_path('foo_10.mol2') == 10


@pytest.mark.order(7)
def test_read_log(monkeypatch, ocplants, tmp_path):
    monkeypatch.setattr(ocplants.pd, 'read_csv', fake_read_csv)
    csv_file = tmp_path / "ranking.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "LIGAND_ENTRY",
            "TOTAL_SCORE",
            "SCORE_RB_PEN",
            "SCORE_NORM_HEVATOMS",
            "SCORE_NORM_CRT_HEVATOMS",
            "SCORE_NORM_WEIGHT",
            "SCORE_NORM_CRT_WEIGHT",
            "SCORE_RB_PEN_NORM_CRT_HEVATOMS",
        ])
        writer.writerow(["lig_split_1.mol2", -10.0, 1, 2, 3, 4, 5, 6])
        writer.writerow(["lig_split_2.mol2", -20.0, 2, 3, 4, 5, 6, 7])

    all_data = ocplants.read_log(str(csv_file))
    assert set(all_data.keys()) == {1, 2}
    assert all_data[1]["PLANTS_TOTAL_SCORE"] == -10.0 # type: ignore

    best = ocplants.read_log(str(csv_file), onlyBest=True)
    assert set(best.keys()) == {1}
    assert best[1]["PLANTS_TOTAL_SCORE"] == [-10.0] # type: ignore


@pytest.mark.order(8)
def test_get_docked_poses_and_write_list(ocplants, tmp_path):
    poses_dir = tmp_path / "run"
    poses_dir.mkdir()
    valid1 = poses_dir / "pose1.mol2"
    valid1.write_text("p1")
    (poses_dir / "pose2_protein.mol2").write_text("p2")
    (poses_dir / "pose3_fixed.mol2").write_text("p3")
    valid2 = poses_dir / "pose4.mol2"
    valid2.write_text("p4")

    poses = ocplants.get_docked_poses(str(poses_dir))
    assert set(map(str, poses)) == {str(valid1), str(valid2)}

    pose_list = tmp_path / "pose_list.txt"
    out = ocplants.write_pose_list(poses, str(pose_list), overwrite=True)
    assert out == str(pose_list)
    contents = pose_list.read_text().splitlines()
    assert set(contents) == {str(valid1), str(valid2)}
