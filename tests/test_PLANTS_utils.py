"""Tests for PLANTS utility helpers."""

import sys
import types
import importlib
import importlib.util as util
import csv
from pathlib import Path
import pytest

def _load_plants(monkeypatch):
    root = importlib.import_module("OCDocker")

    class DummySeries(list):
        @property
        def values(self):
            return list(self)
        def __getitem__(self, item):
            result = list.__getitem__(self, item)
            return DummySeries(result) if isinstance(item, slice) else result

    class DummyDataFrame:
        def __init__(self, rows):
            self._rows = rows
            self.shape = (len(rows), len(rows[0]) if rows else 0)
            self._cols = {k: DummySeries([r[k] for r in rows]) for k in rows[0].keys()} if rows else {}
        def iterrows(self):
            for i, row in enumerate(self._rows):
                yield i, row
        def __getattr__(self, name):
            return self._cols[name]

    def read_csv(path):
        with open(path, newline="") as f:
            rows = []
            for row in csv.DictReader(f):
                for k, v in row.items():
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
                rows.append(row)
        return DummyDataFrame(rows)

    pd_stub = types.ModuleType("pandas")
    pd_stub.read_csv = read_csv  # type: ignore
    monkeypatch.setitem(sys.modules, "pandas", pd_stub)

    class ReportLevel:
        ERROR = 1
    class Error:
        @staticmethod
        def file_not_exist(*a, **k):
            return -1
        @staticmethod
        def read_file(*a, **k):
            return -1
        @staticmethod
        def dir_not_exist(*a, **k):
            return -1
        @staticmethod
        def write_file(*a, **k):
            return -1
        @staticmethod
        def ok():
            return 0
    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.ocerror = types.SimpleNamespace(Error=Error, ReportLevel=ReportLevel)  # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)
    setattr(root, "Initialise", init_mod)  # type: ignore

    lig_mod = types.ModuleType("OCDocker.Ligand")
    class Ligand:
        pass
    lig_mod.Ligand = Ligand  # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Ligand", lig_mod)
    setattr(root, "Ligand", lig_mod)  # type: ignore

    rec_mod = types.ModuleType("OCDocker.Receptor")
    class Receptor:
        pass
    rec_mod.Receptor = Receptor  # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Receptor", rec_mod)
    setattr(root, "Receptor", rec_mod)  # type: ignore

    tb_pkg = types.ModuleType("OCDocker.Toolbox")
    tb_pkg.__path__ = [str(Path("OCDocker") / "Toolbox")]
    sys.modules["OCDocker.Toolbox"] = tb_pkg
    root.Toolbox = tb_pkg # type: ignore

    for name in ["Conversion", "FilesFolders", "Running", "Validation", "Printing"]:
        mod = types.ModuleType(f"OCDocker.Toolbox.{name}")
        setattr(tb_pkg, name, mod)
        monkeypatch.setitem(sys.modules, f"OCDocker.Toolbox.{name}", mod)
    tb_pkg.Printing.printv = lambda *a, **k: None
    tb_pkg.Printing.print_error = lambda *a, **k: None
    tb_pkg.Printing.print_error_log = lambda *a, **k: None

    root_dir = Path(__file__).resolve().parents[1] / "OCDocker"
    spec = util.spec_from_file_location(
        "OCDocker.Docking.PLANTS", root_dir / "Docking" / "PLANTS.py"
    )
    plants = util.module_from_spec(spec) # type: ignore
    assert spec.loader is not None # type: ignore
    spec.loader.exec_module(plants)  # type: ignore
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.PLANTS", plants)
    return plants

@pytest.fixture
def ocplants(monkeypatch):
    """Provide PLANTS module with heavy dependencies stubbed."""
    return _load_plants(monkeypatch)

def test_get_binding_site(ocplants, tmp_path):
    box = tmp_path / "box.pdb"
    header = "HEADER    CORNERS OF BOX      " + "".join(f"{v:8.3f}" for v in [0, 0, 0, 2, 4, 6]) + "\n"
    remark = "REMARK    CENTER (X Y Z)      " + "".join(f"{v:8.3f}" for v in [1, 2, 3]) + "\n"
    box.write_text(header + remark)

    center, radius = ocplants.get_binding_site(str(box)) # type: ignore
    assert center == (1.0, 2.0, 3.0)
    assert radius == 11.7

def test_read_log(ocplants, tmp_path):
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
