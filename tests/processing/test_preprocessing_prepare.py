#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.Preprocessing.Prepare.
'''

# Imports
###############################################################################
import importlib
import importlib.util as util
import sys
import types

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import OCDocker.Error as ocerror

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


class _FakePool:
    def __init__(self, _workers, raise_on_enter=False):
        self._raise_on_enter = raise_on_enter

    def __enter__(self):
        if self._raise_on_enter:
            raise IOError("pool failed")
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False

    def imap_unordered(self, fn, arguments, chunksize=1):
        _ = chunksize
        return (fn(arg) for arg in arguments)


class _LigandOk:
    def __init__(self, _src, _name, sanitize=True):
        _ = sanitize
        self.RadiusOfGyration = 1.0
        self.created_boxes = 0
        self.exported = 0

    def create_box(self, centroid=None, overwrite=False):
        _ = centroid
        _ = overwrite
        self.created_boxes += 1

    def is_valid(self):
        return True

    def to_json(self, overwrite):
        _ = overwrite
        self.exported += 1
        return 0


class _LigandInvalid(_LigandOk):
    def is_valid(self):
        return False


class _LigandNoRadius:
    inputs = []

    def __init__(self, src, _name, sanitize=True):
        _ = sanitize
        self.__class__.inputs.append(src)
        self.RadiusOfGyration = None

    def create_box(self, centroid=None, overwrite=False):
        _ = (centroid, overwrite)

    def is_valid(self):
        return True

    def to_json(self, overwrite):
        _ = overwrite
        return 0


class _LigandStub:
    def __init__(self, _src, _name, sanitize=True):
        _ = sanitize
        self.RadiusOfGyration = 1.0

    def create_box(self, centroid=None, overwrite=False):
        _ = (centroid, overwrite)

    def is_valid(self):
        return True

    def to_json(self, overwrite):
        _ = overwrite
        return 0


class _ReceptorStub:
    def __init__(self, _src, _name, mol2_path=None):
        _ = mol2_path

    def is_valid(self):
        return True

    def to_json(self, overwrite):
        _ = overwrite
        return 0


# Functions
###############################################################################
## Private ##

def _import_prepare(monkeypatch):
    importlib.import_module("OCDocker.Docking")
    importlib.import_module("OCDocker.Docking.Future")
    importlib.import_module("OCDocker.Toolbox")

    rdkit_mod = types.ModuleType("rdkit")
    rdkit_mod.Geometry = types.SimpleNamespace(rdGeometry=types.SimpleNamespace(Point3D=object))
    rdkit_mod.Chem = types.SimpleNamespace(rdchem=types.SimpleNamespace(Mol=object))

    gnina = types.ModuleType("OCDocker.Docking.Gnina")
    gnina.gen_gnina_conf = lambda *a, **k: None  # type: ignore[attr-defined]

    plants = types.ModuleType("OCDocker.Docking.PLANTS")
    plants.generate_plants_files_database = lambda *a, **k: None  # type: ignore[attr-defined]
    plants.box_to_plants = lambda *a, **k: None  # type: ignore[attr-defined]

    smina = types.ModuleType("OCDocker.Docking.Smina")
    smina.gen_smina_conf = lambda *a, **k: None  # type: ignore[attr-defined]

    vina = types.ModuleType("OCDocker.Docking.Vina")
    vina.generate_vina_files_database = lambda *a, **k: None  # type: ignore[attr-defined]
    vina.box_to_vina = lambda *a, **k: None  # type: ignore[attr-defined]

    ligand_mod = types.ModuleType("OCDocker.Ligand")
    ligand_mod.Ligand = _LigandStub  # type: ignore[attr-defined]
    ligand_mod.get_centroid = lambda *a, **k: (0.0, 0.0, 0.0)  # type: ignore[attr-defined]

    receptor_mod = types.ModuleType("OCDocker.Receptor")
    receptor_mod.Receptor = _ReceptorStub  # type: ignore[attr-defined]

    basetools_mod = types.ModuleType("OCDocker.Toolbox.Basetools")
    basetools_mod.redirect_to_tqdm = lambda: nullcontext()  # type: ignore[attr-defined]

    filesfolders_mod = types.ModuleType("OCDocker.Toolbox.FilesFolders")
    filesfolders_mod.safe_create_dir = lambda *a, **k: ocerror.ErrorCode.OK  # type: ignore[attr-defined]

    logging_mod = types.ModuleType("OCDocker.Toolbox.Logging")
    logging_mod.backup_log = lambda *a, **k: None  # type: ignore[attr-defined]

    molproc_mod = types.ModuleType("OCDocker.Toolbox.MoleculeProcessing")
    molproc_mod.clean_for_dssp = lambda *a, **k: None  # type: ignore[attr-defined]

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.print_warning = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_info = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_error = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_error_log = lambda *a, **k: None  # type: ignore[attr-defined]

    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: SimpleNamespace(multiprocess=False, available_cores=1, logdir="/tmp")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Gnina", gnina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.PLANTS", plants)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Smina", smina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Vina", vina)
    monkeypatch.setitem(sys.modules, "rdkit", rdkit_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Ligand", ligand_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Receptor", receptor_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Basetools", basetools_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.FilesFolders", filesfolders_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Logging", logging_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.MoleculeProcessing", molproc_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Config", config_mod)

    path = Path(__file__).resolve().parents[2] / "OCDocker" / "Processing" / "Preprocessing" / "Prepare.py"
    spec = util.spec_from_file_location("ocprepare_coverage_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocprepare(monkeypatch):
    return _import_prepare(monkeypatch)


@pytest.mark.order(134)
def test_prepare_dispatches_between_parallel_serial_and_single(monkeypatch, ocprepare):
    calls = {"backup": 0, "parallel": 0, "serial": 0, "single": 0}
    monkeypatch.setattr(ocprepare.oclogging, "backup_log", lambda *_a, **_k: calls.__setitem__("backup", calls["backup"] + 1))
    monkeypatch.setattr(ocprepare, "__prepare_parallel", lambda *_a, **_k: calls.__setitem__("parallel", calls["parallel"] + 1))
    monkeypatch.setattr(ocprepare, "__prepare_no_parallel", lambda *_a, **_k: calls.__setitem__("serial", calls["serial"] + 1))
    monkeypatch.setattr(ocprepare, "__prepare_single", lambda *_a, **_k: calls.__setitem__("single", calls["single"] + 1))

    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(multiprocess=True))
    ocprepare.prepare(["/tmp/a"], overwrite=False, archive="dudez", sanitize=True, spacing=0.33)

    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(multiprocess=False))
    ocprepare.prepare(["/tmp/a"], overwrite=False, archive="dudez", sanitize=True, spacing=0.33)

    ocprepare.prepare("/tmp/a", overwrite=False, archive="dudez", sanitize=True, spacing=0.33)

    assert calls["backup"] == 2
    assert calls["parallel"] == 1
    assert calls["serial"] == 1
    assert calls["single"] == 1


@pytest.mark.order(135)
def test_thread_prepare_and_single_call_core(monkeypatch, ocprepare):
    monkeypatch.setattr(ocprepare.ocbasetools, "redirect_to_tqdm", lambda: nullcontext())
    monkeypatch.setattr(ocprepare, "__core_prepare", lambda *a, **k: ocerror.ErrorCode.OK)

    rc = ocprepare.__thread_prepare(("/tmp/a", False, "dudez", True, 0.33, False))
    assert rc == ocerror.ErrorCode.OK

    assert ocprepare.__prepare_single("/tmp/a", False, "dudez", True, 0.33, False) is None


@pytest.mark.order(136)
def test_prepare_no_parallel_iterates_all_paths(monkeypatch, ocprepare):
    monkeypatch.setattr(ocprepare.ocbasetools, "redirect_to_tqdm", lambda: nullcontext())
    monkeypatch.setattr(ocprepare, "tqdm", lambda iterable, **kwargs: iterable)
    seen = []
    monkeypatch.setattr(ocprepare, "__core_prepare", lambda path, *_a, **_k: seen.append(path) or ocerror.ErrorCode.OK)

    assert ocprepare.__prepare_no_parallel(["/tmp/a", "/tmp/b"], False, "dudez", True, 0.33, "x", False) is None
    assert seen == ["/tmp/a", "/tmp/b"]


@pytest.mark.order(137)
def test_prepare_parallel_handles_ioerror(monkeypatch, tmp_path, ocprepare):
    logs = []
    errs = []
    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(available_cores=2, logdir=str(tmp_path)))
    monkeypatch.setattr(ocprepare, "Pool", lambda workers: _FakePool(workers, raise_on_enter=True))
    monkeypatch.setattr(ocprepare.ocprint, "print_error_log", lambda msg, path: logs.append((msg, path)))
    monkeypatch.setattr(ocprepare.ocprint, "print_error", lambda msg: errs.append(msg))

    assert ocprepare.__prepare_parallel(["/tmp/a"], False, "dudez", True, 0.33, "x", False) is None
    assert logs
    assert errs


@pytest.mark.order(138)
def test_sub_core_prepare_calls_prepare_molecule_for_pdbbind_and_dudez(monkeypatch, tmp_path, ocprepare):
    root = tmp_path / "ligands"
    process_dir = root / "molA"
    process_dir.mkdir(parents=True, exist_ok=True)

    created = []
    prepared = []
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda p: created.append(p) or ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: prepared.append((a, k)) or None)

    dirs1 = ocprepare.__sub_core_prepare(str(root), "pdbbind", overwrite=False, mols=None, sanitize=True, targetCentroid=(1.0, 2.0, 3.0))
    assert str(process_dir) in dirs1
    assert prepared[-1][0][0].endswith("ligand.sdf")

    dirs2 = ocprepare.__sub_core_prepare(str(root), "dudez", overwrite=False, mols=None, sanitize=True, targetCentroid=(1.0, 2.0, 3.0))
    assert str(process_dir) in dirs2
    assert prepared[-1][0][0].endswith("ligand.smi")
    assert created


@pytest.mark.order(139)
def test_sub_core_prepare_mols_list_branch_moves_files(monkeypatch, tmp_path, ocprepare):
    root = tmp_path / "ligands"
    root.mkdir(parents=True, exist_ok=True)
    ligand_file = root / "lig.and.smi"
    ligand_file.write_text("CCO", encoding="utf-8")
    existing_dir = root / "existing"
    existing_dir.mkdir(parents=True, exist_ok=True)

    dirs_created = []
    moved = []
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda p: dirs_created.append(p) or ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.shutil, "move", lambda src, dst: moved.append((src, dst)))
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)

    out = ocprepare.__sub_core_prepare(str(root), "dudez", overwrite=False, mols=[str(ligand_file)], sanitize=True, targetCentroid=(0.0, 0.0, 0.0))
    assert str(existing_dir) in out
    assert dirs_created
    assert moved


@pytest.mark.order(140)
def test_prepare_molecule_ligand_unknown_and_receptor_wrong_type(monkeypatch, tmp_path, ocprepare):
    calls = {"unknown": 0, "wrong_type": 0}
    monkeypatch.setattr(ocprepare.ocerror.Error, "unknown", lambda *a, **k: calls.__setitem__("unknown", calls["unknown"] + 1) or ocerror.ErrorCode.UNKNOWN)
    monkeypatch.setattr(ocprepare.ocerror.Error, "wrong_type", lambda *a, **k: calls.__setitem__("wrong_type", calls["wrong_type"] + 1) or ocerror.ErrorCode.WRONG_TYPE)

    _ = ocprepare.__prepare_molecule(str(tmp_path / "x.smi"), overwrite=True, moltype="weird", dbName="dudez", sanitize=True)
    _ = ocprepare.__prepare_molecule(123, overwrite=True, moltype="receptor", dbName="dudez", sanitize=True)  # type: ignore[arg-type]

    assert calls["unknown"] == 1
    assert calls["wrong_type"] == 1


@pytest.mark.order(141)
def test_prepare_molecule_ligand_valid_and_invalid_paths(monkeypatch, tmp_path, ocprepare):
    ligand_file = tmp_path / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")

    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.ocl, "Ligand", _LigandOk)

    assert ocprepare.__prepare_molecule(str(ligand_file), overwrite=True, moltype="ligand", dbName="dudez", sanitize=True) is None

    malformed_logs = []
    monkeypatch.setattr(ocprepare.ocl, "Ligand", _LigandInvalid)
    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(logdir=str(tmp_path / "logs")))
    monkeypatch.setattr(ocprepare.ocprint, "print_error_log", lambda msg, path: malformed_logs.append((msg, path)))
    assert ocprepare.__prepare_molecule(str(ligand_file), overwrite=True, moltype="ligand", dbName="dudez", sanitize=True) is None
    assert malformed_logs


@pytest.mark.order(142)
def test_core_prepare_index_and_missing_reference_centroid(monkeypatch, tmp_path, ocprepare):
    rc_index = ocprepare.__core_prepare(str(tmp_path / "index"), False, "dudez", True, 0.33)
    assert rc_index == ocerror.ErrorCode.UNALLOWED_DIR

    work = tmp_path / "ptnA"
    work.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    rc_missing = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=None)
    assert rc_missing == ocerror.ErrorCode.FILE_NOT_EXIST


@pytest.mark.order(143)
def test_core_prepare_generates_configs_single_and_multibox(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnB"
    work.mkdir(parents=True, exist_ok=True)
    (work / "compounds" / "ligands").mkdir(parents=True, exist_ok=True)
    process_dir = tmp_path / "proc1"
    boxes = process_dir / "boxes"
    boxes.mkdir(parents=True, exist_ok=True)
    (boxes / "box0.pdb").write_text("box0", encoding="utf-8")
    (boxes / "box1.pdb").write_text("box1", encoding="utf-8")

    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: [str(process_dir)])

    calls = {"gnina": 0, "vina_db": 0, "plants_db": 0, "smina": 0, "box_vina": 0, "box_plants": 0, "mkdir": 0}
    monkeypatch.setattr(ocprepare.ocgnina, "gen_gnina_conf", lambda *a, **k: calls.__setitem__("gnina", calls["gnina"] + 1))
    monkeypatch.setattr(ocprepare.ocvina, "generate_vina_files_database", lambda *a, **k: calls.__setitem__("vina_db", calls["vina_db"] + 1))
    monkeypatch.setattr(ocprepare.ocplants, "generate_plants_files_database", lambda *a, **k: calls.__setitem__("plants_db", calls["plants_db"] + 1))
    monkeypatch.setattr(ocprepare.ocsmina, "gen_smina_conf", lambda *a, **k: calls.__setitem__("smina", calls["smina"] + 1))
    monkeypatch.setattr(ocprepare.ocvina, "box_to_vina", lambda *a, **k: calls.__setitem__("box_vina", calls["box_vina"] + 1))
    monkeypatch.setattr(ocprepare.ocplants, "box_to_plants", lambda *a, **k: calls.__setitem__("box_plants", calls["box_plants"] + 1))
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: calls.__setitem__("mkdir", calls["mkdir"] + 1) or ocerror.ErrorCode.OK)

    rc_single = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=(1.0, 2.0, 3.0), all_boxes=False)
    assert rc_single == ocerror.ErrorCode.OK
    assert calls["vina_db"] >= 1
    assert calls["plants_db"] >= 1

    rc_multi = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=(1.0, 2.0, 3.0), all_boxes=True)
    assert rc_multi == ocerror.ErrorCode.OK
    assert calls["box_vina"] >= 2
    assert calls["box_plants"] >= 2
    assert calls["mkdir"] >= 8


@pytest.mark.order(164)
def test_core_prepare_reference_ligand_parse_exception_is_logged(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnC"
    work.mkdir(parents=True, exist_ok=True)
    (work / "reference_ligand.mol2").write_text("@<TRIPOS>MOLECULE\nx\n", encoding="utf-8")

    errors = []
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare.ocl, "get_centroid", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad ref")))
    monkeypatch.setattr(ocprepare.ocprint, "print_error", lambda msg: errors.append(msg))

    rc = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=None)
    assert rc == ocerror.ErrorCode.FILE_NOT_EXIST
    assert errors


@pytest.mark.order(165)
def test_core_prepare_no_boxes_logs_warning_and_returns_ok(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnD"
    work.mkdir(parents=True, exist_ok=True)
    (work / "compounds" / "ligands").mkdir(parents=True, exist_ok=True)
    process_dir = tmp_path / "proc_no_box"
    process_dir.mkdir(parents=True, exist_ok=True)

    warnings = []
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: [str(process_dir)])
    monkeypatch.setattr(
        ocprepare.ocprint,
        "print_warning",
        lambda *a, **k: warnings.append(k.get("message", a[0] if a else "")),
    )

    rc = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=(1.0, 2.0, 3.0))
    assert rc == ocerror.ErrorCode.OK
    assert warnings


@pytest.mark.order(166)
def test_core_prepare_skips_existing_generated_configs(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnE"
    work.mkdir(parents=True, exist_ok=True)
    (work / "compounds" / "ligands").mkdir(parents=True, exist_ok=True)
    process_dir = tmp_path / "proc_existing"
    boxes = process_dir / "boxes"
    boxes.mkdir(parents=True, exist_ok=True)
    (boxes / "box0.pdb").write_text("box0", encoding="utf-8")
    (process_dir / "gninaFiles").mkdir(parents=True, exist_ok=True)
    (process_dir / "vinaFiles").mkdir(parents=True, exist_ok=True)
    (process_dir / "plantsFiles").mkdir(parents=True, exist_ok=True)
    (process_dir / "sminaFiles").mkdir(parents=True, exist_ok=True)
    (process_dir / "gninaFiles" / "entry.txt").write_text("x", encoding="utf-8")
    (process_dir / "vinaFiles" / "entry.txt").write_text("x", encoding="utf-8")
    (process_dir / "plantsFiles" / "entry.txt").write_text("x", encoding="utf-8")
    (process_dir / "sminaFiles" / "entry.conf").write_text("x", encoding="utf-8")

    infos = []
    calls = {"gnina": 0, "vina": 0, "plants": 0, "smina": 0}
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: [str(process_dir)])
    monkeypatch.setattr(ocprepare.ocprint, "print_info", lambda msg: infos.append(msg))
    monkeypatch.setattr(ocprepare.ocgnina, "gen_gnina_conf", lambda *a, **k: calls.__setitem__("gnina", calls["gnina"] + 1))
    monkeypatch.setattr(ocprepare.ocvina, "generate_vina_files_database", lambda *a, **k: calls.__setitem__("vina", calls["vina"] + 1))
    monkeypatch.setattr(ocprepare.ocplants, "generate_plants_files_database", lambda *a, **k: calls.__setitem__("plants", calls["plants"] + 1))
    monkeypatch.setattr(ocprepare.ocsmina, "gen_smina_conf", lambda *a, **k: calls.__setitem__("smina", calls["smina"] + 1))

    rc = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=(1.0, 2.0, 3.0), all_boxes=False)
    assert rc == ocerror.ErrorCode.OK
    assert calls == {"gnina": 0, "vina": 0, "plants": 0, "smina": 0}
    assert len(infos) >= 4


@pytest.mark.order(167)
def test_prepare_molecule_ligand_tuple_alternative_and_parse_exception(monkeypatch, tmp_path, ocprepare):
    ligand_file = tmp_path / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")
    warnings = []
    logs = []

    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: ocerror.ErrorCode.OK)
    monkeypatch.setattr(
        ocprepare.ocprint,
        "print_warning",
        lambda *a, **k: warnings.append(k.get("message", a[0] if a else "")),
    )
    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(logdir=str(tmp_path / "logs")))
    monkeypatch.setattr(ocprepare.ocprint, "print_error_log", lambda msg, path: logs.append((msg, path)))

    _LigandNoRadius.inputs = []
    monkeypatch.setattr(ocprepare.ocl, "Ligand", _LigandNoRadius)
    assert ocprepare.__prepare_molecule((str(ligand_file), "ignored"), overwrite=True, moltype="ligand", dbName="dudez", sanitize=True, alternativeLigand="ALT") is None
    assert _LigandNoRadius.inputs == [str(ligand_file), "ALT"]
    assert any("trying to load its alternative ligand" in msg for msg in warnings)
    assert any("even with the alternative ligand" in msg for msg in warnings)

    monkeypatch.setattr(ocprepare.ocl, "Ligand", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("parse failure")))
    assert ocprepare.__prepare_molecule(str(ligand_file), overwrite=True, moltype="ligand", dbName="dudez", sanitize=True) is None
    assert logs


@pytest.mark.order(168)
def test_prepare_molecule_receptor_tuple_and_string_parse_exceptions(monkeypatch, tmp_path, ocprepare):
    receptor_pdb = tmp_path / "receptor.pdb"
    receptor_pdb.write_text("ATOM\n", encoding="utf-8")
    receptor_mol2 = tmp_path / "receptor.mol2"
    receptor_mol2.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    logs = []
    cleaned = {"calls": 0}

    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(logdir=str(tmp_path / "logs")))
    monkeypatch.setattr(ocprepare.ocprint, "print_error_log", lambda msg, path: logs.append((msg, path)))
    monkeypatch.setattr(ocprepare.ocmolproc, "clean_for_dssp", lambda **_k: cleaned.__setitem__("calls", cleaned["calls"] + 1))
    monkeypatch.setattr(ocprepare.ocr, "Receptor", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad receptor")))

    assert ocprepare.__prepare_molecule((str(receptor_pdb), str(receptor_mol2)), overwrite=True, moltype="receptor", dbName="dudez", sanitize=True) is None
    assert ocprepare.__prepare_molecule(str(receptor_pdb), overwrite=True, moltype="receptor", dbName="dudez", sanitize=True) is None
    assert cleaned["calls"] == 1
    assert len(logs) >= 2


@pytest.mark.order(169)
def test_prepare_molecule_descriptor_exists_short_circuits(monkeypatch, tmp_path, ocprepare):
    ligand_file = tmp_path / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")
    (tmp_path / "ligand_descriptors.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ocprepare.ocl, "Ligand", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert ocprepare.__prepare_molecule(str(ligand_file), overwrite=False, moltype="ligand", dbName="dudez", sanitize=True) is None


@pytest.mark.order(170)
def test_prepare_parallel_success_path_calls_gc(monkeypatch, tmp_path, ocprepare):
    gc_calls = {"count": 0}
    monkeypatch.setattr(ocprepare, "get_config", lambda: SimpleNamespace(available_cores=2, logdir=str(tmp_path)))
    monkeypatch.setattr(ocprepare, "Pool", lambda workers: _FakePool(workers, raise_on_enter=False))
    monkeypatch.setattr(ocprepare, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(ocprepare, "__thread_prepare", lambda *_a, **_k: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.gc, "collect", lambda: gc_calls.__setitem__("count", gc_calls["count"] + 1) or 0)

    assert ocprepare.__prepare_parallel(["/tmp/a", "/tmp/b"], False, "dudez", True, 0.33, "x", False) is None
    assert gc_calls["count"] >= 2


@pytest.mark.order(171)
def test_sub_core_prepare_simple_filename_branch(monkeypatch, tmp_path, ocprepare):
    root = tmp_path / "ligands_simple"
    root.mkdir(parents=True, exist_ok=True)
    ligand_file = root / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")

    moved = []
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.shutil, "move", lambda src, dst: moved.append((src, dst)))
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)

    _ = ocprepare.__sub_core_prepare(str(root), "dudez", overwrite=False, mols=[str(ligand_file)], sanitize=True, targetCentroid=(0.0, 0.0, 0.0))
    assert moved
    assert moved[0][1].endswith("/ligand.smi/ligand/ligand.smi")


@pytest.mark.order(192)
def test_core_prepare_reference_ligand_success_and_all_compound_dirs(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnF"
    ligands_dir = work / "compounds" / "ligands"
    decoys_dir = work / "compounds" / "decoys"
    candidates_dir = work / "compounds" / "candidates"
    ligands_dir.mkdir(parents=True, exist_ok=True)
    decoys_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    (work / "reference_ligand.sdf").write_text("MOL", encoding="utf-8")

    calls = []
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare.ocl, "get_centroid", lambda *_a, **_k: (1.0, 2.0, 3.0))
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: calls.append((a, k)) or [])

    rc = ocprepare.__core_prepare(str(work), False, "pdbbind", True, 0.33, targetCentroid=None)
    assert rc == ocerror.ErrorCode.OK
    assert len(calls) == 3
    assert calls[0][0][0] == str(ligands_dir)
    assert calls[1][0][0] == str(decoys_dir)
    assert calls[2][0][0] == str(candidates_dir)
    assert calls[0][1]["targetCentroid"] == (1.0, 2.0, 3.0)


@pytest.mark.order(193)
def test_prepare_molecule_ligand_with_custom_name_and_no_alternative(monkeypatch, tmp_path, ocprepare):
    class _NamedNoRadiusLigand:
        names = []

        def __init__(self, _src, name, sanitize=True):
            _ = sanitize
            self.__class__.names.append(name)
            self.RadiusOfGyration = None

        def create_box(self, centroid=None, overwrite=False):
            _ = (centroid, overwrite)

        def is_valid(self):
            return True

        def to_json(self, overwrite):
            _ = overwrite
            return 0

    ligand_file = tmp_path / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")
    warnings = []
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.ocl, "Ligand", _NamedNoRadiusLigand)
    monkeypatch.setattr(ocprepare.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocprepare.__prepare_molecule(
        str(ligand_file),
        overwrite=True,
        moltype="ligand",
        dbName="dudez",
        sanitize=True,
        molName="custom_ligand",
    )
    assert rc is None
    assert _NamedNoRadiusLigand.names == ["custom_ligand"]
    assert any("no alternative ligand was provided" in msg for msg in warnings)


@pytest.mark.order(194)
def test_prepare_molecule_ligand_alternative_recovers_radius(monkeypatch, tmp_path, ocprepare):
    class _AltRecoversLigand:
        inputs = []
        created_boxes = 0

        def __init__(self, src, _name, sanitize=True):
            _ = sanitize
            self.__class__.inputs.append(src)
            self.RadiusOfGyration = 1.0 if src == "ALT" else None

        def create_box(self, centroid=None, overwrite=False):
            _ = (centroid, overwrite)
            self.__class__.created_boxes += 1

        def is_valid(self):
            return True

        def to_json(self, overwrite):
            _ = overwrite
            return 0

    ligand_file = tmp_path / "ligand.smi"
    ligand_file.write_text("CCO", encoding="utf-8")
    warnings = []
    monkeypatch.setattr(ocprepare.ocff, "safe_create_dir", lambda _p: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocprepare.ocl, "Ligand", _AltRecoversLigand)
    monkeypatch.setattr(ocprepare.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocprepare.__prepare_molecule(
        str(ligand_file),
        overwrite=True,
        moltype="ligand",
        dbName="dudez",
        sanitize=True,
        alternativeLigand="ALT",
    )
    assert rc is None
    assert _AltRecoversLigand.inputs == [str(ligand_file), "ALT"]
    assert _AltRecoversLigand.created_boxes == 1
    assert any("trying to load its alternative ligand" in msg for msg in warnings)
    assert not any("even with the alternative ligand" in msg for msg in warnings)


@pytest.mark.order(195)
def test_prepare_molecule_receptor_tuple_non_pdb_does_not_clean_dssp(monkeypatch, tmp_path, ocprepare):
    receptor_a = tmp_path / "input_a.mol2"
    receptor_b = tmp_path / "input_b.mol2"
    receptor_a.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    receptor_b.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")

    cleaned = {"calls": 0}
    monkeypatch.setattr(ocprepare.ocmolproc, "clean_for_dssp", lambda **_k: cleaned.__setitem__("calls", cleaned["calls"] + 1))
    monkeypatch.setattr(ocprepare.ocr, "Receptor", _ReceptorStub)

    rc = ocprepare.__prepare_molecule(
        (str(receptor_a), str(receptor_b)),
        overwrite=True,
        moltype="receptor",
        dbName="dudez",
        sanitize=True,
    )
    assert rc is None
    assert cleaned["calls"] == 0


@pytest.mark.order(196)
def test_core_prepare_reference_ligand_falsey_centroid_tries_next_extension(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnG"
    work.mkdir(parents=True, exist_ok=True)
    (work / "reference_ligand.mol2").write_text("@<TRIPOS>MOLECULE\nx\n", encoding="utf-8")
    (work / "reference_ligand.pdb").write_text("ATOM\n", encoding="utf-8")

    warnings = []
    centroid_calls = {"count": 0}

    def _fake_centroid(*_a, **_k):
        centroid_calls["count"] += 1
        if centroid_calls["count"] == 1:
            return None
        return (2.0, 3.0, 4.0)

    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare.ocl, "get_centroid", _fake_centroid)
    monkeypatch.setattr(
        ocprepare.ocprint,
        "print_warning",
        lambda *a, **k: warnings.append(k.get("message", a[0] if a else "")),
    )
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: [])

    rc = ocprepare.__core_prepare(str(work), False, "dudez", True, 0.33, targetCentroid=None)
    assert rc == ocerror.ErrorCode.OK
    assert centroid_calls["count"] == 2
    assert any("centroid of the reference ligand" in msg.lower() for msg in warnings)


@pytest.mark.order(197)
def test_core_prepare_handles_missing_ligands_dir_and_processes_decoys(monkeypatch, tmp_path, ocprepare):
    work = tmp_path / "ptnH"
    decoys_dir = work / "compounds" / "decoys"
    decoys_dir.mkdir(parents=True, exist_ok=True)

    calls = []
    monkeypatch.setattr(ocprepare, "__prepare_molecule", lambda *a, **k: None)
    monkeypatch.setattr(ocprepare, "__sub_core_prepare", lambda *a, **k: calls.append((a, k)) or [])

    rc = ocprepare.__core_prepare(
        str(work),
        False,
        "dudez",
        True,
        0.33,
        targetCentroid=(1.0, 2.0, 3.0),
    )
    assert rc == ocerror.ErrorCode.OK
    assert len(calls) == 1
    assert calls[0][0][0] == str(decoys_dir)
