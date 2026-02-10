#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.Dock helper functions.
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
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


class _FakeFcntl:
    LOCK_EX = 1
    LOCK_UN = 2

    def __init__(self, fail_unlock=False):
        self.fail_unlock = fail_unlock
        self.calls = []

    def flock(self, _fd, operation):
        self.calls.append(operation)
        if operation == self.LOCK_UN and self.fail_unlock:
            raise OSError("unlock failed")


class _FakePreparationStrategy:
    def get_ligand_command(self, *_a, **_k):
        return ["prepare_ligand"]

    def get_receptor_command(self, *_a, **_k):
        return ["prepare_receptor"]


class _FakeEngineObject:
    def __init__(self, prepared_receptor, prepared_ligand, input_receptor_path, input_ligand_path, log_path, output_target):
        self.prepared_receptor = prepared_receptor
        self.prepared_ligand = prepared_ligand
        self.input_receptor_path = input_receptor_path
        self.input_ligand_path = input_ligand_path
        self.gnina_log = log_path
        self.plants_log = log_path
        self.smina_log = log_path
        self.vina_log = log_path
        self._output_target = output_target
        self.preparation_strategy = _FakePreparationStrategy()
        self.events = []

    def run_prepare_ligand(self, **_kwargs):
        self.events.append("run_prepare_ligand")
        Path(self.prepared_ligand).parent.mkdir(parents=True, exist_ok=True)
        Path(self.prepared_ligand).write_text("LIGAND", encoding="utf-8")
        return ocerror.ErrorCode.OK

    def run_prepare_receptor(self, **_kwargs):
        self.events.append("run_prepare_receptor")
        Path(self.prepared_receptor).parent.mkdir(parents=True, exist_ok=True)
        Path(self.prepared_receptor).write_text("RECEPTOR", encoding="utf-8")
        return ocerror.ErrorCode.OK

    def run_gnina(self, **_kwargs):
        self.events.append("run_gnina")
        return ocerror.ErrorCode.OK

    def run_plants(self, **_kwargs):
        self.events.append("run_plants")
        Path(self._output_target).mkdir(parents=True, exist_ok=True)
        Path(self._output_target, "ranking.csv").write_text("pose,score\n0,-1.0\n", encoding="utf-8")
        return ocerror.ErrorCode.OK

    def run_smina(self, **_kwargs):
        self.events.append("run_smina")
        return ocerror.ErrorCode.OK

    def run_vina(self, **_kwargs):
        self.events.append("run_vina")
        return ocerror.ErrorCode.OK


# Functions
###############################################################################
## Private ##

def _import_dock(monkeypatch):
    importlib.import_module("OCDocker.Docking")
    importlib.import_module("OCDocker.Docking.Future")
    importlib.import_module("OCDocker.Toolbox")

    gnina = types.ModuleType("OCDocker.Docking.Future.Gnina")
    plants = types.ModuleType("OCDocker.Docking.PLANTS")
    smina = types.ModuleType("OCDocker.Docking.Smina")
    vina = types.ModuleType("OCDocker.Docking.Vina")
    ligand_mod = types.ModuleType("OCDocker.Ligand")
    receptor_mod = types.ModuleType("OCDocker.Receptor")

    basetools = types.ModuleType("OCDocker.Toolbox.Basetools")
    basetools.redirect_to_tqdm = lambda: nullcontext()  # type: ignore[attr-defined]

    logging_mod = types.ModuleType("OCDocker.Toolbox.Logging")
    logging_mod.backup_log = lambda *a, **k: None  # type: ignore[attr-defined]

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.print_error_log = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_warning_log = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_warning = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod.print_error = lambda *a, **k: None  # type: ignore[attr-defined]

    validation_mod = types.ModuleType("OCDocker.Toolbox.Validation")
    validation_mod.is_molecule_valid_with_retry = lambda *_a, **_k: True  # type: ignore[attr-defined]
    validation_mod.is_molecule_valid = lambda *_a, **_k: True  # type: ignore[attr-defined]

    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: SimpleNamespace(available_cores=1, logdir="/tmp", multiprocess=False)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Future.Gnina", gnina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.PLANTS", plants)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Smina", smina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Vina", vina)
    monkeypatch.setitem(sys.modules, "OCDocker.Ligand", ligand_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Receptor", receptor_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Basetools", basetools)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Logging", logging_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Validation", validation_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Config", config_mod)

    path = Path(__file__).resolve().parents[2] / "OCDocker" / "Processing" / "Dock.py"
    spec = util.spec_from_file_location("ocdock_helpers_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_engine_run_environment(tmp_path, engine_dir):
    ligand_dir = tmp_path / f"ligand_{engine_dir}"
    engine_path = ligand_dir / engine_dir
    boxes_path = ligand_dir / "boxes"
    engine_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "receptor.pdb"
    ligand_path = ligand_dir / "ligand.mol2"
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (boxes_path / "box0.pdb").write_text("BOX", encoding="utf-8")
    return ligand_dir, boxes_path, receptor_path, ligand_path


## Public ##

@pytest.fixture
def ocdock_helpers(monkeypatch):
    return _import_dock(monkeypatch)


@pytest.mark.order(150)
def test_needs_receptor_preparation_overwrite_and_missing_file(tmp_path, ocdock_helpers):
    missing = str(tmp_path / "missing.pdbqt")
    assert ocdock_helpers.__needs_receptor_preparation(missing, overwrite=True) is True
    assert ocdock_helpers.__needs_receptor_preparation(missing, overwrite=False) is True


@pytest.mark.order(151)
def test_needs_receptor_preparation_uses_retry_and_non_retry_branches(monkeypatch, tmp_path, ocdock_helpers):
    receptor = tmp_path / "prepared.pdbqt"
    receptor.write_text("ATOM", encoding="utf-8")

    calls = {"retry": 0, "direct": 0}
    monkeypatch.setattr(
        ocdock_helpers.ocvalidation,
        "is_molecule_valid_with_retry",
        lambda *_a, **_k: calls.__setitem__("retry", calls["retry"] + 1) or False,
    )
    monkeypatch.setattr(
        ocdock_helpers.ocvalidation,
        "is_molecule_valid",
        lambda *_a, **_k: calls.__setitem__("direct", calls["direct"] + 1) or True,
    )

    assert ocdock_helpers.__needs_receptor_preparation(str(receptor), overwrite=False, retry=True) is True
    assert ocdock_helpers.__needs_receptor_preparation(str(receptor), overwrite=False, retry=False) is False
    assert calls["retry"] == 1
    assert calls["direct"] == 1


@pytest.mark.order(152)
def test_list_boxes_single_and_all_boxes_modes(tmp_path, ocdock_helpers):
    ligand_dir = tmp_path / "ligand"
    boxes_dir = ligand_dir / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    (boxes_dir / "box1.pdb").write_text("box1", encoding="utf-8")
    (boxes_dir / "box0.pdb").write_text("box0", encoding="utf-8")

    one = ocdock_helpers.__list_boxes(str(ligand_dir), all_boxes=False)
    assert one == [("box0", str(boxes_dir / "box0.pdb"))]

    all_boxes = ocdock_helpers.__list_boxes(str(ligand_dir), all_boxes=True)
    assert all_boxes == [
        ("box0", str(boxes_dir / "box0.pdb")),
        ("box1", str(boxes_dir / "box1.pdb")),
    ]

    (boxes_dir / "box0.pdb").unlink()
    assert ocdock_helpers.__list_boxes(str(ligand_dir), all_boxes=False) == []


@pytest.mark.order(153)
def test_normalize_run_result_for_tuple_and_int(ocdock_helpers):
    assert ocdock_helpers.__normalize_run_result((3, "stderr")) == (3, "stderr")
    assert ocdock_helpers.__normalize_run_result(5) == (5, "")


@pytest.mark.order(154)
def test_receptor_file_lock_without_fcntl_creates_lock_file(tmp_path, monkeypatch, ocdock_helpers):
    receptor_path = tmp_path / "prepared_receptor.pdbqt"
    receptor_path.write_text("prepared", encoding="utf-8")
    monkeypatch.setattr(ocdock_helpers, "fcntl", None)

    with ocdock_helpers.__receptor_file_lock(str(receptor_path)):
        assert (tmp_path / "prepared_receptor.pdbqt.lock").exists()

    assert (tmp_path / "prepared_receptor.pdbqt.lock").exists()


@pytest.mark.order(155)
def test_receptor_file_lock_with_fcntl_unlock_error_is_ignored(tmp_path, monkeypatch, ocdock_helpers):
    receptor_path = tmp_path / "prepared_receptor.pdbqt"
    receptor_path.write_text("prepared", encoding="utf-8")

    fake_fcntl = _FakeFcntl(fail_unlock=True)
    monkeypatch.setattr(ocdock_helpers, "fcntl", fake_fcntl)

    with ocdock_helpers.__receptor_file_lock(str(receptor_path)):
        pass

    assert fake_fcntl.calls[0] == _FakeFcntl.LOCK_EX
    assert fake_fcntl.calls[-1] == _FakeFcntl.LOCK_UN


@pytest.mark.order(164)
def test_core_run_dock_rejects_index_directory(tmp_path, ocdock_helpers):
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    rc = ocdock_helpers.__core_run_dock(
        str(index_dir),
        str(tmp_path),
        "pdbbind",
        "vina",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.UNALLOWED_DIR


@pytest.mark.order(165)
def test_core_run_dock_logs_missing_descriptor_errors(monkeypatch, tmp_path, ocdock_helpers):
    protein_dir = tmp_path / "proteinA"
    ligand_dir = tmp_path / "ligandA"
    protein_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda msg, path: errors.append((msg, path)))

    rc = ocdock_helpers.__core_run_dock(
        str(protein_dir),
        str(ligand_dir),
        "pdbbind",
        "vina",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST
    assert len(errors) == 2
    assert "receptor_descriptors.json" in errors[0][0]
    assert "ligand_descriptors.json" in errors[1][0]


@pytest.mark.order(166)
def test_core_run_dock_rejects_unknown_docking_algorithm(monkeypatch, tmp_path, ocdock_helpers):
    protein_dir = tmp_path / "proteinB"
    ligand_dir = tmp_path / "ligandB"
    (protein_dir / "receptor_descriptors.json").parent.mkdir(parents=True, exist_ok=True)
    (ligand_dir / "boxes").mkdir(parents=True, exist_ok=True)
    (protein_dir / "receptor_descriptors.json").write_text("{}", encoding="utf-8")
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")
    (ligand_dir / "boxes" / "box0.pdb").write_text("BOX", encoding="utf-8")

    errors = []
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda msg, path: errors.append((msg, path)))

    rc = ocdock_helpers.__core_run_dock(
        str(protein_dir),
        str(ligand_dir),
        "pdbbind",
        "badalgo",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM
    assert errors
    assert "Wrong docking algorithm" in errors[0][0]


@pytest.mark.order(167)
def test_core_run_dock_all_boxes_mode_skips_when_no_box_files(monkeypatch, tmp_path, ocdock_helpers):
    protein_dir = tmp_path / "proteinC"
    ligand_dir = tmp_path / "ligandC"
    protein_dir.mkdir(parents=True, exist_ok=True)
    ligand_dir.mkdir(parents=True, exist_ok=True)
    (protein_dir / "receptor_descriptors.json").write_text("{}", encoding="utf-8")
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda msg, path: warnings.append((msg, path)))
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda msg: warnings.append((msg, "")))

    rc = ocdock_helpers.__core_run_dock(
        str(protein_dir),
        str(ligand_dir),
        "pdbbind",
        "vina",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        all_boxes=True,
    )
    assert rc == ocerror.ErrorCode.SKIP
    assert warnings
    assert "No box files found" in warnings[0][0]


@pytest.mark.order(168)
def test_core_run_dock_maps_nonzero_runner_status_to_docking_failed(monkeypatch, tmp_path, ocdock_helpers):
    protein_dir = tmp_path / "proteinD"
    ligand_dir = tmp_path / "ligandD"
    (protein_dir / "receptor_descriptors.json").parent.mkdir(parents=True, exist_ok=True)
    (ligand_dir / "boxes").mkdir(parents=True, exist_ok=True)
    (protein_dir / "receptor_descriptors.json").write_text("{}", encoding="utf-8")
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")
    (ligand_dir / "boxes" / "box0.pdb").write_text("BOX", encoding="utf-8")

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers, "__run_vina", lambda *_a, **_k: ocerror.ErrorCode.SKIP)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)

    rc = ocdock_helpers.__core_run_dock(
        str(protein_dir),
        str(ligand_dir),
        "pdbbind",
        "vina",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.DOCKING_FAILED


@pytest.mark.order(169)
def test_thread_run_dock_parallel_passes_arguments_to_core(monkeypatch, ocdock_helpers):
    captured = {}

    def _fake_core(path, ligand_dir, archive, algo, lock, overwrite, digest_format, all_boxes):
        captured["args"] = (path, ligand_dir, archive, algo, lock, overwrite, digest_format, all_boxes)
        return 123456

    monkeypatch.setattr(ocdock_helpers, "__core_run_dock", _fake_core)

    rc = ocdock_helpers.__thread_run_dock_parallel(
        ("/tmp/p", "/tmp/l", "dudez", "vina", ocdock_helpers._NoOpLock(), True, "json", False)
    )
    assert rc == 123456
    assert captured["args"][0] == "/tmp/p"
    assert captured["args"][1] == "/tmp/l"
    assert captured["args"][2] == "dudez"
    assert captured["args"][3] == "vina"
    assert captured["args"][5] is True
    assert captured["args"][6] == "json"
    assert captured["args"][7] is False


@pytest.mark.order(170)
def test_run_dock_dispatches_parallel_and_sequential_paths(monkeypatch, tmp_path, ocdock_helpers):
    calls = {"parallel": [], "sequential": []}

    def _parallel(paths, archive, dockingAlgorithm, overwrite, digestFormat, desc, all_boxes):
        calls["parallel"].append((paths, archive, dockingAlgorithm, overwrite, digestFormat, desc, all_boxes))
        return 111

    def _sequential(paths, archive, dockingAlgorithm, overwrite, digestFormat, desc, all_boxes):
        calls["sequential"].append((paths, archive, dockingAlgorithm, overwrite, digestFormat, desc, all_boxes))
        return 222

    monkeypatch.setattr(ocdock_helpers.oclogging, "backup_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers, "__run_dock_parallel", _parallel)
    monkeypatch.setattr(ocdock_helpers, "__run_dock_no_parallel", _sequential)

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=True, logdir=str(tmp_path)),
    )
    assert ocdock_helpers.run_dock([("/tmp/p", ["/tmp/l"])], "dudez", "vina", False, "json") == 111
    assert ocdock_helpers.run_dock(("/tmp/p", ["/tmp/l"]), "dudez", "vina", False, "json") == 111
    assert len(calls["parallel"]) == 2

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    assert ocdock_helpers.run_dock([("/tmp/p", ["/tmp/l"])], "dudez", "vina", False, "json") == 222
    assert ocdock_helpers.run_dock(("/tmp/p", ["/tmp/l"]), "dudez", "vina", False, "json") == 222
    assert len(calls["sequential"]) == 2


@pytest.mark.order(171)
def test_run_dock_parallel_uses_noop_lock_and_returns_first_error(monkeypatch, tmp_path, ocdock_helpers):
    class _FakePool:
        captured_args = []

        def __init__(self, cores):
            self.cores = cores

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def imap_unordered(self, fn, arguments):
            _FakePool.captured_args = list(arguments)
            for arg in _FakePool.captured_args:
                yield fn(arg)

    def _fake_core(path, ligand_dir, archive, algo, lock, overwrite, digest_format, all_boxes):
        _ = (path, archive, algo, overwrite, digest_format, all_boxes)
        assert isinstance(lock, ocdock_helpers._NoOpLock)
        if ligand_dir.endswith("lig1"):
            return ocerror.ErrorCode.SKIP
        return ocerror.ErrorCode.OK

    monkeypatch.setattr(ocdock_helpers, "__core_run_dock", _fake_core)
    monkeypatch.setattr(ocdock_helpers, "Pool", _FakePool)
    monkeypatch.setattr(ocdock_helpers, "tqdm", lambda iterable, **_k: iterable)
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=2, multiprocess=True, logdir=str(tmp_path)),
    )

    rc = ocdock_helpers.__run_dock_parallel(
        [("/tmp/protein1", ["/tmp/protein1/lig1", "/tmp/protein1/lig2"])],
        "dudez",
        "vina",
        overwrite=False,
        digestFormat="json",
        desc="dock",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.SKIP
    assert len(_FakePool.captured_args) == 2


@pytest.mark.order(184)
def test_run_dock_no_parallel_returns_first_non_ok_code(monkeypatch, ocdock_helpers):
    responses = [ocerror.ErrorCode.OK, ocerror.ErrorCode.SKIP, ocerror.ErrorCode.DOCKING_FAILED]
    calls = []

    def _fake_core(path, ligand_dir, archive, docking_algorithm, lock, overwrite, digest_format, all_boxes):
        _ = (archive, docking_algorithm, lock, overwrite, digest_format, all_boxes)
        calls.append((path, ligand_dir))
        return responses[len(calls) - 1]

    monkeypatch.setattr(ocdock_helpers, "__core_run_dock", _fake_core)
    monkeypatch.setattr(ocdock_helpers, "tqdm", lambda iterable, **_kwargs: iterable)

    rc = ocdock_helpers.__run_dock_no_parallel(
        [
            ("/tmp/p1", ["/tmp/p1/l1", "/tmp/p1/l2"]),
            ("/tmp/p2", ["/tmp/p2/l1"]),
        ],
        archive="dudez",
        dockingAlgorithm="vina",
        overwrite=False,
        digestFormat="json",
        desc="dock",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.SKIP
    assert len(calls) == 3


@pytest.mark.order(185)
def test_run_dock_no_parallel_returns_ok_when_all_jobs_succeed(monkeypatch, ocdock_helpers):
    monkeypatch.setattr(ocdock_helpers, "__core_run_dock", lambda *_a, **_k: ocerror.ErrorCode.OK)
    monkeypatch.setattr(ocdock_helpers, "tqdm", lambda iterable, **_kwargs: iterable)

    rc = ocdock_helpers.__run_dock_no_parallel(
        [("/tmp/p1", ["/tmp/p1/l1"])],
        archive="dudez",
        dockingAlgorithm="vina",
        overwrite=False,
        digestFormat="json",
        desc="dock",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.OK


@pytest.mark.order(186)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir"),
    [
        ("__run_gnina", "gninaFiles"),
        ("__run_plants", "plantsFiles"),
        ("__run_smina", "sminaFiles"),
        ("__run_vina", "vinaFiles"),
    ],
)
def test_engine_run_reports_missing_engine_folder(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir):
    ligand_dir = tmp_path / "ligand"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    warnings = []

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda msg, path: warnings.append((msg, path)))

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_dir / "ligand.mol2"),
        str(ligand_dir / "ligand_descriptors.json"),
        str(tmp_path / "receptor.pdb"),
        str(tmp_path / "receptor_descriptors.json"),
        str(ligand_dir / "boxes" / "box0.pdb"),
        "ptn1",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.DIR_NOT_EXIST
    assert warnings
    assert engine_dir in warnings[0][0]


@pytest.mark.order(187)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "required_outputs"),
    [
        ("__run_gnina", "gninaFiles", ["gnina_0.log", "gnina_0.pdbqt"]),
        ("__run_plants", "plantsFiles", ["run/ranking.csv"]),
        ("__run_smina", "sminaFiles", ["smina.log", "smina.pdbqt"]),
        ("__run_vina", "vinaFiles", ["vina_0.log", "vina_0.pdbqt"]),
    ],
)
def test_engine_run_skips_when_outputs_already_exist(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, required_outputs):
    ligand_dir = tmp_path / "ligand_ready"
    engine_path = ligand_dir / engine_dir
    boxes_path = ligand_dir / "boxes"
    engine_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)
    (boxes_path / "box0.pdb").write_text("BOX", encoding="utf-8")

    for relpath in required_outputs:
        out_file = engine_path / relpath
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("done", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda msg, path: warnings.append((msg, path)))
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda msg: warnings.append((msg, "")))

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_dir / "ligand.mol2"),
        str(ligand_dir / "ligand_descriptors.json"),
        str(tmp_path / "receptor.pdb"),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnReady",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.OK
    assert warnings


@pytest.mark.order(198)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir"),
    [
        ("__run_gnina", "gninaFiles"),
        ("__run_plants", "plantsFiles"),
        ("__run_smina", "sminaFiles"),
        ("__run_vina", "vinaFiles"),
    ],
)
def test_engine_run_returns_receptor_or_ligand_not_generated(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir):
    ligand_dir = tmp_path / "ligand_missing_obj"
    engine_path = ligand_dir / engine_dir
    boxes_path = ligand_dir / "boxes"
    engine_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)
    (boxes_path / "box0.pdb").write_text("BOX", encoding="utf-8")

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_dir / "ligand.mol2"),
        str(ligand_dir / "ligand_descriptors.json"),
        str(tmp_path / "receptor.pdb"),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnMissingObj",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.RECEPTOR_OR_LIGAND_NOT_GENERATED


@pytest.mark.order(199)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina")),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS")),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina")),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina")),
    ],
)
def test_engine_run_returns_docking_object_not_generated(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr):
    ligand_dir = tmp_path / "ligand_no_dockobj"
    engine_path = ligand_dir / engine_dir
    boxes_path = ligand_dir / "boxes"
    engine_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)
    (boxes_path / "box0.pdb").write_text("BOX", encoding="utf-8")

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], lambda *_a, **_k: None, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_dir / "ligand.mol2"),
        str(ligand_dir / "ligand_descriptors.json"),
        str(tmp_path / "receptor.pdb"),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnNoDockObj",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.DOCKING_OBJECT_NOT_GENERATED


@pytest.mark.order(200)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "required_outputs"),
    [
        ("__run_gnina", "gninaFiles", ["box0/gnina_0.log", "box0/gnina_0.pdbqt", "box1/gnina_0.log", "box1/gnina_0.pdbqt"]),
        ("__run_plants", "plantsFiles", ["box0/run/ranking.csv", "box1/run/ranking.csv"]),
        ("__run_smina", "sminaFiles", ["box0/smina.log", "box0/smina.pdbqt", "box1/smina.log", "box1/smina.pdbqt"]),
        ("__run_vina", "vinaFiles", ["box0/vina_0.log", "box0/vina_0.pdbqt", "box1/vina_0.log", "box1/vina_0.pdbqt"]),
    ],
)
def test_engine_run_all_boxes_uses_box_id_paths(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, required_outputs):
    ligand_dir = tmp_path / "ligand_multibox"
    engine_path = ligand_dir / engine_dir
    boxes_path = ligand_dir / "boxes"
    engine_path.mkdir(parents=True, exist_ok=True)
    boxes_path.mkdir(parents=True, exist_ok=True)
    (boxes_path / "box0.pdb").write_text("BOX0", encoding="utf-8")
    (boxes_path / "box1.pdb").write_text("BOX1", encoding="utf-8")

    for relpath in required_outputs:
        out_file = engine_path / relpath
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("done", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda msg, path: warnings.append((msg, path)))
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda msg: warnings.append((msg, "")))

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_dir / "ligand.mol2"),
        str(ligand_dir / "ligand_descriptors.json"),
        str(tmp_path / "receptor.pdb"),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnMultiBox",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=True,
    )
    assert rc == ocerror.ErrorCode.OK
    assert warnings


@pytest.mark.order(201)
def test_run_dock_parallel_reraises_keyboard_interrupt(monkeypatch, tmp_path, ocdock_helpers):
    class _InterruptPool:
        def __init__(self, _cores):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

        def imap_unordered(self, _fn, _arguments):
            raise KeyboardInterrupt()

    monkeypatch.setattr(ocdock_helpers, "Pool", _InterruptPool)
    monkeypatch.setattr(ocdock_helpers, "tqdm", lambda iterable, **_k: iterable)
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=True, logdir=str(tmp_path)),
    )

    with pytest.raises(KeyboardInterrupt):
        _ = ocdock_helpers.__run_dock_parallel(
            [("/tmp/protein1", ["/tmp/protein1/lig1"])],
            "dudez",
            "vina",
            overwrite=False,
            digestFormat="json",
            desc="dock",
            all_boxes=False,
        )


@pytest.mark.order(240)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr", "run_method"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina"), "run_gnina"),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS"), "run_plants"),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina"), "run_smina"),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina"), "run_vina"),
    ],
)
def test_engine_run_happy_path_executes_prepare_and_run(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr, run_method):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)
    created = []
    digest_calls = []

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        created.append(obj)
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)
    monkeypatch.setattr(factory_module, "generate_digest", lambda *a, **k: digest_calls.append((a, k)) or ocerror.ErrorCode.OK, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnHappy",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )

    assert rc == ocerror.ErrorCode.OK
    assert created
    assert "run_prepare_ligand" in created[0].events
    assert "run_prepare_receptor" in created[0].events
    assert run_method in created[0].events
    if runner_name != "__run_plants":
        assert digest_calls


@pytest.mark.order(241)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina")),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS")),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina")),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina")),
    ],
)
def test_engine_run_returns_ligand_not_prepared_when_prepare_ligand_fails(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        obj.run_prepare_ligand = lambda **_k: (1, "prep failure")  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnLigPrepFail",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.LIGAND_NOT_PREPARED


@pytest.mark.order(242)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina")),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS")),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina")),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina")),
    ],
)
def test_engine_run_propagates_nonzero_engine_exit_code(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        obj.run_gnina = lambda **_k: (9, "gnina failed")  # type: ignore[method-assign]
        obj.run_plants = lambda **_k: (9, "plants failed")  # type: ignore[method-assign]
        obj.run_smina = lambda **_k: (9, "smina failed")  # type: ignore[method-assign]
        obj.run_vina = lambda **_k: (9, "vina failed")  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnRunFail",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == 9


@pytest.mark.order(367)
def test_core_run_dock_uses_smi_path_for_dudez_archive(monkeypatch, tmp_path, ocdock_helpers):
    receptor_dir = tmp_path / "proteinA"
    receptor_dir.mkdir()
    ligand_dir = receptor_dir / "compounds" / "ligandA"
    (ligand_dir / "boxes").mkdir(parents=True)

    receptor_desc = receptor_dir / "receptor_descriptors.json"
    ligand_desc = ligand_dir / "ligand_descriptors.json"
    box0 = ligand_dir / "boxes" / "box0.pdb"
    receptor_desc.write_text("{}", encoding="utf-8")
    ligand_desc.write_text("{}", encoding="utf-8")
    box0.write_text("BOX", encoding="utf-8")

    captured = {}

    def _fake_vina(ligand_path, *_args, **_kwargs):
        captured["ligand_path"] = ligand_path
        return ocerror.ErrorCode.OK

    monkeypatch.setattr(ocdock_helpers, "__run_vina", _fake_vina)
    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocdock_helpers.__core_run_dock(
        str(receptor_dir),
        str(ligand_dir),
        "dudez",
        "vina",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.OK
    assert captured["ligand_path"].endswith("/ligand.smi")


@pytest.mark.order(368)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina")),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS")),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina")),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina")),
    ],
)
def test_engine_run_returns_ligand_not_prepared_when_prepare_ligand_returns_nonzero_int(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        obj.run_prepare_ligand = lambda **_k: 1  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnLigPrepFailInt",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.LIGAND_NOT_PREPARED


@pytest.mark.order(369)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr", "expected_rc"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina"), ocerror.ErrorCode.RECEPTOR_NOT_PREPARED),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS"), ocerror.ErrorCode.LIGAND_NOT_PREPARED),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina"), ocerror.ErrorCode.LIGAND_NOT_PREPARED),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina"), ocerror.ErrorCode.RECEPTOR_NOT_PREPARED),
    ],
)
def test_engine_run_returns_expected_error_when_prepare_receptor_returns_nonzero_int(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr, expected_rc):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        obj.run_prepare_receptor = lambda **_k: 1  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnRecPrepFailInt",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == expected_rc


@pytest.mark.order(370)
@pytest.mark.parametrize(
    ("runner_name", "engine_dir", "factory_attr"),
    [
        ("__run_gnina", "gninaFiles", ("ocgnina", "Gnina")),
        ("__run_plants", "plantsFiles", ("ocplants", "PLANTS")),
        ("__run_smina", "sminaFiles", ("ocsmina", "Smina")),
        ("__run_vina", "vinaFiles", ("ocvina", "Vina")),
    ],
)
def test_engine_run_propagates_nonzero_engine_exit_without_stderr(monkeypatch, tmp_path, ocdock_helpers, runner_name, engine_dir, factory_attr):
    ligand_dir, boxes_path, receptor_path, ligand_path = _prepare_engine_run_environment(tmp_path, engine_dir)

    def _factory(*args, **kwargs):
        _ = kwargs
        obj = _FakeEngineObject(
            prepared_receptor=args[3],
            prepared_ligand=args[5],
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )
        obj.run_gnina = lambda **_k: 9  # type: ignore[method-assign]
        obj.run_plants = lambda **_k: 9  # type: ignore[method-assign]
        obj.run_smina = lambda **_k: 9  # type: ignore[method-assign]
        obj.run_vina = lambda **_k: 9  # type: ignore[method-assign]
        return obj

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())

    factory_module = getattr(ocdock_helpers, factory_attr[0])
    monkeypatch.setattr(factory_module, factory_attr[1], _factory, raising=False)

    runner = getattr(ocdock_helpers, runner_name)
    rc = runner(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnRunFailNoStderr",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=True,
        digestFormat="json",
        all_boxes=False,
    )
    assert rc == 9


@pytest.mark.order(393)
def test_run_vina_mixed_box_outputs_warns_for_existing_and_runs_missing_box(monkeypatch, tmp_path, ocdock_helpers):
    ligand_dir = tmp_path / "ligand_mixed_boxes"
    boxes_path = ligand_dir / "boxes"
    vina_dir = ligand_dir / "vinaFiles"
    box0_dir = vina_dir / "box0"
    box1_dir = vina_dir / "box1"
    boxes_path.mkdir(parents=True, exist_ok=True)
    box0_dir.mkdir(parents=True, exist_ok=True)
    box1_dir.mkdir(parents=True, exist_ok=True)

    receptor_path = tmp_path / "receptor.pdb"
    ligand_path = ligand_dir / "ligand.mol2"
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (boxes_path / "box0.pdb").write_text("BOX0", encoding="utf-8")
    (boxes_path / "box1.pdb").write_text("BOX1", encoding="utf-8")

    # Pre-create only box0 outputs so box0 is skipped and box1 is executed.
    (box0_dir / "vina_0.log").write_text("done", encoding="utf-8")
    (box0_dir / "vina_0.pdbqt").write_text("done", encoding="utf-8")

    warnings = []
    digests = []

    def _factory(*args, **kwargs):
        _ = kwargs
        prepared_receptor = args[3]
        prepared_ligand = args[5]
        Path(prepared_receptor).parent.mkdir(parents=True, exist_ok=True)
        Path(prepared_receptor).write_text("PREP_REC", encoding="utf-8")
        Path(prepared_ligand).parent.mkdir(parents=True, exist_ok=True)
        Path(prepared_ligand).write_text("PREP_LIG", encoding="utf-8")
        return _FakeEngineObject(
            prepared_receptor=prepared_receptor,
            prepared_ligand=prepared_ligand,
            input_receptor_path=str(receptor_path),
            input_ligand_path=str(ligand_path),
            log_path=args[6],
            output_target=args[7],
        )

    monkeypatch.setattr(
        ocdock_helpers,
        "get_config",
        lambda: SimpleNamespace(available_cores=1, multiprocess=False, logdir=str(tmp_path)),
    )
    monkeypatch.setattr(ocdock_helpers.ocr, "Receptor", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocl, "Ligand", lambda *_a, **_k: object(), raising=False)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_error_log", lambda *_a, **_k: None)
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning_log", lambda msg, path: warnings.append((msg, path)))
    monkeypatch.setattr(ocdock_helpers.ocprint, "print_warning", lambda msg: warnings.append((msg, "")))
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers.ocvalidation, "is_molecule_valid_with_retry", lambda *_a, **_k: True)
    monkeypatch.setattr(ocdock_helpers, "__needs_receptor_preparation", lambda *_a, **_k: False)
    monkeypatch.setattr(ocdock_helpers, "__receptor_file_lock", lambda *_a, **_k: nullcontext())
    monkeypatch.setattr(ocdock_helpers.ocvina, "Vina", _factory, raising=False)
    monkeypatch.setattr(
        ocdock_helpers.ocvina,
        "generate_digest",
        lambda *a, **k: digests.append((a, k)) or ocerror.ErrorCode.OK,
        raising=False,
    )

    rc = ocdock_helpers.__run_vina(
        str(ligand_path),
        str(ligand_dir / "ligand_descriptors.json"),
        str(receptor_path),
        str(tmp_path / "receptor_descriptors.json"),
        str(boxes_path / "box0.pdb"),
        "ptnMixed",
        "dudez",
        ocdock_helpers._NoOpLock(),
        overwrite=False,
        digestFormat="json",
        all_boxes=True,
    )

    assert rc == ocerror.ErrorCode.OK
    assert warnings
    assert any("already generated" in msg for msg, _path in warnings)
    assert digests
    assert digests[0][1]["box_id"] == "box1"
