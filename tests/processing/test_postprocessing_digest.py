#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Processing.Postprocessing.Digest.
'''

# Imports
###############################################################################
import builtins
import importlib
import importlib.util as util
import json
import os
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


class _FakePool:
    def __init__(self, _workers, codes=None, raise_on_enter=False):
        self._codes = list(codes or [])
        self._raise_on_enter = raise_on_enter

    def __enter__(self):
        if self._raise_on_enter:
            raise IOError("pool failed")
        return self

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False

    def imap_unordered(self, fn, arguments, chunksize=1):
        _ = fn
        _ = arguments
        _ = chunksize
        return iter(self._codes)


# Functions
###############################################################################
## Private ##

def _import_digest(monkeypatch):
    importlib.import_module("OCDocker.Docking")
    importlib.import_module("OCDocker.Docking.Future")
    importlib.import_module("OCDocker.Toolbox")

    # Lightweight stubs for dependencies used by Digest.py
    gnina = types.ModuleType("OCDocker.Docking.Gnina")
    gnina.read_log = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    gnina.generate_digest = lambda *a, **k: 0  # type: ignore[attr-defined]
    plants = types.ModuleType("OCDocker.Docking.PLANTS")
    plants.read_log = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    plants.generate_digest = lambda *a, **k: 0  # type: ignore[attr-defined]
    smina = types.ModuleType("OCDocker.Docking.Smina")
    smina.read_log = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    smina.generate_digest = lambda *a, **k: 0  # type: ignore[attr-defined]
    vina = types.ModuleType("OCDocker.Docking.Vina")
    vina.read_log = lambda *_a, **_k: {}  # type: ignore[attr-defined]
    vina.generate_digest = lambda *a, **k: 0  # type: ignore[attr-defined]

    basetools = types.ModuleType("OCDocker.Toolbox.Basetools")
    basetools.redirect_to_tqdm = lambda: nullcontext()  # type: ignore[attr-defined]
    logging_mod = types.ModuleType("OCDocker.Toolbox.Logging")
    logging_mod.backup_log = lambda *a, **k: None  # type: ignore[attr-defined]
    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.print_error_log = lambda *a, **k: None  # type: ignore[attr-defined]

    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: SimpleNamespace(multiprocess=False, available_cores=1, logdir="/tmp")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Gnina", gnina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.PLANTS", plants)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Smina", smina)
    monkeypatch.setitem(sys.modules, "OCDocker.Docking.Vina", vina)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Basetools", basetools)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Logging", logging_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Config", config_mod)

    path = Path(__file__).resolve().parents[2] / "OCDocker" / "Processing" / "Postprocessing" / "Digest.py"
    spec = util.spec_from_file_location("ocdigest_coverage_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocdigest(monkeypatch):
    return _import_digest(monkeypatch)


@pytest.mark.order(123)
def test_resolve_smina_log_prefers_primary_file(tmp_path, ocdigest):
    run_dir = tmp_path / "sminaFiles"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smina.log").write_text("ok", encoding="utf-8")
    (run_dir / "smina_0.log").write_text("fallback", encoding="utf-8")

    out = ocdigest._resolve_smina_log(str(run_dir))
    assert out.endswith("smina.log")


@pytest.mark.order(124)
def test_resolve_smina_log_falls_back_to_previous_name(tmp_path, ocdigest):
    run_dir = tmp_path / "sminaFiles"
    run_dir.mkdir(parents=True, exist_ok=True)

    out = ocdigest._resolve_smina_log(str(run_dir))
    assert out.endswith("smina_0.log")


@pytest.mark.order(125)
def test_core_generate_digest_skips_unallowed_dir(tmp_path, ocdigest):
    rc = ocdigest.__core_generate_digest(str(tmp_path / "index"), str(tmp_path), "dudez", overwrite=False)
    assert rc == ocerror.ErrorCode.UNALLOWED_DIR


@pytest.mark.order(126)
def test_core_generate_digest_missing_descriptors_logs_error(tmp_path, monkeypatch, ocdigest):
    ligand_dir = tmp_path / "ligandA"
    ligand_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(logdir=str(tmp_path / "logs")))
    monkeypatch.setattr(ocdigest.ocprint, "print_error_log", lambda msg, path: logs.append((msg, path)))

    rc = ocdigest.__core_generate_digest(str(tmp_path / "ptn"), str(ligand_dir), "dudez", overwrite=False)
    assert rc == ocerror.ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST
    assert logs
    assert "ligand descriptor" in logs[0][0].lower()


@pytest.mark.order(127)
def test_core_generate_digest_single_box_calls_all_engines(tmp_path, monkeypatch, ocdigest):
    ligand_dir = tmp_path / "ligandA"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(ocdigest.ocgnina, "read_log", lambda path, onlyBest=False: calls.append(("gnina", path, onlyBest)) or {"1": {"GNINA_AFFINITY": -7.0}})
    monkeypatch.setattr(ocdigest.ocvina, "read_log", lambda path, onlyBest=False: calls.append(("vina", path, onlyBest)) or {"1": {"VINA_AFFINITY": -6.5}})
    monkeypatch.setattr(ocdigest.ocsmina, "read_log", lambda path, onlyBest=False: calls.append(("smina", path, onlyBest)) or {"1": {"SMINA_AFFINITY": -6.2}})
    monkeypatch.setattr(ocdigest.ocplants, "read_log", lambda path, onlyBest=False: calls.append(("plants", path, onlyBest)) or {"1": {"PLANTS_TOTAL_SCORE": -55.0}})
    monkeypatch.setattr(ocdigest, "_resolve_smina_log", lambda run_dir: f"{run_dir}/resolved.log")

    rc = ocdigest.__core_generate_digest(str(tmp_path / "ptn"), str(ligand_dir), "pdbbind", overwrite=True, digestFormat="json")
    assert rc == ocerror.ErrorCode.OK
    assert [c[0] for c in calls] == ["gnina", "vina", "smina", "plants"]

    digest_file = ligand_dir / "dockingDigest.json"
    assert digest_file.exists()
    digest_data = json.loads(digest_file.read_text(encoding="utf-8"))
    assert digest_data["1"]["GNINA_AFFINITY"] == -7.0
    assert digest_data["1"]["VINA_AFFINITY"] == -6.5
    assert digest_data["1"]["SMINA_AFFINITY"] == -6.2
    assert digest_data["1"]["PLANTS_TOTAL_SCORE"] == -55.0


@pytest.mark.order(128)
def test_core_generate_digest_all_boxes_calls_per_box(tmp_path, monkeypatch, ocdigest):
    ligand_dir = tmp_path / "ligandB"
    boxes_dir = ligand_dir / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")
    (boxes_dir / "box0.pdb").write_text("box0", encoding="utf-8")
    (boxes_dir / "box1.pdb").write_text("box1", encoding="utf-8")

    call_paths = []
    monkeypatch.setattr(ocdigest.ocgnina, "read_log", lambda path, onlyBest=False: call_paths.append(("gnina", path, onlyBest)) or {"1": {"GNINA_AFFINITY": -7.0}})
    monkeypatch.setattr(ocdigest.ocvina, "read_log", lambda path, onlyBest=False: call_paths.append(("vina", path, onlyBest)) or {"1": {"VINA_AFFINITY": -6.5}})
    monkeypatch.setattr(ocdigest.ocsmina, "read_log", lambda path, onlyBest=False: call_paths.append(("smina", path, onlyBest)) or {"1": {"SMINA_AFFINITY": -6.2}})
    monkeypatch.setattr(ocdigest.ocplants, "read_log", lambda path, onlyBest=False: call_paths.append(("plants", path, onlyBest)) or {"1": {"PLANTS_TOTAL_SCORE": -55.0}})
    monkeypatch.setattr(ocdigest, "_resolve_smina_log", lambda run_dir: os.path.join(run_dir, "resolved.log"))

    rc = ocdigest.__core_generate_digest(str(tmp_path / "ptn"), str(ligand_dir), "dudez", overwrite=False, all_boxes=True)
    assert rc == ocerror.ErrorCode.OK
    assert len(call_paths) == 8
    assert sum(1 for _, path, _ in call_paths if "/box0/" in path or "\\box0\\" in path) == 4
    assert sum(1 for _, path, _ in call_paths if "/box1/" in path or "\\box1\\" in path) == 4

    digest_file = ligand_dir / "dockingDigest.json"
    digest_data = json.loads(digest_file.read_text(encoding="utf-8"))
    assert digest_data["box0"]["1"]["GNINA_AFFINITY"] == -7.0
    assert digest_data["box0"]["1"]["PLANTS_TOTAL_SCORE"] == -55.0
    assert digest_data["box1"]["1"]["VINA_AFFINITY"] == -6.5
    assert digest_data["box1"]["1"]["SMINA_AFFINITY"] == -6.2


@pytest.mark.order(175)
def test_core_generate_digest_writes_once_per_ligand(tmp_path, monkeypatch, ocdigest):
    ligand_dir = tmp_path / "ligandC"
    boxes_dir = ligand_dir / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    (ligand_dir / "ligand_descriptors.json").write_text("{}", encoding="utf-8")
    (boxes_dir / "box0.pdb").write_text("box0", encoding="utf-8")
    (boxes_dir / "box1.pdb").write_text("box1", encoding="utf-8")

    monkeypatch.setattr(ocdigest.ocgnina, "read_log", lambda *_a, **_k: {"1": {"GNINA_AFFINITY": -7.0}})
    monkeypatch.setattr(ocdigest.ocvina, "read_log", lambda *_a, **_k: {"1": {"VINA_AFFINITY": -6.5}})
    monkeypatch.setattr(ocdigest.ocsmina, "read_log", lambda *_a, **_k: {"1": {"SMINA_AFFINITY": -6.2}})
    monkeypatch.setattr(ocdigest.ocplants, "read_log", lambda *_a, **_k: {"1": {"PLANTS_TOTAL_SCORE": -55.0}})
    monkeypatch.setattr(ocdigest, "_resolve_smina_log", lambda run_dir: os.path.join(run_dir, "resolved.log"))

    digest_file = ligand_dir / "dockingDigest.json"
    original_open = builtins.open
    writes = {"count": 0}

    def _count_writes(path, mode="r", *args, **kwargs):
        if str(path) == str(digest_file) and "w" in mode:
            writes["count"] += 1
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _count_writes)

    rc = ocdigest.__core_generate_digest(str(tmp_path / "ptn"), str(ligand_dir), "dudez", overwrite=False, all_boxes=True)
    assert rc == ocerror.ErrorCode.OK
    assert writes["count"] == 1


@pytest.mark.order(129)
def test_generate_digest_no_parallel_returns_first_error(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest.ocbasetools, "redirect_to_tqdm", lambda: nullcontext())
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)
    sequence = iter([ocerror.ErrorCode.OK, ocerror.ErrorCode.DOCKING_FAILED, ocerror.ErrorCode.OK])
    monkeypatch.setattr(ocdigest, "__core_generate_digest", lambda *a, **k: next(sequence))

    rc = ocdigest.__generate_digest_no_parallel(
        complexList=[("p1", ["l1", "l2"]), ("p2", ["l3"])],
        archive="pdbbind",
        overwrite=False,
        digestFormat="json",
        desc="x",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.DOCKING_FAILED


@pytest.mark.order(130)
def test_generate_digest_single_and_thread_paths(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest.ocbasetools, "redirect_to_tqdm", lambda: nullcontext())
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)

    monkeypatch.setattr(ocdigest, "__core_generate_digest", lambda *a, **k: ocerror.ErrorCode.DOCKING_FAILED)
    rc_single = ocdigest.__generate_digest_single(("p1", ["l1"]), "dudez", False, "json", "x", False)
    assert rc_single == ocerror.ErrorCode.DOCKING_FAILED

    rc_thread = ocdigest.__thread_generate_digest(("p1", "l1", "dudez", False, "json", False))
    assert rc_thread == ocerror.ErrorCode.DOCKING_FAILED


@pytest.mark.order(131)
def test_generate_digest_parallel_collects_errors(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(available_cores=2, logdir="/tmp"))
    monkeypatch.setattr(ocdigest, "Pool", lambda workers: _FakePool(workers, codes=[ocerror.ErrorCode.OK, ocerror.ErrorCode.DOCKING_FAILED]))
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)

    rc = ocdigest.__generate_digest_parallel(
        complexList=[("p1", ["l1", "l2"])],
        archive="dudez",
        overwrite=False,
        digestFormat="json",
        desc="x",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.DOCKING_FAILED


@pytest.mark.order(132)
def test_generate_digest_parallel_ioerror_logs(monkeypatch, tmp_path, ocdigest):
    logs = []
    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(available_cores=2, logdir=str(tmp_path)))
    monkeypatch.setattr(ocdigest, "Pool", lambda workers: _FakePool(workers, raise_on_enter=True))
    monkeypatch.setattr(ocdigest.ocprint, "print_error_log", lambda msg, path: logs.append((msg, path)))

    rc = ocdigest.__generate_digest_parallel(
        complexList=[("p1", ["l1"])],
        archive="dudez",
        overwrite=False,
        digestFormat="json",
        desc="x",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.DOCKING_FAILED
    assert logs


@pytest.mark.order(133)
def test_public_generate_digest_dispatches(monkeypatch, ocdigest):
    called = {"backup": 0, "parallel": 0, "serial": 0, "single": 0}
    monkeypatch.setattr(ocdigest.oclogging, "backup_log", lambda *_a, **_k: called.__setitem__("backup", called["backup"] + 1))
    monkeypatch.setattr(ocdigest, "__generate_digest_parallel", lambda *_a, **_k: called.__setitem__("parallel", called["parallel"] + 1))
    monkeypatch.setattr(ocdigest, "__generate_digest_no_parallel", lambda *_a, **_k: called.__setitem__("serial", called["serial"] + 1))
    monkeypatch.setattr(ocdigest, "__generate_digest_single", lambda *_a, **_k: called.__setitem__("single", called["single"] + 1))

    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(multiprocess=True))
    ocdigest.generate_digest([("p", ["l"])], "dudez", overwrite=False)

    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(multiprocess=False))
    ocdigest.generate_digest([("p", ["l"])], "dudez", overwrite=False)

    ocdigest.generate_digest(("p", ["l"]), "dudez", overwrite=False)

    assert called["backup"] == 2
    assert called["parallel"] == 1
    assert called["serial"] == 1
    assert called["single"] == 1


@pytest.mark.order(172)
def test_generate_digest_no_parallel_returns_ok_when_all_succeed(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest.ocbasetools, "redirect_to_tqdm", lambda: nullcontext())
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(ocdigest, "__core_generate_digest", lambda *a, **k: ocerror.ErrorCode.OK)

    rc = ocdigest.__generate_digest_no_parallel(
        complexList=[("p1", ["l1", "l2"])],
        archive="pdbbind",
        overwrite=False,
        digestFormat="json",
        desc="x",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.OK


@pytest.mark.order(173)
def test_generate_digest_parallel_returns_ok_when_all_succeed(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest, "get_config", lambda: SimpleNamespace(available_cores=2, logdir="/tmp"))
    monkeypatch.setattr(ocdigest, "Pool", lambda workers: _FakePool(workers, codes=[ocerror.ErrorCode.OK, ocerror.ErrorCode.OK]))
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)

    rc = ocdigest.__generate_digest_parallel(
        complexList=[("p1", ["l1", "l2"])],
        archive="dudez",
        overwrite=False,
        digestFormat="json",
        desc="x",
        all_boxes=False,
    )
    assert rc == ocerror.ErrorCode.OK


@pytest.mark.order(174)
def test_generate_digest_single_returns_ok_when_all_succeed(monkeypatch, ocdigest):
    monkeypatch.setattr(ocdigest, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(ocdigest, "__core_generate_digest", lambda *a, **k: ocerror.ErrorCode.OK)
    rc = ocdigest.__generate_digest_single(("p1", ["l1", "l2"]), "dudez", False, "json", "x", False)
    assert rc == ocerror.ErrorCode.OK
