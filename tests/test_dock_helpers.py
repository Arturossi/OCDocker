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

    path = Path(__file__).resolve().parents[1] / "OCDocker" / "Processing" / "Dock.py"
    spec = util.spec_from_file_location("ocdock_helpers_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
