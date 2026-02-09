#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for Rescoring.ODDT helper functions.
'''

# Imports
###############################################################################
import importlib
import importlib.util as util
import sys
import types

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
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


class _DummyVS:
    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)


# Functions
###############################################################################
## Private ##

def _import_oddt_helpers(monkeypatch):
    importlib.import_module("OCDocker.Rescoring")
    importlib.import_module("OCDocker.Toolbox")

    oddt_mod = types.ModuleType("oddt")
    oddt_mod.toolkit = types.SimpleNamespace(readfile=lambda *_a, **_k: iter([object()]))  # type: ignore[attr-defined]

    oddt_scoring_mod = types.ModuleType("oddt.scoring")
    oddt_scoring_mod.scorer = object  # type: ignore[attr-defined]

    oddt_vs_mod = types.ModuleType("oddt.virtualscreening")
    oddt_vs_mod.virtualscreening = _DummyVS  # type: ignore[attr-defined]

    ligand_mod = types.ModuleType("OCDocker.Ligand")
    receptor_mod = types.ModuleType("OCDocker.Receptor")

    filesfolders_mod = types.ModuleType("OCDocker.Toolbox.FilesFolders")
    filesfolders_mod.safe_create_dir = lambda *_a, **_k: ocerror.ErrorCode.OK  # type: ignore[attr-defined]
    filesfolders_mod.safe_remove_file = lambda *_a, **_k: ocerror.ErrorCode.OK  # type: ignore[attr-defined]

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.print_error = lambda *_a, **_k: None  # type: ignore[attr-defined]
    printing_mod.print_warning = lambda *_a, **_k: None  # type: ignore[attr-defined]
    printing_mod.print_info = lambda *_a, **_k: None  # type: ignore[attr-defined]

    running_mod = types.ModuleType("OCDocker.Toolbox.Running")
    running_mod.run_command = lambda *_a, **_k: 0  # type: ignore[attr-defined]

    config_mod = types.ModuleType("OCDocker.Config")
    config_mod.get_config = lambda: SimpleNamespace(
        oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=[]),
        oddt_models_dir="/tmp",
    )  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "oddt", oddt_mod)
    monkeypatch.setitem(sys.modules, "oddt.scoring", oddt_scoring_mod)
    monkeypatch.setitem(sys.modules, "oddt.virtualscreening", oddt_vs_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Ligand", ligand_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Receptor", receptor_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.FilesFolders", filesfolders_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Printing", printing_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Toolbox.Running", running_mod)
    monkeypatch.setitem(sys.modules, "OCDocker.Config", config_mod)

    path = Path(__file__).resolve().parents[1] / "OCDocker" / "Rescoring" / "ODDT.py"
    spec = util.spec_from_file_location("ocoddt_helpers_module", path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## Public ##

@pytest.fixture
def ocoddt_helpers(monkeypatch):
    return _import_oddt_helpers(monkeypatch)


@pytest.mark.order(156)
def test_build_cmd_rejects_non_csv_output(ocoddt_helpers):
    rc = ocoddt_helpers.__build_cmd("receptor.pdbqt", "ligand.sdf", "output.txt")
    assert rc == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(157)
def test_build_cmd_adds_scoring_functions(monkeypatch, ocoddt_helpers):
    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore", "nnscore"]),
            oddt_models_dir="/tmp",
        ),
    )

    cmd = ocoddt_helpers.__build_cmd("receptor.pdbqt", "ligand.sdf", "output.csv")
    assert cmd[0] == "oddt_cli"
    assert "-i" in cmd and "sdf" in cmd
    assert cmd.count("--score") == 2
    assert "rfscore" in cmd
    assert "nnscore" in cmd


@pytest.mark.order(158)
def test_build_cmd_logs_error_when_scoring_functions_missing(monkeypatch, ocoddt_helpers):
    errors = []
    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=[]),
            oddt_models_dir="/tmp",
        ),
    )
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))

    cmd = ocoddt_helpers.__build_cmd("receptor.pdbqt", "ligand.sdf", "output.csv")
    assert isinstance(cmd, list)
    assert "--score" not in cmd
    assert errors


@pytest.mark.order(159)
def test_read_receptor_with_retry_success(monkeypatch, ocoddt_helpers):
    marker = object()
    monkeypatch.setattr(
        ocoddt_helpers.od.toolkit,
        "readfile",
        lambda *_a, **_k: iter([marker]),
    )
    receptor, err = ocoddt_helpers.__read_receptor_with_retry("pdbqt", "/tmp/r.pdbqt", retries=2, delay=0)
    assert receptor is marker
    assert err is None


@pytest.mark.order(160)
def test_read_receptor_with_retry_exhausts_attempts(monkeypatch, ocoddt_helpers):
    sleeps = []
    monkeypatch.setattr(ocoddt_helpers.time, "sleep", lambda sec: sleeps.append(sec))
    monkeypatch.setattr(
        ocoddt_helpers.od.toolkit,
        "readfile",
        lambda *_a, **_k: iter([]),
    )
    receptor, err = ocoddt_helpers.__read_receptor_with_retry("pdbqt", "/tmp/r.pdbqt", retries=3, delay=0.25)
    assert receptor is None
    assert isinstance(err, StopIteration)
    assert sleeps == [0.25, 0.25]


@pytest.mark.order(161)
def test_df_to_dict_happy_path_and_wrong_type(ocoddt_helpers):
    df = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["pose1"])
    out = ocoddt_helpers.df_to_dict(df)
    assert out == {"pose1": {"a": 1.0, "b": 2.0}}

    rc = ocoddt_helpers.df_to_dict(["not", "a", "dataframe"])  # type: ignore[arg-type]
    assert rc == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(162)
def test_get_models_only_returns_pickle_files(tmp_path, ocoddt_helpers):
    (tmp_path / "model_a.pickle").write_text("x", encoding="utf-8")
    (tmp_path / "model_b.pkl").write_text("x", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    models = ocoddt_helpers.get_models(str(tmp_path))
    assert models == [str(tmp_path / "model_a.pickle")]


@pytest.mark.order(163)
def test_read_log_existing_and_missing_file(tmp_path, ocoddt_helpers):
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame({"A": [1], "B": [2]}).to_csv(csv_path, index=False)
    data = ocoddt_helpers.read_log(str(csv_path))
    assert isinstance(data, pd.DataFrame)
    assert list(data.columns) == ["A", "B"]

    missing = ocoddt_helpers.read_log(str(tmp_path / "missing.csv"))
    assert missing is None
