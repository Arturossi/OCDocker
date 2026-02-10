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


class _FakeMol:
    def __init__(self, title, payload):
        self.title = title
        self.data = SimpleNamespace(to_dict=lambda: dict(payload))


class _ReceptorType:
    def __init__(self, path):
        self.path = path


class _LigandType:
    def __init__(self, path, name="ligand"):
        self.path = path
        self.name = name


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
    ligand_mod.Ligand = _LigandType  # type: ignore[attr-defined]
    receptor_mod.Receptor = _ReceptorType  # type: ignore[attr-defined]

    filesfolders_mod = types.ModuleType("OCDocker.Toolbox.FilesFolders")
    filesfolders_mod.safe_create_dir = lambda *_a, **_k: ocerror.ErrorCode.OK  # type: ignore[attr-defined]
    filesfolders_mod.safe_remove_file = lambda *_a, **_k: ocerror.ErrorCode.OK  # type: ignore[attr-defined]

    printing_mod = types.ModuleType("OCDocker.Toolbox.Printing")
    printing_mod.print_error = lambda *_a, **_k: None  # type: ignore[attr-defined]
    printing_mod.print_warning = lambda *_a, **_k: None  # type: ignore[attr-defined]
    printing_mod.print_info = lambda *_a, **_k: None  # type: ignore[attr-defined]

    running_mod = types.ModuleType("OCDocker.Toolbox.Running")
    running_mod.run_command = lambda *_a, **_k: 0  # type: ignore[attr-defined]
    running_mod.run = lambda *_a, **_k: 0  # type: ignore[attr-defined]

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


def _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path):
    output_dir = tmp_path / "output"
    models_dir = tmp_path / "models"
    receptor_path = tmp_path / "prepared_receptor.pdbqt"
    ligand_path = tmp_path / "prepared_ligand.sdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("RECEPTOR", encoding="utf-8")
    ligand_path.write_text("LIGAND", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    return receptor_path, ligand_path, output_dir, models_dir


def _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers):
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([SimpleNamespace(atoms=[1])]))


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


@pytest.mark.order(172)
def test_run_oddt_returns_dir_not_exist_when_output_path_cannot_be_created(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "missing_output_dir"
    original_isdir = ocoddt_helpers.os.path.isdir
    monkeypatch.setattr(
        ocoddt_helpers.os.path,
        "isdir",
        lambda p: False if str(p) == str(output_dir) else original_isdir(p),
    )
    rc = ocoddt_helpers.run_oddt(
        "receptor.pdbqt",
        "ligand.sdf",
        "lig",
        str(output_dir),
        returnData=True,
    )
    assert rc == ocerror.ErrorCode.DIR_NOT_EXIST


@pytest.mark.order(173)
def test_run_oddt_returns_missing_models_when_models_folder_is_empty(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "output"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "prepared_receptor.pdbqt"
    ligand_path = tmp_path / "prepared_ligand.sdf"
    receptor_path.write_text("RECEPTOR", encoding="utf-8")
    ligand_path.write_text("LIGAND", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "lig",
        str(output_dir),
        returnData=True,
    )
    assert rc == ocerror.ErrorCode.MISSING_ODDT_MODELS


@pytest.mark.order(174)
def test_run_oddt_existing_output_without_overwrite_returns_file_exists(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    (output_dir / "ligA.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligA",
        str(output_dir),
        returnData=False,
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.FILE_EXISTS


@pytest.mark.order(175)
def test_run_oddt_existing_output_with_corrupted_csv_returns_error(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    (output_dir / "ligB.csv").write_text("bad", encoding="utf-8")
    monkeypatch.setattr(ocoddt_helpers.pd, "read_csv", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad csv")))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligB",
        str(output_dir),
        returnData=True,
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.CORRUPTED_FILE


@pytest.mark.order(176)
def test_run_oddt_returns_rescoring_failed_when_receptor_read_retries_exhaust(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    monkeypatch.setattr(
        ocoddt_helpers,
        "__read_receptor_with_retry",
        lambda *_a, **_k: (None, RuntimeError("parse failure")),
    )

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligC",
        str(output_dir),
        returnData=True,
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(177)
def test_run_oddt_reports_missing_ligands_before_pipeline_scoring(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, _ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    missing_ligand = tmp_path / "missing_ligand.sdf"
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(missing_ligand),
        "ligD",
        str(output_dir),
        returnData=True,
        overwrite=False,
    )
    assert rc == ocerror.ErrorCode.FILE_NOT_EXIST


@pytest.mark.order(178)
def test_run_oddt_returns_rescoring_failed_when_ligand_parser_returns_none(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([None]))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligE",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(179)
def test_run_oddt_returns_rescoring_failed_when_ligand_has_no_atoms(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([SimpleNamespace(atoms=[])]))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligF",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(180)
def test_run_oddt_returns_rescoring_failed_when_ligand_file_is_empty(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([]))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligG",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(181)
def test_run_oddt_returns_rescoring_failed_when_ligand_parser_raises(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))

    def _raise_on_read(*_a, **_k):
        raise RuntimeError("ligand parser crash")

    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", _raise_on_read)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligH",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(188)
def test_run_oddt_returns_rescoring_failed_when_no_scoring_model_loads(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _PipelineWithLigandLoader:
        def load_ligands(self, *_a, **_k):
            return None

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _PipelineWithLigandLoader())
    monkeypatch.setattr(
        ocoddt_helpers,
        "scorer",
        SimpleNamespace(load=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad model"))),
    )

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligI",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(189)
def test_run_oddt_uses_individual_fallback_and_cleans_models(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _Pipeline:
        def __init__(self, molecules):
            self._molecules = molecules

        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter(self._molecules)

    pipelines = [
        _Pipeline([]),
        _Pipeline([_FakeMol("ligJ.sdf", {"rfscore_value": -7.1, "vina_aux": 0.0})]),
    ]
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: pipelines.pop(0))
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    removable = [output_dir / "tmp_model1.pickle", output_dir / "tmp_model2.pickle"]
    for file in removable:
        file.write_text("x", encoding="utf-8")

    removed = []
    monkeypatch.setattr(ocoddt_helpers.ocff, "safe_remove_file", lambda path: removed.append(path) or ocerror.ErrorCode.OK)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligJ",
        str(output_dir),
        returnData=False,
        overwrite=True,
        cleanModels=True,
    )
    assert rc == ocerror.ErrorCode.OK
    assert sorted(removed) == sorted([str(removable[0]), str(removable[1])])


@pytest.mark.order(190)
def test_run_oddt_reports_missing_expected_scoring_family(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore", "nnscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    class _SinglePassPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter([_FakeMol("ligK.sdf", {"rfscore_metric": 1.5})])

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _SinglePassPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    errors = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligK",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert isinstance(result, pd.DataFrame)
    assert "rfscore_metric" in result.columns
    assert any("Missing scoring functions in results" in msg for msg in errors)


@pytest.mark.order(191)
def test_run_oddt_warns_when_parallel_context_close_fails(monkeypatch, tmp_path, ocoddt_helpers):
    import joblib

    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _SinglePassPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter([_FakeMol("ligL.sdf", {"rfscore_metric": 0.9})])

    class _BadContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            raise RuntimeError("cannot close")

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _SinglePassPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))
    monkeypatch.setattr(joblib, "parallel_backend", lambda *_a, **_k: _BadContext())

    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligL",
        str(output_dir),
        returnData=False,
        overwrite=True,
        n_cpu=2,
    )
    assert rc == ocerror.ErrorCode.OK
    assert any("Failed to close ODDT parallel context cleanly" in msg for msg in warnings)


@pytest.mark.order(192)
def test_run_oddt_from_cli_rejects_missing_output_dir(tmp_path, ocoddt_helpers):
    rc = ocoddt_helpers.run_oddt_from_cli(
        receptor="receptor.pdbqt",
        ligand="ligand.sdf",
        outputPath=str(tmp_path / "missing"),
    )
    assert rc == ocerror.ErrorCode.DIR_NOT_EXIST


@pytest.mark.order(193)
def test_run_oddt_from_cli_rejects_wrong_input_types(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    rc_bad_receptor = ocoddt_helpers.run_oddt_from_cli(
        receptor=123,  # type: ignore[arg-type]
        ligand="ligand.sdf",
        outputPath=str(output_dir),
    )
    assert rc_bad_receptor == ocerror.ErrorCode.WRONG_TYPE

    rc_bad_ligand = ocoddt_helpers.run_oddt_from_cli(
        receptor="receptor.pdbqt",
        ligand=123,  # type: ignore[arg-type]
        outputPath=str(output_dir),
    )
    assert rc_bad_ligand == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(194)
def test_run_oddt_from_cli_validates_output_and_input_files(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "rec.pdbqt"
    ligand_path = tmp_path / "lig.sdf"

    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (output_dir / "lig.csv").write_text("already", encoding="utf-8")

    rc_exists = ocoddt_helpers.run_oddt_from_cli(
        receptor=str(receptor_path),
        ligand=str(ligand_path),
        outputPath=str(output_dir),
        overwrite=False,
    )
    assert rc_exists == ocerror.ErrorCode.FILE_EXISTS

    (output_dir / "lig.csv").unlink()
    rc_missing_rec = ocoddt_helpers.run_oddt_from_cli(
        receptor=str(tmp_path / "missing_rec.pdbqt"),
        ligand=str(ligand_path),
        outputPath=str(output_dir),
    )
    assert rc_missing_rec == ocerror.ErrorCode.FILE_NOT_EXIST

    rc_missing_lig = ocoddt_helpers.run_oddt_from_cli(
        receptor=str(receptor_path),
        ligand=str(tmp_path / "missing_lig.sdf"),
        outputPath=str(output_dir),
    )
    assert rc_missing_lig == ocerror.ErrorCode.FILE_NOT_EXIST


@pytest.mark.order(195)
def test_run_oddt_from_cli_propagates_build_cmd_error(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "rec.pdbqt"
    ligand_path = tmp_path / "lig.sdf"
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")

    monkeypatch.setattr(ocoddt_helpers, "__build_cmd", lambda *_a, **_k: ocerror.ErrorCode.UNSUPPORTED_EXTENSION)
    rc = ocoddt_helpers.run_oddt_from_cli(
        receptor=str(receptor_path),
        ligand=str(ligand_path),
        outputPath=str(output_dir),
    )
    assert rc == ocerror.ErrorCode.UNSUPPORTED_EXTENSION


@pytest.mark.order(196)
def test_run_oddt_from_cli_success_with_objects_and_clean_models(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "rec.pdbqt"
    ligand_path = tmp_path / "lig.sdf"
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")

    class _ReceptorObj:
        def __init__(self, path):
            self.path = path

    class _LigandObj:
        def __init__(self, path, name):
            self.path = path
            self.name = name

    monkeypatch.setattr(ocoddt_helpers.ocr, "Receptor", _ReceptorObj, raising=False)
    monkeypatch.setattr(ocoddt_helpers.ocl, "Ligand", _LigandObj, raising=False)

    receptor_obj = _ReceptorObj(str(receptor_path))
    ligand_obj = _LigandObj(str(ligand_path), "ligObj")

    built = {}
    removed = []

    monkeypatch.setattr(
        ocoddt_helpers,
        "__build_cmd",
        lambda rec, lig, out: built.update({"receptor": rec, "ligand": lig, "out": out}) or ["oddt_cli", lig],
    )
    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore"]),
            oddt_models_dir=str(tmp_path),
            multiprocess=False,
        ),
    )
    monkeypatch.setattr(ocoddt_helpers.ocrun, "run", lambda *_a, **_k: (ocerror.ErrorCode.OK, ""))
    monkeypatch.setattr(ocoddt_helpers, "get_models", lambda _p: [str(output_dir / "m1.pickle"), str(output_dir / "m2.pickle")])
    monkeypatch.setattr(ocoddt_helpers.ocff, "safe_remove_file", lambda p: removed.append(p) or ocerror.ErrorCode.OK)

    rc = ocoddt_helpers.run_oddt_from_cli(
        receptor=receptor_obj,
        ligand=ligand_obj,
        outputPath=str(output_dir),
        cleanModels=True,
    )
    assert rc == ocerror.ErrorCode.OK
    assert built["receptor"] == str(receptor_path)
    assert built["ligand"] == str(ligand_path)
    assert built["out"].endswith("/ligObj.csv")
    assert len(removed) == 2


@pytest.mark.order(197)
def test_run_oddt_from_cli_returns_nonzero_exit_code_from_runner(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_path = tmp_path / "rec.pdbqt"
    ligand_path = tmp_path / "lig2.sdf"
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")

    monkeypatch.setattr(ocoddt_helpers, "__build_cmd", lambda *_a, **_k: ["oddt_cli"])
    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="oddt_cli", scoring_functions=["rfscore"]),
            oddt_models_dir=str(tmp_path),
            multiprocess=False,
        ),
    )
    monkeypatch.setattr(ocoddt_helpers.ocrun, "run", lambda *_a, **_k: 9)

    rc = ocoddt_helpers.run_oddt_from_cli(
        receptor=str(receptor_path),
        ligand=str(ligand_path),
        outputPath=str(output_dir),
    )
    assert rc == 9


@pytest.mark.order(243)
def test_run_oddt_logs_group_attribute_error_and_returns_failure(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _GroupAttrErrorPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            raise AttributeError("missing descriptor attribute")

        def fetch(self):
            return iter([])

    errors = []
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _GroupAttrErrorPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligGroupAttr",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED
    assert any("Group processing failed with AttributeError" in msg for msg in errors)


@pytest.mark.order(244)
def test_run_oddt_logs_group_type_error_and_returns_failure(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _GroupTypeErrorPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            raise TypeError("unexpected scoring type")

        def fetch(self):
            return iter([])

    errors = []
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _GroupTypeErrorPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligGroupType",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED
    assert any("Group processing failed with TypeError" in msg for msg in errors)


@pytest.mark.order(245)
def test_run_oddt_logs_group_unexpected_error_and_returns_failure(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _GroupRuntimeErrorPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            raise RuntimeError("runtime crash")

        def fetch(self):
            return iter([])

    errors = []
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _GroupRuntimeErrorPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligGroupRuntime",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED
    assert any("Group processing failed with unexpected error" in msg for msg in errors)


@pytest.mark.order(246)
def test_run_oddt_logs_individual_monotonic_attribute_error(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _GroupEmptyPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter([])

    class _IndividualAttrErrorPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            raise AttributeError("monotonic_cst is missing")

        def fetch(self):
            return iter([])

    pipelines = [_GroupEmptyPipeline(), _IndividualAttrErrorPipeline()]
    errors = []
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: pipelines.pop(0))
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligMonotonic",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED
    assert any("version mismatch" in msg.lower() for msg in errors)
