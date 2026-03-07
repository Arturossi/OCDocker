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


class _ZeroLenDict(dict):
    def __len__(self):
        return 0


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
        oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=[]),
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

    path = Path(__file__).resolve().parents[2] / "OCDocker" / "Rescoring" / "ODDT.py"
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
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    return receptor_path, ligand_path, output_dir, models_dir


def _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers):
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([SimpleNamespace(atoms=[1])]))


def _make_pipeline(*, score_exc=None, fetch_items=None):
    class _Pipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            if score_exc is not None:
                raise score_exc
            return None

        def fetch(self):
            return iter(fetch_items or [])

    return _Pipeline()


## Public ##

@pytest.fixture
def ocoddt_helpers(monkeypatch):
    return _import_oddt_helpers(monkeypatch)


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
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
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
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore", "nnscore"]),
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


@pytest.mark.order(248)
def test_read_receptor_with_retry_tracks_none_and_exception(monkeypatch, ocoddt_helpers):
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([None]))
    receptor_none, err_none = ocoddt_helpers.__read_receptor_with_retry("pdbqt", "/tmp/receptor.pdbqt", retries=1, delay=0)
    assert receptor_none is None
    assert isinstance(err_none, ValueError)

    def _raise(*_a, **_k):
        raise RuntimeError("reader failed")

    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", _raise)
    receptor_exc, err_exc = ocoddt_helpers.__read_receptor_with_retry("pdbqt", "/tmp/receptor.pdbqt", retries=1, delay=0)
    assert receptor_exc is None
    assert isinstance(err_exc, RuntimeError)


@pytest.mark.order(249)
def test_run_oddt_existing_output_dir_with_list_ligands_hits_receptor_type_guard(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=123,  # type: ignore[arg-type]
        preparedLigandPath=[str(tmp_path / "lig.sdf")],
        ligandName="lig",
        outputPath=str(output_dir),
    )
    assert rc == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(250)
def test_run_oddt_input_guards_missing_receptor_and_wrong_ligand_type(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    rc_missing_receptor = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(tmp_path / "missing_receptor.pdbqt"),
        preparedLigandPath=str(tmp_path / "lig.sdf"),
        ligandName="lig",
        outputPath=str(output_dir),
    )
    assert rc_missing_receptor == ocerror.ErrorCode.FILE_NOT_EXIST

    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("REC", encoding="utf-8")
    rc_wrong_ligand = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(receptor),
        preparedLigandPath=("lig1.sdf",),  # type: ignore[arg-type]
        ligandName="lig",
        outputPath=str(output_dir),
    )
    assert rc_wrong_ligand == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(251)
def test_run_oddt_handles_receptor_and_ligand_without_extensions(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")

    receptor = tmp_path / "receptor_no_ext"
    ligand = tmp_path / "ligand_no_ext"
    receptor.write_text("REC", encoding="utf-8")
    ligand.write_text("LIG", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    captured = {"receptor_format": None}

    def _capture_receptor_fmt(fmt, *_a, **_k):
        captured["receptor_format"] = fmt
        return (SimpleNamespace(), None)

    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", _capture_receptor_fmt)
    monkeypatch.setattr(ocoddt_helpers.od.toolkit, "readfile", lambda *_a, **_k: iter([SimpleNamespace(atoms=[1])]))
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _make_pipeline(fetch_items=[]))
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("bad model"))))

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(receptor),
        preparedLigandPath=str(ligand),
        ligandName="ligNoExt",
        outputPath=str(output_dir),
        overwrite=True,
    )

    assert captured["receptor_format"] == ""
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(252)
def test_run_oddt_returns_failure_when_ligand_list_is_empty(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("REC", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )
    monkeypatch.setattr(ocoddt_helpers, "__read_receptor_with_retry", lambda *_a, **_k: (SimpleNamespace(), None))
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _make_pipeline(fetch_items=[]))

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(receptor),
        preparedLigandPath=[],
        ligandName="ligEmpty",
        outputPath=str(output_dir),
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.RESCORING_FAILED


@pytest.mark.order(253)
def test_run_oddt_attempts_to_initialize_missing_exact_models(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore_v2_pdbbind2016", "nnscore_v1"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    init_calls = []
    init_mod = types.ModuleType("OCDocker.Initialise")

    def _initialise_missing(path, names):
        init_calls.append((path, list(names)))
        for model_name in names:
            (Path(path) / f"{model_name}.pickle").write_text("generated", encoding="utf-8")

    init_mod.initialise_oddt_models = _initialise_missing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=999,  # type: ignore[arg-type]
        preparedLigandPath="ligand.sdf",
        ligandName="lig",
        outputPath=str(output_dir),
        overwrite=True,
    )

    assert init_calls
    assert any("rfscore_v2_pdbbind2016" in call[1] for call in init_calls)
    assert rc == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(2531)
def test_run_oddt_single_ligand_forces_single_cpu_even_when_requested_parallel(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    captured_n_cpu = []

    class _CaptureVS:
        def __init__(self, *args, **kwargs):
            _ = args
            captured_n_cpu.append(kwargs.get("n_cpu"))

        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter([_FakeMol("ligand.sdf", {"rfscore_v1": -7.5})])

    monkeypatch.setattr(ocoddt_helpers, "vs", _CaptureVS)
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    result = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(receptor_path),
        preparedLigandPath=str(ligand_path),
        ligandName="lig",
        outputPath=str(output_dir),
        overwrite=True,
        returnData=True,
        n_cpu=-1,
    )

    assert isinstance(result, pd.DataFrame)
    assert captured_n_cpu
    assert captured_n_cpu[0] == 1


@pytest.mark.order(2532)
def test_run_oddt_plecrf_alias_does_not_trigger_missing_model_initialization(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "plecrf_p5_l1_pdbbind2016_s65536.pickle").write_text("model", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["plecrf_pdbbind2016"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    init_calls = []
    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.initialise_oddt_models = lambda *_a, **_k: init_calls.append(True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=123,  # type: ignore[arg-type]
        preparedLigandPath="ligand.sdf",
        ligandName="lig",
        outputPath=str(output_dir),
        overwrite=True,
    )

    assert rc == ocerror.ErrorCode.WRONG_TYPE
    assert not init_calls


@pytest.mark.order(254)
def test_run_oddt_warns_when_exact_models_missing_and_models_dir_invalid(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_models_dir = tmp_path / "not_created_models"

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore_v9"]),
            oddt_models_dir=str(invalid_models_dir),
            multiprocess=False,
        ),
    )

    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath="receptor.pdbqt",
        preparedLigandPath="ligand.sdf",
        ligandName="lig",
        outputPath=str(output_dir),
        overwrite=True,
    )

    assert rc == ocerror.ErrorCode.MISSING_ODDT_MODELS
    assert any("models directory is not set or does not exist" in msg.lower() for msg in warnings)


@pytest.mark.order(255)
def test_run_oddt_descriptor_patch_paths_and_model_family_filters(monkeypatch, tmp_path, ocoddt_helpers):
    np = pytest.importorskip("numpy")
    scipy_sparse = pytest.importorskip("scipy.sparse")

    output_dir = tmp_path / "out"
    models_dir = tmp_path / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Include models for branch coverage in model-family detection and exact matching
    (models_dir / "nnscore_v1.pickle").write_text("model", encoding="utf-8")
    (models_dir / "plec_v2.pickle").write_text("model", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("model", encoding="utf-8")
    (models_dir / "unknown_model.pickle").write_text("model", encoding="utf-8")
    receptor = tmp_path / "receptor.pdbqt"
    ligand = tmp_path / "ligand.sdf"
    receptor.write_text("REC", encoding="utf-8")
    ligand.write_text("LIG", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(
                executable="unused_oddt_command",
                # exact request covers map assignment branch at line 586
                scoring_functions=["nnscore_v1", "plec_v2", "rfscore_v1"],
            ),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    # Provide patchable oddt submodules used inside run_oddt patch block
    fp_mod = types.ModuleType("oddt.fingerprints")
    fp_mod.sparse_to_csr_matrix = lambda fp, size, count_bits=True: scipy_sparse.csr_matrix((1, size), dtype=np.uint8)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.fingerprints", fp_mod)

    desc_mod = types.ModuleType("oddt.scoring.descriptors")

    class _UniversalDescriptor:
        def __init__(self):
            self.protein = None
            self.sparse = True
            self.shape = 8
            self.func = lambda mol, protein=None: 0

        def build(self, ligands, protein=None):
            _ = (ligands, protein)
            return np.zeros((1, 8), dtype=np.float32)

    desc_mod.universal_descriptor = _UniversalDescriptor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.scoring.descriptors", desc_mod)

    utils_mod = types.ModuleType("oddt.utils")
    utils_mod.is_molecule = lambda _x: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.utils", utils_mod)

    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _make_pipeline(fetch_items=[_FakeMol("ligPatch.sdf", {"nnscore_v1_metric": 1.0, "REMARK": "x"})]))
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    result = ocoddt_helpers.run_oddt(
        preparedReceptorPath=str(receptor),
        preparedLigandPath=str(ligand),
        ligandName="ligPatch",
        outputPath=str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert isinstance(result, pd.DataFrame)

    # Exercise patched sparse_to_csr_matrix success and error branches
    stcsr = fp_mod.sparse_to_csr_matrix
    _ = stcsr(np.array([], dtype=np.uint64), size=8, count_bits=True)
    with pytest.raises(ValueError):
        _ = stcsr(np.array([[1, 2]], dtype=np.uint64), size=8, count_bits=True)

    # Exercise patched universal_descriptor.build sparse and dense paths
    ud_sparse = desc_mod.universal_descriptor()
    ud_sparse.sparse = True
    ud_sparse.shape = 8
    ud_sparse.func = lambda mol, protein=None: np.array([], dtype=np.uint64)
    sparse_out = ud_sparse.build([SimpleNamespace(title="lig1")], protein=None)
    assert sparse_out.shape[1] == 8

    ud_dense = desc_mod.universal_descriptor()
    ud_dense.sparse = False
    ud_dense.shape = 4
    ud_dense.func = lambda mol, protein=None: np.array([], dtype=np.uint64)
    dense_out = ud_dense.build([SimpleNamespace(title="lig2")], protein=None)
    assert dense_out.shape[1] == 4


@pytest.mark.order(256)
def test_run_oddt_group_monotonic_attribute_and_iteration_type_warnings(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda *_a, **_k: None)
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _make_pipeline(score_exc=AttributeError("monotonic_cst missing"), fetch_items=[]))
    rc_attr = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligGroupMono",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc_attr == ocerror.ErrorCode.RESCORING_FAILED

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _make_pipeline(score_exc=TypeError("0-d array conversion failed"), fetch_items=[]))
    rc_type = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligGroupTypeIter",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert rc_type == ocerror.ErrorCode.RESCORING_FAILED
    assert any("version incompatibility" in msg.lower() or "trying each scoring function individually" in msg.lower() for msg in warnings)


@pytest.mark.order(257)
def test_run_oddt_threading_backend_fallbacks_and_individual_error_paths(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    (models_dir / "rfscore_v2.pickle").write_text("model2", encoding="utf-8")

    calls = {"n": 0}

    def _get_config_twice_then_import_error():
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(
                oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
                oddt_models_dir=str(models_dir),
                multiprocess=True,
            )
        raise ImportError("config unavailable on second call")

    monkeypatch.setattr(ocoddt_helpers, "get_config", _get_config_twice_then_import_error)

    # Force "from joblib import parallel_backend" ImportError branch
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))

    class _IndividualPipeline:
        def __init__(self, mode):
            self.mode = mode

        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            if self.mode == "attr":
                raise AttributeError("some_attribute_missing")
            if self.mode == "type":
                raise TypeError("invalid type in model")
            if self.mode == "runtime":
                raise RuntimeError("unexpected crash")
            return None

        def fetch(self):
            if self.mode == "empty":
                return iter([])
            if self.mode == "zerolen":
                d = _ZeroLenDict({"rfscore_value": 2.0, "OpenBabel Symmetry Classes": "x"})
                return iter([_FakeMol("ligThread.sdf", d)])
            if self.mode == "success1":
                return iter([_FakeMol("ligThread.sdf", {"rfscore_a": 1.0, "REMARK": "x"})])
            if self.mode == "success2":
                return iter([_FakeMol("ligThread.sdf", {"rfscore_b": 2.0, "TORSDO": "x"})])
            return iter([])

    # First pipeline is group (empty), then individual pipelines for models
    pipeline_modes = ["empty", "success1", "type"]

    def _vs_factory(*_a, **_k):
        return _IndividualPipeline(pipeline_modes.pop(0))

    monkeypatch.setattr(ocoddt_helpers, "vs", _vs_factory)
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    errors = []
    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligThread",
        str(output_dir),
        returnData=True,
        overwrite=True,
        n_cpu=2,
    )

    assert isinstance(result, pd.DataFrame)
    assert "rfscore_a" in result.columns
    assert any("joblib not available" in msg.lower() for msg in warnings)
    assert any("some scoring functions failed" in msg.lower() for msg in errors)


@pytest.mark.order(258)
def test_run_oddt_individual_attribute_and_runtime_errors(monkeypatch, tmp_path, ocoddt_helpers):
    receptor_path, ligand_path, output_dir, _models_dir = _prepare_run_oddt_inputs(monkeypatch, ocoddt_helpers, tmp_path)
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    # Group pipeline returns empty to force individual fallback.
    # Individual pipeline then raises non-monotonic AttributeError and RuntimeError.
    pipeline_sequence = [
        _make_pipeline(fetch_items=[]),
        _make_pipeline(score_exc=AttributeError("generic missing attr"), fetch_items=[]),
        _make_pipeline(score_exc=RuntimeError("generic runtime"), fetch_items=[]),
    ]
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: pipeline_sequence.pop(0))

    # Two models to process in fallback
    models_dir = tmp_path / "models_attr_runtime"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")
    (models_dir / "rfscore_v2.pickle").write_text("m2", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    errors = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda *_a, **_k: None)

    rc = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligAttrRuntime",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )

    assert rc == ocerror.ErrorCode.RESCORING_FAILED
    assert any("failed with AttributeError" in msg for msg in errors)
    assert any("failed for ligand 'ligAttrRuntime'" in msg for msg in errors)


@pytest.mark.order(259)
def test_run_oddt_creates_output_dir_and_uses_default_family_matching(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "new_out"
    models_dir = tmp_path / "models_default"
    receptor_path = tmp_path / "receptor.pdbqt"
    ligand_path = tmp_path / "ligand.sdf"

    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")
    (models_dir / "unknown_model.pickle").write_text("mx", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers.ocff,
        "safe_create_dir",
        lambda p: Path(p).mkdir(parents=True, exist_ok=True) or ocerror.ErrorCode.OK,
    )
    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=[]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)
    monkeypatch.setattr(
        ocoddt_helpers,
        "vs",
        lambda *_a, **_k: _make_pipeline(fetch_items=[_FakeMol("ligDefault.sdf", {"rfscore_metric": 1.0})]),
    )
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligDefault",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert output_dir.is_dir()


@pytest.mark.order(260)
def test_run_oddt_requested_plec_and_unknown_score_branches(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out_plec"
    models_dir = tmp_path / "models_plec"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "plec_v1.pickle").write_text("m1", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["plec", "unknown_score_family"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=123,  # type: ignore[arg-type]
        preparedLigandPath="ligand.sdf",
        ligandName="ligPlec",
        outputPath=str(output_dir),
        overwrite=True,
    )
    assert rc == ocerror.ErrorCode.WRONG_TYPE


@pytest.mark.order(261)
def test_run_oddt_warns_when_missing_exact_model_initialization_fails(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out_exact_missing"
    models_dir = tmp_path / "models_exact_missing"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore_v999"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    init_mod = types.ModuleType("OCDocker.Initialise")
    init_mod.initialise_oddt_models = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("init failed"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "OCDocker.Initialise", init_mod)

    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))

    rc = ocoddt_helpers.run_oddt(
        preparedReceptorPath=123,  # type: ignore[arg-type]
        preparedLigandPath="ligand.sdf",
        ligandName="ligExact",
        outputPath=str(output_dir),
        overwrite=True,
    )

    assert rc == ocerror.ErrorCode.WRONG_TYPE
    assert any("Failed to initialize missing ODDT models" in msg for msg in warnings)


@pytest.mark.order(262)
def test_run_oddt_descriptor_patch_deep_branches(monkeypatch, tmp_path, ocoddt_helpers):
    np = pytest.importorskip("numpy")
    scipy_sparse = pytest.importorskip("scipy.sparse")

    output_dir = tmp_path / "out_patch_deep"
    models_dir = tmp_path / "models_patch_deep"
    receptor_path = tmp_path / "receptor.pdbqt"
    ligand_path = tmp_path / "ligand.sdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )

    fp_mod = types.ModuleType("oddt.fingerprints")
    fp_mod.sparse_to_csr_matrix = lambda fp, size, count_bits=True: scipy_sparse.csr_matrix((1, size), dtype=np.uint8)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.fingerprints", fp_mod)

    desc_mod = types.ModuleType("oddt.scoring.descriptors")

    class _UniversalDescriptor:
        def __init__(self):
            self.protein = None
            self.sparse = True
            self.shape = 4
            self.func = lambda mol, protein=None: np.array([1], dtype=np.uint64)

        def build(self, ligands, protein=None):
            _ = (ligands, protein)
            return np.zeros((1, self.shape), dtype=np.float32)

    desc_mod.universal_descriptor = _UniversalDescriptor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.scoring.descriptors", desc_mod)

    utils_mod = types.ModuleType("oddt.utils")
    utils_mod.is_molecule = lambda x: not isinstance(x, list)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "oddt.utils", utils_mod)

    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)
    monkeypatch.setattr(
        ocoddt_helpers,
        "vs",
        lambda *_a, **_k: _make_pipeline(fetch_items=[_FakeMol("ligPatchDeep.sdf", {"rfscore_metric": 1.0, "REMARK": "x"})]),
    )
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligPatchDeep",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )
    assert isinstance(result, pd.DataFrame)

    # Patched sparse-to-CSR helper branches.
    patched_stcsr = fp_mod.sparse_to_csr_matrix
    _ = patched_stcsr(np.uint64(7), size=4, count_bits=True)
    _ = patched_stcsr(np.array([1, 2], dtype=np.uint64), size=4, count_bits=True)

    # Patched universal descriptor branches.
    ud_cls = desc_mod.universal_descriptor
    ud1 = ud_cls()
    ud1.sparse = True
    ud1.shape = 4
    ud1.func = lambda mol, protein=None: 1
    _ = ud1.build(SimpleNamespace(title="mol_scalar"), protein=SimpleNamespace())

    ud2 = ud_cls()
    ud2.sparse = True
    ud2.shape = 4

    def _func_by_title(mol, protein=None):
        _ = protein
        if mol.title == "mol_vec":
            return np.array([1, 2], dtype=np.uint64)
        if mol.title == "mol_matrix":
            return np.array([[1, 2], [3, 4]], dtype=np.uint64)
        raise RuntimeError("descriptor failure")

    ud2.func = _func_by_title

    real_vstack = scipy_sparse.vstack
    vstack_calls = {"n": 0}

    def _fail_then_ok_vstack(mats, format="csr"):
        vstack_calls["n"] += 1
        if vstack_calls["n"] == 1:
            raise RuntimeError("vstack fail once")
        return real_vstack(mats, format=format)

    ud2.build.__globals__["sparse_vstack"] = _fail_then_ok_vstack

    stcsr_calls = {"n": 0}
    real_after_patch = fp_mod.sparse_to_csr_matrix

    def _custom_stcsr(arr, size, count_bits=True):
        stcsr_calls["n"] += 1
        if stcsr_calls["n"] == 1:
            return np.array([[1, 0]], dtype=np.uint8)
        if stcsr_calls["n"] == 2:
            raise RuntimeError("csr conversion fail")
        return real_after_patch(arr, size=size, count_bits=count_bits)

    fp_mod.sparse_to_csr_matrix = _custom_stcsr  # type: ignore[attr-defined]

    _ = ud2.build(
        [
            SimpleNamespace(title="mol_vec"),
            SimpleNamespace(title="mol_matrix"),
            SimpleNamespace(title="mol_error"),
        ],
        protein=SimpleNamespace(),
    )

    ud3 = ud_cls()
    ud3.sparse = True
    ud3.shape = 4
    _ = ud3.build([], protein=None)

    ud4 = ud_cls()
    ud4.sparse = False
    ud4.shape = 4
    ud4.func = lambda mol, protein=None: np.array([1, 2, 3, 4], dtype=np.float32)
    dense_non_empty = ud4.build([SimpleNamespace(title="mol_dense")], protein=None)
    assert dense_non_empty.shape[1] == 4

    ud5 = ud_cls()
    ud5.sparse = False
    ud5.shape = 4
    dense_empty = ud5.build([], protein=None)
    assert dense_empty.shape[1] == 4


@pytest.mark.order(263)
def test_run_oddt_individual_fallback_merge_noext_and_empty_data_paths(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out_merge_paths"
    models_dir = tmp_path / "models_merge_paths"
    receptor_path = tmp_path / "receptor.pdbqt"
    ligand_path = tmp_path / "ligand_noext"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")
    (models_dir / "rfscore_v2.pickle").write_text("m2", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _Pipeline:
        def __init__(self, rows):
            self.rows = rows

        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter(self.rows)

    class _MolZeroLen:
        def __init__(self, title, payload):
            self.title = title
            self.data = SimpleNamespace(to_dict=lambda: _ZeroLenDict(payload))

    pipeline_sequence = [
        _Pipeline(
            [
                _MolZeroLen("ligMerge.sdf", {"REMARK": "a"}),
                _MolZeroLen("ligMerge.sdf", {"REMARK": "b"}),
            ]
        ),
        _Pipeline([_MolZeroLen("ligMerge.sdf", {"REMARK": "c"})]),
        _Pipeline([_FakeMol("ligMerge.sdf", {"rfscore_a": 1.0}), _FakeMol("ligMerge.sdf", {"rfscore_b": 2.0})]),
    ]
    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: pipeline_sequence.pop(0))
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    warnings = []
    monkeypatch.setattr(ocoddt_helpers.ocprint, "print_warning", lambda msg: warnings.append(msg))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligMerge",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert any("No data collected" in msg for msg in warnings)


@pytest.mark.order(264)
def test_run_oddt_merges_duplicate_ligand_rows_from_group_fetch(monkeypatch, tmp_path, ocoddt_helpers):
    output_dir = tmp_path / "out_group_merge"
    models_dir = tmp_path / "models_group_merge"
    receptor_path = tmp_path / "receptor.pdbqt"
    ligand_path = tmp_path / "ligand.sdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("REC", encoding="utf-8")
    ligand_path.write_text("LIG", encoding="utf-8")
    (models_dir / "rfscore_v1.pickle").write_text("m1", encoding="utf-8")

    monkeypatch.setattr(
        ocoddt_helpers,
        "get_config",
        lambda: SimpleNamespace(
            oddt=SimpleNamespace(executable="unused_oddt_command", scoring_functions=["rfscore"]),
            oddt_models_dir=str(models_dir),
            multiprocess=False,
        ),
    )
    _patch_valid_receptor_and_ligand(monkeypatch, ocoddt_helpers)

    class _GroupPipeline:
        def load_ligands(self, *_a, **_k):
            return None

        def score(self, *_a, **_k):
            return None

        def fetch(self):
            return iter(
                [
                    _FakeMol("ligMergeGroup.sdf", {"rfscore_a": 1.0}),
                    _FakeMol("ligMergeGroup.sdf", {"rfscore_b": 2.0}),
                ]
            )

    monkeypatch.setattr(ocoddt_helpers, "vs", lambda *_a, **_k: _GroupPipeline())
    monkeypatch.setattr(ocoddt_helpers, "scorer", SimpleNamespace(load=lambda *_a, **_k: object()))

    result = ocoddt_helpers.run_oddt(
        str(receptor_path),
        str(ligand_path),
        "ligMergeGroup",
        str(output_dir),
        returnData=True,
        overwrite=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert "rfscore_a" in result.columns
    assert "rfscore_b" in result.columns
