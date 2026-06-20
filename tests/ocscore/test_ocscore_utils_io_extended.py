#!/usr/bin/env python3

# Description
###############################################################################
'''
Additional coverage tests for OCScore.Utils.IO edge branches.
'''

# Imports
###############################################################################
import builtins
import pickle
import sys

import numpy as np
import pandas as pd

import pytest

import OCDocker.OCScore.Utils.IO as ocscoreio

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


# Functions
###############################################################################
## Private ##


## Public ##

@pytest.mark.order(330)
def test_get_models_dir_creates_directory_when_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(ocscoreio.os.path, "isdir", lambda _p: False)
    monkeypatch.setattr(ocscoreio.os, "makedirs", lambda path, exist_ok=False: calls.append((path, exist_ok)))

    models_dir = ocscoreio.get_models_dir()

    assert models_dir.endswith("OCScore_models")
    assert len(calls) == 1
    assert calls[0][0].endswith("OCScore_models")
    assert calls[0][1] is True


@pytest.mark.order(331)
def test_load_data_without_nans_does_not_drop_rows(tmp_path):
    csv_path = tmp_path / "clean.csv"
    data = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, 30.0],
            "experimental": [0.1, 0.2, 0.3],
        }
    )
    data.to_csv(csv_path, index=False)

    loaded = ocscoreio.load_data(str(csv_path), exclude_column="experimental")
    assert loaded.shape[0] == 3
    assert loaded.equals(data)


@pytest.mark.order(332)
def test_load_mask_uses_default_models_dir_when_none(monkeypatch, tmp_path):
    default_models = tmp_path / "default_models"
    default_models.mkdir(parents=True, exist_ok=True)
    mask_file = default_models / "default_mask.pkl"
    mask_file.write_text("placeholder", encoding="utf-8")

    seen = {}
    monkeypatch.setattr(ocscoreio, "get_models_dir", lambda: str(default_models))

    def _fake_load(path, serialization_method="auto", trusted=False):
        seen["call"] = (path, serialization_method, trusted)
        return [1, 0, 1]

    monkeypatch.setattr(ocscoreio, "load_object", _fake_load)

    loaded = ocscoreio.load_mask("default", models_dir=None)
    np.testing.assert_array_equal(loaded, np.array([1, 0, 1], dtype=int))
    assert seen["call"][0].endswith("default_mask.pkl")
    assert seen["call"][1] == "joblib"
    assert seen["call"][2] is True


@pytest.mark.order(333)
def test_load_mask_raises_when_joblib_and_pickle_fallback_both_fail(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mask_file = models_dir / "broken_mask.pkl"
    mask_file.write_text("placeholder", encoding="utf-8")

    def _fake_load(_path, serialization_method="auto", trusted=False):
        _ = trusted
        if serialization_method == "joblib":
            raise ValueError("joblib fail")
        raise pickle.UnpicklingError("pickle fail")

    monkeypatch.setattr(ocscoreio, "load_object", _fake_load)

    with pytest.raises(ValueError, match="Failed to load mask"):
        ocscoreio.load_mask("broken", models_dir=str(models_dir))


@pytest.mark.order(334)
def test_load_mask_dict_mask_and_array_keys(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "withmask_mask.pkl").write_text("placeholder", encoding="utf-8")
    (models_dir / "witharray_mask.pkl").write_text("placeholder", encoding="utf-8")

    def _fake_load(path, serialization_method="auto", trusted=False):
        _ = serialization_method
        _ = trusted
        if path.endswith("withmask_mask.pkl"):
            return {"mask": [1, 0, 1]}
        return {"array": [0, 1, 0]}

    monkeypatch.setattr(ocscoreio, "load_object", _fake_load)

    a = ocscoreio.load_mask("withmask", models_dir=str(models_dir))
    b = ocscoreio.load_mask("witharray", models_dir=str(models_dir))

    np.testing.assert_array_equal(a, np.array([1, 0, 1], dtype=int))
    np.testing.assert_array_equal(b, np.array([0, 1, 0], dtype=int))


@pytest.mark.order(335)
def test_load_mask_dict_first_array_like_value_branch(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "firstvalue_mask.pkl").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(
        ocscoreio,
        "load_object",
        lambda *_a, **_k: {"meta": "x", "payload": [1, 1, 0]},
    )

    loaded = ocscoreio.load_mask("firstvalue", models_dir=str(models_dir))
    np.testing.assert_array_equal(loaded, np.array([1, 1, 0], dtype=int))


@pytest.mark.order(336)
def test_load_mask_rejects_values_other_than_zero_or_one(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "invalidbits_mask.pkl").write_text("placeholder", encoding="utf-8")

    monkeypatch.setattr(ocscoreio, "load_object", lambda *_a, **_k: [1, 2, 0])

    with pytest.raises(ValueError, match="only 0s and 1s"):
        ocscoreio.load_mask("invalidbits", models_dir=str(models_dir))


@pytest.mark.order(337)
def test_save_mask_uses_default_models_dir_and_existing_custom_dir(monkeypatch, tmp_path):
    default_models = tmp_path / "default_models"
    custom_models = tmp_path / "existing_custom"
    default_models.mkdir(parents=True, exist_ok=True)
    custom_models.mkdir(parents=True, exist_ok=True)

    calls = []
    monkeypatch.setattr(ocscoreio, "get_models_dir", lambda: str(default_models))
    monkeypatch.setattr(ocscoreio, "save_object", lambda obj, filename: calls.append((obj, filename)))
    monkeypatch.setattr(
        ocscoreio.os,
        "makedirs",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("makedirs should not be called")),
    )

    default_path = ocscoreio.save_mask([1, 0, 1], "default", models_dir=None)
    custom_path = ocscoreio.save_mask([0, 1, 0], "custom", models_dir=str(custom_models))

    assert default_path.endswith("default_mask.pkl")
    assert custom_path.endswith("custom_mask.pkl")
    assert len(calls) == 2


@pytest.mark.order(338)
def test_save_object_auto_torch_import_error_raises_value_error(monkeypatch, tmp_path):
    file_path = tmp_path / "model.pt"

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ValueError, match="PyTorch is not installed"):
        ocscoreio.save_object({"w": 1}, str(file_path), serialization_method="auto")


@pytest.mark.order(339)
def test_save_object_auto_unknown_extension_falls_back_to_joblib(monkeypatch, tmp_path):
    file_path = tmp_path / "object.bin"
    calls = []
    monkeypatch.setattr(ocscoreio.joblib, "dump", lambda obj, path: calls.append((obj, path)))

    ocscoreio.save_object({"a": 1}, str(file_path), serialization_method="auto")

    assert len(calls) == 1
    assert calls[0][0] == {"a": 1}
    assert calls[0][1] == str(file_path)


@pytest.mark.order(340)
def test_save_object_auto_torch_success_with_fake_torch(monkeypatch, tmp_path):
    file_path = tmp_path / "model.pth"
    calls = []

    class _FakeTorch:
        @staticmethod
        def save(obj, path):
            calls.append((obj, path))

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    ocscoreio.save_object({"w": [1, 2, 3]}, str(file_path), serialization_method="auto")

    assert len(calls) == 1
    assert calls[0][0] == {"w": [1, 2, 3]}
    assert calls[0][1] == str(file_path)

@pytest.mark.order(334)
def test_load_pipeline_results_drops_completely_empty_rows(tmp_path):
    csv_path = tmp_path / "pipeline_results.csv"
    csv_path.write_text("f1,f2\n1,2\n , \n3,4\n", encoding="utf-8")

    loaded = ocscoreio.load_pipeline_results_from_archive(csv_path)

    assert loaded.shape == (2, 2)
    assert loaded["f1"].tolist() == ["1", "3"]
