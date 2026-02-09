#!/usr/bin/env python3

# Description
###############################################################################
'''
Functional tests for OCScore.Utils.IO helpers.
'''

# Imports
###############################################################################
import builtins
import pickle

import numpy as np
import pandas as pd
import pytest

import OCDocker.OCScore.Utils.IO as ocscoreio

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


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(87)
def test_get_models_dir_returns_existing_directory():
    models_dir = ocscoreio.get_models_dir()
    assert models_dir.endswith("OCScore_models")
    assert ocscoreio.os.path.isdir(models_dir)


@pytest.mark.order(88)
def test_load_data_drops_nans_except_excluded_column(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0],
            "f2": [10.0, 20.0, np.nan],
            "experimental": [np.nan, 2.0, 3.0],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = ocscoreio.load_data(str(csv_path), exclude_column="experimental")
    # rows with NaN in f1/f2 are removed; NaN in excluded column is allowed
    assert loaded.shape[0] == 1
    assert loaded.iloc[0]["f1"] == 1.0
    assert loaded.iloc[0]["f2"] == 10.0


@pytest.mark.order(89)
def test_save_and_load_mask_roundtrip_custom_dir(tmp_path):
    models_dir = tmp_path / "models"
    saved = ocscoreio.save_mask([1, 0, 1, 1], "toy", models_dir=str(models_dir))
    assert saved.endswith("toy_mask.pkl")

    loaded = ocscoreio.load_mask("toy", models_dir=str(models_dir))
    np.testing.assert_array_equal(loaded, np.array([1, 0, 1, 1], dtype=int))


@pytest.mark.order(90)
def test_load_mask_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _ = ocscoreio.load_mask("missing", models_dir=str(tmp_path))


@pytest.mark.order(91)
def test_load_mask_dict_without_array_raises(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    bad_path = models_dir / "bad_mask.pkl"
    with open(bad_path, "wb") as f:
        pickle.dump({"unexpected": "value"}, f)

    with pytest.raises(ValueError):
        _ = ocscoreio.load_mask("bad", models_dir=str(models_dir))


@pytest.mark.order(92)
def test_save_mask_invalid_values_raises(tmp_path):
    with pytest.raises(ValueError):
        _ = ocscoreio.save_mask([1, 2, 0], "bad", models_dir=str(tmp_path))


@pytest.mark.order(93)
def test_save_object_pickle_and_load_auto(tmp_path):
    payload = {"a": 1, "b": [1, 2]}
    file_path = tmp_path / "obj.pkl"
    ocscoreio.save_object(payload, str(file_path), serialization_method="pickle")

    loaded = ocscoreio.load_object(str(file_path), serialization_method="auto", trusted=True)
    assert loaded == payload


@pytest.mark.order(94)
def test_save_object_invalid_serialization_raises(tmp_path):
    with pytest.raises(ValueError):
        ocscoreio.save_object({"x": 1}, str(tmp_path / "x.bin"), serialization_method="invalid")


@pytest.mark.order(95)
def test_load_object_torch_import_error_raises_value_error(monkeypatch, tmp_path):
    # Ensure trust gate passes; we only want to test torch ImportError branch.
    file_path = tmp_path / "fake.pt"
    file_path.write_text("not-a-real-model", encoding="utf-8")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError):
        _ = ocscoreio.load_object(str(file_path), serialization_method="torch", trusted=True)

