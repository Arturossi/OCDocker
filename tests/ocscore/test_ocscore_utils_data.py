#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Utils.Data inference helpers.
'''

# Imports
###############################################################################
import sys
import types

import pandas as pd

from sklearn.decomposition import PCA

import pytest

import OCDocker.OCScore.Utils.Data as ocscoredata

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

# Functions
###############################################################################
## Public ##

@pytest.mark.order(307)
def test_apply_pca_with_model_object_and_inplace_modes():
    base_df = pd.DataFrame(
        {
            "meta": ["a", "b", "c", "d"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 3.0, 4.0, 5.0],
        }
    )
    pca = PCA(n_components=1)
    pca.fit(base_df[["f1", "f2"]])

    transformed = ocscoredata.apply_pca(
        base_df.copy(),
        pca_model=pca,
        columns_to_skip_pca=["meta"],
        inplace=False,
    )
    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == ["meta", "PC_0"]

    inplace_df = base_df.copy()
    ret = ocscoredata.apply_pca(
        inplace_df,
        pca_model=pca,
        columns_to_skip_pca=["meta"],
        inplace=True,
    )
    assert ret is None
    assert list(inplace_df.columns) == ["meta", "PC_0"]


@pytest.mark.order(308)
def test_apply_pca_rejects_missing_file_and_invalid_model_type():
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})

    with pytest.raises(FileNotFoundError):
        ocscoredata.apply_pca(df.copy(), pca_model="does-not-exist.pkl")

    with pytest.raises(TypeError, match="Invalid PCA model type"):
        ocscoredata.apply_pca(df.copy(), pca_model=3.14)


@pytest.mark.order(309)
def test_get_column_order_handles_dataframe_file_none_and_type_errors(monkeypatch, tmp_path):
    df = pd.DataFrame({"c1": [1], "c2": [2]})
    assert ocscoredata.get_column_order(df) == ["c1", "c2"]

    csv_path = tmp_path / "columns.csv"
    pd.DataFrame({"x": [1], "y": [2]}).to_csv(csv_path, index=False)
    assert ocscoredata.get_column_order(str(csv_path)) == ["x", "y"]

    with pytest.raises(FileNotFoundError):
        ocscoredata.get_column_order(str(tmp_path / "missing.csv"))

    with pytest.raises(TypeError, match="Expected str"):
        ocscoredata.get_column_order(123)

    fake_config = types.ModuleType("OCDocker.Config")
    fake_config.get_config = lambda: types.SimpleNamespace(
        paths=types.SimpleNamespace(reference_column_order=["a", "b", "c"])
    )
    monkeypatch.setitem(sys.modules, "OCDocker.Config", fake_config)
    assert ocscoredata.get_column_order(None) == ["a", "b", "c"]

    fake_config.get_config = lambda: types.SimpleNamespace()
    with pytest.raises(ValueError, match="Could not load config"):
        ocscoredata.get_column_order(None)


@pytest.mark.order(310)
def test_invert_norm_and_remove_other_columns_branches():
    df = pd.DataFrame(
        {
            "receptor": ["r1", "r2"],
            "ligand": ["l1", "l2"],
            "name": ["n1", "n2"],
            "type": ["ligand", "decoy"],
            "db": ["PDBbind", "DUDEz"],
            "experimental": [1.0, 0.0],
            "VINA_score": [-7.0, -6.0],
            "SMINA_score": [-8.0, -7.0],
            "feat1": [10.0, 12.0],
        }
    )

    copied = df.copy()
    inverted = ocscoredata.invert_values_conditionally(copied, inplace=False)
    assert inverted.loc[0, "VINA_score"] == pytest.approx(7.0)
    assert copied.loc[0, "VINA_score"] == pytest.approx(-7.0)

    inplace_df = df.copy()
    ret = ocscoredata.invert_values_conditionally(inplace_df, inplace=True)
    assert ret is None
    assert inplace_df.loc[0, "SMINA_score"] == pytest.approx(8.0)

    normed, fitted_scaler = ocscoredata.norm_data(df.copy(), scaler="standard", inplace=False)
    assert isinstance(normed, pd.DataFrame)
    assert hasattr(fitted_scaler, "transform")

    transformed = ocscoredata.norm_data(df.copy(), scaler=fitted_scaler, inplace=False)
    assert isinstance(transformed, pd.DataFrame)

    inplace_norm = ocscoredata.norm_data(df.copy(), scaler="minmax", inplace=True)
    assert isinstance(inplace_norm, pd.DataFrame)

    with pytest.raises(ValueError, match="Invalid scaler"):
        ocscoredata.norm_data(df.copy(), scaler="bad-scaler", inplace=False)

    reduced = ocscoredata.remove_other_columns(
        df.copy(),
        ["receptor", "ligand", "name", "type", "db", "experimental", "feat1"],
        inplace=False,
    )
    assert list(reduced.columns) == ["receptor", "ligand", "name", "type", "db", "experimental", "feat1"]
    assert "VINA_score" not in reduced.columns

    inplace_reduced_df = df.copy()
    returned = ocscoredata.remove_other_columns(
        inplace_reduced_df,
        ["receptor", "ligand", "name", "type", "db", "experimental", "feat1"],
        inplace=True,
    )
    assert returned is inplace_reduced_df
    assert list(inplace_reduced_df.columns) == ["receptor", "ligand", "name", "type", "db", "experimental", "feat1"]

    with pytest.raises(ValueError, match="not found"):
        ocscoredata.remove_other_columns(df.copy(), ["missing"], inplace=False)


@pytest.mark.order(311)
def test_reorder_columns_to_match_data_order():
    source = pd.DataFrame(columns=["a", "b", "c"])
    df = pd.DataFrame({"b": [2, 3], "a": [1, 4], "extra": [9, 8]})

    reordered_keep_extra = ocscoredata.reorder_columns_to_match_data_order(
        df.copy(),
        data_source=source,
        keep_extra_columns=True,
        fill_missing_columns=False,
    )
    assert reordered_keep_extra.columns.tolist() == ["a", "b", "extra"]

    reordered_fill = ocscoredata.reorder_columns_to_match_data_order(
        df.copy(),
        data_source=source,
        keep_extra_columns=False,
        fill_missing_columns=True,
    )
    assert reordered_fill.columns.tolist() == ["a", "b", "c"]
    assert reordered_fill["c"].isna().all()
