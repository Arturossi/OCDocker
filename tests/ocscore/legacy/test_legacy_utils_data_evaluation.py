#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Utils.legacy.Data and Utils.legacy.Evaluation helpers.
'''

# Imports
###############################################################################
import os

import pandas as pd

import pytest

import OCDocker.OCScore.Utils.legacy.Data as ocscoredata_legacy
import OCDocker.OCScore.Utils.legacy.Evaluation as ocseval

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
## Private ##

def _raw_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "receptor": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "ligand": ["l1", "l2", "l3", "l4", "l5", "l6"],
            "name": ["n1", "n2", "n3", "n4", "n5", "n6"],
            "type": ["ligand", "decoy", "ligand", "decoy", "ligand", "decoy"],
            "db": ["DUDEz", "DUDEz", "DUDEz", "PDBbind", "PDBbind", "PDBbind"],
            "experimental": [1.0, 0.0, 1.5, 2.0, 3.0, 4.0],
            "SMINA_1": [-7.2, -6.1, -7.8, -8.5, -7.9, -7.0],
            "VINA_1": [-6.8, -5.9, -7.1, -8.0, -7.6, -6.9],
            "feat1": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "feat2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def _mock_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    score_columns = ["SMINA_1", "VINA_1"]
    dudez = pd.DataFrame(
        {
            "receptor": ["r1", "r2", "r3", "r4"],
            "ligand": ["l1", "l2", "l3", "l4"],
            "name": ["n1", "n2", "n3", "n4"],
            "type": ["ligand", "decoy", "ligand", "decoy"],
            "db": ["DUDEz", "DUDEz", "DUDEz", "DUDEz"],
            "experimental": [1.0, 0.0, 1.5, 0.5],
            "SMINA_1": [-7.2, -6.1, -7.8, -6.4],
            "VINA_1": [-6.8, -5.9, -7.1, -6.0],
            "feat1": [10.0, 11.0, 12.0, 13.0],
        }
    )
    pdbbind = pd.DataFrame(
        {
            "receptor": ["p1", "p2", "p3", "p4"],
            "ligand": ["q1", "q2", "q3", "q4"],
            "name": ["m1", "m2", "m3", "m4"],
            "type": ["ligand", "decoy", "ligand", "decoy"],
            "db": ["PDBbind", "PDBbind", "PDBbind", "PDBbind"],
            "experimental": [2.0, 3.0, 2.5, 3.5],
            "SMINA_1": [-8.5, -7.9, -8.1, -7.8],
            "VINA_1": [-8.0, -7.6, -7.9, -7.4],
            "feat1": [20.0, 21.0, 22.0, 23.0],
        }
    )
    return dudez, pdbbind, score_columns


## Public ##

@pytest.mark.order(715)
def test_compute_auc_and_rmse_basic_behaviour():
    auc_df = pd.DataFrame(
        {
            "type": ["ligand", "decoy", "ligand", "decoy"],
            "s1": [0.9, 0.1, 0.8, 0.2],
            "s2": [0.2, 0.8, 0.3, 0.7],
        }
    )
    out_auc = ocseval.compute_auc(auc_df.copy(), "ligand", ["s1", "s2"], "type")
    assert set(out_auc["score_column"]) == {"s1", "s2"}
    assert out_auc["AUC"].between(0.0, 1.0).all()

    rmse_df = pd.DataFrame(
        {
            "experimental": [1.0, 2.0, 3.0, 4.0],
            "p1": [1.1, 2.2, 3.1, 3.8],
            "p2": [0.9, 1.8, 3.4, 4.3],
        }
    )
    out_rmse = ocseval.compute_rmse(rmse_df, ["p1", "p2"], "experimental")
    assert set(out_rmse["score_column"]) == {"p1", "p2"}
    assert (out_rmse["RMSE"] >= 0).all()


@pytest.mark.order(716)
def test_compute_metrics_handles_inversion_toggle_and_validates_metric_db_name(monkeypatch):
    df = pd.DataFrame(
        {
            "db": ["PDBbind", "PDBbind", "DUDEz", "DUDEz"],
            "experimental": [1.0, 2.0, 0.0, 1.0],
            "type": ["ligand", "decoy", "ligand", "decoy"],
            "s1": [1.1, 1.9, 0.8, 0.2],
            "s2": [0.9, 2.2, 0.7, 0.3],
        }
    )

    calls = {"invert": 0}

    def _invert(dataframe):
        calls["invert"] += 1
        return dataframe

    monkeypatch.setattr(ocseval.ocscoredata, "invert_values_conditionally", _invert)

    out = ocseval.compute_metrics(
        df=df.copy(),
        score_columns=["s1", "s2"],
        target_column_name="experimental",
        db_column_name="db",
        metric_db_name=("PDBbind", "DUDEz"),
        class_column_name="type",
        positive_class_names="ligand",
        invert_conditionally=True,
    )
    assert calls["invert"] == 1
    assert set(out.columns) == {"score_column", "RMSE", "AUC"}
    assert len(out) == 2

    calls["invert"] = 0
    ocseval.compute_metrics(
        df=df.copy(),
        score_columns=["s1"],
        target_column_name="experimental",
        db_column_name="db",
        metric_db_name=("PDBbind", "DUDEz"),
        class_column_name="type",
        positive_class_names=["ligand"],
        invert_conditionally=False,
    )
    assert calls["invert"] == 0

    with pytest.raises(ValueError, match="two elements"):
        ocseval.compute_metrics(
            df=df,
            score_columns=["s1"],
            target_column_name="experimental",
            db_column_name="db",
            metric_db_name=("only_one",),
            class_column_name="type",
            positive_class_names="ligand",
        )


@pytest.mark.order(717)
def test_metrics_zscore_and_chunk_helpers_cover_success_and_errors():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    metrics_df, metric_names = ocscoredata_legacy.calculate_metrics(df.copy(), ["a", "b"])
    assert "iqr" in metric_names
    assert all(name in metrics_df.columns for name in metric_names)

    with pytest.raises(ValueError, match="not found"):
        ocscoredata_legacy.calculate_metrics(df.copy(), ["missing"])

    z_df = ocscoredata_legacy.compute_zscore(df.copy(), ["a", "b"])
    assert "z_a" in z_df.columns
    assert "z_b" in z_df.columns

    with pytest.raises(ValueError, match="not found"):
        ocscoredata_legacy.compute_zscore(df.copy(), ["missing"])

    chunk = ocscoredata_legacy.chunkenize_dataset([1, 2, 3, 4, 5], id=2, num_machines=2)
    assert chunk == [4, 5]

    with pytest.raises(ValueError, match="between 1"):
        ocscoredata_legacy.chunkenize_dataset([1, 2], id=0, num_machines=2)


@pytest.mark.order(718)
def test_outlier_helpers_and_generate_mask():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 1000.0],
            "y": [-1.0, 2.0, 3.0, 4.0, 5.0],
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    outliers = ocscoredata_legacy.detect_extreme_outliers_iqr_columns_positive(df, ["x", "y"], extreme_factor=1.5)
    assert "x" in outliers
    assert not outliers["x"].empty

    cleaned = ocscoredata_legacy.remove_extreme_outliers_iqr_columns_positive(df, ["x"], extreme_factor=1.5)
    assert len(cleaned) == 4
    assert 1000.0 not in cleaned["x"].tolist()

    masks = ocscoredata_legacy.generate_mask(["A", "B", "C"], ["A", "C"])
    assert len(masks) == 4
    assert all(mask[1] == 1 for mask in masks)


@pytest.mark.order(719)
def test_split_dataset():
    X = pd.DataFrame({"f1": list(range(10)), "f2": list(range(10, 20))})
    y = pd.Series(list(range(10)))
    X_train, X_test, y_train, y_test = ocscoredata_legacy.split_dataset(X, y, test_size=0.3, random_state=1)
    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10


@pytest.mark.order(720)
def test_preprocess_df_supports_outlier_flag_and_return_scaler(monkeypatch):
    calls = {"outlier": 0}

    def _remove_outliers(df, _columns, extreme_factor=3.0):
        calls["outlier"] += 1
        return df

    monkeypatch.setattr(ocscoredata_legacy.ocscoreio, "load_data", lambda _p: _raw_scores_df())
    monkeypatch.setattr(ocscoredata_legacy, "remove_extreme_outliers_iqr_columns_positive", _remove_outliers)

    dudez, pdbbind, score_columns = ocscoredata_legacy.preprocess_df(
        file_name="dummy.csv",
        score_columns_list=["SMINA", "VINA"],
        outliers_columns_list=["SMINA_1"],
        invert_conditionally=False,
        normalize=False,
    )
    assert calls["outlier"] == 1
    assert score_columns == ["SMINA_1", "VINA_1"]
    assert "experimental" not in dudez.columns
    assert "experimental" in pdbbind.columns

    dudez_n, pdbbind_n, score_columns_n, scaler = ocscoredata_legacy.preprocess_df(
        file_name="dummy.csv",
        score_columns_list=["SMINA", "VINA"],
        outliers_columns_list=None,
        invert_conditionally=True,
        normalize=True,
        return_scaler=True,
    )
    assert score_columns_n == ["SMINA_1", "VINA_1"]
    assert hasattr(scaler, "transform")
    assert dudez_n.shape[0] > 0
    assert pdbbind_n.shape[0] > 0


@pytest.mark.order(721)
def test_load_data_covers_pca_no_scores_and_non_pdb_training(monkeypatch, tmp_path):
    def _preprocess_stub(*_a, **_k):
        dudez, pdbbind, score_columns = _mock_preprocessed_data()
        return dudez.copy(), pdbbind.copy(), score_columns.copy()

    pca_calls = {"count": 0}

    def _apply_pca_stub(*_a, **_k):
        pca_calls["count"] += 1
        return None

    monkeypatch.setattr(ocscoredata_legacy, "preprocess_df", _preprocess_stub)
    monkeypatch.setattr(ocscoredata_legacy, "apply_pca", _apply_pca_stub)

    out_pca = ocscoredata_legacy.load_data(
        base_models_folder=str(tmp_path),
        storage_id=11,
        df_path="dummy.csv",
        optimization_type="NN",
        use_PCA=True,
        pca_type=90,
        use_pdb_train=True,
        random_seed=1,
    )
    assert out_pca["study_name"].startswith("PCA90_")
    assert pca_calls["count"] == 2
    assert out_pca["X_val"] is not None
    assert os.path.isdir(out_pca["models_folder"])

    out_no_scores = ocscoredata_legacy.load_data(
        base_models_folder=str(tmp_path),
        storage_id=12,
        df_path="dummy.csv",
        optimization_type="XGB",
        no_scores=True,
        use_PCA=False,
        use_pdb_train=True,
        random_seed=1,
    )
    assert out_no_scores["study_name"].startswith("NoScores_")
    assert "SMINA_1" not in out_no_scores["X_train"].columns
    assert "VINA_1" not in out_no_scores["X_train"].columns

    out_non_pdb = ocscoredata_legacy.load_data(
        base_models_folder=str(tmp_path),
        storage_id=13,
        df_path="dummy.csv",
        optimization_type="TRANS",
        use_PCA=False,
        use_pdb_train=False,
        random_seed=1,
    )
    assert out_non_pdb["X_val"] is None
    assert out_non_pdb["y_val"] is None
    assert set(out_non_pdb["y_test"].dropna().unique()).issubset({0, 1})
