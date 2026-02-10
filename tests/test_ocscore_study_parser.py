#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Utils.StudyParser helpers.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Utils.StudyParser as ocstudy

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


class _FakeStudy:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def trials_dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# Functions
###############################################################################
## Private ##

def _trials_df(with_feature_mask: bool = False) -> pd.DataFrame:
    data = {
        "number": [0, 1, 2, 3],
        "state": ["COMPLETE", "COMPLETE", "FAIL", "COMPLETE"],
        "value": [1.20, 1.00, 0.95, 1.00],
        "user_attrs_AUC": [0.70, 0.80, 0.90, 0.80],
    }
    if with_feature_mask:
        data["user_attrs_Feature_Mask"] = ["111000", "110000", "000111", "110000"]
    return pd.DataFrame(data)


## Public ##

@pytest.mark.order(324)
def test_analyze_studies_filters_skips_handles_errors_and_includes_ablation_features(monkeypatch):
    studies = {
        "NN_Optimization_1": _FakeStudy(_trials_df(with_feature_mask=False)),
        "XGB_PCA95_Ablation_1": _FakeStudy(_trials_df(with_feature_mask=True)),
    }
    load_calls = []

    def _load_study(study_name, storage):
        _ = storage
        load_calls.append(study_name)
        if study_name == "Broken_Study":
            raise RuntimeError("boom")
        return studies[study_name]

    monkeypatch.setattr(ocstudy.optuna, "load_study", _load_study)

    df_rmse, df_auc, df_combined = ocstudy.analyze_studies(
        snames=[
            "AO_ignored",
            "Pre_ignore",
            "NN_Optimization_1",
            "Broken_Study",
            "XGB_PCA95_Ablation_1",
        ],
        storage="sqlite://",
        n_trials=2,
        verbose=False,
    )

    assert not df_rmse.empty
    assert not df_auc.empty
    assert not df_combined.empty
    assert set(df_rmse["study_name"].unique()) == {"NN_Optimization_1", "XGB_PCA95_Ablation_1"}
    assert "features" in df_rmse.columns
    assert "combined_metric" in df_combined.columns
    assert "Broken_Study" in load_calls
    assert "AO_ignored" not in load_calls
    assert "Pre_ignore" not in load_calls
    assert "XGB + PCA95" in set(df_rmse["study_type"])


@pytest.mark.order(325)
def test_analyze_studies_uses_all_trials_when_n_trials_minus_one(monkeypatch):
    monkeypatch.setattr(
        ocstudy.optuna,
        "load_study",
        lambda **_k: _FakeStudy(_trials_df(with_feature_mask=False)),
    )

    df_rmse, df_auc, df_combined = ocstudy.analyze_studies(
        snames=["NN_Optimization_1"],
        storage="sqlite://",
        n_trials=-1,
        verbose=True,
    )

    # COMPLETE + dedupe leaves two rows in our fixture.
    assert len(df_rmse) == 2
    assert len(df_auc) == 2
    assert len(df_combined) == 2


@pytest.mark.order(326)
def test_analyze_studies_old_tracks_flags_and_ablation_columns(monkeypatch):
    studies = {
        "NN_Optimization_1": _FakeStudy(_trials_df(with_feature_mask=False)),
        "XGB_Optimization_1": _FakeStudy(_trials_df(with_feature_mask=False)),
        "Trans_Optimization_1": _FakeStudy(_trials_df(with_feature_mask=False)),
        "NN_Ablation_1": _FakeStudy(_trials_df(with_feature_mask=True)),
    }
    printed = []

    def _load_study(study_name, storage):
        _ = storage
        if study_name == "BrokenOld":
            raise RuntimeError("boom")
        return studies[study_name]

    monkeypatch.setattr(ocstudy.optuna, "load_study", _load_study)
    monkeypatch.setattr(ocstudy.ocprint, "printv", lambda msg: printed.append(msg))

    out = ocstudy.analyze_studies_old(
        snames=[
            "AO_flag",
            "NN_Optimization_1",
            "feature_selection_flag",
            "XGB_Optimization_1",
            "AO_LIG_flag",
            "Trans_Optimization_1",
            "pre-test",
            "BrokenOld",
            "NN_Ablation_1",
        ],
        storage="sqlite://",
        n_trials=5,
        verbose=True,
    )

    assert not out.empty
    assert {"best_rmse_number", "best_auc_number", "best_combined_number"}.issubset(out.columns)
    assert "best_rmse_features" in out.columns

    nn_row = out[out["study_name"] == "NN_Optimization_1"].iloc[0]
    xgb_row = out[out["study_name"] == "XGB_Optimization_1"].iloc[0]
    trans_row = out[out["study_name"] == "Trans_Optimization_1"].iloc[0]
    ablation_rows = out[out["study_name"] == "NN_Ablation_1"]

    assert nn_row["study_type"] == "NN + AE"
    assert xgb_row["study_type"] == "XGB + GA"
    assert trans_row["study_type"] == "Transformer + MAE"
    assert ablation_rows["best_rmse_features"].notna().all()
    assert len(printed) > 0


@pytest.mark.order(327)
def test_parse_study_type_variants_cover_dimensional_and_ml_paths():
    assert ocstudy.parse_study_type("XGB_any", autoencoder=True) == "XGB + AE"
    assert ocstudy.parse_study_type("XGB_any", genetic_algorithm=True) == "XGB + GA"
    assert ocstudy.parse_study_type("Trans_any", multiple_autoencoders=True) == "Transformer + MAE"

    assert ocstudy.parse_study_type("XGB_PCA95") == "XGB + PCA95"
    assert ocstudy.parse_study_type("NN_PCA90") == "NN + PCA90"
    assert ocstudy.parse_study_type("NN_PCA85") == "NN + PCA85"
    assert ocstudy.parse_study_type("NN_PCA80") == "NN + PCA80"
    assert ocstudy.parse_study_type("NN_ScoreOnly") == "NN + Scores Only"
    assert ocstudy.parse_study_type("NN_NoScores") == "NN + No Scores"

    assert ocstudy.parse_study_type("NN_Optimization") == "NN"
    assert ocstudy.parse_study_type("Trans_Optimization") == "Transformer"
    assert ocstudy.parse_study_type("UnknownStudy") == ""
