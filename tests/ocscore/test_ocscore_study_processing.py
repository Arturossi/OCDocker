#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.StudyProcessing helpers.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.StudyProcessing as ocstudyproc

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

def _results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_name": ["NN_Optimization_1", "XGB_Optimization_1", "Transformer_Optimization_1"],
            "study_type": ["NN", "XGB", "Transformer"],
            "best_rmse_number": [10, 20, 30],
            "best_rmse_value": [1.10, 1.35, 1.20],
            "best_rmse_auc": [0.72, 0.68, 0.74],
            "best_auc_number": [11, 21, 31],
            "best_auc_value": [1.15, 1.40, 1.25],
            "best_auc": [0.78, 0.60, 0.80],
            "best_combined_number": [12, 22, 32],
            "best_combined_metric": [0.30, 0.80, 0.45],
            "best_combined_value": [1.08, 1.31, 1.18],
            "best_combined_auc": [0.78, 0.51, 0.73],
        }
    )


def _final_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_name": "Mean_consensus",
                "Methodology": "Mean consensus",
                "Experiment": "consensus",
                "RMSE": 1.05,
                "AUC": 0.49,
                "combined_metric": 0.56,
            }
        ]
    )


## Public ##

@pytest.mark.order(304)
def test_get_study_data_happy_path_with_labels_and_ranges(monkeypatch):
    monkeypatch.setattr(ocstudyproc.ocstudy, "analyze_studies_old", lambda *_a, **_k: _results_df())

    (
        best_rmse_df,
        best_auc_df,
        best_combined_df,
        results_df,
        min_auc,
        max_auc,
        min_error,
        max_error,
        error_range,
        auc_range,
    ) = ocstudyproc.get_study_data(
        snames=["s1", "s2"],
        storage="sqlite://",
        final_metrics=_final_metrics_df(),
        n_trials=10,
        error_threshold=1.25,
        nn_ae_start=0,
        nn_ae_end=1,
        xgb_ga_start=1,
        xgb_ga_end=2,
    )

    assert results_df.loc[0, "study_type"] == "NN + AE"
    assert results_df.loc[1, "study_type"] == "XGB + GA"

    assert (best_rmse_df["RMSE"] <= 1.25).all()
    assert (best_auc_df["RMSE"] <= 1.25).all()
    assert (best_combined_df["RMSE"] <= 1.25).all()

    assert "AUC New" in best_rmse_df.columns
    assert "AUC New" in best_auc_df.columns
    assert "AUC New" in best_combined_df.columns

    assert min_auc <= max_auc
    assert min_error <= max_error
    assert error_range == pytest.approx(max_error - min_error)
    assert auc_range == pytest.approx(max_auc - min_auc)


@pytest.mark.order(305)
def test_get_study_data_rejects_invalid_nn_ae_range(monkeypatch):
    monkeypatch.setattr(ocstudyproc.ocstudy, "analyze_studies_old", lambda *_a, **_k: _results_df())

    with pytest.raises(ValueError, match="NN \\+ AE"):
        ocstudyproc.get_study_data(
            snames=["s1"],
            storage="sqlite://",
            final_metrics=_final_metrics_df(),
            n_trials=5,
            nn_ae_start=2,
            nn_ae_end=2,
        )


@pytest.mark.order(306)
def test_get_study_data_rejects_invalid_xgb_ga_range(monkeypatch):
    monkeypatch.setattr(ocstudyproc.ocstudy, "analyze_studies_old", lambda *_a, **_k: _results_df())

    with pytest.raises(ValueError, match="XGB \\+ GA"):
        ocstudyproc.get_study_data(
            snames=["s1"],
            storage="sqlite://",
            final_metrics=_final_metrics_df(),
            n_trials=5,
            xgb_ga_start=3,
            xgb_ga_end=1,
        )
