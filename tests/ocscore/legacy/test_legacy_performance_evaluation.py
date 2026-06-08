#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore analysis performance-evaluation helpers.
'''

# Imports
###############################################################################
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.legacy.PerformanceEvaluation as ocperf

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

def _metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_name": ["s1", "s2"],
            "Methodology": ["Raw Scoring Function", "Mean consensus"],
            "RMSE": [1.1, 1.0],
            "AUC": [0.7, 0.75],
            "combined_metric": [0.4, 0.25],
        }
    )


def _studies_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_name": ["NN_Optimization_1", "XGB_Optimization_1"],
            "study_type": ["NN", "XGB"],
            "rmse": [1.2, 1.3],
            "auc": [0.71, 0.73],
        }
    )


## Public ##

@pytest.mark.order(670)
def test_format_consensus_label_variants():
    assert ocperf._format_consensus_label("mean") == "Mean consensus"
    assert ocperf._format_consensus_label("iqr") == "IQR consensus"
    assert ocperf._format_consensus_label("quantile_25") == "Quantile 25 consensus"


@pytest.mark.order(671)
def test_compute_combined_metrics_merges_raw_and_consensus(monkeypatch):
    monkeypatch.setattr(
        ocperf.ocscoredata,
        "preprocess_df",
        lambda _p: (pd.DataFrame({"ligand": ["L"]}), pd.DataFrame({"name": ["N"]}), ["SMINA_score", "VINA_score"]),
    )
    monkeypatch.setattr(
        ocperf.ocseval,
        "compute_auc",
        lambda *_a, **_k: pd.DataFrame({"score_column": ["SMINA_score", "VINA_score"], "AUC": [0.70, 0.75]}),
    )
    monkeypatch.setattr(
        ocperf.ocseval,
        "compute_rmse",
        lambda *_a, **_k: pd.DataFrame({"score_column": ["SMINA_score", "VINA_score"], "RMSE": [1.20, 1.10]}),
    )
    monkeypatch.setattr(
        ocperf.ocsimple,
        "perform_simple_consensus",
        lambda *_a, **_k: pd.DataFrame({"RMSE": [1.0, 1.1], "AUC": [0.76, 0.74]}, index=["mean", "iqr"]),
    )

    out = ocperf.compute_combined_metrics("dummy.csv.gz", metrics=["mean", "iqr"])

    assert {"study_name", "Methodology", "RMSE", "AUC", "combined_metric"}.issubset(out.columns)
    assert "Raw Scoring Function" in out["Methodology"].values
    assert "Mean consensus" in out["Methodology"].values
    assert "IQR consensus" in out["Methodology"].values


@pytest.mark.order(672)
def test_get_all_lists_lengths_are_consistent():
    snames, nn_len, xgb_len = ocperf.get_all_lists()
    assert isinstance(snames, list)
    assert len(snames) > 50
    assert nn_len == 5
    assert xgb_len == 5
    assert any(name.startswith("NN_Optimization_") for name in snames)
    assert any(name.startswith("XGB_Optimization_") for name in snames)


@pytest.mark.order(673)
def test_get_feature_matrix_drops_known_metadata_and_score_columns(tmp_path):
    p = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "receptor": ["r"],
            "ligand": ["l"],
            "name": ["n"],
            "type": ["active"],
            "db": ["d"],
            "experimental": [1.0],
            "label": [1],
            "Methodology": ["Raw"],
            "study_name": ["s1"],
            "RMSE": [1.2],
            "AUC": [0.7],
            "combined_metric": [0.5],
            "SMINA_score_1": [-7.2],
            "score_custom": [3.1],
            "desc1": [10.0],
            "desc2": [20.0],
        }
    ).to_csv(p, index=False)

    feat = ocperf.get_feature_matrix(str(p))
    assert list(feat.columns) == ["desc1", "desc2"]


@pytest.mark.order(674)
def test_setup_dirs_respects_doc_build_env(monkeypatch):
    calls = []
    monkeypatch.setenv("OC_BUILD_DOCS", "1")
    monkeypatch.setattr(ocperf.os, "makedirs", lambda *a, **k: calls.append((a, k)))
    ocperf.setup_dirs()
    assert calls == []

    monkeypatch.delenv("OC_BUILD_DOCS", raising=False)
    ocperf.setup_dirs()
    assert len(calls) == 2
    assert calls[0][0][0] == "plots"
    assert calls[1][0][0] == "csvs"


@pytest.mark.order(675)
def test_run_full_analysis_executes_summary_and_feature_paths(monkeypatch, tmp_path):
    calls = {
        "setup": 0,
        "plot_scatter": 0,
        "plot_box": 0,
        "plot_panels": 0,
        "stats": 0,
        "corr": 0,
        "pca": 0,
    }

    monkeypatch.setattr(ocperf, "setup_dirs", lambda: calls.__setitem__("setup", calls["setup"] + 1))
    monkeypatch.setattr(ocperf, "compute_combined_metrics", lambda *_a, **_k: _metrics_df())
    monkeypatch.setattr(ocperf, "get_all_lists", lambda: (["NN_Optimization_1", "XGB_Optimization_1"], 1, 1))
    monkeypatch.setattr(ocperf.ocstudy, "analyze_studies", lambda *_a, **_k: (_studies_df(), _studies_df(), _studies_df()))
    monkeypatch.setattr(ocperf.ocstatcolour, "set_color_mapping", lambda *_a, **_k: {"NN": "blue", "XGB": "green"})
    monkeypatch.setattr(ocperf.ocstatplot, "plot_combined_metric_scatter", lambda *_a, **_k: calls.__setitem__("plot_scatter", calls["plot_scatter"] + 1))
    monkeypatch.setattr(ocperf.ocstatplot, "plot_boxplots", lambda *_a, **_k: calls.__setitem__("plot_box", calls["plot_box"] + 1))
    monkeypatch.setattr(ocperf.ocstatplot, "plot_scatterplot", lambda *_a, **_k: calls.__setitem__("plot_panels", calls["plot_panels"] + 1))
    monkeypatch.setattr(ocperf.ocstat, "run_statistical_tests", lambda *_a, **_k: calls.__setitem__("stats", calls["stats"] + 1))
    monkeypatch.setattr(ocperf.occorrana, "correlation_analysis", lambda *_a, **_k: calls.__setitem__("corr", calls["corr"] + 1))
    monkeypatch.setattr(ocperf, "get_feature_matrix", lambda *_a, **_k: pd.DataFrame({"d1": [1.0], "d2": [2.0]}))
    monkeypatch.setattr(ocperf.ocstat, "run_pca_analysis", lambda *_a, **_k: calls.__setitem__("pca", calls["pca"] + 1))

    ocperf.run_full_analysis(
        df_path="dummy.csv.gz",
        base_path=str(tmp_path),
        storage_str="sqlite://",
        trials_list=[1],
        output_dir=str(tmp_path),
        consensus_metrics=["mean"],
        show_consensus=True,
        feature_analysis=True,
        plot_summary=True,
    )

    assert calls["setup"] == 1
    assert calls["plot_scatter"] == 1
    assert calls["plot_box"] == 1
    assert calls["plot_panels"] == 1
    assert calls["stats"] == 1
    assert calls["corr"] == 1
    assert calls["pca"] == 1


@pytest.mark.order(676)
def test_run_full_analysis_skips_optional_paths_when_disabled(monkeypatch):
    calls = {"plot": 0, "pca": 0}
    monkeypatch.setattr(ocperf, "setup_dirs", lambda: None)
    monkeypatch.setattr(ocperf, "compute_combined_metrics", lambda *_a, **_k: _metrics_df())
    monkeypatch.setattr(ocperf, "get_all_lists", lambda: (["NN_Optimization_1"], 1, 1))
    monkeypatch.setattr(ocperf.ocstudy, "analyze_studies", lambda *_a, **_k: (_studies_df(), _studies_df(), _studies_df()))
    monkeypatch.setattr(ocperf.ocstatcolour, "set_color_mapping", lambda *_a, **_k: {"NN": "blue"})
    monkeypatch.setattr(ocperf.ocstatplot, "plot_combined_metric_scatter", lambda *_a, **_k: calls.__setitem__("plot", calls["plot"] + 1))
    monkeypatch.setattr(ocperf.ocstatplot, "plot_boxplots", lambda *_a, **_k: calls.__setitem__("plot", calls["plot"] + 1))
    monkeypatch.setattr(ocperf.ocstatplot, "plot_scatterplot", lambda *_a, **_k: calls.__setitem__("plot", calls["plot"] + 1))
    monkeypatch.setattr(ocperf.ocstat, "run_statistical_tests", lambda *_a, **_k: None)
    monkeypatch.setattr(ocperf.occorrana, "correlation_analysis", lambda *_a, **_k: None)
    monkeypatch.setattr(ocperf.ocstat, "run_pca_analysis", lambda *_a, **_k: calls.__setitem__("pca", calls["pca"] + 1))

    ocperf.run_full_analysis(
        df_path="dummy.csv.gz",
        base_path=".",
        storage_str="sqlite://",
        trials_list=[1],
        feature_analysis=False,
        plot_summary=False,
    )

    assert calls["plot"] == 0
    assert calls["pca"] == 0
