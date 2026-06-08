#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis.legacy.RankingMetrics module.
'''

# Imports
###############################################################################
import numpy as np
import pandas as pd

import pytest

import OCDocker.OCScore.Analysis.legacy.RankingMetrics as ocrank

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

def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "active": [1, 0, 1, 0, 1, 0],
            "m1": [0.9, 0.2, 0.8, 0.1, 0.7, 0.3],
            "m2": [0.1, 0.8, 0.2, 0.9, 0.3, 0.7],
        }
    )


## Public ##

@pytest.mark.order(677)
def test_binary_conversion_and_flip_helpers():
    assert np.array_equal(ocrank._to_binary(pd.Series([True, False]), None), np.array([1, 0]))
    assert np.array_equal(ocrank._to_binary(pd.Series([0, 1, 1]), None), np.array([0, 1, 1]))
    assert np.array_equal(ocrank._to_binary(pd.Series(["inactive", "active"]), "active"), np.array([0, 1]))
    assert np.array_equal(ocrank._to_binary(pd.Series(["ligand", "decoy"]), None), np.array([1, 0]))

    y = np.array([0, 1, 0, 1], dtype=int)
    s = np.array([0.9, 0.1, 0.8, 0.2], dtype=float)  # deliberately inverted
    assert ocrank._decide_flip(y, s) is True
    assert np.allclose(ocrank._apply_flip(s, True), -s)
    assert np.allclose(ocrank._apply_flip(s, False), s)


@pytest.mark.order(678)
def test_safe_metric_and_bootstrap_ci_edge_paths():
    y = np.array([0, 1, 0, 1], dtype=int)
    s = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    assert np.isnan(ocrank._safe_metric(lambda *_a: (_ for _ in ()).throw(ValueError("x")), y, s))
    assert np.isnan(ocrank._safe_metric(lambda *_a: (_ for _ in ()).throw(TypeError("x")), y, s))

    deg_ci = ocrank._bootstrap_ci_on_scores(np.array([1, 1]), np.array([0.2, 0.8]), ocrank.roc_auc_score, n_boot=8, seed=3)
    assert np.isnan(deg_ci.point)

    ok_ci = ocrank._bootstrap_ci_on_scores(y, s, ocrank.roc_auc_score, n_boot=8, seed=3)
    assert np.isfinite(ok_ci.point)


@pytest.mark.order(679)
def test_efroc_helpers_cover_regular_and_degenerate_cases():
    y_ok = np.array([0, 1, 0, 1], dtype=int)
    s_ok = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)
    out_ok = ocrank._efroc(y_ok, s_ok, epsilons=[0.01, 0.1])
    assert list(out_ok.columns) == ["epsilon", "ef_roc", "tpr_at_epsilon"]
    assert out_ok.shape[0] == 2

    y_bad = np.array([1, 1, 1], dtype=int)
    s_bad = np.array([0.1, 0.2, 0.3], dtype=float)
    out_bad = ocrank._efroc(y_bad, s_bad, epsilons=[0.05])
    assert np.isnan(out_bad.iloc[0]["ef_roc"])

    ci_ok = ocrank._efroc_bootstrap_ci(y_ok, s_ok, epsilons=[0.05], n_boot=8, seed=4)
    assert {"ci_low", "ci_high"}.issubset(ci_ok.columns)

    ci_bad = ocrank._efroc_bootstrap_ci(y_bad, s_bad, epsilons=[0.05], n_boot=8, seed=4)
    assert np.isnan(ci_bad.iloc[0]["ci_low"])
    assert np.isnan(ci_bad.iloc[0]["ci_high"])


@pytest.mark.order(680)
def test_public_metric_tables_per_target_and_pooled():
    df = _toy_df()
    score_cols = ["m1", "m2"]

    roc_t = ocrank.roc_auc_per_target(df, "target", "active", score_cols, n_boot=10, seed=1, auto_flip=True)
    pr_t = ocrank.pr_auc_per_target(df, "target", "active", score_cols, n_boot=10, seed=1, auto_flip=True)
    ef_t = ocrank.efroc_per_target(df, "target", "active", score_cols, epsilons=[0.01, 0.05], n_boot=8, seed=1, auto_flip=True)

    roc_p = ocrank.roc_auc_pooled(df, "active", score_cols, n_boot=10, seed=1, auto_flip=True)
    pr_p = ocrank.pr_auc_pooled(df, "active", score_cols, n_boot=10, seed=1, auto_flip=True)
    ef_p = ocrank.efroc_pooled(df, "active", score_cols, epsilons=[0.01, 0.05], n_boot=8, seed=1, auto_flip=True)

    assert not roc_t.empty
    assert not pr_t.empty
    assert not ef_t.empty
    assert not roc_p.empty
    assert not pr_p.empty
    assert not ef_p.empty
    assert {"target", "model", "roc_auc"}.issubset(roc_t.columns)
    assert {"model", "pr_auc"}.issubset(pr_p.columns)
    assert {"model", "epsilon", "ef_roc"}.issubset(ef_p.columns)


@pytest.mark.order(681)
def test_build_test2_tables_and_summary_table_with_pr_auc():
    df = _toy_df()
    models = ["m1", "m2"]
    tables = ocrank.build_test2_tables(
        df=df,
        models=models,
        target_col="target",
        label_col="active",
        n_boot=10,
        seed=2,
        epsilons=(0.01, 0.05),
        auto_flip=True,
    )

    assert {
        "roc_auc_per_target",
        "pr_auc_per_target",
        "efroc_per_target",
        "roc_auc_pooled",
        "pr_auc_pooled",
        "efroc_pooled",
        "summary",
    }.issubset(tables.keys())
    assert not tables["summary"].empty

    ef_t = tables["efroc_per_target"].copy()
    ef_t["metric"] = ef_t["epsilon"].map(lambda e: f"EF_ROC_{int(round(float(e) * 100))}%")
    summary_targets = (
        ef_t.groupby(["model", "metric"], as_index=False)
        .agg(
            median_across_targets=("ef_roc", "median"),
            CI95_lo=("ci_low", "median"),
            CI95_hi=("ci_high", "median"),
        )
    )

    ef_p = tables["efroc_pooled"].copy()
    ef_p["metric"] = ef_p["epsilon"].map(lambda e: f"EF_ROC_{int(round(float(e) * 100))}%")
    summary_pooled = ef_p.rename(columns={"ef_roc": "pooled_value", "ci_low": "CI95_lo", "ci_high": "CI95_hi"})[
        ["model", "metric", "pooled_value", "CI95_lo", "CI95_hi"]
    ]

    pr_t = (
        tables["pr_auc_per_target"]
        .groupby("model", as_index=False)
        .agg(median_across_targets=("pr_auc", "median"), CI95_lo=("ci_low", "median"), CI95_hi=("ci_high", "median"))
        .assign(metric="PR_AUC")
    )
    pr_p = tables["pr_auc_pooled"].rename(columns={"pr_auc": "pooled_value", "ci_low": "CI95_lo", "ci_high": "CI95_hi"}).assign(metric="PR_AUC")[
        ["model", "metric", "pooled_value", "CI95_lo", "CI95_hi"]
    ]

    out = ocrank.build_summary_table(
        summary_targets=summary_targets,
        summary_pooled=summary_pooled,
        models=models,
        eps=(1, 5),
        include_pr_auc=True,
        pr_summary_targets=pr_t,
        pr_summary_pooled=pr_p,
    )

    assert "Median EF-ROC 1%" in out.columns
    assert "Pooled EF-ROC 5%" in out.columns
    assert "Median PR-AUC" in out.columns
    assert "Pooled PR-AUC" in out.columns
