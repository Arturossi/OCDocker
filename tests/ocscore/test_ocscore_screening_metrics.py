#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for DUDEz screening metric evaluation helpers."""

# Imports
###############################################################################
import numpy as np
import pytest

import optuna

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocranking
import OCDocker.OCScore.Optimization.StagedOptuna as ocstaged

# License
###############################################################################
"""
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
"""


# Functions
###############################################################################
## Public ##

@pytest.mark.order(259)
def test_bedroc_ranking_sanity_perfect_random_worst_and_inverted():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    perfect_scores = np.array([10.0, 9.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    worst_scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    random_scores = np.array([0.2, 0.9, 0.1, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0])

    perfect = ocranking.bedroc(y_true, perfect_scores)
    random_value = ocranking.bedroc(y_true, random_scores)
    worst = ocranking.bedroc(y_true, worst_scores)
    inverted = ocranking.bedroc(y_true, ocranking.orient_scores(worst_scores, higher_is_better=False))

    assert perfect > random_value > worst
    assert inverted > random_value


@pytest.mark.order(260)
def test_constant_global_scores_are_invalid_for_early_enrichment_metrics():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    constant_scores = np.full(10, 0.25)

    assert not ocranking.is_valid_ranking_scores(constant_scores)
    assert np.isnan(ocranking.bedroc(y_true, constant_scores))
    assert np.isnan(ocranking.enrichment_factor(y_true, constant_scores, 0.01))
    assert np.isnan(ocranking.ndcg_at_fraction(y_true, constant_scores, 0.01))

    metrics = ocstaged.evaluate_screening_metrics(y_true, constant_scores, groups=None)
    assert metrics["ranking_metrics_valid"] == 0.0
    assert np.isnan(metrics["BEDROC"])
    assert np.isnan(metrics["EF1%"])
    assert np.isnan(metrics["NDCG@1%"])


@pytest.mark.order(261)
def test_constant_grouped_scores_are_invalid_when_ligands_precede_decoys():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=int)
    groups = np.array(["r1", "r1", "r1", "r1", "r2", "r2", "r2", "r2"])
    constant_scores = np.full(8, 0.5)

    metrics = ocstaged.evaluate_screening_metrics(
        y_true,
        constant_scores,
        groups=groups,
        higher_is_better=True,
    )

    assert metrics["ranking_metrics_valid"] == 0.0
    assert metrics["n_groups_invalid_constant_score"] == 2.0
    assert metrics["n_groups_used"] == 0.0
    assert np.isnan(metrics["BEDROC"])
    assert np.isnan(metrics["NDCG@1%"])
    assert np.isnan(metrics["EF1%"])


@pytest.mark.order(262)
def test_constant_scores_metrics_are_invariant_to_row_order():
    y_true = np.array([1, 1, 0, 0, 0, 0], dtype=int)
    constant_scores = np.full(6, 0.25)
    reversed_true = y_true[::-1]
    reversed_scores = constant_scores[::-1]

    first = ocstaged.evaluate_screening_metrics(y_true, constant_scores, groups=None)
    second = ocstaged.evaluate_screening_metrics(reversed_true, reversed_scores, groups=None)

    for key in ("BEDROC", "EF1%", "NDCG@1%", "ranking_metrics_valid"):
        left = first[key]
        right = second[key]
        if np.isnan(left) and np.isnan(right):
            continue
        assert left == right


@pytest.mark.order(263)
def test_dudez_objective_prunes_when_bedroc_ranking_is_invalid():
    metrics = {
        "BEDROC": 0.9999999999999997,
        "PR-AUC": 0.25,
        "ranking_metrics_valid": 0.0,
        "n_groups_used": 0.0,
        "score_std": 0.0,
        "n_unique_scores": 1.0,
    }

    with pytest.raises(optuna.exceptions.TrialPruned):
        ocstaged._resolve_dudez_objective_value("BEDROC", metrics)


@pytest.mark.order(264)
def test_evaluate_screening_metrics_uses_grouped_enrichment_when_groups_present():
    y_true = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
    per_target_good = np.array([10.0, 0.0, 0.0] * 4)
    per_target_bad = np.array([0.0, 10.0, 10.0] * 4)

    grouped_good = ocstaged.evaluate_screening_metrics(
        y_true,
        per_target_good,
        groups=groups,
        higher_is_better=True,
    )
    grouped_bad = ocstaged.evaluate_screening_metrics(
        y_true,
        per_target_bad,
        groups=groups,
        higher_is_better=True,
    )

    assert grouped_good["n_groups_used"] == 4.0
    assert grouped_good["ranking_metrics_valid"] == 1.0
    assert grouped_good["BEDROC"] > grouped_bad["BEDROC"]
    assert grouped_bad["BEDROC"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.order(265)
def test_bedroc_missing_or_invalid_groups_fall_back_to_pr_auc_only_when_ranking_valid():
    metrics = {"BEDROC": float("nan"), "PR-AUC": 0.42, "ranking_metrics_valid": 1.0}
    assert ocstaged.resolve_dudez_primary_metric("BEDROC", metrics) == "PR-AUC"

    invalid_groups = {
        "BEDROC": 0.1,
        "PR-AUC": 0.42,
        "n_groups_used": 0.0,
        "ranking_metrics_valid": 0.0,
    }
    assert ocstaged.resolve_dudez_primary_metric("BEDROC", invalid_groups) == "BEDROC"
    assert ocstaged.resolve_dudez_primary_metric("PR-AUC", invalid_groups) == "PR-AUC"


@pytest.mark.order(266)
def test_evaluate_screening_metrics_exposes_ranking_diagnostics():
    y_true = np.array([1, 0, 0, 1, 0, 0], dtype=int)
    scores = np.array([3.0, 1.0, 0.5, 2.5, 0.2, 0.1])
    groups = np.array(["a", "a", "a", "b", "b", "b"])

    metrics = ocstaged.evaluate_screening_metrics(y_true, scores, groups=groups)

    assert metrics["score_std"] > 0.0
    assert metrics["n_unique_scores"] >= 2.0
    assert metrics["ranking_metrics_valid"] == 1.0
    assert metrics["n_groups_invalid_constant_score"] == 0.0


@pytest.mark.order(266)
def test_grouped_screening_metrics_do_not_raise_for_one_class_validation_fold():
    y_true = np.array([0, 0, 0, 0], dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    groups = np.array(["r1", "r1", "r2", "r2"])

    metrics = ocranking.evaluate_screening_metrics(y_true, scores, groups=groups)

    assert metrics["ranking_metrics_valid"] == 0.0
    assert metrics["n_groups_used"] == 0.0
    assert metrics["n_groups_invalid_one_class"] == 2.0
    assert np.isnan(metrics["ROC-AUC_group_mean"])
    assert np.isnan(metrics["PR-AUC_group_mean"])
    assert np.isnan(metrics["BEDROC"])


@pytest.mark.order(267)
def test_summarize_dudez_split_diagnostics_flags_single_class_targets():
    labels = np.array([1, 0, 0, 1, 0, 0], dtype=int)
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    diagnostics = ocstaged.summarize_dudez_split_diagnostics(
        df=None,
        labels=labels,
        train_idx=np.array([0, 1, 2]),
        val_idx=np.array([3]),
        test_idx=np.array([4, 5]),
        groups=groups,
        target_group_column="receptor",
    )

    assert diagnostics["target_group_column"] == "receptor"
    assert diagnostics["splits"]["validation"]["targets_with_zero_decoys"] == ["b"]
    assert diagnostics["splits"]["test"]["targets_with_zero_actives"] == ["b"]


@pytest.mark.order(268)
def test_evaluate_screening_metrics_by_group_returns_one_row_per_receptor():
    y_true = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    scores = np.array([5.0, 1.0, 0.5, 4.0, 0.2, 0.1, 3.0, 0.3])
    groups = np.array(["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3"])

    result = ocranking.evaluate_screening_metrics_by_group(y_true, scores, groups)

    assert len(result) == 3
    assert set(result["group"]) == {"r1", "r2", "r3"}
    assert np.isfinite(result["BEDROC"]).all()


@pytest.mark.order(269)
def test_evaluate_screening_metrics_by_group_rejects_length_mismatch():
    with pytest.raises(ValueError, match="groups must have the same length"):
        ocranking.evaluate_screening_metrics_by_group(
            np.array([1, 0]),
            np.array([0.1, 0.2]),
            np.array(["a"]),
        )


@pytest.mark.order(270)
def test_evaluate_screening_metrics_includes_classification_and_confusion():
    y_true = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=int)
    scores = np.array([10.0, 9.0, 1.0, 2.0, 3.0, 4.0, 8.0, 5.0, 6.0, 7.0])

    metrics = ocranking.evaluate_screening_metrics(y_true, scores, groups=None)

    assert metrics["PR-AUC"] > 0.5
    assert 0.0 <= metrics["Precision"] <= 1.0
    assert 0.0 <= metrics["Recall"] <= 1.0
    assert 0.0 <= metrics["F1"] <= 1.0
    assert metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"] == pytest.approx(len(y_true))
    assert np.isfinite(metrics["classification_threshold"])


@pytest.mark.order(271)
def test_evaluate_screening_metrics_by_group_includes_per_receptor_confusion():
    y_true = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    scores = np.array([5.0, 1.0, 0.5, 4.0, 0.2, 0.1, 3.0, 0.3, 0.2])
    groups = np.array(["r1", "r1", "r1", "r2", "r2", "r2", "r3", "r3", "r3"])

    result = ocranking.evaluate_screening_metrics_by_group(
        y_true,
        scores,
        groups,
        metric_names=(
            "ROC-AUC",
            "PR-AUC",
            "Precision",
            "Recall",
            "F1",
            "MCC",
            "TP",
            "FP",
            "TN",
            "FN",
        ),
    )

    assert len(result) == 3
    group_sizes = (
        result.groupby("group")[["TP", "FP", "TN", "FN"]].sum().sum(axis=1).to_dict()
    )
    assert group_sizes == {"r1": 3.0, "r2": 3.0, "r3": 3.0}
    for _, row in result.iterrows():
        assert 0.0 <= row["Precision"] <= 1.0


@pytest.mark.order(272)
def test_grouped_screening_metrics_use_macro_classification_and_pooled_confusion():
    y_true = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=int)
    groups = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c", "d", "d", "d"])
    scores = np.array([10.0, 0.0, 0.0] * 4)

    metrics = ocranking.evaluate_screening_metrics(y_true, scores, groups=groups)

    assert metrics["TP"] + metrics["FP"] + metrics["TN"] + metrics["FN"] == pytest.approx(len(y_true))
    assert np.isfinite(metrics["Precision"])
    assert np.isfinite(metrics["Precision_global"])
    assert metrics["n_groups_classification_used"] == 4.0
