#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore ranking metrics helpers.
'''

# Imports
###############################################################################
import numpy as np

import pytest

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocranking

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

@pytest.mark.order(255)
def test_validate_raises_for_length_mismatch_and_single_class(monkeypatch):
    calls = []
    monkeypatch.setattr(ocranking.ocerror.Error, "value_error", lambda msg: calls.append(msg))

    with pytest.raises(ValueError, match="same length"):
        ocranking._validate(np.array([0, 1]), np.array([0.8]))

    with pytest.raises(ValueError, match="both classes"):
        ocranking._validate(np.array([1, 1]), np.array([0.2, 0.8]))

    assert len(calls) == 2


@pytest.mark.order(256)
def test_ranking_metrics_happy_paths_and_clamped_fractions():
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=float)

    assert ocranking.roc_auc(y_true, y_score) == pytest.approx(1.0)
    assert ocranking.pr_auc(y_true, y_score) == pytest.approx(1.0)
    assert ocranking.top_k_precision(y_true, y_score, k=2) == pytest.approx(1.0)
    assert ocranking.top_fraction_precision(y_true, y_score, frac=0.5) == pytest.approx(1.0)
    assert ocranking.top_fraction_precision(y_true, y_score, frac=-5.0) == pytest.approx(1.0)
    assert ocranking.top_fraction_precision(y_true, y_score, frac=2.0) == pytest.approx(0.5)
    assert ocranking.riep(y_true, y_score, k=2) == pytest.approx(2.0 / 3.0)
    assert ocranking.enrichment_factor(y_true, y_score, fraction=0.5) == pytest.approx(2.0)
    assert np.isfinite(ocranking.bedroc(y_true, y_score, alpha=20.0))


@pytest.mark.order(257)
def test_threshold_and_groupwise_macro_micro_paths():
    y_true = np.array([1, 1, 1, 0, 1, 0], dtype=int)
    y_score = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45], dtype=float)

    thr, prec, rec = ocranking.threshold_at_precision(y_true, y_score, target_precision=0.7)
    assert np.isfinite(thr)
    assert np.isfinite(prec)
    assert np.isfinite(rec)
    assert prec >= 0.7

    thr2, prec2, rec2 = ocranking.threshold_at_precision(y_true, y_score, target_precision=1.1)
    assert np.isnan(thr2)
    assert np.isnan(prec2)
    assert np.isnan(rec2)

    grouped = ocranking.groupwise(y_true, y_score, groups=["a", "a", "a", "b", "b", "b"])
    assert set(grouped) == {
        "roc_auc_macro",
        "pr_auc_macro",
        "roc_auc_micro",
        "pr_auc_micro",
    }
    assert 0.0 <= grouped["roc_auc_micro"] <= 1.0
    assert 0.0 <= grouped["pr_auc_micro"] <= 1.0


@pytest.mark.order(258)
def test_groupwise_all_single_class_groups_yields_nan_macro_and_bedroc_nan_branch(monkeypatch):
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.2, 0.8, 0.3, 0.7], dtype=float)

    grouped = ocranking.groupwise(y_true, y_score, groups=["g1", "g2", "g3", "g4"])
    assert np.isnan(grouped["roc_auc_macro"])
    assert np.isnan(grouped["pr_auc_macro"])

    monkeypatch.setattr(
        ocranking,
        "_validate",
        lambda _y_true, _y_score: (
            np.array([0, 0, 0], dtype=int),
            np.array([0.1, 0.2, 0.3], dtype=float),
        ),
    )
    assert np.isnan(ocranking.bedroc(y_true, y_score))


@pytest.mark.order(258)
def test_groupwise_global_single_class_yields_nan_auc_values():
    y_true = np.array([0, 0, 0, 0], dtype=int)
    y_score = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    grouped = ocranking.groupwise(y_true, y_score, groups=["g1", "g1", "g2", "g2"])

    assert np.isnan(grouped["roc_auc_macro"])
    assert np.isnan(grouped["pr_auc_macro"])
    assert np.isnan(grouped["roc_auc_micro"])
    assert np.isnan(grouped["pr_auc_micro"])
