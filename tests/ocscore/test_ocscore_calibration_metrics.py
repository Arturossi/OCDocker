#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for OCScore probability calibration metrics and calibrators."""

# Imports
###############################################################################
import numpy as np
import pytest

import OCDocker.OCScore.Analysis.Metrics.Calibration as occal
import OCDocker.OCScore.Analysis.Metrics.Bootstrap as ocboot


# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""


# Functions
###############################################################################
## Public ##

@pytest.mark.order(283)
def test_platt_calibration_improves_brier_on_shifted_logits():
    rng = np.random.default_rng(0)
    n = 400
    logits = rng.normal(size=n)
    y_true = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    shifted = logits * 2.5

    uncal = occal.evaluate_calibration_metrics(y_true, occal.logits_to_probabilities(shifted))
    calibrator = occal.ProbabilityCalibrator.fit(y_true, shifted, method="platt")
    calibrated = occal.evaluate_calibration_metrics(y_true, calibrator.predict(shifted))

    assert np.isfinite(uncal["Brier"])
    assert calibrated["Brier"] <= uncal["Brier"]


@pytest.mark.order(284)
def test_merge_calibration_metrics_adds_calibrated_suffix():
    y_true = np.array([1, 1, 0, 0, 0, 0], dtype=int)
    logits = np.array([2.0, 1.0, -1.0, -2.0, -0.5, -1.5])
    metrics: dict[str, float] = {}
    calibrator = occal.ProbabilityCalibrator.fit(y_true[:4], logits[:4], method="isotonic")
    occal.merge_calibration_metrics(metrics, y_true, logits, calibrator=calibrator)

    assert "Brier" in metrics
    assert "Brier_calibrated" in metrics
    assert metrics["calibration_method"] == "isotonic"


@pytest.mark.order(285)
def test_enrich_dudez_export_metrics_fits_on_validation_only():
    val_true = np.array([1, 0, 0, 1, 0, 0], dtype=int)
    val_logits = np.array([1.5, -1.0, -0.5, 2.0, -2.0, -1.5])
    test_true = np.array([1, 0, 0, 0], dtype=int)
    test_logits = np.array([1.0, -0.5, -1.0, -2.0])

    val_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}
    calibrator = occal.enrich_dudez_export_metrics(
        val_metrics,
        test_metrics,
        val_true=val_true,
        val_scores=val_logits,
        test_true=test_true,
        test_scores=test_logits,
        calibration_method="platt",
    )
    assert calibrator.method == "platt"
    assert np.isfinite(val_metrics["diagnostic_ECE"])
    assert np.isfinite(test_metrics["diagnostic_ECE_calibrated"])
    assert "ECE" not in val_metrics
    assert occal.validate_calibration_report_mode(val_metrics, "ranking_only") == []


@pytest.mark.order(286)
def test_ranking_only_strict_rejects_unprefixed_calibration_keys():
    metrics = {"Brier": 0.2, "ROC-AUC": 0.9}
    with pytest.raises(ValueError, match="diagnostic_"):
        occal.validate_calibration_report_mode(metrics, "ranking_only", strict=True)


@pytest.mark.order(287)
def test_calibration_validated_mode_keeps_calibration_keys():
    metrics = {"Brier": 0.2, "ROC-AUC": 0.9}
    occal.apply_calibration_report_mode_to_metrics(metrics, "calibration_validated")
    assert metrics["Brier"] == 0.2


@pytest.mark.order(288)
def test_build_calibration_report_section_ranking_only():
    val_metrics = {"diagnostic_Brier": 0.2, "ROC-AUC": 0.9}
    test_metrics = {"diagnostic_Brier_calibrated": 0.15}
    section = occal.build_calibration_report_section(val_metrics, test_metrics, mode="ranking_only")
    assert section["mode"] == "ranking_only"
    assert section["primary_claim"] == "ranking_screening"
    assert "disclaimer" in section
    assert section["validation"]["diagnostic_Brier"] == 0.2

@pytest.mark.order(289)
def test_bootstrap_ci_unstratified_and_no_bootstrap_paths():
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_score = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    def metric_fn(labels, scores):
        oriented = np.where(labels == 1, scores, -scores)
        return float(np.mean(oriented))

    estimate, low, high = ocboot.bootstrap_ci(
        y_true,
        y_score,
        metric_fn,
        n_boot=20,
        random_state=0,
    )
    assert estimate == pytest.approx(0.35)
    assert low <= estimate <= high

    estimate_no_boot, low_no_boot, high_no_boot = ocboot.bootstrap_ci(
        y_true,
        y_score,
        metric_fn,
        n_boot=0,
        random_state=0,
    )
    assert estimate_no_boot == pytest.approx(0.35)
    assert np.isnan(low_no_boot)
    assert np.isnan(high_no_boot)


@pytest.mark.order(290)
def test_bootstrap_ci_stratified_preserves_group_sampling():
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    y_score = np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.9], dtype=float)
    strata = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)

    def metric_fn(labels, scores):
        oriented = np.where(labels == 1, scores, -scores)
        return float(np.mean(oriented))

    estimate, low, high = ocboot.bootstrap_ci(
        y_true,
        y_score,
        metric_fn,
        n_boot=20,
        alpha=0.1,
        random_state=1,
        strata=strata,
    )

    assert np.isfinite(estimate)
    assert np.isfinite(low)
    assert np.isfinite(high)
    assert low <= high

