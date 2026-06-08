#!/usr/bin/env python3

# Description
###############################################################################
"""Tests for OCScore probability calibration metrics and calibrators."""

# Imports
###############################################################################
import numpy as np
import pytest

import OCDocker.OCScore.Analysis.Metrics.Calibration as occal


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
