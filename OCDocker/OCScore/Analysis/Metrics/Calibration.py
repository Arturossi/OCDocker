#!/usr/bin/env python3

# Description
###############################################################################
'''
Probability calibration metrics and post-hoc calibrators for OCScore classifiers.

Fits Platt scaling or isotonic regression on training/validation logits only, then
reports Brier score, log loss, and expected calibration error (ECE) on evaluation splits.
'''

# Imports
###############################################################################
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

import OCDocker.Error as ocerror

CalibrationMethod = Literal["platt", "isotonic"]
CalibrationReportMode = Literal["ranking_only", "calibration_validated"]

DEFAULT_CALIBRATION_METHOD: CalibrationMethod = "platt"
DEFAULT_CALIBRATION_REPORT_MODE: CalibrationReportMode = "ranking_only"
DEFAULT_CALIBRATION_N_BINS = 10
CALIBRATION_METRIC_NAMES = ("Brier", "Log-loss", "ECE")
CALIBRATED_METRIC_SUFFIX = "_calibrated"
DIAGNOSTIC_CALIBRATION_PREFIX = "diagnostic_"
RANKING_ONLY_CALIBRATION_DISCLAIMER = (
    "Post-hoc calibration metrics are diagnostic only and must not be interpreted "
    "as validated probability estimates for screening claims."
)

PROBABILITY_CLIP_EPSILON = 1e-15
LOGIT_CLIP = 50.0


# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''


# Functions
###############################################################################
## Public ##

def logits_to_probabilities(logits: np.ndarray) -> np.ndarray:
    '''Map classifier logits to probabilities with a numerically stable sigmoid.

    Parameters
    ----------
    logits : np.ndarray
        One-dimensional classifier logits.

    Returns
    -------
    np.ndarray
        Probabilities in ``(0, 1)`` with clipping applied for stability.
    '''

    logits = np.asarray(logits, dtype=float).reshape(-1)
    clipped = np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


def clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    '''Clip probabilities into the open unit interval for log loss.

    Parameters
    ----------
    probabilities : np.ndarray
        Raw predicted probabilities.

    Returns
    -------
    np.ndarray
        Probabilities clipped to ``(eps, 1 - eps)``.
    '''

    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    return np.clip(probs, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)


def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        n_bins: int = DEFAULT_CALIBRATION_N_BINS,
    ) -> float:
    '''Compute expected calibration error with uniform probability bins.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, optional
        Number of uniform bins on ``[0, 1]``, by default 10.

    Returns
    -------
    float
        Weighted mean absolute difference between bin accuracy and confidence,
        or ``nan`` when the input is empty or single-class.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = clip_probabilities(y_prob)
    if len(y_true) == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    n_samples = float(len(y_true))
    for bin_index in range(int(n_bins)):
        low = bin_edges[bin_index]
        high = bin_edges[bin_index + 1]
        if bin_index == int(n_bins) - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += float(np.sum(mask)) / n_samples * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_calibration_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        n_bins: int = DEFAULT_CALIBRATION_N_BINS,
    ) -> dict[str, float]:
    '''Return Brier score, log loss, and ECE for probabilistic predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, optional
        Bin count for ECE, by default 10.

    Returns
    -------
    dict[str, float]
        Mapping with keys ``"Brier"``, ``"Log-loss"``, and ``"ECE"``. Values are
        ``nan`` when the input is empty or single-class.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = clip_probabilities(y_prob)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {name: float("nan") for name in CALIBRATION_METRIC_NAMES}

    return {
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Log-loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


def reliability_curve_points(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        *,
        n_bins: int = DEFAULT_CALIBRATION_N_BINS,
    ) -> tuple[np.ndarray, np.ndarray]:
    '''Return mean predicted probability and fraction of positives per bin.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, optional
        Number of calibration bins, by default 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_predicted, fraction_positives)`` per bin; empty arrays when
        calibration cannot be computed.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = clip_probabilities(y_prob)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.array([]), np.array([])
    fraction_positives, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=int(n_bins),
        strategy="uniform",
    )
    return np.asarray(mean_predicted, dtype=float), np.asarray(fraction_positives, dtype=float)


@dataclass
class ProbabilityCalibrator:
    """Post-hoc probability calibrator fit on one split and applied to others.

    Parameters
    ----------
    method : CalibrationMethod
        Calibration method: ``"platt"`` or ``"isotonic"``.
    scores_are_logits : bool, optional
        If True, ``scores`` are raw logits; otherwise they are probabilities,
        by default True.
    """

    method: CalibrationMethod
    scores_are_logits: bool = True
    _platt: Optional[LogisticRegression] = None
    _isotonic: Optional[IsotonicRegression] = None

    @classmethod
    def fit(
            cls,
            y_true: np.ndarray,
            scores: np.ndarray,
            *,
            method: CalibrationMethod = DEFAULT_CALIBRATION_METHOD,
            scores_are_logits: bool = True,
        ) -> "ProbabilityCalibrator":
        '''Fit Platt or isotonic calibration on reference labels and scores.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground-truth labels (0/1) from the fit split only.
        scores : np.ndarray
            Model scores or logits aligned with ``y_true``.
        method : CalibrationMethod, optional
            Calibration method, by default ``"platt"``.
        scores_are_logits : bool, optional
            Whether ``scores`` are logits, by default True.

        Returns
        -------
        ProbabilityCalibrator
            Fitted calibrator ready for :meth:`predict`.

        Raises
        ------
        ValueError
            If ``y_true`` is single-class or ``method`` is unsupported.
        '''

        y_true = np.asarray(y_true, dtype=int).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(np.unique(y_true)) < 2:
            ocerror.Error.value_error(
                "Calibration requires both positive and negative labels in the fit split."
            )
            raise ValueError("Calibration requires both positive and negative labels.")

        calibrator = cls(method=str(method), scores_are_logits=bool(scores_are_logits))
        uncalibrated = (
            logits_to_probabilities(scores) if scores_are_logits else clip_probabilities(scores)
        )
        if method == "platt":
            features = scores.reshape(-1, 1) if scores_are_logits else uncalibrated.reshape(-1, 1)
            model = LogisticRegression(solver="lbfgs", max_iter=1000)
            model.fit(features, y_true)
            calibrator._platt = model
        elif method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(uncalibrated, y_true)
            calibrator._isotonic = model
        else:
            raise ValueError(f"Unsupported calibration method: {method!r}")
        return calibrator

    def predict(self, scores: np.ndarray) -> np.ndarray:
        '''Return calibrated probabilities for new scores or logits.

        Parameters
        ----------
        scores : np.ndarray
            Model scores or logits for the evaluation split.

        Returns
        -------
        np.ndarray
            Calibrated probabilities in ``(0, 1)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        '''

        scores = np.asarray(scores, dtype=float).reshape(-1)
        if self.method == "platt" and self._platt is not None:
            features = (
                scores.reshape(-1, 1)
                if self.scores_are_logits
                else clip_probabilities(scores).reshape(-1, 1)
            )
            return clip_probabilities(self._platt.predict_proba(features)[:, 1])
        if self.method == "isotonic" and self._isotonic is not None:
            uncalibrated = (
                logits_to_probabilities(scores) if self.scores_are_logits else clip_probabilities(scores)
            )
            return clip_probabilities(self._isotonic.predict(uncalibrated))
        raise RuntimeError("Calibrator was not fit.")

    def to_dict(self) -> dict[str, Any]:
        '''Serialize calibrator metadata for JSON export.

        Returns
        -------
        dict[str, Any]
            JSON-serializable method name and ``scores_are_logits`` flag.
        '''

        return {
            "method": self.method,
            "scores_are_logits": self.scores_are_logits,
        }


def is_calibration_metric_key(key: str) -> bool:
    '''Return True when ``key`` names a calibration export metric.'''

    if key == "calibration_method":
        return True
    if key in CALIBRATION_METRIC_NAMES:
        return True
    return any(key == f"{name}{CALIBRATED_METRIC_SUFFIX}" for name in CALIBRATION_METRIC_NAMES)


def apply_calibration_report_mode_to_metrics(
        metrics: dict[str, Any],
        mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
    ) -> dict[str, Any]:
    '''Rename calibration keys with ``diagnostic_`` when reporting ranking-only claims.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary updated in place.
    mode : CalibrationReportMode, optional
        Report mode; ``ranking_only`` prefixes calibration keys.

    Returns
    -------
    dict[str, Any]
        The same ``metrics`` dict.
    '''

    if mode == "calibration_validated":
        return metrics
    for key in list(metrics.keys()):
        if is_calibration_metric_key(key):
            metrics[f"{DIAGNOSTIC_CALIBRATION_PREFIX}{key}"] = metrics.pop(key)
    return metrics


def collect_calibration_report_issues(
        metrics: Mapping[str, Any],
        mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
    ) -> list[str]:
    '''Return human-readable issues when calibration keys violate report mode.'''

    if mode != "ranking_only":
        return []
    issues: list[str] = []
    for key in metrics:
        if is_calibration_metric_key(key):
            issues.append(
                f"Calibration metric '{key}' is present without '{DIAGNOSTIC_CALIBRATION_PREFIX}' "
                "prefix in ranking_only report mode."
            )
    return issues


def validate_calibration_report_mode(
        metrics: Mapping[str, Any],
        mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
        *,
        strict: bool = False,
    ) -> list[str]:
    '''Validate calibration metric naming for the selected report mode.

    Raises
    ------
    ValueError
        When ``strict`` is True and ranking-only violations are found.
    '''

    issues = collect_calibration_report_issues(metrics, mode)
    if strict and issues:
        raise ValueError("; ".join(issues))
    return issues


def extract_calibration_metrics(
        metrics: Mapping[str, Any],
        mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
    ) -> dict[str, Any]:
    '''Return calibration-related entries from a metrics mapping.'''

    if mode == "calibration_validated":
        return {
            key: value
            for key, value in metrics.items()
            if is_calibration_metric_key(key)
        }
    prefix = DIAGNOSTIC_CALIBRATION_PREFIX
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(prefix)
    }


def build_calibration_report_section(
        validation_metrics: Mapping[str, Any],
        test_metrics: Mapping[str, Any],
        *,
        mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
        calibrator: Optional["ProbabilityCalibrator"] = None,
        val_true: Optional[np.ndarray] = None,
        val_scores: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
    '''Build a JSON-friendly calibration subsection for production-grade reports.'''

    section: dict[str, Any] = {
        "mode": mode,
        "primary_claim": "ranking_screening" if mode == "ranking_only" else "ranking_and_calibration",
        "validation": extract_calibration_metrics(validation_metrics, mode),
        "test": extract_calibration_metrics(test_metrics, mode),
    }
    if mode == "ranking_only":
        section["disclaimer"] = RANKING_ONLY_CALIBRATION_DISCLAIMER
    if (
        mode == "calibration_validated"
        and calibrator is not None
        and val_true is not None
        and val_scores is not None
    ):
        probabilities = calibrator.predict(val_scores)
        mean_predicted, fraction_positives = reliability_curve_points(val_true, probabilities)
        section["reliability_curve"] = {
            "mean_predicted_probability": mean_predicted.tolist(),
            "fraction_positives": fraction_positives.tolist(),
            "n_bins": int(DEFAULT_CALIBRATION_N_BINS),
        }
        section["calibration_method"] = calibrator.method
    return section


def merge_calibration_metrics(
        metrics: dict[str, Any],
        y_true: np.ndarray,
        scores: np.ndarray,
        *,
        calibrator: Optional[ProbabilityCalibrator] = None,
        scores_are_logits: bool = True,
        n_bins: int = DEFAULT_CALIBRATION_N_BINS,
        include_uncalibrated: bool = True,
        include_calibrated: bool = True,
    ) -> dict[str, Any]:
    '''Add calibration metrics to an existing metrics mapping (in place).

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary updated in place.
    y_true : np.ndarray
        Binary ground-truth labels.
    scores : np.ndarray
        Model scores or logits aligned with ``y_true``.
    calibrator : ProbabilityCalibrator, optional
        Fitted calibrator for calibrated metrics, by default None.
    scores_are_logits : bool, optional
        Whether ``scores`` are logits when no calibrator is supplied, by default True.
    n_bins : int, optional
        Bin count for ECE, by default 10.
    include_uncalibrated : bool, optional
        Write uncalibrated Brier/log-loss/ECE keys, by default True.
    include_calibrated : bool, optional
        Write ``*_calibrated`` keys when ``calibrator`` is set, by default True.

    Returns
    -------
    dict[str, Any]
        The same ``metrics`` dict, updated in place.
    '''

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    uncalibrated_prob = (
        logits_to_probabilities(scores) if scores_are_logits else clip_probabilities(scores)
    )

    if include_uncalibrated:
        for key, value in evaluate_calibration_metrics(
            y_true,
            uncalibrated_prob,
            n_bins=n_bins,
        ).items():
            metrics[key] = value

    if include_calibrated and calibrator is not None:
        calibrated_prob = calibrator.predict(scores)
        for key, value in evaluate_calibration_metrics(
            y_true,
            calibrated_prob,
            n_bins=n_bins,
        ).items():
            metrics[f"{key}{CALIBRATED_METRIC_SUFFIX}"] = value
        metrics["calibration_method"] = calibrator.method

    return metrics


def enrich_dudez_export_metrics(
        validation_metrics: dict[str, Any],
        test_metrics: dict[str, Any],
        *,
        val_true: np.ndarray,
        val_scores: np.ndarray,
        test_true: np.ndarray,
        test_scores: np.ndarray,
        calibration_method: CalibrationMethod = DEFAULT_CALIBRATION_METHOD,
        report_mode: CalibrationReportMode = DEFAULT_CALIBRATION_REPORT_MODE,
    ) -> ProbabilityCalibrator:
    '''Fit calibration on validation logits and enrich val/test metric dicts.

    Parameters
    ----------
    validation_metrics : dict[str, Any]
        Validation metrics dict updated in place.
    test_metrics : dict[str, Any]
        Test metrics dict updated in place.
    val_true, val_scores : np.ndarray
        Validation labels and logits used to fit the calibrator.
    test_true, test_scores : np.ndarray
        Test labels and logits used for calibrated test metrics.
    calibration_method : CalibrationMethod, optional
        Calibration method, by default ``"platt"``.
    report_mode : CalibrationReportMode, optional
        Controls whether calibration keys are prefixed as diagnostic-only.

    Returns
    -------
    ProbabilityCalibrator
        Fitted calibrator applied to both splits.
    '''

    calibrator = ProbabilityCalibrator.fit(
        val_true,
        val_scores,
        method=calibration_method,
        scores_are_logits=True,
    )
    merge_calibration_metrics(validation_metrics, val_true, val_scores, calibrator=calibrator)
    merge_calibration_metrics(
        test_metrics,
        test_true,
        test_scores,
        calibrator=calibrator,
        include_uncalibrated=True,
        include_calibrated=True,
    )
    apply_calibration_report_mode_to_metrics(validation_metrics, report_mode)
    apply_calibration_report_mode_to_metrics(test_metrics, report_mode)
    return calibrator


def calibration_metric_names(*, include_calibrated: bool = True) -> tuple[str, ...]:
    '''Return calibration metric column names for CSV/JSON exports.

    Parameters
    ----------
    include_calibrated : bool, optional
        Include ``*_calibrated`` suffix variants, by default True.

    Returns
    -------
    tuple[str, ...]
        Ordered metric key names.
    '''

    names = list(CALIBRATION_METRIC_NAMES)
    if include_calibrated:
        names.extend(f"{name}{CALIBRATED_METRIC_SUFFIX}" for name in CALIBRATION_METRIC_NAMES)
    return tuple(names)
