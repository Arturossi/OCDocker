#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench run comparison payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_run_comparison
from OCDocker.Workbench import parse_comparison_metric
from OCDocker.Workbench import write_model

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


def _write_comparison_workspace(tmp_path) -> None:
    '''Write result manifests for comparison tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    payloads = {
        "baseline": {"auc": 0.85, "validation": {"loss": 0.20}, "runtime": 20},
        "better": {"auc": 0.90, "validation": {"loss": 0.18}, "runtime": 18},
        "worse": {"auc": 0.80, "validation": {"loss": 0.24}, "runtime": 25},
        "partial": {"auc": 0.88, "runtime": 16},
    }
    for run_id, metrics in payloads.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        artifacts = ()
        if run_id == "better":
            artifacts = (
                ResultArtifact(name="missing", path="missing.csv", kind="csv"),
            )
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(
                run_id=run_id, status="completed", metrics=metrics, artifacts=artifacts
            ),
        )


## Public ##


def test_build_run_comparison_scores_candidates(tmp_path) -> None:
    '''Run comparisons score candidates against explicit metric objectives.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_comparison_workspace(tmp_path)

    comparison = build_run_comparison(
        tmp_path,
        baseline_run_id="baseline",
        metrics=(
            parse_comparison_metric("auc:max"),
            parse_comparison_metric("validation.loss:min"),
        ),
        max_depth=2,
    )

    assert comparison.baseline_run_id == "baseline"
    assert comparison.result_manifest_count == 4
    assert comparison.candidate_count == 3
    assert comparison.best_candidate.run_id == "better"
    assert comparison.best_candidate.net_score == 2
    better = comparison.candidates[0]
    assert [metric.direction for metric in better.metrics] == ["improved", "improved"]
    assert better.missing_artifact_count == 1
    worse = next(
        candidate for candidate in comparison.candidates if candidate.run_id == "worse"
    )
    assert worse.net_score == -2
    partial = next(
        candidate
        for candidate in comparison.candidates
        if candidate.run_id == "partial"
    )
    assert partial.incomplete_count == 1
    loss_metric = next(
        metric for metric in partial.metrics if metric.metric_name == "validation.loss"
    )
    assert loss_metric.candidate_missing is True
    assert loss_metric.direction == "incomplete"


def test_build_run_comparison_can_filter_candidates(tmp_path) -> None:
    '''Run comparisons can restrict candidates by run id.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_comparison_workspace(tmp_path)

    comparison = build_run_comparison(
        tmp_path,
        baseline_run_id="baseline",
        candidates=("worse", "missing", "baseline"),
        metrics=(parse_comparison_metric("auc:max"),),
        max_depth=2,
    )

    assert [candidate.run_id for candidate in comparison.candidates] == ["worse"]
    assert comparison.issue_count == 2
    assert any(
        "Candidate run_id not found" in issue.message for issue in comparison.issues
    )
    assert any("Skipping baseline" in issue.message for issue in comparison.issues)


def test_build_run_comparison_infers_numeric_metrics(tmp_path) -> None:
    '''Run comparisons infer numeric metrics when none are supplied.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_comparison_workspace(tmp_path)

    comparison = build_run_comparison(tmp_path, baseline_run_id="baseline", max_depth=2)

    assert [metric.metric_name for metric in comparison.metrics] == [
        "auc",
        "runtime",
        "validation.loss",
    ]
    modes = {metric.metric_name: metric.mode for metric in comparison.metrics}
    assert modes["auc"] == "max"
    assert modes["runtime"] == "min"
    assert modes["validation.loss"] == "min"


def test_build_run_comparison_rejects_missing_baseline(tmp_path) -> None:
    '''Run comparisons require an existing baseline run id.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_comparison_workspace(tmp_path)

    with pytest.raises(ValueError, match="Baseline run_id not found"):
        build_run_comparison(tmp_path, baseline_run_id="unknown", max_depth=2)
