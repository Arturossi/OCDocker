#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench OCScore ablation analysis payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_ablation_analysis
from OCDocker.Workbench import parse_ablation_metric
from OCDocker.Workbench import write_model

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
## Private ##


def _write_result_pair(root, run_id: str, source_path: str, metrics: dict) -> None:
    '''Write a synthetic adopted run/result manifest pair.

    Parameters
    ----------
    root : pathlib.Path
        Workspace root.
    run_id : str
        Run id.
    source_path : str
        Adopted source path recorded in run metadata.
    metrics : dict
        Result metrics.
    '''

    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id=run_id,
            spec_type="ocscore_ablation",
            name=run_id,
            status="completed",
            workspace=source_path,
            metadata={"adopted": True, "source_path": source_path},
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(run_id=run_id, status="completed", metrics=metrics),
    )


def _write_ablation_workspace(tmp_path) -> None:
    '''Write a synthetic adopted ablation workspace.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    _write_result_pair(
        tmp_path,
        "train",
        "/source/output/train",
        {"auc": 0.88, "validation": {"loss": 0.20}},
    )
    _write_result_pair(
        tmp_path,
        "row_cleanup",
        "/source/output/train/row_cleanup",
        {"auc": 0.10, "validation": {"loss": 0.90}},
    )
    _write_result_pair(
        tmp_path,
        "shape_only",
        "/source/output/train/ablations/shape_only",
        {"auc": 0.91, "validation": {"loss": 0.18}},
    )
    _write_result_pair(
        tmp_path,
        "no_shape_core",
        "/source/output/train/ablations/no_shape_core",
        {"auc": 0.82, "validation": {"loss": 0.24}},
    )


## Public ##


def test_build_ablation_analysis_auto_selects_train_reference(tmp_path) -> None:
    '''Ablation analysis detects policies and compares them to train.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    _write_ablation_workspace(tmp_path)

    analysis = build_ablation_analysis(
        tmp_path,
        metrics=(
            parse_ablation_metric("auc:max"),
            parse_ablation_metric("validation.loss:min"),
        ),
        max_depth=2,
    )

    assert analysis.baseline_run_id == "train"
    assert analysis.detected_ablation_count == 2
    assert analysis.candidate_count == 2
    assert analysis.best_candidate.run_id == "shape_only"
    assert analysis.best_candidate.policy_name == "shape_only"
    assert analysis.best_candidate.net_score == 2
    no_shape_core = next(candidate for candidate in analysis.candidates if candidate.policy_name == "no_shape_core")
    assert no_shape_core.net_score == -2
    assert [metric.direction for metric in no_shape_core.metrics] == ["regressed", "regressed"]
    assert any("Auto-selected" in issue.message for issue in analysis.issues)


def test_build_ablation_analysis_accepts_policy_candidate_filter(tmp_path) -> None:
    '''Ablation analysis can filter candidates by policy name.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    _write_ablation_workspace(tmp_path)

    analysis = build_ablation_analysis(
        tmp_path,
        candidates=("no_shape_core",),
        metrics=(parse_ablation_metric("auc:max"),),
        max_depth=2,
    )

    assert [candidate.policy_name for candidate in analysis.candidates] == ["no_shape_core"]
    assert analysis.candidates[0].metrics[0].delta == pytest.approx(-0.06)


def test_build_ablation_analysis_requires_reference_when_missing(tmp_path) -> None:
    '''Ablation analysis reports when no reference run can be inferred.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    _write_result_pair(
        tmp_path,
        "shape_only",
        "/source/output/train/ablations/shape_only",
        {"auc": 0.91},
    )

    with pytest.raises(ValueError, match="No non-ablation reference run"):
        build_ablation_analysis(tmp_path, max_depth=2)


def test_build_ablation_analysis_returns_empty_payload_without_candidates(tmp_path) -> None:
    '''Ablation analysis does not compare unrelated rows when no ablations exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary workspace root.
    '''

    _write_result_pair(
        tmp_path,
        "train",
        "/source/output/train",
        {"auc": 0.91},
    )

    analysis = build_ablation_analysis(
        tmp_path,
        baseline_run_id="train",
        metrics=(parse_ablation_metric("auc:max"),),
        max_depth=2,
    )

    assert analysis.detected_ablation_count == 0
    assert analysis.candidate_count == 0
    assert analysis.candidates == ()
    assert any("No ablation candidates" in issue.message for issue in analysis.issues)
