#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench adoption of existing output directories.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import build_adoption_plan
from OCDocker.Workbench import read_result_manifest
from OCDocker.Workbench import read_run_manifest
from OCDocker.Workbench import scan_workspace
from OCDocker.Workbench import write_adoption_workspace

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


def _write_existing_output(tmp_path):
    '''Write a synthetic existing output tree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Existing output root.
    '''

    source = tmp_path / "existing"
    run_dir = source / "ablation A"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("metric,value\nauc,0.91\nloss,0.12\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text("started\ncompleted\n", encoding="utf-8")
    (run_dir / "report.html").write_text("<html></html>\n", encoding="utf-8")
    return source


## Public ##


def test_build_adoption_plan_discovers_metrics_logs_and_artifacts(tmp_path) -> None:
    '''Adoption plans discover candidate run evidence without writing manifests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    source = _write_existing_output(tmp_path)

    plan = build_adoption_plan(source, max_depth=1, run_id_prefix="adopted-")

    assert plan.candidate_count == 1
    candidate = plan.candidates[0]
    assert candidate.run_id == "adopted-ablation-A"
    assert candidate.status == "completed"
    assert candidate.metrics == {"auc": 0.91, "loss": 0.12}
    assert len(candidate.log_files) == 1
    assert {artifact.kind for artifact in candidate.artifacts} >= {"csv", "html", "log"}
    assert not (source / "ablation A" / "run_manifest.yml").exists()


def test_build_adoption_plan_discovers_ocscore_ablation_policy_dirs_beyond_depth(tmp_path) -> None:
    '''Adoption promotes OCScore ablation policy directories at shallow depth.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    source = tmp_path / "existing"
    policy_dir = source / "train" / "ablations" / "no_shape_core"
    policy_dir.mkdir(parents=True)
    (policy_dir / "replicas_summary.json").write_text(
        '{"metrics": {"validation_auc": 0.82}}\n',
        encoding="utf-8",
    )

    output_plan = build_adoption_plan(source, max_depth=1)
    train_plan = build_adoption_plan(source / "train", max_depth=0)

    for plan in (output_plan, train_plan):
        assert plan.candidate_count == 1
        candidate = plan.candidates[0]
        assert candidate.run_id == "no_shape_core"
        assert candidate.name == "no_shape_core"
        assert candidate.source_path == policy_dir
        assert candidate.metrics == {"validation_auc": 0.82}


def test_build_adoption_plan_can_require_metrics_before_reserving_run_ids(tmp_path) -> None:
    '''Metric-only adoption skips placeholders without consuming run ids.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    source = tmp_path / "existing"
    export_policy = source / "export" / "ablations" / "no_shape_core"
    train_policy = source / "train" / "ablations" / "no_shape_core"
    export_policy.mkdir(parents=True)
    train_policy.mkdir(parents=True)
    (export_policy / "report.html").write_text("<html></html>\n", encoding="utf-8")
    (train_policy / "replicas_summary.json").write_text(
        '{"metrics": {"validation_auc": 0.82}}\n',
        encoding="utf-8",
    )

    all_plan = build_adoption_plan(source, max_depth=1)
    metric_plan = build_adoption_plan(source, max_depth=1, require_metrics=True)

    assert [candidate.run_id for candidate in all_plan.candidates] == ["no_shape_core", "no_shape_core-2"]
    assert metric_plan.require_metrics is True
    assert metric_plan.candidate_count == 1
    candidate = metric_plan.candidates[0]
    assert candidate.run_id == "no_shape_core"
    assert candidate.source_path == train_policy
    assert candidate.metrics == {"validation_auc": 0.82}


def test_write_adoption_workspace_creates_manifests_without_touching_source(tmp_path) -> None:
    '''Adoption writes Workbench manifests into a separate destination workspace.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    source = _write_existing_output(tmp_path)
    destination = tmp_path / "workbench-runs"

    result = write_adoption_workspace(source, destination, max_depth=1)

    assert result.run_count == 1
    adopted = result.runs[0]
    assert adopted.metric_count == 2
    assert adopted.log_count == 1
    assert adopted.run_manifest_path == destination / "ablation-A" / "run_manifest.yml"
    assert adopted.result_manifest_path == destination / "ablation-A" / "result_manifest.yml"
    assert not (source / "ablation A" / "run_manifest.yml").exists()

    run_manifest = read_run_manifest(adopted.run_manifest_path)
    result_manifest = read_result_manifest(adopted.result_manifest_path)
    assert run_manifest.workspace == (source / "ablation A").resolve(strict=False)
    assert run_manifest.log_files[0] == (source / "ablation A" / "stdout.log").resolve(strict=False)
    assert run_manifest.metadata["adopted"] is True
    assert result_manifest.metrics == {"auc": 0.91, "loss": 0.12}
    assert result_manifest.artifacts[0].path.is_absolute()

    inventory = scan_workspace(destination, max_depth=2)
    assert inventory.runs[0].run_id == "ablation-A"
    assert inventory.result_manifests == (adopted.result_manifest_path,)
