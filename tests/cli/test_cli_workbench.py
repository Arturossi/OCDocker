#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for the read-only Workbench CLI commands.
'''

from __future__ import annotations

# Imports
###############################################################################
import json
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import OCDocker.CLI as cli
from OCDocker.CLI import workbench as cli_workbench
from OCDocker.Workbench import OCScoreInputSpec
from OCDocker.Workbench import OCScoreStudySpec
from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import read_result_manifest
from OCDocker.Workbench import read_run_manifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import write_model

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Private ##


def _write_study_spec(tmp_path) -> Path:
    '''Write a minimal OCScore study spec for CLI tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Written spec path.
    '''

    spec = OCScoreStudySpec(
        name="cli-study",
        protocol="smoke-test",
        inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
        output_dir="out/smoke",
    )
    return write_model(tmp_path / "study.yml", spec)


def _write_existing_adoption_source(tmp_path) -> Path:
    '''Write a synthetic existing output tree for CLI adoption tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Existing output root.
    '''

    source = tmp_path / "existing-cli"
    run_dir = source / "run one"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("metric,value\nauc,0.88\n", encoding="utf-8")
    (run_dir / "stdout.log").write_text("done\n", encoding="utf-8")
    return source


def _write_existing_evidence_workspace(tmp_path) -> Path:
    '''Write adopted OCScore evidence manifests for CLI tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Workspace root.
    '''

    source_root = tmp_path / "source" / "output"
    train_source = source_root / "train"
    train_source.mkdir(parents=True)
    (train_source / "baselines_per_fold.csv").write_text(
        "baseline,baseline_family,split,BEDROC\nvina,scoring_function,validation,0.30\n",
        encoding="utf-8",
    )
    shap_dir = source_root / "export" / "dudez" / "shap"
    shap_dir.mkdir(parents=True)
    (shap_dir / "shap_values.csv").write_text("feature_a,feature_b\n1.0,-2.0\n", encoding="utf-8")
    (shap_dir / "shap_feature_importance.png").write_bytes(b"png")

    workspace = tmp_path / "evidence-runs"
    run_dir = workspace / "train"
    run_dir.mkdir(parents=True)
    write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="train",
            spec_type="ocscore_study",
            name="train",
            status="completed",
            workspace=train_source,
            metadata={"adopted": True, "source_path": str(train_source)},
        ),
    )
    write_model(
        run_dir / "result_manifest.yml",
        ResultManifest(run_id="train", status="completed", metrics={"auc": 0.9}),
    )
    return workspace


def _write_existing_ablation_workspace(tmp_path) -> Path:
    '''Write adopted ablation manifests for CLI tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.

    Returns
    -------
    pathlib.Path
        Workspace root.
    '''

    workspace = tmp_path / "ablation-runs"
    for run_id, source_path, metrics in (
        ("train", "/source/output/train", {"auc": 0.88}),
        ("shape_only", "/source/output/train/ablations/shape_only", {"auc": 0.91}),
    ):
        run_dir = workspace / run_id
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
    return workspace


## Public ##


@pytest.mark.order(461)
def test_workbench_subcommands_registered(tmp_path) -> None:
    '''Test parser registration for Workbench subcommands.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    parser = cli.build_parser()
    spec_path = tmp_path / "study.yml"
    args = parser.parse_args(["workbench", "validate", str(spec_path), "--output", "valid.json"])

    assert args.workbench_command == "validate"
    assert args.spec == str(spec_path)
    assert args.output == "valid.json"
    assert args.func is cli_workbench.cmd_validate

    ablations_args = parser.parse_args(
        [
            "workbench",
            "ablations",
            str(tmp_path),
            "--baseline",
            "train",
            "--candidate",
            "shape_only",
            "--metric",
            "auc:max",
        ]
    )
    assert ablations_args.workbench_command == "ablations"
    assert ablations_args.root == str(tmp_path)
    assert ablations_args.baseline == "train"
    assert ablations_args.candidates == ["shape_only"]
    assert ablations_args.metrics == ["auc:max"]
    assert ablations_args.func is cli_workbench.cmd_ablations

    adopt_plan_args = parser.parse_args(
        [
            "workbench",
            "adopt-plan",
            str(tmp_path),
            "--max-depth",
            "2",
            "--spec-type",
            "ocscore_ablation",
            "--run-id-prefix",
            "adopted-",
        ]
    )
    assert adopt_plan_args.workbench_command == "adopt-plan"
    assert adopt_plan_args.source == str(tmp_path)
    assert adopt_plan_args.max_depth == 2
    assert adopt_plan_args.spec_type == "ocscore_ablation"
    assert adopt_plan_args.run_id_prefix == "adopted-"
    assert adopt_plan_args.func is cli_workbench.cmd_adopt_plan

    adopt_args = parser.parse_args(
        [
            "workbench",
            "adopt",
            str(tmp_path),
            str(tmp_path / "workbench-runs"),
            "--overwrite",
        ]
    )
    assert adopt_args.workbench_command == "adopt"
    assert adopt_args.source == str(tmp_path)
    assert adopt_args.destination == str(tmp_path / "workbench-runs")
    assert adopt_args.overwrite is True
    assert adopt_args.func is cli_workbench.cmd_adopt

    plan_args = parser.parse_args(["workbench", "plan", str(spec_path), "--run-id", "run-001"])
    assert plan_args.workbench_command == "plan"
    assert plan_args.run_id == "run-001"
    assert plan_args.func is cli_workbench.cmd_plan

    artifacts_args = parser.parse_args(["workbench", "artifacts", str(tmp_path), "--kind", "csv", "--role", "metrics"])
    assert artifacts_args.workbench_command == "artifacts"
    assert artifacts_args.kinds == ["csv"]
    assert artifacts_args.roles == ["metrics"]
    assert artifacts_args.func is cli_workbench.cmd_artifacts

    evidence_args = parser.parse_args(
        ["workbench", "evidence", str(tmp_path), "--source-depth", "5", "--max-entries", "20"]
    )
    assert evidence_args.workbench_command == "evidence"
    assert evidence_args.source_depth == 5
    assert evidence_args.max_entries == 20
    assert evidence_args.func is cli_workbench.cmd_evidence

    build_args = parser.parse_args(
        [
            "workbench",
            "build",
            str(spec_path),
            str(tmp_path / "run"),
            "--run-id",
            "run-001",
        ]
    )
    assert build_args.workbench_command == "build"
    assert build_args.run_id == "run-001"
    assert build_args.func is cli_workbench.cmd_build

    check_args = parser.parse_args(["workbench", "check", str(spec_path)])
    assert check_args.workbench_command == "check"
    assert check_args.spec == str(spec_path)
    assert check_args.func is cli_workbench.cmd_check

    compare_args = parser.parse_args(
        [
            "workbench",
            "compare",
            str(tmp_path),
            "--baseline",
            "baseline",
            "--candidate",
            "candidate",
            "--metric",
            "auc:max",
        ]
    )
    assert compare_args.workbench_command == "compare"
    assert compare_args.baseline == "baseline"
    assert compare_args.candidates == ["candidate"]
    assert compare_args.metrics == ["auc:max"]
    assert compare_args.func is cli_workbench.cmd_compare

    export_args = parser.parse_args(
        [
            "workbench",
            "export",
            str(tmp_path / "result_manifest.yml"),
            str(tmp_path / "export"),
        ]
    )
    assert export_args.workbench_command == "export"
    assert export_args.func is cli_workbench.cmd_export

    overview_args = parser.parse_args(["workbench", "overview", str(tmp_path), "--recent-limit", "3"])
    assert overview_args.workbench_command == "overview"
    assert overview_args.recent_limit == 3
    assert overview_args.func is cli_workbench.cmd_overview

    inventory_args = parser.parse_args(["workbench", "inventory", str(tmp_path), "--max-depth", "1"])
    assert inventory_args.workbench_command == "inventory"
    assert inventory_args.max_depth == 1
    assert inventory_args.func is cli_workbench.cmd_inventory

    status_args = parser.parse_args(["workbench", "status", str(tmp_path / "run")])
    assert status_args.workbench_command == "status"
    assert status_args.target == str(tmp_path / "run")
    assert status_args.func is cli_workbench.cmd_status

    launch_args = parser.parse_args(
        [
            "workbench",
            "launch-plan",
            str(tmp_path / "run"),
            "--script-output",
            "run.sh",
        ]
    )
    assert launch_args.workbench_command == "launch-plan"
    assert launch_args.target == str(tmp_path / "run")
    assert launch_args.script_output == "run.sh"
    assert launch_args.func is cli_workbench.cmd_launch_plan

    catalog_args = parser.parse_args(["workbench", "metrics-catalog", str(tmp_path)])
    assert catalog_args.workbench_command == "metrics-catalog"
    assert catalog_args.func is cli_workbench.cmd_metrics_catalog

    pareto_args = parser.parse_args(
        [
            "workbench",
            "pareto",
            str(tmp_path),
            "--objective",
            "auc:max",
            "--objective",
            "loss:min",
        ]
    )
    assert pareto_args.workbench_command == "pareto"
    assert pareto_args.objectives == ["auc:max", "loss:min"]
    assert pareto_args.func is cli_workbench.cmd_pareto

    leaderboard_args = parser.parse_args(["workbench", "leaderboard", str(tmp_path), "--metric", "auc"])
    assert leaderboard_args.workbench_command == "leaderboard"
    assert leaderboard_args.metric == "auc"
    assert leaderboard_args.func is cli_workbench.cmd_leaderboard

    matrix_args = parser.parse_args(["workbench", "metrics-matrix", str(tmp_path), "--metric", "auc"])
    assert matrix_args.workbench_command == "metrics-matrix"
    assert matrix_args.metrics == ["auc"]
    assert matrix_args.func is cli_workbench.cmd_metrics_matrix

    logs_args = parser.parse_args(["workbench", "logs", str(tmp_path / "run"), "--lines", "5"])
    assert logs_args.workbench_command == "logs"
    assert logs_args.target == str(tmp_path / "run")
    assert logs_args.lines == 5
    assert logs_args.func is cli_workbench.cmd_logs

    results_args = parser.parse_args(["workbench", "results", str(tmp_path / "result_manifest.yml")])
    assert results_args.workbench_command == "results"
    assert results_args.manifest == str(tmp_path / "result_manifest.yml")
    assert results_args.func is cli_workbench.cmd_results

    plot_args = parser.parse_args(
        [
            "workbench",
            "plot",
            str(tmp_path),
            "--kind",
            "scatter",
            "--x-metric",
            "auc",
            "--y-metric",
            "loss",
        ]
    )
    assert plot_args.workbench_command == "plot"
    assert plot_args.kind == "scatter"
    assert plot_args.x_metric == "auc"
    assert plot_args.y_metric == "loss"
    assert plot_args.func is cli_workbench.cmd_plot

    report_args = parser.parse_args(
        [
            "workbench",
            "report",
            str(tmp_path),
            "--leaderboard",
            "auc:max",
            "--objective",
            "loss:min",
            "--format",
            "markdown",
        ]
    )
    assert report_args.workbench_command == "report"
    assert report_args.leaderboards == ["auc:max"]
    assert report_args.objectives == ["loss:min"]
    assert report_args.output_format == "markdown"
    assert report_args.func is cli_workbench.cmd_report

    serve_args = parser.parse_args(
        [
            "workbench",
            "serve",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--max-depth",
            "2",
            "--verbose",
        ]
    )
    assert serve_args.workbench_command == "serve"
    assert serve_args.host == "127.0.0.1"
    assert serve_args.port == 9000
    assert serve_args.max_depth == 2
    assert serve_args.verbose is True
    assert serve_args.func is cli_workbench.cmd_serve

    schema_args = parser.parse_args(["workbench", "schema", "ocscore_study"])
    assert schema_args.workbench_command == "schema"
    assert schema_args.names == ["ocscore_study"]
    assert schema_args.func is cli_workbench.cmd_schema

    template_args = parser.parse_args(["workbench", "template", "ocscore_study", "--format", "json"])
    assert template_args.workbench_command == "template"
    assert template_args.template_name == "ocscore_study"
    assert template_args.output_format == "json"
    assert template_args.func is cli_workbench.cmd_template


def test_cmd_ablations_prints_policy_deltas(tmp_path, capsys) -> None:
    '''Test Workbench ablations output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    workspace = _write_existing_ablation_workspace(tmp_path)

    rc = cli_workbench.cmd_ablations(
        SimpleNamespace(
            root=str(workspace),
            baseline=None,
            candidates=None,
            metrics=("auc:max",),
            max_depth=2,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_run_id"] == "train"
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["policy_name"] == "shape_only"
    assert payload["candidates"][0]["metrics"][0]["direction"] == "improved"


def test_cmd_adopt_plan_prints_dry_run_payload(tmp_path, capsys) -> None:
    '''Test Workbench adopt-plan output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    source = _write_existing_adoption_source(tmp_path)

    rc = cli_workbench.cmd_adopt_plan(
        SimpleNamespace(
            source=str(source),
            max_depth=1,
            spec_type="ocscore_ablation",
            status=None,
            run_id_prefix="",
            max_metric_bytes=1048576,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["metrics"] == {"auc": 0.88}
    assert not (source / "run one" / "run_manifest.yml").exists()


def test_cmd_adopt_writes_destination_manifests(tmp_path, capsys) -> None:
    '''Test Workbench adopt writes manifests outside the source tree.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    source = _write_existing_adoption_source(tmp_path)
    destination = tmp_path / "workbench-runs"

    rc = cli_workbench.cmd_adopt(
        SimpleNamespace(
            source=str(source),
            destination=str(destination),
            max_depth=1,
            spec_type="ocscore_ablation",
            status=None,
            run_id_prefix="",
            max_metric_bytes=1048576,
            overwrite=False,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_count"] == 1
    run_manifest = read_run_manifest(destination / "run-one" / "run_manifest.yml")
    result_manifest = read_result_manifest(destination / "run-one" / "result_manifest.yml")
    assert run_manifest.workspace == (source / "run one").resolve(strict=False)
    assert result_manifest.metrics == {"auc": 0.88}
    assert not (source / "run one" / "run_manifest.yml").exists()

    rc = cli_workbench.cmd_adopt(
        SimpleNamespace(
            source=str(source),
            destination=str(destination),
            max_depth=1,
            spec_type="ocscore_ablation",
            status=None,
            run_id_prefix="",
            max_metric_bytes=1048576,
            overwrite=False,
            output=None,
        )
    )
    assert rc == 2
    assert "could not adopt existing outputs" in capsys.readouterr().out


def test_cmd_artifacts_prints_index_payload(tmp_path, capsys) -> None:
    '''Test Workbench artifact index output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    (tmp_path / "metrics.csv").write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(
            run_id="run-cli-artifacts",
            status="completed",
            artifacts=(
                ResultArtifact(name="metrics", path="metrics.csv", kind="csv", role="metrics"),
                ResultArtifact(name="plot", path="missing.png", kind="image", role="plot"),
            ),
        ),
    )

    rc = cli_workbench.cmd_artifacts(
        SimpleNamespace(
            root=str(tmp_path),
            kinds=("csv",),
            roles=None,
            require_existing=True,
            max_depth=1,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_count"] == 1
    assert payload["entries"][0]["name"] == "metrics"
    assert payload["entries"][0]["exists"] is True
    assert payload["kind_counts"] == {"csv": 1}


def test_cmd_evidence_prints_index_payload(tmp_path, capsys) -> None:
    '''Test Workbench evidence index output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    workspace = _write_existing_evidence_workspace(tmp_path)

    rc = cli_workbench.cmd_evidence(
        SimpleNamespace(
            root=str(workspace),
            max_depth=2,
            source_depth=4,
            max_entries=20,
            max_csv_rows=20,
            max_series=4,
            max_shap_features=8,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evidence_count"] == 3
    assert payload["kind_counts"]["performance"] == 1
    assert payload["kind_counts"]["shap"] == 2
    assert payload["performance_points"][0]["metric_name"] == "BEDROC"
    assert payload["shap_features"][0]["feature"] == "feature_b"


def test_cmd_template_prints_valid_yaml(tmp_path, capsys) -> None:
    '''Test Workbench starter template output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    rc = cli_workbench.cmd_template(
        SimpleNamespace(
            template_name="ocscore_study",
            output_format="yaml",
            output=None,
        )
    )

    assert rc == 0
    payload = yaml.safe_load(capsys.readouterr().out)
    assert payload["type"] == "ocscore_study"
    assert payload["inputs"]["raw_input_dir"] == "path/to/raw_prepare"


def test_cmd_template_writes_json_file(tmp_path, capsys) -> None:
    '''Test Workbench starter template file output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    output_path = tmp_path / "template.json"

    rc = cli_workbench.cmd_template(
        SimpleNamespace(
            template_name="vs_campaign",
            output_format="json",
            output=str(output_path),
        )
    )

    assert rc == 0
    assert "Workbench template written" in capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["type"] == "vs_campaign"
    assert payload["inputs"][0]["sample"] == "sample_001"


def test_cmd_check_prints_preflight_report(tmp_path, capsys) -> None:
    '''Test Workbench preflight output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    raw_dir = tmp_path / "raw_prepare"
    raw_dir.mkdir()
    spec_path = write_model(
        tmp_path / "study.yml",
        OCScoreStudySpec(
            name="cli-check",
            protocol="smoke-test",
            inputs=OCScoreInputSpec(raw_input_dir="raw_prepare"),
            output_dir="out/cli-check",
        ),
    )

    rc = cli_workbench.cmd_check(
        SimpleNamespace(
            spec=str(spec_path),
            output=None,
            ocdocker_executable=sys.executable,
            snakemake_executable="snakemake",
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
    assert payload["error_count"] == 0
    assert payload["planned_command"][0] == sys.executable


def test_cmd_validate_prints_json_payload(tmp_path, capsys) -> None:
    '''Test successful Workbench spec validation output.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    spec_path = _write_study_spec(tmp_path)

    rc = cli_workbench.cmd_validate(SimpleNamespace(spec=str(spec_path), output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["spec_type"] == "ocscore_study"
    assert payload["name"] == "cli-study"


def test_cmd_plan_writes_payload_and_manifest(tmp_path, capsys) -> None:
    '''Test command planning and optional run-manifest writing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    spec_path = _write_study_spec(tmp_path)
    output_path = tmp_path / "plan.json"
    manifest_path = tmp_path / "run_manifest.yml"

    rc = cli_workbench.cmd_plan(
        SimpleNamespace(
            spec=str(spec_path),
            output=str(output_path),
            run_id="run-001",
            manifest_output=str(manifest_path),
            ocdocker_executable="ocdocker-dev",
            snakemake_executable="snakemake-dev",
        )
    )

    assert rc == 0
    assert "Workbench payload written" in capsys.readouterr().out
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["plan"]["command"][:3] == ["ocdocker-dev", "ocscore", "train"]
    assert payload["run_manifest"]["run_id"] == "run-001"
    assert manifest_path.is_file()


def test_cmd_validate_reports_invalid_spec(tmp_path, capsys) -> None:
    '''Test invalid Workbench spec reporting.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    path = tmp_path / "bad.yml"
    path.write_text("type: unknown\nname: bad\n", encoding="utf-8")

    rc = cli_workbench.cmd_validate(SimpleNamespace(spec=str(path), output=None))

    assert rc == 2
    assert "invalid Workbench spec" in capsys.readouterr().out


def test_cmd_overview_prints_workspace_payload(tmp_path, capsys) -> None:
    '''Test read-only Workbench workspace overview output.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-cli-overview",
            spec_type="ocscore_study",
            name="cli-overview",
            status="running",
            workspace=tmp_path,
        ),
    )

    rc = cli_workbench.cmd_overview(SimpleNamespace(root=str(tmp_path), max_depth=1, recent_limit=5, output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_count"] == 1
    assert payload["status_counts"]["running"] == 1
    assert payload["recent_runs"][0]["run_id"] == "run-cli-overview"
    assert payload["issues"] == []


def test_cmd_inventory_prints_workspace_payload(tmp_path, capsys) -> None:
    '''Test read-only Workbench workspace inventory output.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-cli-inventory",
            spec_type="ocscore_study",
            name="cli-inventory",
            workspace=tmp_path,
        ),
    )

    rc = cli_workbench.cmd_inventory(SimpleNamespace(root=str(tmp_path), max_depth=1, output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["runs"][0]["run_id"] == "run-cli-inventory"
    assert payload["issues"] == []


def test_cmd_serve_invokes_server(tmp_path, monkeypatch, capsys) -> None:
    '''Test Workbench serve command wiring without starting a real server.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    calls = []

    def fake_serve(
        root,
        *,
        host,
        port,
        max_depth,
        optuna_dashboard_port_start,
        optuna_dashboard_port_end,
        optuna_dashboard_slots,
        verbose,
    ):
        calls.append(
            (
                root,
                host,
                port,
                max_depth,
                optuna_dashboard_port_start,
                optuna_dashboard_port_end,
                optuna_dashboard_slots,
                verbose,
            )
        )

    monkeypatch.setattr(cli_workbench, "serve_workbench_api", fake_serve)

    rc = cli_workbench.cmd_serve(
        SimpleNamespace(
            root=str(tmp_path),
            host="127.0.0.1",
            port=8765,
            max_depth=2,
            optuna_port_start=8790,
            optuna_port_end=8819,
            optuna_slots=None,
            verbose=True,
        )
    )

    assert rc == 0
    assert calls == [(str(tmp_path), "127.0.0.1", 8765, 2, 8790, 8819, None, True)]


def test_cmd_schema_prints_selected_catalog(tmp_path, capsys) -> None:
    '''Test Workbench schema catalog output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    rc = cli_workbench.cmd_schema(SimpleNamespace(names=("ocscore_study",), output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["schemas"]) == {"ocscore_study"}
    assert payload["schemas"]["ocscore_study"]["title"] == "OCScoreStudySpec"


def test_cmd_status_prints_single_run_payload(tmp_path, capsys) -> None:
    '''Test Workbench single-run status output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    (tmp_path / "run.log").write_text("started\n", encoding="utf-8")
    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-cli-status",
            spec_type="ocscore_study",
            name="cli-status",
            workspace=tmp_path,
            log_files=("run.log",),
        ),
    )

    rc = cli_workbench.cmd_status(SimpleNamespace(target=str(tmp_path), output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-status"
    assert payload["workspace_status"]["exists"] is True
    assert payload["log_files"][0]["exists"] is True


def test_cmd_build_writes_run_bundle(tmp_path, capsys) -> None:
    '''Test Workbench run bundle creation from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    spec_path = _write_study_spec(tmp_path)
    bundle_dir = tmp_path / "bundle"

    rc = cli_workbench.cmd_build(
        SimpleNamespace(
            spec=str(spec_path),
            bundle_dir=str(bundle_dir),
            run_id="run-cli-build",
            overwrite=False,
            output=None,
            ocdocker_executable="ocdocker-dev",
            snakemake_executable="snakemake-dev",
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-build"
    assert (bundle_dir / "spec.yml").is_file()
    assert (bundle_dir / "plan.json").is_file()
    assert (bundle_dir / "run_manifest.yml").is_file()
    assert (bundle_dir / "bundle.json").is_file()

    rc = cli_workbench.cmd_build(
        SimpleNamespace(
            spec=str(spec_path),
            bundle_dir=str(bundle_dir),
            run_id="run-cli-build",
            overwrite=False,
            output=None,
            ocdocker_executable="ocdocker-dev",
            snakemake_executable="snakemake-dev",
        )
    )
    assert rc == 2
    assert "could not build Workbench bundle" in capsys.readouterr().out


def test_cmd_launch_plan_prints_execution_envelope(tmp_path, capsys) -> None:
    '''Test Workbench launch-plan output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-cli-launch",
            spec_type="ocscore_study",
            name="cli-launch",
            workspace=Path("."),
            command=("ocdocker", "--help"),
        ),
    )

    rc = cli_workbench.cmd_launch_plan(
        SimpleNamespace(
            target=str(tmp_path),
            log_dir="logs",
            script_output="run.sh",
            overwrite=False,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-launch"
    assert payload["shell_command"] == "ocdocker --help"
    assert payload["script_written"] is True
    assert (tmp_path / "run.sh").is_file()
    assert not (tmp_path / "logs" / "run-cli-launch.pid").exists()


def test_cmd_metrics_catalog_prints_metric_coverage(tmp_path, capsys) -> None:
    '''Test Workbench metric catalog output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(run_id="run-cli-catalog", status="completed", metrics={"auc": 0.91}),
    )

    rc = cli_workbench.cmd_metrics_catalog(SimpleNamespace(root=str(tmp_path), max_depth=1, output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric_count"] == 1
    assert payload["metrics"][0]["metric_name"] == "auc"
    assert payload["metrics"][0]["numeric_count"] == 1


def test_cmd_pareto_prints_front(tmp_path, capsys) -> None:
    '''Test Workbench Pareto front output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    for run_id, metrics in {
        "run-front": {"auc": 0.9, "loss": 0.2},
        "run-dominated": {"auc": 0.8, "loss": 0.3},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )

    rc = cli_workbench.cmd_pareto(
        SimpleNamespace(
            root=str(tmp_path),
            objectives=("auc:max", "loss:min"),
            max_depth=2,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["run_id"] for entry in payload["front_entries"]] == ["run-front"]
    assert [entry["run_id"] for entry in payload["dominated_entries"]] == ["run-dominated"]


def test_cmd_leaderboard_prints_metric_ranking(tmp_path, capsys) -> None:
    '''Test Workbench metric leaderboard output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(run_id="run-cli-leader", status="completed", metrics={"auc": 0.91}),
    )

    rc = cli_workbench.cmd_leaderboard(
        SimpleNamespace(
            root=str(tmp_path),
            metric="auc",
            mode="max",
            max_depth=1,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric_name"] == "auc"
    assert payload["best_entry"]["run_id"] == "run-cli-leader"
    assert payload["ranked_entries"][0]["metric_value"] == 0.91


def test_cmd_metrics_matrix_prints_metric_rows(tmp_path, capsys) -> None:
    '''Test Workbench metrics matrix output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(run_id="run-cli-matrix", status="completed", metrics={"auc": 0.91}),
    )

    rc = cli_workbench.cmd_metrics_matrix(
        SimpleNamespace(
            root=str(tmp_path),
            metrics=("auc",),
            max_depth=1,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metric_names"] == ["auc"]
    assert payload["rows"][0]["run_id"] == "run-cli-matrix"
    assert payload["rows"][0]["metric_values"] == {"auc": 0.91}


def test_cmd_plot_prints_scatter_payload(tmp_path, capsys) -> None:
    '''Test Workbench plot JSON output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    for run_id, metrics in {
        "run-a": {"auc": 0.9, "loss": 0.2},
        "run-b": {"auc": 0.8, "loss": 0.3},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )

    rc = cli_workbench.cmd_plot(
        SimpleNamespace(
            root=str(tmp_path),
            kind="scatter",
            metrics=None,
            x_metric="auc",
            y_metric="loss",
            color_metric=None,
            objectives=None,
            mode="max",
            max_depth=2,
            top_n=20,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plot_kind"] == "metric_scatter"
    assert payload["metric_names"] == ["auc", "loss"]
    assert payload["data"][0]["type"] == "scatter"
    assert payload["included_count"] == 2


def test_cmd_plot_reports_invalid_arguments(capsys) -> None:
    '''Test Workbench plot validation errors from the CLI layer.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    rc = cli_workbench.cmd_plot(
        SimpleNamespace(
            root=".",
            kind="leaderboard",
            metrics=None,
            x_metric=None,
            y_metric=None,
            color_metric=None,
            objectives=None,
            mode="max",
            max_depth=1,
            top_n=20,
            output=None,
        )
    )

    assert rc == 2
    assert "leaderboard plots require exactly one --metric" in capsys.readouterr().out


def test_cmd_report_prints_analysis_payload(tmp_path, capsys) -> None:
    '''Test Workbench analysis report JSON output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    for run_id, metrics in {
        "run-cli-front": {"auc": 0.9, "loss": 0.2},
        "run-cli-dominated": {"auc": 0.8, "loss": 0.3},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "run_manifest.yml",
            RunManifest(
                run_id=run_id,
                spec_type="ocscore_study",
                name=run_id,
                status="completed",
                workspace=run_dir,
            ),
        )
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )

    rc = cli_workbench.cmd_report(
        SimpleNamespace(
            root=str(tmp_path),
            leaderboards=("auc:max",),
            objectives=("auc:max", "loss:min"),
            max_depth=2,
            recent_limit=5,
            top_n=3,
            output_format="json",
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overview"]["run_count"] == 2
    assert payload["leaderboards"][0]["best_entry"]["run_id"] == "run-cli-front"
    assert payload["pareto_front"]["front_entries"][0]["run_id"] == "run-cli-front"
    assert "OCDocker Workbench Analysis Report" in payload["markdown"]


def test_cmd_report_prints_markdown(tmp_path, capsys) -> None:
    '''Test Workbench analysis report Markdown output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(run_id="run-cli-report-md", status="completed", metrics={"auc": 0.91}),
    )

    rc = cli_workbench.cmd_report(
        SimpleNamespace(
            root=str(tmp_path),
            leaderboards=("auc:max",),
            objectives=(),
            max_depth=1,
            recent_limit=5,
            top_n=3,
            output_format="markdown",
            output=None,
        )
    )

    assert rc == 0
    markdown = capsys.readouterr().out
    assert markdown.startswith("# OCDocker Workbench Analysis Report")
    assert "### auc (max)" in markdown


def test_cmd_logs_prints_bounded_preview(tmp_path, capsys) -> None:
    '''Test Workbench log preview output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    (tmp_path / "run.log").write_text("one\ntwo\nthree\n", encoding="utf-8")
    write_model(
        tmp_path / "run_manifest.yml",
        RunManifest(
            run_id="run-cli-logs",
            spec_type="ocscore_study",
            name="cli-logs",
            workspace=tmp_path,
            log_files=("run.log",),
        ),
    )

    rc = cli_workbench.cmd_logs(
        SimpleNamespace(
            target=str(tmp_path),
            lines=2,
            max_bytes=1024,
            encoding="utf-8",
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-logs"
    assert payload["logs"][0]["lines"] == ["two", "three"]
    assert payload["logs"][0]["truncated"] is True


def test_cmd_results_prints_artifact_summary(tmp_path, capsys) -> None:
    '''Test Workbench result summary output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    (tmp_path / "metrics.csv").write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    manifest_path = write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(
            run_id="run-cli-results",
            status="completed",
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
            metrics={"auc": 0.9},
        ),
    )

    rc = cli_workbench.cmd_results(SimpleNamespace(manifest=str(manifest_path), output=None))

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-results"
    assert payload["metrics"] == {"auc": 0.9}
    assert payload["artifact_count"] == 1
    assert payload["existing_artifact_count"] == 1
    assert payload["artifacts"][0]["exists"] is True


def test_cmd_compare_prints_payload(tmp_path, capsys) -> None:
    '''Test Workbench comparison output from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    for run_id, metrics in {
        "baseline": {"auc": 0.85, "loss": 0.2},
        "candidate": {"auc": 0.9, "loss": 0.18},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )

    rc = cli_workbench.cmd_compare(
        SimpleNamespace(
            root=str(tmp_path),
            baseline="baseline",
            candidates=("candidate",),
            metrics=("auc:max", "loss:min"),
            max_depth=2,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_run_id"] == "baseline"
    assert payload["candidate_count"] == 1
    assert payload["best_candidate"]["run_id"] == "candidate"
    assert payload["best_candidate"]["net_score"] == 2
    assert [metric["direction"] for metric in payload["candidates"][0]["metrics"]] == [
        "improved",
        "improved",
    ]


def test_cmd_export_writes_publication_scaffold(tmp_path, capsys) -> None:
    '''Test Workbench publication export creation from the CLI layer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    capsys : pytest.CaptureFixture
        Pytest stdout/stderr capture fixture.
    '''

    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nauc,0.9\n", encoding="utf-8")
    manifest_path = write_model(
        tmp_path / "result_manifest.yml",
        ResultManifest(
            run_id="run-cli-export",
            status="completed",
            artifacts=(ResultArtifact(name="metrics", path="metrics.csv", kind="csv"),),
        ),
    )
    export_dir = tmp_path / "export"

    rc = cli_workbench.cmd_export(
        SimpleNamespace(
            manifest=str(manifest_path),
            export_dir=str(export_dir),
            copy_artifacts=False,
            overwrite=False,
            output=None,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run-cli-export"
    assert (export_dir / "README.md").is_file()
    assert (export_dir / "publication_manifest.json").is_file()
    assert not (export_dir / "artifacts").exists()
