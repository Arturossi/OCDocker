#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for composed Workbench analysis reports.
'''

# Imports
###############################################################################
from __future__ import annotations

from OCDocker.Workbench import ResultArtifact
from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_analysis_report
from OCDocker.Workbench import parse_report_metric
from OCDocker.Workbench import render_analysis_report_markdown
from OCDocker.Workbench import write_model

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
## Private ##


def _write_report_workspace(tmp_path) -> None:
    '''Write a small manifest workspace for report tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    for run_id, metrics in {
        "run-balanced": {"auc": 0.88, "loss": 0.18},
        "run-fast": {"auc": 0.84, "loss": 0.10},
        "run-dominated": {"auc": 0.82, "loss": 0.20},
        "run-missing": {"auc": 0.95},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        artifact = ResultArtifact(
            name="metrics",
            path="metrics.csv",
            kind="csv",
            role="analysis",
        )
        write_model(
            run_dir / "run_manifest.yml",
            RunManifest(
                run_id=run_id,
                spec_type="ocscore_study",
                name=run_id,
                status="completed",
                workspace=run_dir,
                artifacts=(artifact,),
            ),
        )
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(
                run_id=run_id,
                status="completed",
                artifacts=(artifact,),
                metrics=metrics,
            ),
        )


## Public ##


def test_build_analysis_report_composes_decision_payload(tmp_path) -> None:
    '''Analysis reports compose overview, metrics, leaderboards, and Pareto data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_report_workspace(tmp_path)

    report = build_analysis_report(
        tmp_path,
        leaderboards=(parse_report_metric("auc:max"), parse_report_metric("loss:min")),
        pareto_objectives=(
            parse_report_metric("auc:max"),
            parse_report_metric("loss:min"),
        ),
        max_depth=2,
        recent_limit=3,
        top_n=2,
    )

    assert report.overview.run_count == 4
    assert report.overview.result_manifest_count == 4
    assert report.metrics_catalog.metric_count == 2
    assert [leaderboard.metric_name for leaderboard in report.leaderboards] == [
        "auc",
        "loss",
    ]
    assert report.leaderboards[0].best_entry.run_id == "run-missing"
    assert report.leaderboards[1].best_entry.run_id == "run-fast"
    assert report.metric_matrix.metric_names == ("auc", "loss")
    assert [entry.run_id for entry in report.pareto_front.front_entries] == [
        "run-balanced",
        "run-fast",
    ]
    assert any(finding.kind == "best_metric" for finding in report.findings)
    assert any(finding.kind == "pareto_candidate" for finding in report.findings)
    assert any(finding.kind == "missing_artifact" for finding in report.findings)
    assert "# OCDocker Workbench Analysis Report" in report.markdown


def test_build_analysis_report_infers_default_leaderboards(tmp_path) -> None:
    '''Analysis reports infer leaderboards from numeric metric coverage.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_report_workspace(tmp_path)

    report = build_analysis_report(tmp_path, max_depth=2, top_n=1)

    assert len(report.leaderboards) == 1
    assert report.leaderboards[0].metric_name == "auc"
    assert report.leaderboards[0].mode == "max"
    assert report.metric_matrix.metric_names == ("auc",)


def test_render_analysis_report_markdown_is_stable(tmp_path) -> None:
    '''Markdown rendering includes the main report sections.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_report_workspace(tmp_path)
    report = build_analysis_report(
        tmp_path,
        leaderboards=(parse_report_metric("loss:min"),),
        max_depth=2,
    )

    markdown = render_analysis_report_markdown(report)

    assert "## Metric Coverage" in markdown
    assert "## Leaderboards" in markdown
    assert "### loss (min)" in markdown
    assert "## Findings" in markdown
