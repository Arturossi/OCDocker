#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench decision-analysis payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_metrics_catalog
from OCDocker.Workbench import build_pareto_front
from OCDocker.Workbench import parse_pareto_objective
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
## Public ##


def test_build_metrics_catalog_summarizes_metric_coverage(tmp_path) -> None:
    '''Metric catalogs report numeric coverage and summary values.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    first = tmp_path / "first"
    second = tmp_path / "second"
    skipped = tmp_path / "bad"
    first.mkdir()
    second.mkdir()
    skipped.mkdir()
    write_model(
        first / "result_manifest.yml",
        ResultManifest(
            run_id="run-first",
            status="completed",
            metrics={"auc": 0.8, "validation": {"loss": 0.3}, "label": "ok"},
        ),
    )
    write_model(
        second / "result_manifest.yml",
        ResultManifest(
            run_id="run-second",
            status="completed",
            metrics={"auc": 0.9, "validation": {"loss": 0.2}},
        ),
    )
    (skipped / "result_manifest.yml").write_text("run_id: broken\n", encoding="utf-8")

    catalog = build_metrics_catalog(tmp_path, max_depth=2)
    entries = {entry.metric_name: entry for entry in catalog.metrics}

    assert catalog.result_manifest_count == 2
    assert catalog.metric_count == 3
    assert catalog.issue_count == 1
    assert entries["auc"].numeric_count == 2
    assert entries["auc"].min_value == 0.8
    assert entries["auc"].max_value == 0.9
    assert entries["auc"].mean_value == pytest.approx(0.85)
    assert entries["label"].observed_count == 1
    assert entries["label"].non_numeric_count == 1
    assert entries["label"].missing_count == 1
    assert entries["validation.loss"].numeric_count == 2


def test_build_pareto_front_identifies_non_dominated_runs(tmp_path) -> None:
    '''Pareto fronts separate non-dominated and dominated result manifests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    metrics = {
        "run-balanced": {"auc": 0.88, "loss": 0.18},
        "run-fast": {"auc": 0.84, "loss": 0.10},
        "run-dominated": {"auc": 0.82, "loss": 0.20},
        "run-missing": {"auc": 0.95},
    }
    for run_id, metric_values in metrics.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metric_values),
        )

    front = build_pareto_front(
        tmp_path,
        objectives=(
            parse_pareto_objective("auc:max"),
            parse_pareto_objective("loss:min"),
        ),
        max_depth=2,
    )

    assert [entry.run_id for entry in front.front_entries] == [
        "run-balanced",
        "run-fast",
    ]
    assert [entry.run_id for entry in front.dominated_entries] == ["run-dominated"]
    assert front.dominated_entries[0].dominated_by == ("run-balanced", "run-fast")
    assert [entry.run_id for entry in front.skipped_entries] == ["run-missing"]
    assert front.skipped_entries[0].missing_metrics == ("loss",)
    assert front.issue_count == 0


def test_parse_pareto_objective_defaults_and_validates_modes() -> None:
    '''Pareto objective parsing supports defaults and rejects invalid modes.'''

    default = parse_pareto_objective("auc")
    explicit = parse_pareto_objective("loss:min")

    assert default.metric_name == "auc"
    assert default.mode == "max"
    assert explicit.metric_name == "loss"
    assert explicit.mode == "min"
    with pytest.raises(ValueError, match="mode"):
        parse_pareto_objective("auc:median")


def test_build_pareto_front_rejects_duplicate_objectives(tmp_path) -> None:
    '''Pareto fronts require unique objective metric names.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    objective = parse_pareto_objective("auc:max")
    with pytest.raises(ValueError, match="unique"):
        build_pareto_front(tmp_path, objectives=(objective, objective))
