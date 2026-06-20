#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench plot-ready payloads.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import ResultManifest
from OCDocker.Workbench import build_leaderboard_plot
from OCDocker.Workbench import build_metric_scatter_plot
from OCDocker.Workbench import build_parallel_coordinates_plot
from OCDocker.Workbench import build_pareto_scatter_plot
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
## Private ##


def _write_plot_workspace(tmp_path) -> None:
    '''Write result manifests for plot tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    for run_id, metrics in {
        "balanced": {"auc": 0.88, "loss": 0.18, "runtime": 30},
        "fast": {"auc": 0.84, "loss": 0.10, "runtime": 12},
        "dominated": {"auc": 0.82, "loss": 0.20, "runtime": 35},
        "missing": {"auc": 0.95},
    }.items():
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        write_model(
            run_dir / "result_manifest.yml",
            ResultManifest(run_id=run_id, status="completed", metrics=metrics),
        )


## Public ##


def test_build_leaderboard_plot_emits_bar_payload(tmp_path) -> None:
    '''Leaderboard plots emit bar traces ordered by rank.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_plot_workspace(tmp_path)

    plot = build_leaderboard_plot(
        tmp_path,
        metric_name="auc",
        mode="max",
        max_depth=2,
        top_n=2,
    )

    assert plot.plot_kind == "leaderboard_bar"
    assert plot.metric_names == ("auc",)
    assert plot.data[0]["type"] == "bar"
    assert plot.data[0]["x"] == ["missing", "balanced"]
    assert plot.included_count == 4
    assert plot.skipped_count == 0


def test_build_metric_scatter_plot_emits_marker_trace(tmp_path) -> None:
    '''Metric scatter plots include only rows with all requested numeric metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_plot_workspace(tmp_path)

    plot = build_metric_scatter_plot(
        tmp_path,
        x_metric="auc",
        y_metric="loss",
        color_metric="runtime",
        max_depth=2,
    )

    assert plot.plot_kind == "metric_scatter"
    assert plot.metric_names == ("auc", "loss", "runtime")
    assert plot.data[0]["type"] == "scatter"
    assert plot.data[0]["mode"] == "markers"
    assert plot.included_count == 3
    assert plot.skipped_count == 1
    assert plot.data[0]["marker"]["color"] == [30.0, 35.0, 12.0]


def test_build_parallel_coordinates_plot_emits_dimensions(tmp_path) -> None:
    '''Parallel-coordinate plots emit one dimension per requested metric.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_plot_workspace(tmp_path)

    plot = build_parallel_coordinates_plot(
        tmp_path,
        metric_names=("auc", "loss", "runtime"),
        max_depth=2,
    )

    assert plot.plot_kind == "parallel_coordinates"
    assert plot.data[0]["type"] == "parcoords"
    assert [dimension["label"] for dimension in plot.data[0]["dimensions"]] == [
        "auc",
        "loss",
        "runtime",
    ]
    assert plot.included_count == 3
    assert plot.skipped_count == 1


def test_build_parallel_coordinates_plot_requires_two_metrics(tmp_path) -> None:
    '''Parallel-coordinate plots require at least two metric dimensions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    with pytest.raises(ValueError, match="at least two"):
        build_parallel_coordinates_plot(tmp_path, metric_names=("auc",))


def test_build_pareto_scatter_plot_emits_front_and_dominated_traces(tmp_path) -> None:
    '''Pareto scatter plots emit front and dominated traces.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    _write_plot_workspace(tmp_path)

    plot = build_pareto_scatter_plot(
        tmp_path,
        objectives=(
            parse_pareto_objective("auc:max"),
            parse_pareto_objective("loss:min"),
        ),
        max_depth=2,
    )

    assert plot.plot_kind == "pareto_scatter"
    assert plot.metric_names == ("auc", "loss")
    assert [trace["name"] for trace in plot.data] == ["Pareto front", "Dominated"]
    assert plot.data[0]["text"] == ["balanced", "fast"]
    assert plot.data[1]["text"] == ["dominated"]
    assert plot.skipped_count == 1
