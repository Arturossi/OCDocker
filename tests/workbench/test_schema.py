#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Workbench JSON Schema helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import available_schema_names
from OCDocker.Workbench import build_json_schema
from OCDocker.Workbench import build_schema_catalog

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Functions
###############################################################################
## Public ##


def test_available_schema_names_include_gui_entrypoints() -> None:
    '''Registered schemas include specs, manifests, and read-only reports.'''

    names = available_schema_names()

    assert "ocscore_study" in names
    assert "vs_campaign" in names
    assert "run_manifest" in names
    assert "result_summary" in names
    assert "run_detail" in names
    assert "metric_leaderboard" in names
    assert "metric_matrix" in names
    assert "pareto_front" in names
    assert "metric_catalog" in names
    assert "workbench_adoption_plan" in names
    assert "workbench_adoption_result" in names
    assert "workbench_analysis_report" in names
    assert "workbench_artifact_index" in names
    assert "workbench_comparison" in names
    assert "workbench_plot" in names
    assert names == tuple(sorted(names))


def test_build_json_schema_returns_model_schema() -> None:
    '''Single-model schema output exposes expected fields.'''

    schema = build_json_schema("ocscore_study")

    assert schema["title"] == "OCScoreStudySpec"
    assert "properties" in schema
    assert "inputs" in schema["properties"]
    assert "output_dir" in schema["properties"]


def test_build_schema_catalog_can_select_subset() -> None:
    '''Schema catalogs can include only requested model names.'''

    catalog = build_schema_catalog(("ocscore_study", "result_summary"))

    assert catalog["schema_version"] == 1
    assert set(catalog["schemas"]) == {"ocscore_study", "result_summary"}
    assert "run_manifest" in catalog["available_schemas"]


def test_build_schema_catalog_rejects_unknown_names() -> None:
    '''Unknown schema names fail before producing partial catalogs.'''

    with pytest.raises(ValueError, match="Unknown Workbench schema"):
        build_schema_catalog(("unknown",))
