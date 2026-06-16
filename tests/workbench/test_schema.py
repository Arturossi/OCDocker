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
## Public ##


def test_available_schema_names_include_gui_entrypoints() -> None:
    '''Registered schemas include specs, manifests, and read-only reports.'''

    names = available_schema_names()

    assert "ocscore_study" in names
    assert "vs_campaign" in names
    assert "run_manifest" in names
    assert "result_summary" in names
    assert "metric_leaderboard" in names
    assert "metric_matrix" in names
    assert "pareto_front" in names
    assert "metric_catalog" in names
    assert "workbench_analysis_report" in names
    assert "workbench_artifact_index" in names
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
