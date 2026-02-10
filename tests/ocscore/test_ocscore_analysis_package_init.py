#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for OCScore.Analysis package-level exports.
'''

# Imports
###############################################################################
import importlib
import sys
import types

import pytest

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

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##


@pytest.mark.order(351)
def test_analysis_package_exposes_shap_exports_when_shap_module_is_available(monkeypatch):
    analysis_pkg = importlib.import_module("OCDocker.OCScore.Analysis")

    run_marker = object()
    output_marker = type("OutputPathsMarker", (), {})
    studies_marker = type("StudyHandlesMarker", (), {})
    best_marker = type("BestSelectionsMarker", (), {})
    select_marker = object()
    data_marker = type("DataHandlesMarker", (), {})
    prepare_marker = object()
    nn_marker = object()
    compute_marker = object()
    plots_marker = object()

    fake_shap = types.ModuleType("OCDocker.OCScore.Analysis.SHAP")
    fake_shap.run_shap_analysis = run_marker
    fake_shap.OutputPaths = output_marker
    fake_shap.StudyHandles = studies_marker
    fake_shap.BestSelections = best_marker
    fake_shap.select_best_from_studies = select_marker
    fake_shap.DataHandles = data_marker
    fake_shap.load_and_prepare_data = prepare_marker
    fake_shap.build_neural_net = nn_marker
    fake_shap.compute_shap_values = compute_marker
    fake_shap.plots = plots_marker

    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", fake_shap)

    reloaded = importlib.reload(analysis_pkg)
    assert reloaded.run_shap_analysis is run_marker
    assert reloaded.OutputPaths is output_marker
    assert reloaded.StudyHandles is studies_marker
    assert reloaded.BestSelections is best_marker
    assert reloaded.select_best_from_studies is select_marker
    assert reloaded.DataHandles is data_marker
    assert reloaded.load_and_prepare_data is prepare_marker
    assert reloaded.build_neural_net is nn_marker
    assert reloaded.compute_shap_values is compute_marker
    assert reloaded.shap_plots is plots_marker

    monkeypatch.delitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", raising=False)
    importlib.reload(reloaded)


@pytest.mark.order(352)
def test_analysis_package_sets_shap_exports_to_none_when_shap_import_fails(monkeypatch):
    analysis_pkg = importlib.import_module("OCDocker.OCScore.Analysis")

    # Missing required names causes `from .SHAP import ...` to fail.
    fake_shap = types.ModuleType("OCDocker.OCScore.Analysis.SHAP")
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", fake_shap)

    reloaded = importlib.reload(analysis_pkg)
    assert reloaded.run_shap_analysis is None
    assert reloaded.OutputPaths is None
    assert reloaded.StudyHandles is None
    assert reloaded.BestSelections is None
    assert reloaded.select_best_from_studies is None
    assert reloaded.DataHandles is None
    assert reloaded.load_and_prepare_data is None
    assert reloaded.build_neural_net is None
    assert reloaded.compute_shap_values is None
    assert reloaded.shap_plots is None

    monkeypatch.delitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", raising=False)
    importlib.reload(reloaded)


@pytest.mark.order(353)
def test_analysis_package_sets_impact_exports_to_none_when_impact_import_fails(monkeypatch):
    analysis_pkg = importlib.import_module("OCDocker.OCScore.Analysis")

    # Keep SHAP import successful so we isolate the Impact fallback branch.
    fake_shap = types.ModuleType("OCDocker.OCScore.Analysis.SHAP")
    fake_shap.run_shap_analysis = object()
    fake_shap.OutputPaths = type("OutputPathsMarker", (), {})
    fake_shap.StudyHandles = type("StudyHandlesMarker", (), {})
    fake_shap.BestSelections = type("BestSelectionsMarker", (), {})
    fake_shap.select_best_from_studies = object()
    fake_shap.DataHandles = type("DataHandlesMarker", (), {})
    fake_shap.load_and_prepare_data = object()
    fake_shap.build_neural_net = object()
    fake_shap.compute_shap_values = object()
    fake_shap.plots = object()
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", fake_shap)

    # Missing required Impact names triggers fallback to None exports.
    fake_impact = types.ModuleType("OCDocker.OCScore.Analysis.Impact")
    monkeypatch.setitem(sys.modules, "OCDocker.OCScore.Analysis.Impact", fake_impact)

    reloaded = importlib.reload(analysis_pkg)
    assert reloaded.build_impact_overview is None
    assert reloaded.plot_impact_arrows_inline_labels is None
    assert reloaded.get_neutral_features is None

    monkeypatch.delitem(sys.modules, "OCDocker.OCScore.Analysis.SHAP", raising=False)
    monkeypatch.delitem(sys.modules, "OCDocker.OCScore.Analysis.Impact", raising=False)
    importlib.reload(reloaded)
