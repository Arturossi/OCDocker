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
