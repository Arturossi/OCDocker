#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for embedded Workbench browser assets.
'''

# Imports
###############################################################################
from __future__ import annotations

import pytest

from OCDocker.Workbench import build_workbench_web_asset
from OCDocker.Workbench import is_workbench_web_asset_path

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
## Public ##


def test_build_workbench_web_asset_serves_browser_entrypoint() -> None:
    '''Workbench web assets include the browser shell and API calls.'''

    content_type, body = build_workbench_web_asset("/app")
    script_type, script = build_workbench_web_asset("/app.js")
    style_type, style = build_workbench_web_asset("/app.css")

    assert content_type == "text/html; charset=utf-8"
    assert script_type == "text/javascript; charset=utf-8"
    assert style_type == "text/css; charset=utf-8"
    assert b"Decision Console" in body
    assert b"Run Detail" in body
    assert b"Ablations" in body
    assert b"Ablation Delta" in body
    assert b"Metric Direction Heatmap" in body
    assert b"Evidence Explorer" in body
    assert b"Performance Profile" in body
    assert b"Optuna Trace" in body
    assert b"SHAP Importance" in body
    assert b"Figure Gallery" in body
    assert b"ablation-delta-plot" in body
    assert b"ablation-heatmap" in body
    assert b"evidence-performance-plot" in body
    assert b"evidence-gallery" in body
    assert b"/api/overview" in script
    assert b"/api/run-detail" in script
    assert b"/api/ablations" in script
    assert b"/api/evidence" in script
    assert b"/api/evidence-asset" in script
    assert b"renderAblationDeltaPlot" in script
    assert b"renderAblationHeatmap" in script
    assert b"renderEvidencePerformancePlot" in script
    assert b"renderEvidenceTracePlot" in script
    assert b"renderShapFeaturePlot" in script
    assert b"renderEvidenceGallery" in script
    assert b"plot-axis-label" in script
    assert b"x axis" in script
    assert b".stat-grid" in style
    assert b".detail-grid" in style
    assert b"decision-grid" in style
    assert b".plot-split" in style
    assert b".evidence-grid" in style
    assert b".evidence-gallery" in style
    assert b".heat-cell" in style
    assert b".plot-zero-line" in style


def test_is_workbench_web_asset_path_recognizes_known_routes() -> None:
    '''Workbench web asset route detection keeps API root available.'''

    assert is_workbench_web_asset_path("/app") is True
    assert is_workbench_web_asset_path("/app/") is True
    assert is_workbench_web_asset_path("/app.css") is True
    assert is_workbench_web_asset_path("/app.js") is True
    assert is_workbench_web_asset_path("/") is False
    assert is_workbench_web_asset_path("/api/health") is False


def test_build_workbench_web_asset_rejects_unknown_route() -> None:
    '''Workbench web asset builder rejects unknown routes.'''

    with pytest.raises(KeyError, match="Unknown Workbench web asset"):
        build_workbench_web_asset("/missing.js")
