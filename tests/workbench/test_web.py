#!/usr/bin/env python3

# Description
###############################################################################
"""
Tests for packaged Workbench browser assets.
"""

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

import pytest

from OCDocker.Workbench import build_workbench_web_asset
from OCDocker.Workbench import is_workbench_web_asset_path
from OCDocker.Workbench.Web import WORKBENCH_STATIC_DIR

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


def test_build_workbench_web_asset_serves_strict_ocscore_dashboard() -> None:
    '''Workbench web assets include only the strict OCScore dashboard shell.'''

    content_type, body = build_workbench_web_asset("/app")
    script_type, script = build_workbench_web_asset("/app.js")
    style_type, style = build_workbench_web_asset("/app.css")

    assert content_type == "text/html; charset=utf-8"
    assert script_type == "text/javascript; charset=utf-8"
    assert style_type == "text/css; charset=utf-8"
    assert b"OCScore Control Dashboard" in body
    assert b"/app-favicon.png" in body
    assert b"/app-brand-logo.png" in body
    assert b'class="brand-logo"' in body
    assert b".brand-logo" in style
    assert b'role="tablist"' in body
    assert b"tab-ablation" in body
    assert b"tab-design" in body
    assert b"panel-design" in body
    assert b">Design</button>" in body
    assert b"Design ablation" in body
    assert b"bindAblationDesignPanel" in script
    assert b"/api/ablation-design" in script
    assert b"ablation-design-panel" in style
    assert b"setActiveTab" in script
    assert b"bindAppTabs" in script
    assert b"bindCollapsibleZones" in script
    assert b"bindThemeToggle" in script
    assert b"applyTheme" in script
    assert b'data-theme="dark"' in body
    assert b"#1e1e1e" in style
    assert b"PLOTLY_EXPORT_LAYOUT" in script
    assert b"zone-collapsible" in style
    assert b"Ablation protocol</span>" in body
    assert b"renderProtocolPanel" in script
    assert b"protocol-grid" in style
    assert b"renderRunContext" in script
    assert b"run-context" in style
    assert b"loadPersistedUiState" in script
    assert b"localStorage" in script
    assert b"sessionStorage" in script
    assert b"buildSimpleBarPlotlySpec" in script
    assert b"synthesized-baseline" in style
    assert b"delta-within-noise" in style
    assert b"data-zone=\"cvTable\"" in body
    assert b"detailReplicaSort" in script
    assert b"renderDetailReplicaTable" in script
    assert b"bindOptunaDashboardButtons" in script
    assert b"apiPost" in script
    assert b"optuna-open" in script
    assert b"replicaMetricColumns" in script
    assert b".app-tabs" in style
    assert b".tab-panel" in style
    assert b"Results</h2>" in body
    assert b"comparison-legend-top" in body
    assert b"comparison-legend-bottom" in body
    assert b"comparison-color-legend-top" in body
    assert b"comparison-color-legend-bottom" in body
    assert b"renderComparisonColorLegend" in script
    assert b"color-legend-grid" in style
    assert b"metric-legend" in style
    assert b"metricStatMarkup" in script
    assert b"metric-stat" in style
    assert b"metric-cell-mu" in style
    assert b"metric-stat-aggregate" in style
    assert b"metric-std" in style
    assert b"cv-panel" in body
    assert b"renderCrossValidationPanel" in script
    assert b"plot-span-full" in style
    assert b"cdn.plot.ly/plotly" in body
    assert b"buildRankPlotlySpec" in script
    assert b"barmode" in script
    assert b"handleRankPlotRowClick" in script
    assert b"rankPlotExpandLabels" in script
    assert b"scheduleAblationDesignPreview" in script
    assert b"ensureAblationDesignFeatures" in script
    assert b"mountVirtualFeatureList" in script
    assert b"renderAblationDesignTemplateDiff" in script
    assert b"writeAblationDesignPolicy" in script
    assert b"/api/ablation-design/write" in script
    assert b"ablation-design-write-policy" in body
    assert b"ablation-design-template-diff" in body
    assert b"ablation-design-case-warning" in body
    assert b"ablationDesignUpdateCaseAndFilterWarnings" in script
    assert b"mountPendingPlotlyCharts" in script
    assert b"plotly-host" in style
    assert b"externalBaselineKindLabel" in script
    assert b"RANK_BAR_LABELS" in script
    assert b"Other consensus" in script
    assert b'"SF"' in script
    assert b"#9BD4EF" in script
    assert b"barLabel" in script
    assert b"rankPlotLabelOffset" in script
    assert b"replica values" in script
    assert b"replica values" in script
    assert b"cross-validation" in body.lower() or b"Cross-validation" in body
    assert b"renderComparisonExportActions" in script
    assert b"objectsToCsv" in script
    assert b"externalEntryId" in script
    assert b"normalizeComparisonBaseline" in script
    assert b"findExternalBaselineByEntryId" in script
    assert b"data-comparison-export" in script
    assert b"comparison-charts" in body
    assert b"result-scope-select" in body
    assert b"comparison-baseline-select" in body
    assert b"Workspace Issues" in body
    assert b"comparisonEntries" in script
    assert b"renderComparisonTable" in script
    assert b"generatedAllDeltasPlot" in script
    assert b"renderGlobalControls" in script
    assert b"full_ocscore" in script
    assert b"Full model" in script
    assert b"modelDisplayName" in script
    assert b"modelDescription" in script
    assert b"ABLATION_DESCRIPTIONS" in script
    assert b"lr_sf" in script
    assert b"renderIssues" in script
    assert b"metricColumns" in script
    assert b"bindSortButtons" in script
    assert b"comparisonSort" in script
    assert b"comparisonBaseline" in script
    assert b"scopedExternalBaselines" in script
    assert b"external_baselines" in script
    assert b"rankedMetricCell" in script
    assert b"Rank ${rank} of ${total}" in script
    assert b"metric-rank" not in script
    assert b"rank-mark-1" in body
    assert b"reference row" in body
    assert b".rank-top" in style
    assert b".rank-second" in style
    assert b".rank-third" in style
    assert b"generatedRankPlot" in script
    assert b"generatedStabilityPlot" in script
    assert b"shapFigureSection" in script
    assert b"shap-figure-grid" in style
    assert b"renderFigureControls" in script
    assert b"FIGURE_RENDER_LIMIT" in script
    assert b"figureFilters" in script
    assert b"resultScope" in script
    assert b"Test" in script
    assert b"Validation" in script
    assert b"Combined" in script
    assert b"Model comparison" in script
    assert b"Selected model" in script
    assert b"registerPlotExport" in script
    assert b"downloadText" in script
    assert b"svgToPngBlob" in script
    assert b"copyPngToClipboard" in script
    assert b'data-export-kind="png"' in script
    assert b'data-export-kind="copy"' in script
    assert b'return /[",\\n]/.test(text)' in script
    assert b'return /[",\n]/.test(text)' not in script
    assert b"Metric Delta vs" not in script
    assert b"async function refresh" in script
    assert b'\nfunction refresh()' not in script
    assert b'lines.join("\\n")' in script
    from py_mini_racer import MiniRacer

    parse_only = script.decode("utf-8").replace('$("refresh").addEventListener("click", refresh);', "")
    parse_only = parse_only.replace("bindAppTabs();", "")
    parse_only = parse_only.replace("bindAblationDesignPanel();", "")
    parse_only = parse_only.replace("bindCollapsibleZones();", "")
    parse_only = parse_only.replace("bindThemeToggle();", "")
    parse_only = parse_only.replace("loadPersistedUiState();", "")
    parse_only = parse_only.replace("uiStateHydrated = true;", "")
    parse_only = parse_only.replace('setActiveTab(state.activeTab || "ablation");', "")
    parse_only = parse_only.replace("refresh();", "")
    ctx = MiniRacer()
    ctx.eval(parse_only)
    ctx.eval(
        """
(function(){
  var rows = [
    {label:'study_a', value:0.5, std:0.01, count:3, display:'0.5', barLabel:'mu 0.5', external:false, study_name:'study_a'},
    {label:'vina_vina', value:0.4, std:0.02, count:1, display:'0.4', barLabel:'mu 0.4', external:true, baseline_family:'scoring_function', study_name:'vina_vina'}
  ];
  var metric = {name:'test_bedroc', label:'Test BEDROC', direction:'max'};
  var spec = buildRankPlotlySpec(rows, metric);
  var barTraces = spec.data.filter(function(trace){ return trace.type === 'bar'; });
  if (barTraces.length < 1) throw new Error('buildRankPlotlySpec must emit at least one bar trace');
  if (spec.data.some(function(trace){ return trace.type === 'scatter'; })) {
    throw new Error('buildRankPlotlySpec must not use scatter legend traces');
  }
  if (spec.layout.barmode !== 'overlay') throw new Error('buildRankPlotlySpec must set barmode overlay');
  if (!spec.layout.title || spec.layout.title.text.indexOf('Error bars') === -1) {
    throw new Error('buildRankPlotlySpec must include the error-bar subtitle');
  }
  if (spec.layout.legend.itemdoubleclick !== 'toggleothers') {
    throw new Error('buildRankPlotlySpec legend must support double-click isolate');
  }
  var simple = buildSimpleBarPlotlySpec(rows, metric);
  if (simple.layout.barmode !== 'overlay') throw new Error('buildSimpleBarPlotlySpec must set barmode overlay');
})();
"""
    )
    assert b"/api/figure-asset" in script
    assert b"data-sort-key" in script
    assert b"metric:" in script
    assert b"/api/ocscore-workspace" in script
    assert b"/api/evidence" not in script
    assert b"Metric Scatter" not in body
    assert b"Figure Gallery" not in body
    assert b".figure-list" in style
    assert b".filter-grid" in style
    assert b".decision-plots" in style
    assert b".sort-button" in style
    assert b".figure-preview" in style
    assert b".generated-plot" in style
    assert b".decision-svg" in style
    assert b".export-actions" in style
    assert b"figure-lightbox" in body
    assert b"bindFigureLightbox" in script
    assert b"figure-preview-button" in script
    assert b".figure-lightbox" in style
    assert b".gallery-note" in style


def test_workbench_static_assets_exist_on_disk() -> None:
    '''Packaged Workbench UI assets live beside Web.py for pip installs.'''

    for name in ("index.html", "app.css", "app.js"):
        assert (WORKBENCH_STATIC_DIR / name).is_file()


def test_build_workbench_web_asset_matches_static_files_on_disk() -> None:
    '''HTTP asset payloads match the packaged static files on disk.'''

    for route, filename in (
        ("/app", "index.html"),
        ("/app.css", "app.css"),
        ("/app.js", "app.js"),
    ):
        _, body = build_workbench_web_asset(route)
        assert body == (WORKBENCH_STATIC_DIR / filename).read_bytes()


def test_is_workbench_web_asset_path_recognizes_known_routes() -> None:
    '''Workbench web asset route detection keeps API root available.'''

    assert is_workbench_web_asset_path("/app") is True
    assert is_workbench_web_asset_path("/app/") is True
    assert is_workbench_web_asset_path("/app.css") is True
    assert is_workbench_web_asset_path("/app.js") is True
    assert is_workbench_web_asset_path("/app-favicon.png") is True
    assert is_workbench_web_asset_path("/app-brand-logo.png") is True
    assert is_workbench_web_asset_path("/") is False
    assert is_workbench_web_asset_path("/api/health") is False


def test_build_workbench_web_asset_rejects_unknown_route() -> None:
    '''Workbench web asset builder rejects unknown routes.'''

    with pytest.raises(KeyError, match="Unknown Workbench web asset"):
        build_workbench_web_asset("/missing.js")


def test_build_workbench_web_asset_serves_favicon_png() -> None:
    '''Workbench serves the small browser tab icon.'''

    content_type, body = build_workbench_web_asset("/app-favicon.png")

    assert content_type == "image/png"
    assert body.startswith(b"\x89PNG\r\n\x1a\n")
    favicon_path = Path(__file__).resolve().parents[2] / "ocdocker_small_logo.png"
    if favicon_path.is_file():
        assert body == favicon_path.read_bytes()


def test_build_workbench_web_asset_serves_brand_logo_png() -> None:
    '''Workbench serves the in-page OCDocker wordmark.'''

    content_type, body = build_workbench_web_asset("/app-brand-logo.png")

    assert content_type == "image/png"
    assert body.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(body) > 1024
    brand_path = Path(__file__).resolve().parents[2] / "OCDocker.png"
    if brand_path.is_file():
        assert body == brand_path.read_bytes()
