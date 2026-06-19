#!/usr/bin/env python3

# Description
###############################################################################
'''
Embedded browser assets for the strict OCScore Workbench dashboard.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path
from typing import Final

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

# Constants
###############################################################################

WORKBENCH_WEB_INDEX_ROUTE: Final[str] = "/app"
WORKBENCH_WEB_FAVICON_ROUTE: Final[str] = "/app-favicon.png"
WORKBENCH_WEB_BRAND_LOGO_ROUTE: Final[str] = "/app-brand-logo.png"
WORKBENCH_WEB_ROUTES: Final[tuple[str, ...]] = (
    "/app",
    "/app/",
    "/app.css",
    "/app.js",
    WORKBENCH_WEB_FAVICON_ROUTE,
    WORKBENCH_WEB_BRAND_LOGO_ROUTE,
)

_INDEX_HTML: Final[str] = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCDocker Workbench</title>
  <link rel="icon" type="image/png" href="/app-favicon.png">
  <link rel="stylesheet" href="/app.css">
  <script>
    (function () {
      try {
        var saved = JSON.parse(sessionStorage.getItem("ocscore-workbench-ui") || "{}");
        document.documentElement.setAttribute("data-theme", saved.theme === "light" ? "light" : "dark");
      } catch (_error) {
        document.documentElement.setAttribute("data-theme", "dark");
      }
    })();
  </script>
</head>
<body>
  <div class="app-header">
    <header class="topbar">
      <div class="brand">
        <img class="brand-logo" src="/app-brand-logo.png" height="28" alt="OCDocker">
        <div class="brand-text">
          <h1>OCScore Control Dashboard</h1>
        </div>
      </div>
      <div class="toolbar" data-tab-toolbar="ablation">
        <label class="field" for="result-scope-select"><span>Scope</span><select id="result-scope-select"></select></label>
        <label class="field" for="comparison-baseline-select"><span>Reference</span><select id="comparison-baseline-select"></select></label>
        <label class="field" for="decision-metric-select"><span>Metric</span><select id="decision-metric-select"></select></label>
        <button id="theme-toggle" type="button" class="ghost-button theme-toggle" aria-pressed="true" title="Toggle light/dark appearance">Theme</button>
        <button id="refresh" type="button">Refresh</button>
      </div>
    </header>
    <section class="run-context" id="run-context" aria-label="Run context">
      <span id="health-dot" class="dot pending"></span>
      <span id="health-label" class="run-context-health"></span>
      <span id="root-label" class="root-label run-context-root"></span>
      <div id="run-context-scroll" class="run-context-scroll">
        <div id="run-context-items" class="run-context-items"></div>
      </div>
    </section>
    <nav class="app-tabs" role="tablist" aria-label="Dashboard sections">
      <button type="button" class="tab active" role="tab" id="tab-ablation" aria-selected="true" aria-controls="panel-ablation" data-tab="ablation">Ablation</button>
    </nav>
  </div>
  <main>
    <section id="panel-ablation" class="tab-panel active" role="tabpanel" aria-labelledby="tab-ablation" data-tab-panel="ablation">
    <section class="stat-grid">
      <article class="stat"><span>Studies</span><strong id="study-count">-</strong></article>
      <article class="stat"><span>Completed</span><strong id="completed-count">-</strong></article>
      <article class="stat"><span>Failed</span><strong id="failed-count">-</strong></article>
      <article class="stat"><span>Missing</span><strong id="missing-count">-</strong></article>
    </section>
    <section class="panel protocol-panel">
      <div class="zone-block zone-collapsible" data-zone="protocol">
        <div class="zone-head zone-head-split">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="protocol-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Ablation protocol</span>
          </button>
          <span id="protocol-summary" class="muted"></span>
        </div>
        <div id="protocol-zone" class="zone-body">
          <div id="protocol-content" class="protocol-grid"></div>
        </div>
      </div>
    </section>
    <section id="issue-panel" class="panel issue-panel" hidden>
      <div class="panel-head panel-head-split">
        <h2>Workspace Issues</h2>
        <span id="issue-summary" class="muted"></span>
      </div>
      <div id="issue-list" class="issue-list"></div>
    </section>

    <section class="panel comparison-panel">
      <div class="panel-head">
        <h2>Results</h2>
      </div>
      <div class="zone-block zone-collapsible" data-zone="comparisonTable">
        <div class="zone-head">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="comparison-table-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Table</span>
          </button>
        </div>
        <div id="comparison-table-zone" class="zone-body">
          <div class="zone-toolbar">
            <span id="comparison-summary" class="muted"></span>
            <div class="export-actions" id="comparison-export-actions"></div>
          </div>
          <div id="comparison-table" class="table-wrap"></div>
          <div id="comparison-color-legend" class="metric-legend color-legend" aria-label="Model color key"></div>
          <div id="comparison-legend" class="metric-legend" aria-label="Metric notation">
            <span class="legend-item"><span class="legend-mark mean-mark">μ</span> replica mean (multi-replica cells only)</span>
            <span class="legend-item"><span class="legend-mark std-mark">σ</span> std dev in those cells</span>
            <span class="legend-item"><span class="legend-mark delta-mark">Δ</span> vs reference (column)</span>
            <span class="legend-item"><span class="legend-mark noise-mark">~</span> |Δ| &lt; σ (within replica noise)</span>
            <span class="legend-item"><span class="legend-mark synth-mark">≈</span> synthesized SF consensus (approximate)</span>
          </div>
        </div>
      </div>
      <div class="zone-block zone-collapsible" data-zone="comparisonCharts">
        <div class="zone-head">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="comparison-charts-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Charts</span>
          </button>
        </div>
        <div id="comparison-charts-zone" class="zone-body">
          <div id="comparison-charts" class="decision-plots"></div>
        </div>
      </div>
    </section>

    <section class="panel cv-panel" id="cv-panel" hidden>
      <div class="panel-head">
        <h2>Cross-validation</h2>
      </div>
      <div class="zone-block zone-collapsible" data-zone="cvTable">
        <div class="zone-head zone-head-split">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="cv-table-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Table</span>
          </button>
          <span id="cv-summary" class="muted"></span>
        </div>
        <div id="cv-table-zone" class="zone-body">
          <div id="cv-table" class="table-wrap"></div>
        </div>
      </div>
    </section>

    <section class="panel detail-panel" id="detail-panel">
      <div class="panel-head">
        <h2 id="detail-title">Details</h2>
      </div>
      <div class="zone-block zone-collapsible" data-zone="detailReplicas">
        <div class="zone-head">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="detail-replicas-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Replicas</span>
          </button>
        </div>
        <div id="detail-replicas-zone" class="zone-body">
          <div id="detail-replicas" class="table-wrap"></div>
        </div>
      </div>
      <div class="zone-block zone-collapsible" data-zone="detailCharts">
        <div class="zone-head">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="detail-charts-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Charts</span>
          </button>
        </div>
        <div id="detail-charts-zone" class="zone-body">
          <div id="detail-plots" class="decision-plots"></div>
        </div>
      </div>
      <div class="zone-block zone-collapsible" data-zone="detailFigures">
        <div class="zone-head">
          <button type="button" class="zone-toggle" aria-expanded="true" aria-controls="detail-figures-zone">
            <span class="zone-chevron" aria-hidden="true">▾</span>
            <span class="zone-title">Figures</span>
          </button>
        </div>
        <div id="detail-figures-zone" class="zone-body">
          <div class="figure-filters-bar">
            <span class="zone-title">Figure filters</span>
            <span id="figure-filter-summary" class="muted"></span>
          </div>
          <div id="figure-controls" class="filter-grid"></div>
          <div id="figure-list" class="figure-list"></div>
        </div>
      </div>
    </section>
    </section>
  </main>
  <div id="figure-lightbox" class="figure-lightbox" hidden>
    <button type="button" class="figure-lightbox-close" aria-label="Close expanded figure">×</button>
    <img id="figure-lightbox-image" class="figure-lightbox-image" alt="">
    <p id="figure-lightbox-caption" class="figure-lightbox-caption"></p>
  </div>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" defer></script>
  <script src="/app.js" defer></script>
</body>
</html>
"""

_STYLE_CSS: Final[str] = """* { box-sizing: border-box; }
:root,
[data-theme="dark"] {
  color-scheme: dark;
  --bg: #1e1e1e;
  --surface: #252526;
  --surface-raised: #2d2d30;
  --surface-input: #3c3c3c;
  --ink: #cccccc;
  --muted: #858585;
  --line: #3e3e42;
  --accent: #4db6ac;
  --danger: #f48771;
  --warn: #cca700;
  --ok: #89d185;
  --plot-bg: #ffffff;
  --plot-border: #464647;
  --header-shadow: rgba(0, 0, 0, 0.35);
  --rank-top-bg: rgba(137, 209, 133, 0.14);
  --rank-second-bg: rgba(77, 182, 172, 0.12);
  --rank-third-bg: rgba(77, 182, 172, 0.08);
  --row-hover: rgba(77, 182, 172, 0.08);
  --reference-row-bg: rgba(77, 182, 172, 0.1);
  --ghost-bg: #2d2d30;
  --disabled-bg: #3a3a3a;
  --disabled-fg: #6e6e6e;
  --badge-neutral-bg: #3a3a3a;
  --protocol-card-border: #3e3e42;
  --issue-bg: rgba(204, 167, 0, 0.1);
  --issue-border: #6b5a2a;
  --pill-text: #1e1e1e;
  --table-border: #3e3e42;
  --accent-soft-bg: rgba(77, 182, 172, 0.14);
  --accent-soft-border: rgba(77, 182, 172, 0.45);
  --legend-mean-bg: rgba(77, 182, 172, 0.16);
  --legend-std-bg: #333337;
  --legend-delta-bg: rgba(137, 209, 133, 0.16);
  --legend-synth-bg: rgba(204, 167, 0, 0.16);
  --kind-synth-bg: rgba(204, 167, 0, 0.14);
  --kind-synth-border: rgba(204, 167, 0, 0.45);
  --badge-completed-bg: rgba(137, 209, 133, 0.16);
  --badge-failed-bg: rgba(244, 135, 113, 0.16);
  --badge-missing-bg: rgba(204, 167, 0, 0.16);
  --badge-running-bg: rgba(100, 149, 237, 0.16);
  --badge-running-fg: #9cdcfe;
  --run-context-bg: #252526;
  --tabs-bg: #252526;
  --tab-hover-bg: rgba(77, 182, 172, 0.08);
  --protocol-variant-bg: #333337;
  --protocol-variant-active-bg: rgba(77, 182, 172, 0.14);
  --plot-track-bg: #3c3c3c;
  --figure-card-border: #464647;
  --lightbox-overlay: rgba(0, 0, 0, 0.88);
  --lightbox-caption-bg: rgba(37, 37, 38, 0.94);
  --lightbox-close-bg: rgba(60, 60, 60, 0.94);
  --toast-bg: #007acc;
}
[data-theme="light"] {
  color-scheme: light;
  --bg: #f6f5f1;
  --surface: #ffffff;
  --surface-raised: #fbfaf7;
  --surface-input: #ffffff;
  --ink: #202833;
  --muted: #667085;
  --line: #d8d1c5;
  --accent: #087f7b;
  --danger: #b42318;
  --warn: #a15c00;
  --ok: #16703f;
  --plot-bg: #ffffff;
  --plot-border: #ebe5dc;
  --header-shadow: rgba(32, 40, 51, 0.06);
  --rank-top-bg: #edf7f1;
  --rank-second-bg: #f4faf8;
  --rank-third-bg: #f8fbfa;
  --row-hover: #f4faf8;
  --reference-row-bg: #f8fbfa;
  --ghost-bg: #fffdf8;
  --disabled-bg: #ece8df;
  --disabled-fg: #667085;
  --badge-neutral-bg: #eef1f4;
  --protocol-card-border: #ebe5dc;
  --issue-bg: #fffaf0;
  --issue-border: #e8c178;
  --pill-text: #202833;
  --table-border: #ebe5dc;
  --accent-soft-bg: #e8f4f3;
  --accent-soft-border: #9ec5c3;
  --legend-mean-bg: #eef4f4;
  --legend-std-bg: #f3f4f6;
  --legend-delta-bg: #edf7f1;
  --legend-synth-bg: #fff1d8;
  --kind-synth-bg: #fff6e8;
  --kind-synth-border: #e8c178;
  --badge-completed-bg: #e8f5ee;
  --badge-failed-bg: #fce9e7;
  --badge-missing-bg: #fff1d8;
  --badge-running-bg: #eaf2ff;
  --badge-running-fg: #315c9c;
  --run-context-bg: #f3f8f7;
  --tabs-bg: #fffdf8;
  --tab-hover-bg: rgba(8, 127, 123, 0.04);
  --protocol-variant-bg: #eef1f4;
  --protocol-variant-active-bg: #e8f4f3;
  --plot-track-bg: #ebe5dc;
  --figure-card-border: #ebe5dc;
  --lightbox-overlay: rgba(32, 40, 51, 0.92);
  --lightbox-caption-bg: rgba(255, 255, 255, 0.92);
  --lightbox-close-bg: rgba(255, 255, 255, 0.92);
  --toast-bg: #202833;
}
body {
  margin: 0;
  min-width: 320px;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.topbar, main { padding-left: clamp(16px, 4vw, 40px); padding-right: clamp(16px, 4vw, 40px); }
.app-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--surface-raised);
  border-bottom: 1px solid var(--line);
  box-shadow: 0 2px 10px var(--header-shadow);
}
.topbar {
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding-top: 6px;
  padding-bottom: 6px;
  border-bottom: none;
  background: var(--surface-raised);
}
.brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
.brand-logo { height: 28px; width: auto; max-width: min(168px, 38vw); object-fit: contain; object-position: left center; flex-shrink: 0; display: block; filter: brightness(0) invert(1); opacity: 0.92; }
[data-theme="light"] .brand-logo { filter: brightness(0); opacity: 0.9; }
.brand-text { min-width: 0; }
.brand-text h1 { margin: 0; }
h1 { margin: 0; font-size: 15px; font-weight: 700; letter-spacing: 0; line-height: 1.2; }
h2 { margin: 0; font-size: 16px; letter-spacing: 0; }
h3 { margin: 0; font-size: 14px; letter-spacing: 0; }
.product { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; white-space: nowrap; }
.toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.topbar .field { display: flex; flex-direction: row; align-items: center; gap: 6px; color: var(--muted); font-size: 11px; font-weight: 700; white-space: nowrap; }
.topbar input, .topbar select { width: auto; min-width: 108px; padding: 5px 8px; font-size: 13px; }
.topbar button { padding: 5px 10px; font-size: 12px; }
.field, .filter-field { display: grid; gap: 4px; color: var(--muted); font-size: 12px; font-weight: 700; }
input, select { width: 92px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; background: var(--surface-input); color: var(--ink); }
select { width: 100%; min-width: 132px; }
button { border: 1px solid var(--accent); border-radius: 6px; padding: 9px 13px; background: var(--accent); color: #ffffff; font-weight: 700; cursor: pointer; }
button:disabled { border-color: var(--line); background: var(--disabled-bg); color: var(--disabled-fg); cursor: not-allowed; }
.connection { display: none; }
.run-context {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px clamp(16px, 4vw, 40px) 5px;
  background: var(--run-context-bg);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  font-size: 11px;
  overflow: hidden;
}
.run-context-health { color: var(--muted); white-space: nowrap; flex-shrink: 0; }
.run-context-root {
  color: var(--muted);
  white-space: nowrap;
  max-width: 16ch;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 0;
}
.run-context-root::before { content: "·"; margin-right: 8px; color: var(--line); }
.run-context-scroll {
  flex: 1;
  min-width: 0;
  overflow: hidden;
}
.run-context-items {
  display: flex;
  flex-wrap: nowrap;
  gap: 6px 12px;
  align-items: center;
  width: max-content;
  min-width: 100%;
  will-change: transform;
}
.run-context-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  flex-shrink: 0;
  white-space: nowrap;
}
.run-context-item strong {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .05em;
  text-transform: uppercase;
  color: var(--muted);
}
.run-context-item span:last-child {
  white-space: nowrap;
}
.run-context-item.path-item span:last-child { color: var(--muted); }
.app-tabs:has(.tab:only-child) { display: none; }
.app-tabs {
  display: flex;
  align-items: stretch;
  gap: 2px;
  padding: 0 clamp(16px, 4vw, 40px);
  background: var(--tabs-bg);
  border-top: 1px solid var(--line);
  overflow-x: auto;
}
.tab {
  border: 0;
  border-bottom: 2px solid transparent;
  border-radius: 0;
  margin-bottom: -1px;
  padding: 6px 12px;
  background: transparent;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
}
.tab:hover { color: var(--ink); background: var(--tab-hover-bg); }
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: transparent;
}
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: #a3a3a3; }
.dot.ok { background: var(--ok); }
.dot.error { background: var(--danger); }
.root-label, .muted { color: var(--muted); }
main { padding-top: 16px; padding-bottom: 32px; }
.tab-panel { display: grid; gap: 18px; }
.tab-panel[hidden] { display: none; }
.stat-grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; }
.stat, .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
.stat { padding: 14px; display: grid; gap: 4px; }
.stat span { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
.stat strong { font-size: 28px; line-height: 1; }
.panel { padding: 14px; }
.comparison-panel .zone-block + .zone-block { margin-top: 12px; }
.comparison-panel .decision-plots { margin-top: 0; }
.detail-panel .zone-block + .zone-block { margin-top: 12px; }
.zone-block { display: grid; gap: 8px; }
.zone-head { display: flex; align-items: center; }
.zone-head-split { gap: 10px; flex-wrap: wrap; }
.zone-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 0;
  background: transparent;
  padding: 0;
  margin: 0;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-align: left;
}
.zone-toggle:hover .zone-title, .zone-toggle:hover .zone-chevron { color: var(--accent); }
.zone-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--muted);
}
.zone-chevron {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1;
  transition: transform 0.15s ease;
}
.zone-collapsible.is-collapsed .zone-chevron { transform: rotate(-90deg); }
.zone-body { display: grid; gap: 10px; }
.zone-collapsible.is-collapsed > .zone-body { display: none; }
.zone-collapsible.is-collapsed > .zone-head { margin-bottom: 0; }
.zone-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}
.zone-toolbar .export-actions { margin-left: auto; }
.protocol-panel { border-color: var(--line); }
.protocol-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
.protocol-card { border: 1px solid var(--protocol-card-border); border-radius: 6px; padding: 10px; display: grid; gap: 6px; align-content: start; }
.protocol-card h3 { margin: 0; font-size: 11px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
.protocol-card-full { grid-column: 1 / -1; }
.protocol-facts { display: grid; gap: 4px; margin: 0; }
.protocol-facts div { display: grid; gap: 1px; }
.protocol-facts dt { margin: 0; font-size: 11px; color: var(--muted); }
.protocol-facts dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.protocol-variants { display: flex; flex-wrap: wrap; gap: 6px; }
.protocol-variant { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; background: var(--protocol-variant-bg); color: var(--ink); }
.protocol-variant.active { background: var(--protocol-variant-active-bg); color: var(--accent); border: 1px solid var(--accent-soft-border); }
.metric-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin: 0 0 10px;
  font-size: 11px;
  color: var(--muted);
}
.legend-item { display: inline-flex; align-items: center; gap: 5px; }
.legend-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 800;
  line-height: 1.2;
}
.legend-mark.mean-mark { background: var(--legend-mean-bg); color: var(--accent); }
.legend-mark.std-mark { background: var(--legend-std-bg); color: var(--muted); }
.legend-mark.delta-mark { background: var(--legend-delta-bg); color: var(--ok); }
.legend-mark.noise-mark { background: var(--legend-std-bg); color: var(--muted); }
.legend-mark.synth-mark { background: var(--legend-synth-bg); color: var(--warn); }
.color-legend { margin-top: 8px; }
.color-legend-grid { display: flex; flex-wrap: wrap; gap: 6px 10px; max-height: 132px; overflow-y: auto; padding: 2px 0; }
.legend-intro { flex: 1 1 100%; font-size: 11px; color: var(--muted); }
.legend-swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 999px;
  border: 1px solid transparent;
  flex-shrink: 0;
}
.color-legend-item { max-width: min(28ch, 100%); }
.color-legend-item > span:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.metric-delta.delta-within-noise { color: var(--muted); text-decoration: underline dotted; text-underline-offset: 2px; }
.kind-badge.synthesized-baseline { background: var(--kind-synth-bg); color: var(--warn); border: 1px dashed var(--kind-synth-border); }
.cv-panel .zone-block { margin-top: 0; }
.metric-stat { display: inline-grid; gap: 1px; justify-items: end; line-height: 1.15; text-align: right; }
.metric-stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }
.metric-stat-aggregate { gap: 2px; }
.metric-stat-line { display: inline-flex; align-items: baseline; gap: 4px; justify-content: flex-end; }
.metric-cell-mu { font-size: 9px; font-weight: 800; color: var(--accent); line-height: 1; }
.metric-mean { font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-std { font-size: 10px; font-weight: 600; color: var(--muted); font-variant-numeric: tabular-nums; letter-spacing: 0.01em; }
.decision-plots { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-bottom: 10px; align-items: start; }
.decision-plots.layout-single { grid-template-columns: 1fr; }
.generated-plot.plot-span-full { grid-column: 1 / -1; width: 100%; min-width: 0; }
.generated-plot.plot-span-half { grid-column: span 1; width: 100%; min-width: 0; }
.generated-plot .decision-svg { width: 100%; height: auto; display: block; }
.plotly-host { width: 100%; min-height: 280px; min-width: 0; background: var(--plot-bg); border: 1px solid var(--plot-border); border-radius: 5px; }
.plotly-host .main-svg { border-radius: 5px; }
.cv-panel { border-color: var(--line); }
.cv-study-block { display: grid; gap: 8px; margin-bottom: 14px; }
.cv-study-block:last-child { margin-bottom: 0; }
.cv-study-block h3 { margin: 0; font-size: 14px; }
.panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.panel-head-split { align-items: center; flex-wrap: wrap; }
.panel-head-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; margin-left: auto; }
.detail-panel { border-color: var(--line); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; min-width: 720px; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--table-border); text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
th.numeric, td.numeric { text-align: right; font-variant-numeric: tabular-nums; vertical-align: middle; }
th.delta-col, td.delta-col { width: 1%; white-space: nowrap; }
td.rank-top { background: var(--rank-top-bg); box-shadow: inset 3px 0 0 var(--ok); font-weight: 700; }
td.rank-second { background: var(--rank-second-bg); box-shadow: inset 3px 0 0 var(--accent); }
td.rank-third { background: var(--rank-third-bg); box-shadow: inset 3px 0 0 var(--accent); }
.sort-button { width: 100%; border: 0; border-radius: 4px; padding: 0; background: transparent; color: inherit; font: inherit; font-weight: 800; text-align: inherit; text-transform: inherit; letter-spacing: inherit; cursor: pointer; }
.sort-button:hover, .sort-button.active { color: var(--accent); }
.sort-button .sort-indicator { margin-left: 4px; color: var(--accent); }
tr.selectable { cursor: pointer; }
tr.selectable:hover { background: var(--row-hover); }
tr.reference-row { background: var(--reference-row-bg); box-shadow: inset 3px 0 0 var(--accent); }
tr.reference-row:hover { background: var(--row-hover); }
.metric-delta { display: inline-block; font-weight: 700; line-height: 1.35; font-variant-numeric: tabular-nums; }
.delta-positive { color: var(--ok); }
.delta-negative { color: var(--danger); }
.delta-neutral { color: var(--muted); }
.kind-badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; border: 1px solid transparent; color: var(--pill-text); white-space: nowrap; }
.model-cell { display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap; max-width: min(40ch, 100%); }
button.model-pill {
  border: 1px solid transparent;
  border-radius: 999px;
  padding: 3px 9px;
  color: var(--pill-text);
  font: inherit;
  font-size: 11px;
  font-weight: 700;
  text-align: left;
  white-space: normal;
  overflow-wrap: anywhere;
  line-height: 1.35;
  cursor: pointer;
}
button.model-pill:hover { filter: brightness(0.96); border-color: var(--accent) !important; }
.role-badge { display: inline-block; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; padding: 2px 6px; border-radius: 999px; background: var(--accent-soft-bg); color: var(--accent); border: 1px solid var(--accent-soft-border); vertical-align: middle; }
.toolbar select { width: auto; min-width: 120px; }
.badge { display: inline-block; min-width: 74px; padding: 3px 7px; border-radius: 999px; font-size: 12px; font-weight: 700; text-align: center; background: var(--badge-neutral-bg); color: var(--ink); }
.badge.completed { background: var(--badge-completed-bg); color: var(--ok); }
.badge.failed { background: var(--badge-failed-bg); color: var(--danger); }
.badge.missing { background: var(--badge-missing-bg); color: var(--warn); }
.badge.running { background: var(--badge-running-bg); color: var(--badge-running-fg); }
.split-panel { display: grid; grid-template-columns: 1fr; gap: 18px; }
.split-panel > div { min-width: 0; }
.filter-grid { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 8px; margin-bottom: 10px; }
.figure-filters-bar {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.figure-filters-bar .zone-title { margin: 0; }
.figure-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 12px; align-items: start; }
.figure-item, .generated-plot { border: 1px solid var(--figure-card-border); border-radius: 6px; padding: 10px; display: grid; gap: 7px; min-width: 0; width: 100%; }
.chart-head { display: flex; align-items: start; justify-content: space-between; gap: 10px; }
.export-actions { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
.ghost-button, .asset-link { border: 1px solid var(--line); border-radius: 5px; padding: 5px 8px; background: var(--ghost-bg); color: var(--ink); font-size: 12px; font-weight: 700; text-decoration: none; }
.ghost-button:hover, .asset-link:hover { border-color: var(--accent); color: var(--accent); }
.theme-toggle { min-width: 72px; }
.scope-note { color: var(--muted); font-size: 12px; }
.decision-svg { width: 100%; height: auto; display: block; background: var(--plot-bg); border: 1px solid var(--plot-border); border-radius: 5px; }
.figure-section { grid-column: 1 / -1; display: grid; gap: 10px; }
.figure-section h3 { margin: 4px 0 0; font-size: 14px; letter-spacing: 0; }
.figure-section-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 12px; align-items: start; }
.shap-section { grid-column: 1 / -1; }
.shap-figure-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; align-items: start; }
.figure-preview { width: 100%; max-height: 620px; object-fit: contain; border: 1px solid var(--plot-border); border-radius: 5px; background: var(--plot-bg); }
.figure-preview-button { border: 0; padding: 0; background: transparent; cursor: zoom-in; display: block; width: 100%; }
.figure-preview-expandable { cursor: zoom-in; }
.figure-preview-button:hover .figure-preview { border-color: var(--accent); }
.figure-lightbox {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--lightbox-overlay);
}
.figure-lightbox[hidden] { display: none !important; }
.figure-lightbox-image {
  max-width: min(96vw, 100%);
  max-height: calc(100vh - 96px);
  object-fit: contain;
  border-radius: 6px;
  background: var(--plot-bg);
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35);
}
.figure-lightbox-caption {
  position: absolute;
  left: 50%;
  bottom: 18px;
  transform: translateX(-50%);
  max-width: min(920px, calc(100vw - 48px));
  margin: 0;
  padding: 8px 12px;
  border-radius: 6px;
  background: var(--lightbox-caption-bg);
  color: var(--ink);
  font-size: 12px;
  text-align: center;
}
.figure-lightbox-close {
  position: absolute;
  top: 16px;
  right: 16px;
  width: 36px;
  height: 36px;
  border: 0;
  border-radius: 999px;
  background: var(--lightbox-close-bg);
  color: var(--ink);
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
}
.figure-lightbox-close:hover { background: var(--surface-input); color: var(--accent); }
body.figure-lightbox-open { overflow: hidden; }
.shap-preview { max-height: 480px; height: auto; }
.generated-plot .figure-preview { max-height: 520px; }
.figure-meta { display: flex; gap: 6px; flex-wrap: wrap; color: var(--muted); font-size: 12px; }
.plot-row { display: grid; grid-template-columns: minmax(180px, .34fr) minmax(220px, 1fr) minmax(82px, max-content); gap: 10px; align-items: center; }
.plot-track { position: relative; height: 10px; border-radius: 999px; background: var(--plot-track-bg); overflow: hidden; }
.plot-zero { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: var(--muted); }
.plot-bar { position: absolute; top: 0; bottom: 0; background: var(--accent); }
.plot-bar.negative { background: var(--danger); }
.plot-dot { position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); transform: translate(-50%, -50%); }
.issue-panel { border-color: var(--issue-border); background: var(--issue-bg); }
.issue-list { display: grid; gap: 8px; }
.issue-item { border-left: 3px solid var(--warn); padding: 6px 0 6px 10px; }
.figure-item strong, .path { overflow-wrap: anywhere; }
.path { color: var(--muted); font-size: 12px; }
.gallery-note { border: 1px dashed var(--line); border-radius: 6px; padding: 9px; color: var(--muted); }
.control-grid { display: flex; gap: 10px; flex-wrap: wrap; }
.toast { position: fixed; right: 16px; bottom: 16px; max-width: min(420px, calc(100vw - 32px)); padding: 12px 14px; border-radius: 8px; background: var(--toast-bg); color: #ffffff; opacity: 0; transform: translateY(8px); transition: .16s ease; pointer-events: none; }
.toast.show { opacity: 1; transform: translateY(0); }
@media (max-width: 1180px) {
  .decision-plots { grid-template-columns: 1fr; }
  .filter-grid { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
}
@media (max-width: 880px) {
  .topbar { align-items: flex-start; flex-direction: column; padding-top: 10px; padding-bottom: 10px; }
  .toolbar { justify-content: flex-start; width: 100%; }
  .stat-grid { grid-template-columns: 1fr 1fr; }
  .filter-grid { grid-template-columns: 1fr; }
  .plot-row { grid-template-columns: 1fr; gap: 4px; }
  table { min-width: 640px; }
  .shap-figure-grid { grid-template-columns: 1fr; }
}
"""

_SCRIPT_JS: Final[str] = """const FIGURE_RENDER_LIMIT = 36;
const SHAP_RENDER_LIMIT = 8;
const TEST_SCOPE = "test";
const VALIDATION_SCOPE = "validation";
const COMBINED_SCOPE = "combined";
const MODEL_COMPARISON_ROLES = new Set(["performance", "cv_mean_std", "cv_heatmap", "cv_fold_comparison", "per_target_validation", "optuna"]);
const SELECTED_MODEL_ROLES = new Set(["shap", "shap_beeswarm", "shap_importance", "shap_dependence", "architecture"]);
const UI_STATE_KEY = "ocscore-workbench-ui";
const MODEL_CATEGORY_COLORS = {
  full_ocscore: "#74c476",
  ablation: "#9ecae1",
  sf: "#f4b183",
  consensus: "#c9b1d4",
};
const MODEL_CATEGORY_LABELS = {
  full_ocscore: "full_ocscore",
  ablation: "Ablation",
  sf: "SF",
  consensus: "Other consensus",
};
const RANK_BAR_COLORS = MODEL_CATEGORY_COLORS;
const RANK_BAR_LABELS = MODEL_CATEGORY_LABELS;

const state = {
  workspace: null,
  selectedStudy: null,
  selectedMetric: null,
  activeTab: "ablation",
  resultScope: TEST_SCOPE,
  figureFilters: { dataset: "all", role: "all", metric: "all", group: "comparison" },
  plotExports: {},
  pendingPlotly: [],
  zoneCollapsed: {
    protocol: false,
    comparisonTable: false,
    comparisonCharts: false,
    cvTable: false,
    detailReplicas: false,
    detailCharts: false,
    detailFigures: false,
  },
  comparisonSort: { key: "delta", direction: "desc" },
  detailReplicaSort: { key: "replica", direction: "asc" },
  comparisonBaseline: "internal",
  theme: "dark",
  _persistedSelectedStudyName: null,
};
let uiStateHydrated = false;
const $ = (id) => document.getElementById(id);

function loadPersistedUiState() {
  try {
    const raw = sessionStorage.getItem(UI_STATE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.zoneCollapsed && typeof saved.zoneCollapsed === "object") {
      state.zoneCollapsed = { ...state.zoneCollapsed, ...saved.zoneCollapsed };
    }
    if (saved.resultScope) state.resultScope = saved.resultScope;
    if (saved.comparisonBaseline) state.comparisonBaseline = saved.comparisonBaseline;
    if (saved.selectedMetric) state.selectedMetric = saved.selectedMetric;
    if (saved.comparisonSort) state.comparisonSort = saved.comparisonSort;
    if (saved.detailReplicaSort) state.detailReplicaSort = saved.detailReplicaSort;
    if (saved.figureFilters) {
      state.figureFilters = { ...state.figureFilters, ...saved.figureFilters };
      if (state.figureFilters.role === "recommended") state.figureFilters.role = "all";
    }
    if (saved.theme === "light" || saved.theme === "dark") state.theme = saved.theme;
    state._persistedSelectedStudyName = saved.selectedStudyName || null;
  } catch (_) {
    /* ignore corrupt session state */
  }
}

function persistUiState() {
  if (!uiStateHydrated) return;
  try {
    sessionStorage.setItem(UI_STATE_KEY, JSON.stringify({
      zoneCollapsed: state.zoneCollapsed,
      selectedStudyName: state.selectedStudy?.study_name || null,
      resultScope: state.resultScope,
      comparisonBaseline: state.comparisonBaseline,
      selectedMetric: state.selectedMetric,
      comparisonSort: state.comparisonSort,
      detailReplicaSort: state.detailReplicaSort,
      figureFilters: state.figureFilters,
      theme: state.theme,
    }));
  } catch (_) {
    /* ignore quota errors */
  }
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  state.theme = normalized;
  if (document.documentElement) {
    document.documentElement.setAttribute("data-theme", normalized);
  }
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.textContent = normalized === "dark" ? "Light" : "Dark";
    toggle.setAttribute("aria-pressed", normalized === "dark" ? "true" : "false");
    toggle.title = normalized === "dark" ? "Switch to light appearance" : "Switch to dark appearance";
  }
}

function bindThemeToggle() {
  applyTheme(state.theme);
  const toggle = $("theme-toggle");
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
    persistUiState();
  });
}

function restoreSelectedStudyFromPersisted() {
  const name = state._persistedSelectedStudyName;
  if (!name || state.selectedStudy) return;
  const study = allStudies().find((item) => item.study_name === name);
  if (study) state.selectedStudy = study;
}

function setActiveTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll(".app-tabs .tab").forEach((button) => {
    const active = button.dataset.tab === tabId;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
    const active = panel.dataset.tabPanel === tabId;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  document.querySelectorAll("[data-tab-toolbar]").forEach((toolbar) => {
    toolbar.hidden = toolbar.dataset.tabToolbar !== tabId;
  });
}

function bindAppTabs() {
  document.querySelectorAll(".app-tabs .tab").forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button.dataset.tab));
  });
}

function setZoneCollapsed(zoneId, collapsed) {
  state.zoneCollapsed[zoneId] = collapsed;
  const zone = document.querySelector(`[data-zone="${zoneId}"]`);
  if (!zone) return;
  zone.classList.toggle("is-collapsed", collapsed);
  const toggle = zone.querySelector(".zone-toggle");
  if (toggle) toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  persistUiState();
  if (!collapsed && (zoneId === "comparisonCharts" || zoneId === "detailCharts")) {
    requestAnimationFrame(() => resizePlotlyHosts(zone));
  }
}

function bindCollapsibleZones() {
  document.querySelectorAll(".zone-collapsible[data-zone]").forEach((zone) => {
    const zoneId = zone.dataset.zone;
    if (!zoneId) return;
    setZoneCollapsed(zoneId, Boolean(state.zoneCollapsed[zoneId]));
    const toggle = zone.querySelector(".zone-toggle");
    if (!toggle || toggle.dataset.bound === "true") return;
    toggle.dataset.bound = "true";
    toggle.addEventListener("click", () => {
      setZoneCollapsed(zoneId, !state.zoneCollapsed[zoneId]);
    });
  });
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  window.setTimeout(() => node.classList.remove("show"), 3200);
}

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    if (Array.isArray(value)) value.forEach((item) => url.searchParams.append(key, item));
    else url.searchParams.set(key, value);
  });
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

async function apiPost(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function escapeHtml(value) {
  const replacements = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value ?? "").replace(/[&<>"']/g, (char) => replacements[char]);
}

function numeric(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 5 });
}

function titleCase(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(/\\b\\w/g, (char) => char.toUpperCase());
}

function metricSummary(summary) {
  const names = Object.keys(summary || {});
  if (!names.length) return "-";
  return names.map((name) => `${summary[name].label || name}: ${numeric(summary[name].mean)}`).join(" | ");
}

function statusBadge(status) {
  return `<span class="badge ${status}">${status}</span>`;
}

function table(target, headers, rows, rowClass = () => "", sortState = state.comparisonSort) {
  if (!rows.length) {
    target.innerHTML = '<div class="path">No records detected.</div>';
    return;
  }
  const normalizedHeaders = headers.map((item) => (typeof item === "string" ? { label: item } : item));
  const head = normalizedHeaders.map((item) => {
    const classes = [item.numeric ? "numeric" : "", item.headerClass || ""].filter(Boolean).join(" ");
    const classAttr = classes ? ` class="${classes}"` : "";
    const labelContent = item.labelHtml ?? escapeHtml(String(item.label ?? ""));
    if (!item.sortKey) return `<th${classAttr}>${labelContent}</th>`;
    const active = sortState.key === item.sortKey;
    const indicator = active ? `<span class="sort-indicator">${sortState.direction === "asc" ? "asc" : "desc"}</span>` : "";
    return `<th${classAttr}><button class="sort-button${active ? " active" : ""}" type="button" data-sort-key="${item.sortKey}" data-sort-default="${item.defaultDirection || "asc"}">${labelContent}${indicator}</button></th>`;
  }).join("");
  const body = rows.map((row, index) => {
    const cells = row.map((item) => {
      if (item && typeof item === "object" && "value" in item) {
        const classes = [item.numeric ? "numeric" : "", item.className || ""].filter(Boolean).join(" ");
        const title = item.title ? ` title="${escapeHtml(item.title)}"` : "";
        return `<td${classes ? ` class="${classes}"` : ""}${title}>${item.value}</td>`;
      }
      return `<td>${item}</td>`;
    }).join("");
    return `<tr class="${rowClass(row, index)}">${cells}</tr>`;
  }).join("");
  target.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function compareValues(left, right, direction) {
  const leftMissing = left === null || left === undefined || left === "";
  const rightMissing = right === null || right === undefined || right === "";
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (typeof left === "number" && typeof right === "number") {
    return direction === "asc" ? left - right : right - left;
  }
  return direction === "asc" ? String(left).localeCompare(String(right)) : String(right).localeCompare(String(left));
}

function isFullModelStudy(study) {
  return !!(study && (study.study_name === "baseline" || study.policy_name === "baseline"));
}

function externalEntryId(baseline) {
  return `${baseline.baseline_name}::${baseline.split || "test"}`;
}

function findExternalBaselineByEntryId(entryId) {
  return scopedExternalBaselines().find((item) => externalEntryId(item) === entryId) || null;
}

function externalDisplayName(baseline) {
  const name = baseline.baseline_name || "";
  if (state.resultScope === COMBINED_SCOPE) {
    return `${name} (${titleCase(baseline.split || "test")})`;
  }
  return name;
}

function normalizeComparisonBaseline() {
  if (state.comparisonBaseline === "internal") return;
  const baselines = scopedExternalBaselines();
  if (baselines.some((item) => externalEntryId(item) === state.comparisonBaseline)) return;
  const legacyMatches = baselines.filter((item) => item.baseline_name === state.comparisonBaseline);
  if (legacyMatches.length === 1) {
    state.comparisonBaseline = externalEntryId(legacyMatches[0]);
    return;
  }
  if (legacyMatches.length > 1) {
    const preferred = legacyMatches.find((item) => item.split === resultSplit()) || legacyMatches[0];
    state.comparisonBaseline = externalEntryId(preferred);
    return;
  }
  state.comparisonBaseline = "internal";
}

function modelDisplayName(item) {
  if (item.isFullModel) return "full_ocscore";
  if (item.external) return externalDisplayName(item.entry || item.study);
  return item.id;
}

function studyDisplayName(study) {
  if (isFullModelStudy(study)) return "full_ocscore";
  return study.study_name || study.baseline_name || "";
}

const ABLATION_DESCRIPTIONS = {
  full_ocscore: "Full OCScore feature set after standard metadata/target exclusion.",
  no_pmi: "Remove direct PMI descriptors to test direct PMI dependence.",
  no_shape_core: "Remove core ligand 3D shape descriptors.",
  no_ligand_shape_size: "Remove ligand shape descriptors and common ligand size/topology proxies.",
  shape_only: "Use only the core ligand 3D shape descriptors.",
  scoring_function_only: "Use only classical docking/scoring-function columns.",
  ligand_plus_scoring_function: "Ligand descriptors plus classical scoring-function columns, excluding receptor descriptors.",
  ligand_plus_scoring_function_no_shape_core: "Ligand descriptors plus scoring functions, excluding core ligand 3D shape descriptors.",
  ligand_plus_scoring_function_no_shape_size: "Ligand descriptors plus scoring functions, excluding ligand shape/size/topology proxies.",
  ligand_plus_scoring_function_no_shape_size_no_autocorr2d: "Ligand + scoring functions without shape/size descriptors and without AUTOCORR2D.",
  ligand_plus_scoring_function_no_pmi: "Ligand descriptors and scoring functions, excluding principal moments of inertia.",
  ligand_plus_scoring_function_no_plants: "Ligand descriptors and scoring functions, excluding PLANTS-derived scores.",
  ligand_plus_scoring_function_clean_receptor: "Ligand + scoring functions with receptor descriptors, excluding amino-acid count and length proxies.",
  no_scoring_function: "Remove classical docking/scoring-function columns.",
  ligand_only: "Use only ligand descriptor columns.",
  receptor_plus_scoring_function: "Receptor descriptors plus classical scoring-function columns, excluding ligand descriptors.",
  no_shape_core_no_receptor_length_pair: "Remove core ligand 3D shape descriptors plus receptor whole-chain length descriptors.",
  no_shape_core_no_receptor_surface_counts: "Remove core ligand 3D shape plus receptor surface amino-acid composition counts.",
  no_shape_core_no_receptor_surface_size: "Remove core ligand 3D shape plus receptor surface residue count and SASA size proxies.",
};

const CONSENSUS_DESCRIPTIONS = {
  sf_mean: "Row-wise mean across all docking-score columns, used directly as a ranker.",
  sf_median: "Row-wise median across all docking-score columns, used directly as a ranker.",
  sf_max: "Row-wise max across all docking-score columns, used directly as a ranker.",
  sf_min: "Row-wise min across all docking-score columns, used directly as a ranker.",
  desc_mean: "Row-wise mean across selected model input features, used directly as a ranker.",
  desc_median: "Row-wise median across selected model input features, used directly as a ranker.",
  desc_max: "Row-wise max across selected model input features, used directly as a ranker.",
  desc_min: "Row-wise min across selected model input features, used directly as a ranker.",
};

const LEARNED_SF_DESCRIPTIONS = {
  lr_sf: "Logistic regression trained on docking-score columns only (train-only fit, same splits as OCScore).",
  rf_sf: "Random forest trained on docking-score columns only.",
  xgb_sf: "XGBoost trained on docking-score columns only.",
  lgbm_sf: "LightGBM trained on docking-score columns only.",
  shuffled_lr_sf: "Negative control: logistic regression with shuffled training labels.",
};

const SF_ENGINE_LABELS = {
  vina: "AutoDock Vina",
  gnina: "GNINA",
  smina: "Smina",
  plants: "PLANTS",
  oddt: "ODDT",
};

function aggregateDescription(prefix, name) {
  const agg = name.slice(prefix.length);
  const labels = { mean: "mean", median: "median", max: "max", min: "min" };
  if (!labels[agg]) return null;
  const scope = prefix === "sf_" ? "docking-score columns" : "selected model input features";
  return `Row-wise ${labels[agg]} across ${scope}, used directly as a ranker.`;
}

function scoringFunctionDescription(name) {
  const parts = String(name || "").split("_");
  const engine = parts[0] || "";
  const term = parts.slice(1).join("_").replaceAll("_", " ") || "score";
  const engineLabel = SF_ENGINE_LABELS[engine] || titleCase(engine);
  return `${engineLabel} ${titleCase(term)} — raw docking score used as a ranker (no ML).`;
}

function modelDescription(item) {
  if (item.isFullModel) {
    return `${ABLATION_DESCRIPTIONS.full_ocscore} OCScore reference run (train/replica_*).`;
  }
  if (!item.external) {
    const key = item.study?.policy_name || item.id;
    return ABLATION_DESCRIPTIONS[key] || ABLATION_DESCRIPTIONS[item.id] || `Feature-policy ablation (${key}).`;
  }
  const name = item.entry?.baseline_name || item.id.split("::")[0] || item.id;
  const family = item.entry?.baseline_family || "";
  if (CONSENSUS_DESCRIPTIONS[name]) return CONSENSUS_DESCRIPTIONS[name];
  if (LEARNED_SF_DESCRIPTIONS[name]) return LEARNED_SF_DESCRIPTIONS[name];
  if (name.startsWith("sf_")) return aggregateDescription("sf_", name) || `SF consensus baseline (${name}).`;
  if (name.startsWith("desc_")) return aggregateDescription("desc_", name) || `Descriptor aggregate baseline (${name}).`;
  if (family === "scoring_function") return scoringFunctionDescription(name);
  if (/^(vina|gnina|smina|plants|oddt)_/.test(name)) return scoringFunctionDescription(name);
  if (family === "sf_consensus") {
    const base = aggregateDescription("sf_", name) || `Consensus across scoring functions (${name}).`;
    return item.synthesized || item.entry?.synthesized
      ? `${base} Approximate aggregate synthesized from individual SF rows when absent from baseline CSV.`
      : base;
  }
  if (family === "descriptor_aggregate") return aggregateDescription("desc_", name) || `Aggregate over model input features (${name}).`;
  if (family === "learned_sf") return `Learned classifier on docking-score columns (${name}).`;
  return `External baseline (${externalBaselineFamilyLabel(family)}).`;
}

function comparisonEntries() {
  const entries = [];
  const baseline = state.workspace?.baseline_study;
  if (baseline) {
    entries.push({
      entry: baseline,
      kind: "Full model",
      kindClass: "full-model",
      id: baseline.study_name,
      study: baseline,
      external: false,
      isFullModel: true,
    });
  }
  (state.workspace?.ablation_studies || []).forEach((study) => {
    entries.push({ entry: study, kind: "Ablation", kindClass: "ablation", id: study.study_name, study, external: false, isFullModel: false });
  });
  scopedExternalBaselines().forEach((item) => {
    entries.push({
      entry: item,
      kind: externalBaselineKindLabel(item.baseline_family),
      kindClass: "external",
      id: externalEntryId(item),
      study: item,
      external: true,
      isFullModel: false,
      synthesized: Boolean(item.synthesized),
      baseline_family: item.baseline_family,
    });
  });
  return entries;
}

function isReferenceEntry(item) {
  if (state.comparisonBaseline === "internal") {
    return item.isFullModel === true;
  }
  return item.id === state.comparisonBaseline;
}

function entryLegendCategory(item) {
  const family = item.baseline_family || item.entry?.baseline_family || "";
  if (family === "scoring_function") return "sf";
  return "consensus";
}

function entryModelCategory(item) {
  if (item.external) {
    return entryLegendCategory(item) === "sf" ? "sf" : "consensus";
  }
  if (item.isFullModel) return "full_ocscore";
  return "ablation";
}

function entryPaletteColor(item) {
  return MODEL_CATEGORY_COLORS[entryModelCategory(item)] || MODEL_CATEGORY_COLORS.ablation;
}

function entryPaletteStyle(item) {
  const fill = entryPaletteColor(item);
  return `background:${fill};border-color:${fill};color:#202833`;
}

function modelCell(item) {
  const label = modelDisplayName(item);
  const refBadge = isReferenceEntry(item) ? '<span class="role-badge reference">Reference</span>' : "";
  const tip = escapeHtml(modelDescription(item));
  const style = entryPaletteStyle(item);
  return `<span class="model-cell" title="${tip}"><button type="button" class="model-pill" style="${style}" data-entry-id="${escapeHtml(item.id)}" title="${tip}">${escapeHtml(label)}</button>${refBadge}</span>`;
}

function entryMetricStd(entry, metricName) {
  const metric = metricSummaryLookup(entry?.metric_summary, metricName);
  if (!metric || Number(metric.count) <= 1 || !Number.isFinite(Number(metric.std))) return null;
  return Number(metric.std);
}

function metricDecisionDelta(entry, metricName) {
  const reference = comparisonReferenceSummary();
  const value = metricValue(entry, metricName);
  const referenceMetric = metricSummaryLookup(reference, metricName);
  if (value === null || !referenceMetric || !Number.isFinite(Number(referenceMetric.mean))) return null;
  let delta = value - referenceMetric.mean;
  if (metricMeta(metricName).direction === "min") delta = -delta;
  return delta;
}

function comparisonMetricCell(item, metricName, rankLookup, total) {
  const summary = item.entry?.metric_summary || null;
  const value = metricValue(item.entry, metricName);
  const rank = rankLookup ? rankLookup.get(item) : null;
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { value: "-", numeric: true };
  }
  const titleParts = [];
  if (rank && total) titleParts.push(`Rank ${rank} of ${total}`);
  const metric = metricSummaryLookup(summary, metricName);
  if (metric) titleParts.push(metricStatTitle(metric, item));
  const title = titleParts.length ? titleParts.join(" · ") : undefined;
  const markup = metricStatMarkup(summary, metricName, item);
  return {
    value: markup || numeric(value),
    numeric: true,
    className: rankHighlightClass(rank),
    title,
  };
}

function comparisonSortValue(item, key, metricName) {
  if (key === "model") return modelDisplayName(item) || "";
  if (key === "kind") return item.kind || "";
  if (key === "replicas") return item.study?.detected_replica_count || 0;
  if (key === "delta") return metricDecisionDelta(item.entry, metricName);
  if (key.startsWith("metric:")) return metricValue(item.entry, key.slice(7));
  return "";
}

function sortedComparisonEntries(entries, metricName) {
  const sort = state.comparisonSort;
  return [...entries].sort((left, right) => {
    const leftRef = isReferenceEntry(left);
    const rightRef = isReferenceEntry(right);
    if (leftRef !== rightRef) return leftRef ? -1 : 1;
    const compared = compareValues(
      comparisonSortValue(left, sort.key, metricName),
      comparisonSortValue(right, sort.key, metricName),
      sort.direction,
    );
    return compared || String(left.id || "").localeCompare(String(right.id || ""));
  });
}

function kindBadge(item) {
  const tip = escapeHtml(modelDescription(item));
  const synthClass = item.synthesized ? " synthesized-baseline" : "";
  const label = item.synthesized ? `${item.kind} · approx` : item.kind;
  const style = item.synthesized ? "" : ` style="${entryPaletteStyle(item)}"`;
  return `<span class="kind-badge${synthClass}"${style} title="${tip}">${escapeHtml(label)}</span>`;
}

function comparisonRowClass(item) {
  const classes = ["selectable"];
  if (isReferenceEntry(item)) classes.push("reference-row");
  return classes.join(" ");
}

function bindSortButtons(target, sortStateKey = "comparisonSort") {
  target.querySelectorAll("button[data-sort-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sortKey;
      const defaultDirection = button.dataset.sortDefault || "asc";
      const current = state[sortStateKey];
      state[sortStateKey] = {
        key,
        direction: current.key === key && current.direction === defaultDirection ? (defaultDirection === "asc" ? "desc" : "asc") : defaultDirection,
      };
      renderWorkspace(state.workspace);
    });
  });
}

function metricColumns(payload, studies) {
  const fromStudies = payload.metric_names && payload.metric_names.length
    ? [...payload.metric_names]
    : Array.from(new Set(studies.flatMap((study) => Object.keys(study.metric_summary || {}))));
  const fromExternals = Array.from(new Set(
    (payload.external_baselines || []).flatMap((baseline) => Object.keys(baseline.metric_summary || {})),
  ));
  const names = Array.from(new Set([...fromStudies, ...fromExternals]));
  return names.map((name) => {
    const ownerStudy = studies.find((study) => study.metric_summary && study.metric_summary[name]);
    const ownerExternal = (payload.external_baselines || []).find((baseline) => baseline.metric_summary && baseline.metric_summary[name]);
    const summary = ownerStudy?.metric_summary?.[name] || ownerExternal?.metric_summary?.[name] || {};
    return {
      name,
      label: summary.label || name.replaceAll("_", " ").toUpperCase(),
      direction: summary.direction || "max",
    };
  });
}

function buildMetricRankLookup(items, valueFn, direction = "max") {
  const entries = items
    .map((item) => ({ item, value: valueFn(item) }))
    .filter((entry) => entry.value !== null && Number.isFinite(entry.value));
  entries.sort((left, right) => (direction === "min" ? left.value - right.value : right.value - left.value));
  const lookup = new Map();
  let rank = 0;
  let lastValue = null;
  entries.forEach((entry, index) => {
    if (lastValue === null || entry.value !== lastValue) {
      rank = index + 1;
      lastValue = entry.value;
    }
    lookup.set(entry.item, rank);
  });
  return lookup;
}

function rankHighlightClass(rank) {
  if (rank === 1) return "rank-top";
  if (rank === 2) return "rank-second";
  if (rank === 3) return "rank-third";
  return "";
}

function rankedMetricCell(value, rank, total) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return { value: "-", numeric: true };
  }
  const title = rank && total ? `Rank ${rank} of ${total}` : undefined;
  return {
    value: numeric(value),
    numeric: true,
    className: rankHighlightClass(rank),
    title,
  };
}

function replicaMetricCell(replica, name, rankLookup, total) {
  const value = replicaMetricValue(replica, name);
  const rank = rankLookup ? rankLookup.get(replica) : null;
  return rankedMetricCell(value, rank, total);
}

function externalMetricCell(baseline, name, rankLookup, total) {
  const value = metricValue(baseline, name);
  const rank = rankLookup ? rankLookup.get(baseline) : null;
  return rankedMetricCell(value, rank, total);
}

function allStudies() {
  if (!state.workspace) return [];
  return [state.workspace.baseline_study, ...(state.workspace.ablation_studies || [])];
}

function resultSplit(scope = state.resultScope) {
  if (scope === VALIDATION_SCOPE) return "validation";
  return "test";
}

function scopedExternalBaselines(scope = state.resultScope) {
  const baselines = state.workspace?.external_baselines || [];
  if (scope === COMBINED_SCOPE) return baselines;
  const split = resultSplit(scope);
  return baselines.filter((item) => item.split === split);
}

function externalBaselineFamilyLabel(family) {
  const labels = {
    sf_consensus: "SF consensus",
    descriptor_aggregate: "Descriptor aggregate",
    scoring_function: "Scoring function",
    learned_sf: "Learned SF",
  };
  return labels[family] || titleCase(family);
}

function externalBaselineKindLabel(family) {
  if (family === "scoring_function") return "SF";
  return "Other consensus";
}

function externalBaselineLabel(baseline) {
  return `${baseline.baseline_name} (${externalBaselineFamilyLabel(baseline.baseline_family)})`;
}

function comparisonReferenceSummary() {
  if (state.comparisonBaseline === "internal") {
    return state.workspace?.baseline_study?.metric_summary || {};
  }
  const external = findExternalBaselineByEntryId(state.comparisonBaseline);
  return external?.metric_summary || {};
}

function comparisonReferenceLabel() {
  if (state.comparisonBaseline === "internal") return "full_ocscore";
  const external = findExternalBaselineByEntryId(state.comparisonBaseline);
  return external ? externalDisplayName(external) : state.comparisonBaseline;
}

function rankComparisonEntries() {
  const studies = allStudies().map((study) => ({
    study_name: study.study_name,
    policy_name: study.policy_name,
    metric_summary: study.metric_summary || {},
    external: false,
    baseline_family: null,
  }));
  const externals = scopedExternalBaselines().map((baseline) => ({
    study_name: baseline.baseline_name,
    metric_summary: baseline.metric_summary || {},
    external: true,
    baseline_family: baseline.baseline_family || "",
    synthesized: Boolean(baseline.synthesized),
  }));
  return [...studies, ...externals];
}

function externalSortValue(baseline, key) {
  if (key === "baseline") return baseline.baseline_name || "";
  if (key === "family") return baseline.baseline_family || "";
  if (key === "split") return baseline.split || "";
  if (key.startsWith("metric:")) {
    const metricName = key.slice(7);
    return baseline.metric_summary && baseline.metric_summary[metricName] ? baseline.metric_summary[metricName].mean : null;
  }
  return "";
}

function sortExternalBaselines(baselines) {
  const sort = state.externalSort;
  return [...baselines].sort((left, right) => {
    const compared = compareValues(externalSortValue(left, sort.key), externalSortValue(right, sort.key), sort.direction);
    return compared || String(left.baseline_name || "").localeCompare(String(right.baseline_name || ""));
  });
}

function availableMetrics() {
  if (!state.workspace) return [];
  return metricColumns(state.workspace, allStudies());
}

function metricScope(name) {
  const normalized = String(name || "").toLowerCase();
  if (normalized.startsWith("test_")) return TEST_SCOPE;
  if (normalized.startsWith("validation_")) return VALIDATION_SCOPE;
  if (normalized.includes("validation") || normalized === "validation_metric") return VALIDATION_SCOPE;
  if (["rmse", "mae", "r2"].includes(normalized)) return VALIDATION_SCOPE;
  return TEST_SCOPE;
}

function metricSummaryLookup(summary, metricName) {
  if (!summary) return null;
  if (summary[metricName]) return summary[metricName];
  const base = String(metricName || "").replace(/^(test|validation)_/, "");
  return summary[base] || null;
}

function metricAllowedByScope(name) {
  return state.resultScope === COMBINED_SCOPE || metricScope(name) === state.resultScope;
}

function scopedMetrics() {
  return availableMetrics().filter((metric) => metricAllowedByScope(metric.name));
}

function scopeLabel(scope = state.resultScope) {
  if (scope === TEST_SCOPE) return "Test";
  if (scope === VALIDATION_SCOPE) return "Validation";
  return "Combined";
}

function metricMeta(name) {
  return availableMetrics().find((metric) => metric.name === name) || { name, label: titleCase(name), direction: "max" };
}

function ensureSelectedMetric() {
  const metrics = scopedMetrics();
  if (!metrics.length) {
    state.selectedMetric = null;
    return null;
  }
  if (!state.selectedMetric || !metrics.some((metric) => metric.name === state.selectedMetric)) {
    state.selectedMetric = metrics[0].name;
  }
  return state.selectedMetric;
}

function isReplicaAggregateCell(item, metric) {
  if (!item || item.external) return false;
  return Number(metric?.count) > 1;
}

function metricStatTitle(metric, item) {
  if (!metric) return "";
  if (item?.external) return "Precomputed external baseline value";
  const count = Number(metric.count) || 0;
  const parts = [];
  if (count > 1) parts.push(`Mean (μ) of ${count} replicas`);
  else if (count === 1) parts.push("Single replica value");
  else parts.push("Value");
  const std = Number(metric.std);
  if (count > 1 && Number.isFinite(std)) parts.push(`Std dev (σ) ${numeric(std)}`);
  return parts.join(" · ");
}

function metricStatMarkup(summary, metricName, item) {
  const metric = metricSummaryLookup(summary, metricName);
  if (!metric || !Number.isFinite(Number(metric.mean))) return null;
  const valueText = numeric(metric.mean);
  if (!isReplicaAggregateCell(item, metric)) {
    const hint = item?.external ? "External baseline value" : Number(metric.count) === 1 ? "Single replica value" : "Value";
    return `<span class="metric-stat metric-stat-value" title="${escapeHtml(hint)}">${valueText}</span>`;
  }
  const std = Number(metric.std);
  const stdMarkup = Number.isFinite(std)
    ? `<span class="metric-std" title="Standard deviation (σ) across ${metric.count} replicas">σ ${numeric(std)}</span>`
    : "";
  return `<span class="metric-stat metric-stat-aggregate" title="${escapeHtml(metricStatTitle(metric, item))}"><span class="metric-stat-line"><span class="metric-cell-mu">μ</span><span class="metric-mean">${valueText}</span></span>${stdMarkup}</span>`;
}

function plotMetricLabel(metric) {
  return metric.label || metric.name;
}

function plotMetricValueDisplay(entry, metricName) {
  const summary = entry?.metric_summary || null;
  const metric = metricSummaryLookup(summary, metricName);
  if (!metric || !Number.isFinite(Number(metric.mean))) return null;
  const mean = Number(metric.mean);
  const count = Number(metric.count) || 0;
  const item = { external: Boolean(entry.external) };
  const aggregate = isReplicaAggregateCell(item, metric);
  const std = aggregate && Number.isFinite(Number(metric.std)) ? Number(metric.std) : null;
  if (aggregate) {
    const meanText = numeric(mean);
    const barLabel = std !== null ? `μ ${meanText}<br>σ ${numeric(std)}` : `μ ${meanText}`;
    const hoverLabel = std !== null ? `μ ${meanText}<br>σ ${numeric(std)}` : `μ ${meanText}`;
    const display = std !== null ? `μ ${meanText} σ ${numeric(std)}` : `μ ${meanText}`;
    return { value: mean, std, count, display, barLabel, hoverLabel };
  }
  const plain = numeric(mean);
  return { value: mean, std: null, count, display: plain, barLabel: plain, hoverLabel: plain };
}

function rankPlotLabelOffset(rows) {
  const extents = rows.map((row) => row.value + (Number(row.std) || 0));
  const maxExtent = Math.max(...extents);
  const minExtent = Math.min(0, ...rows.map((row) => row.value));
  return Math.max((maxExtent - minExtent) * 0.012, maxExtent * 0.004, 1e-4);
}

function rankBarCategory(row) {
  if (!row.external) {
    return isFullModelStudy(row) ? "full_ocscore" : "ablation";
  }
  if (String(row.baseline_family || "") === "scoring_function") return "sf";
  return "consensus";
}

function rankBarFillColor(row) {
  return MODEL_CATEGORY_COLORS[rankBarCategory(row)] || MODEL_CATEGORY_COLORS.ablation;
}

function rankBarHoverKind(row) {
  if (!row.external) return row.count > 1 ? `${row.count} replicas` : "Single value";
  return RANK_BAR_LABELS[rankBarCategory(row)];
}

function buildRankPlotlySpec(rows, metric) {
  const title = `${plotMetricLabel(metric)} rank across studies`;
  const subtitle = "Error bars = σ across replicas";
  const labelOffset = rankPlotLabelOffset(rows);
  const maxLabelLen = rows.reduce((longest, row) => {
    const lines = String(row.barLabel || row.display || "").split("<br>");
    return Math.max(longest, ...lines.map((line) => line.length));
  }, 8);
  const legendCategories = ["full_ocscore", "ablation", "sf", "consensus"].filter((category) => rows.some((row) => rankBarCategory(row) === category));
  const mainTrace = {
    type: "bar",
    orientation: "h",
    y: rows.map((row) => row.label),
    x: rows.map((row) => row.value),
    customdata: rows.map((row) => [row.std, row.count, row.hoverLabel || row.display, rankBarHoverKind(row)]),
    error_x: {
      type: "data",
      array: rows.map((row) => row.std),
      color: "#8899a6",
      thickness: 1.2,
      width: 5,
    },
    marker: {
      color: rows.map((row) => rankBarFillColor(row)),
      line: { color: "#ffffff", width: 1 },
    },
    showlegend: false,
    hovertemplate: "<b>%{y}</b><br>%{customdata[2]}<br><span style='color:#667085'>%{customdata[3]}</span><extra></extra>",
  };
  const legendTraces = legendCategories.map((category) => ({
    type: "bar",
    orientation: "h",
    x: [null],
    y: [null],
    name: RANK_BAR_LABELS[category],
    marker: { color: RANK_BAR_COLORS[category] },
    showlegend: true,
    hoverinfo: "skip",
  }));
  return {
    data: [mainTrace, ...legendTraces],
    layout: {
      template: "plotly_white",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      autosize: true,
      margin: { l: 16, r: Math.max(88, maxLabelLen * 7), t: 56, b: 44 },
      annotations: rows.map((row) => ({
        x: row.value + (Number(row.std) || 0) + labelOffset,
        y: row.label,
        text: row.barLabel || row.display,
        showarrow: false,
        xanchor: "left",
        yanchor: "middle",
        align: "left",
        font: { size: 11, color: "#202833", family: "system-ui, sans-serif" },
        xref: "x",
        yref: "y",
      })),
      legend: {
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "left",
        x: 0,
        font: { color: "#667085", size: 12 },
      },
      xaxis: {
        title: plotMetricLabel(metric),
        titlefont: { color: "#667085" },
        tickfont: { color: "#667085" },
        gridcolor: "#ebe5dc",
        zerolinecolor: "#d8d1c5",
        rangemode: metric.direction !== "min" ? "tozero" : "normal",
        zeroline: true,
      },
      yaxis: {
        automargin: true,
        autorange: "reversed",
        tickfont: { color: "#202833", size: 12 },
      },
      height: Math.max(280, rows.length * 40 + 120),
    },
    config: {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: slug(title), scale: 2 },
    },
  };
}

async function mountPendingPlotlyCharts() {
  const queue = state.pendingPlotly;
  state.pendingPlotly = [];
  for (const item of queue) {
    const host = document.getElementById(item.divId);
    if (!host) continue;
    const payload = state.plotExports[item.key];
    if (!payload) continue;
    if (!window.Plotly) {
      host.innerHTML = '<span class="path">Plotly failed to load. Check your network connection and refresh.</span>';
      continue;
    }
    if (host.data) Plotly.purge(host);
    await Plotly.newPlot(host, item.spec.data, item.spec.layout, item.spec.config);
    payload.plotlyDivId = item.divId;
  }
  requestAnimationFrame(() => {
    resizePlotlyHosts();
    requestAnimationFrame(resizePlotlyHosts);
  });
}

function resizePlotlyHosts(root = document) {
  if (!window.Plotly) return;
  const scope = root instanceof Element ? root : document;
  scope.querySelectorAll(".plotly-host").forEach((host) => {
    if (!host.data) return;
    try {
      Plotly.Plots.resize(host);
    } catch (_error) {
      /* ignore resize races while charts mount */
    }
  });
}

function buildSimpleBarPlotlySpec(rows, metric, options = {}) {
  const title = options.title || plotMetricLabel(metric);
  const subtitle = options.subtitle || "";
  const zeroCentered = Boolean(options.zeroCentered);
  const colorForRow = options.colorForRow || (() => "#9ecae1");
  const maxLabelLen = rows.reduce((longest, row) => Math.max(longest, String(row.label || "").length), 8);
  const trace = {
    type: "bar",
    orientation: "h",
    y: rows.map((row) => row.label),
    x: rows.map((row) => row.value),
    customdata: rows.map((row) => row.display || row.value),
    marker: {
      color: rows.map((row) => colorForRow(row)),
      line: { color: "#ffffff", width: 1 },
    },
    hovertemplate: "<b>%{y}</b><br>%{customdata}<extra></extra>",
  };
  return {
    data: [trace],
    layout: {
      template: "plotly_white",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      autosize: true,
      title: subtitle
        ? { text: `${title}<br><sup>${subtitle}</sup>`, x: 0, xanchor: "left", font: { size: 16, color: "#202833" } }
        : { text: title, x: 0, xanchor: "left", font: { size: 16, color: "#202833" } },
      margin: { l: 16, r: 24, t: subtitle ? 88 : 64, b: 36 },
      xaxis: {
        title: options.xTitle || plotMetricLabel(metric),
        titlefont: { color: "#667085" },
        tickfont: { color: "#667085" },
        gridcolor: "#ebe5dc",
        zeroline: true,
        zerolinecolor: zeroCentered ? "#8a93a0" : "#d8d1c5",
        rangemode: !zeroCentered && metric.direction !== "min" ? "tozero" : "normal",
      },
      yaxis: {
        automargin: true,
        autorange: "reversed",
        tickfont: { color: "#202833", size: Math.max(10, 12 - Math.floor(maxLabelLen / 28)) },
      },
      height: Math.max(280, rows.length * 28 + 120),
    },
    config: {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: slug(title), scale: 2 },
    },
  };
}

function plotlyChartMarkup(key, title, subtitle, note, rows, spec) {
  const divId = `plot-${slug(key)}`;
  state.pendingPlotly.push({ key, divId, spec });
  return `
    <div class="generated-plot">
      <div class="chart-head"><div><strong>${escapeHtml(title)}</strong>${note ? `<div class="scope-note">${escapeHtml(note)}</div>` : ""}</div>${registerPlotExport(key, title, rows, "plotly")}</div>
      <div id="${divId}" class="plotly-host" role="img" aria-label="${escapeHtml(title)}"></div>
    </div>
  `;
}

function applyPlotSpan(html, span) {
  if (!html) return html;
  const cls = span === "full"
    ? "generated-plot plot-span-full"
    : span === "half"
      ? "generated-plot plot-span-half"
      : "generated-plot";
  return html.replace(/class="generated-plot(?: plot-span-(?:full|half))?"/, `class="${cls}"`);
}

function metricValue(study, name) {
  const summary = study && study.metric_summary ? study.metric_summary : null;
  const metric = metricSummaryLookup(summary, name);
  return metric && Number.isFinite(Number(metric.mean)) ? Number(metric.mean) : null;
}

function replicaMetricValue(replica, name) {
  const metric = (replica.metrics || []).find((item) => item.name === name);
  return metric && Number.isFinite(Number(metric.value)) ? Number(metric.value) : null;
}

function figureAssetUrl(figure) {
  return `/api/figure-asset?path=${encodeURIComponent(figure.path)}`;
}

function isPreviewableFigure(figure) {
  return [".png", ".jpg", ".jpeg", ".svg"].includes((figure.suffix || "").toLowerCase());
}

function collectStudyFigures(study) {
  const figures = [];
  (study.figures || []).forEach((figure) => figures.push({ label: "study", figure, study }));
  (study.replicas || []).forEach((replica) => {
    (replica.figures || []).forEach((figure) => figures.push({ label: replica.replica_name, figure, study }));
  });
  return figures;
}

function collectComparisonFigures(study) {
  const baseline = state.workspace ? state.workspace.baseline_study : null;
  if (!baseline || baseline.study_name === study.study_name) return collectStudyFigures(study);
  return [...collectStudyFigures(baseline), ...collectStudyFigures(study)];
}

function figureGroup(figure) {
  const role = figure.role || "figure";
  const path = String(figure.path || "").toLowerCase();
  if (SELECTED_MODEL_ROLES.has(role) || path.includes("best_model") || path.includes("selected")) return "selected";
  if (MODEL_COMPARISON_ROLES.has(role) || path.includes("cv_") || path.includes("comparison")) return "comparison";
  return "other";
}

function figureMatchesFilters(item) {
  const figure = item.figure;
  const filters = state.figureFilters;
  if (filters.group !== "all" && figureGroup(figure) !== filters.group) return false;
  if (filters.dataset !== "all" && (figure.dataset || "") !== filters.dataset) return false;
  if (filters.metric !== "all" && (figure.metric_name || "") !== filters.metric) return false;
  if (filters.role !== "all" && (figure.role || "figure") !== filters.role) return false;
  return true;
}

function optionHtml(value, label, selected) {
  return `<option value="${escapeHtml(value)}"${value === selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Boolean))).sort((left, right) => left.localeCompare(right));
}

function renderGlobalControls() {
  ensureSelectedMetric();
  normalizeComparisonBaseline();
  const scopeOptions = [
    optionHtml(TEST_SCOPE, "Test", state.resultScope),
    optionHtml(VALIDATION_SCOPE, "Validation", state.resultScope),
    optionHtml(COMBINED_SCOPE, "Combined", state.resultScope),
  ].join("");
  const comparisonOptions = [
    optionHtml("internal", "full_ocscore", state.comparisonBaseline),
    ...scopedExternalBaselines().map((item) => optionHtml(externalEntryId(item), externalDisplayName(item), state.comparisonBaseline)),
  ].join("");
  const metricOptions = scopedMetrics().map((item) => optionHtml(item.name, item.label, state.selectedMetric)).join("");
  $("result-scope-select").innerHTML = scopeOptions;
  $("comparison-baseline-select").innerHTML = comparisonOptions || '<option value="internal">full_ocscore</option>';
  $("decision-metric-select").innerHTML = metricOptions || '<option value="">No metrics</option>';
  $("result-scope-select").onchange = (event) => {
    state.resultScope = event.target.value;
    state.selectedMetric = null;
    if (state.comparisonBaseline !== "internal" && !scopedExternalBaselines().some((item) => externalEntryId(item) === state.comparisonBaseline)) {
      state.comparisonBaseline = "internal";
    }
    renderWorkspace(state.workspace);
  };
  $("comparison-baseline-select").onchange = (event) => {
    state.comparisonBaseline = event.target.value || "internal";
    renderWorkspace(state.workspace);
  };
  $("decision-metric-select").onchange = (event) => {
    state.selectedMetric = event.target.value || null;
    renderComparisonTable();
    renderComparisonCharts();
    if (state.selectedStudy) renderDetailPlots(state.selectedStudy);
  };
}

function comparisonDeltaCell(item, metricName) {
  if (isReferenceEntry(item)) return { value: "—", numeric: true, className: "delta-col" };
  const delta = metricDecisionDelta(item.entry, metricName);
  if (delta === null) return { value: "-", numeric: true, className: "delta-col" };
  const deltaClass = delta > 0 ? "delta-positive" : delta < 0 ? "delta-negative" : "delta-neutral";
  const std = item.external ? null : entryMetricStd(item.entry, metricName);
  const withinNoise = std !== null && Math.abs(delta) < std;
  const spanClass = ["metric-delta", deltaClass, withinNoise ? "delta-within-noise" : ""].filter(Boolean).join(" ");
  const deltaText = delta === 0 ? "±0" : `${delta > 0 ? "+" : ""}${numeric(delta)}`;
  const title = withinNoise ? `|Δ| < σ (${numeric(std)}) — within replica noise` : undefined;
  return {
    value: `<span class="${spanClass}">${escapeHtml(deltaText)}</span>`,
    numeric: true,
    className: "delta-col",
    title,
  };
}

function comparisonColorLegendItem(label, color, options = {}) {
  const title = options.title ? ` title="${escapeHtml(options.title)}"` : "";
  return `<span class="legend-item color-legend-item"${title}><span class="legend-swatch" style="background:${color};border-color:${color};"></span><span>${escapeHtml(label)}</span></span>`;
}

function renderComparisonColorLegend(entries) {
  const node = $("comparison-color-legend");
  if (!node) return;
  if (!entries.length) {
    node.innerHTML = "";
    return;
  }
  const categoriesPresent = new Set(entries.map((item) => entryModelCategory(item)));
  const items = [
    '<span class="legend-intro">Model and Type pill colors by category (same palette as Charts).</span>',
  ];
  ["full_ocscore", "ablation", "sf", "consensus"].forEach((category) => {
    if (!categoriesPresent.has(category)) return;
    items.push(comparisonColorLegendItem(MODEL_CATEGORY_LABELS[category], MODEL_CATEGORY_COLORS[category], {
      title: category === "full_ocscore"
        ? "Full OCScore baseline reference model"
        : category === "ablation"
          ? "Feature-policy ablation studies"
          : category === "sf"
            ? "Scoring-function external baselines"
            : "Other consensus external baselines",
    }));
  });
  node.innerHTML = `<div class="color-legend-grid">${items.join("")}</div>`;
}

function renderComparisonTable() {
  const metricHeaders = metricColumns(state.workspace, allStudies()).filter((metric) => metricAllowedByScope(metric.name));
  const selectedMetric = ensureSelectedMetric();
  const entries = comparisonEntries();
  const metricRanks = Object.fromEntries(
    metricHeaders.map((metric) => [
      metric.name,
      buildMetricRankLookup(entries, (item) => metricValue(item.entry, metric.name), metric.direction),
    ])
  );
  const sortedEntries = sortedComparisonEntries(entries, selectedMetric);
  const referenceLabel = comparisonReferenceLabel();
  $("comparison-summary").textContent = `${sortedEntries.length} models · ${scopeLabel()} · vs ${referenceLabel}`;
  table(
    $("comparison-table"),
    [
      { label: "Model", sortKey: "model", defaultDirection: "asc" },
      { label: "Type", sortKey: "kind", defaultDirection: "asc" },
      { label: "Replicas", sortKey: "replicas", defaultDirection: "desc", numeric: true },
      { label: `Δ ${selectedMetric ? metricMeta(selectedMetric).label : "Metric"}`, sortKey: "delta", defaultDirection: "desc", numeric: true, headerClass: "delta-col" },
      ...metricHeaders.map((metric) => ({
        label: metric.label,
        sortKey: `metric:${metric.name}`,
        defaultDirection: metric.direction === "min" ? "asc" : "desc",
        numeric: true,
      })),
    ],
    sortedEntries.map((item) => [
      modelCell(item),
      kindBadge(item),
      { value: item.study?.detected_replica_count ? `${item.study.detected_replica_count}/${item.study.expected_replica_count || item.study.detected_replica_count}` : "—", numeric: !!item.study?.detected_replica_count, title: modelDescription(item) },
      comparisonDeltaCell(item, selectedMetric),
      ...metricHeaders.map((metric) => comparisonMetricCell(item, metric.name, metricRanks[metric.name], metricRanks[metric.name].size)),
    ]),
    (row, index) => comparisonRowClass(sortedEntries[index]),
    state.comparisonSort,
  );
  bindSortButtons($("comparison-table"), "comparisonSort");
  renderComparisonExportActions(metricHeaders, sortedEntries, selectedMetric);
  renderComparisonColorLegend(sortedEntries);
  $("comparison-table").querySelectorAll("button[data-entry-id]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const item = entries.find((entry) => entry.id === button.dataset.entryId);
      if (!item) return;
      if (item.external) {
        state.comparisonBaseline = item.id;
        renderWorkspace(state.workspace);
        return;
      }
      if (item.isFullModel && state.comparisonBaseline !== "internal") {
        state.comparisonBaseline = "internal";
        renderDetail(item.study);
        renderGlobalControls();
        renderComparisonTable();
        renderComparisonCharts();
        return;
      }
      renderDetail(item.study);
    });
  });
}

function renderComparisonCharts() {
  const selectedMetric = ensureSelectedMetric();
  state.plotExports = {};
  state.pendingPlotly = [];
  const rankPlot = generatedRankPlot(selectedMetric);
  const spreadPlot = generatedReplicaSpreadPlot(selectedMetric);
  const deltaPlot = generatedAllDeltasPlot(selectedMetric);
  const container = $("comparison-charts");
  container.className = "decision-plots";
  const parts = [];
  if (rankPlot) parts.push(applyPlotSpan(rankPlot, "full"));
  if (spreadPlot && deltaPlot) {
    parts.push(applyPlotSpan(spreadPlot, "half"));
    parts.push(applyPlotSpan(deltaPlot, "half"));
  } else if (spreadPlot) {
    parts.push(applyPlotSpan(spreadPlot, "full"));
  } else if (deltaPlot) {
    parts.push(applyPlotSpan(deltaPlot, "full"));
  }
  if (parts.length === 1) container.classList.add("layout-single");
  container.innerHTML = parts.join("");
  void mountPendingPlotlyCharts();
  bindPlotExportButtons();
}

function generatedReplicaSpreadPlot(metricName) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const rows = comparisonEntries()
    .filter((item) => !item.external)
    .flatMap((item) => (item.study?.replicas || [])
      .map((replica) => {
        const value = replicaMetricValue(replica, metricName);
        if (value === null) return null;
        return {
          label: `${modelDisplayName(item)} · ${replica.replica_name}`,
          value,
          display: numeric(value),
          scope: metricScope(metricName),
          study: item.id,
          metric: metricName,
          color: entryPaletteColor(item),
        };
      })
      .filter(Boolean));
  if (rows.length < 2) return "";
  rows.sort((left, right) => right.value - left.value);
  const title = `${plotMetricLabel(metric)} · replica values`;
  const key = `replica_spread_${slug(metricName)}_${state.resultScope}`;
  const spec = buildSimpleBarPlotlySpec(rows, metric, {
    title,
    subtitle: "Each bar is one replica. Table cells show μ only when averaged over multiple replicas.",
    colorForRow: (row) => row.color || MODEL_CATEGORY_COLORS.ablation,
  });
  return plotlyChartMarkup(key, title, "", "Green = full_ocscore · blue = ablation", rows, spec);
}

function generatedAllDeltasPlot(metricName) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const reference = comparisonReferenceSummary();
  const referenceMetric = metricSummaryLookup(reference, metricName);
  if (!referenceMetric) return "";
  const rows = comparisonEntries()
    .filter((item) => !isReferenceEntry(item))
    .map((item) => {
      const delta = metricDecisionDelta(item.entry, metricName);
      if (delta === null) return null;
      const std = item.external ? null : entryMetricStd(item.entry, metricName);
      const withinNoise = std !== null && Math.abs(delta) < std;
      const label = modelDisplayName(item) + (item.synthesized ? " (approx)" : "") + (withinNoise ? " · ~noise" : "");
      return {
        label,
        value: delta,
        display: `${delta > 0 ? "+" : ""}${numeric(delta)}${withinNoise ? " (< σ)" : ""}`,
        scope: metricScope(metricName),
        study: item.id,
        metric: metricName,
        withinNoise,
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.value - left.value);
  if (!rows.length) return "";
  const title = `${plotMetricLabel(metric)} vs ${comparisonReferenceLabel()}`;
  const key = `all_delta_${slug(metricName)}_${state.resultScope}_${slug(state.comparisonBaseline)}`;
  const spec = buildSimpleBarPlotlySpec(rows, metric, {
    title,
    subtitle: "Positive = improvement vs reference under metric direction",
    zeroCentered: true,
    xTitle: "Δ vs reference",
    colorForRow: (row) => {
      if (row.withinNoise) return "#c5cad1";
      return row.value > 0 ? "#16703f" : row.value < 0 ? "#b42318" : "#667085";
    },
  });
  return plotlyChartMarkup(key, title, "", "Green/red = directionally better/worse · grey = within replica σ", rows, spec);
}

function cvMetricMatchesScope(metricName) {
  const normalized = String(metricName || "").toLowerCase();
  if (state.resultScope === COMBINED_SCOPE) return true;
  const regression = ["rmse", "mae", "r2", "pearson r", "spearman rho"].some((token) => normalized.includes(token));
  const ranking = ["bedroc", "roc-auc", "pr-auc", "ef", "ndcg"].some((token) => normalized.includes(token));
  if (state.resultScope === VALIDATION_SCOPE) return regression || normalized.includes("validation");
  return ranking || !regression;
}

function renderCrossValidationPanel() {
  const panel = $("cv-panel");
  const sections = allStudies()
    .map((study) => ({ study, cv: study.cross_validation }))
    .filter((item) => item.cv && (item.cv.metrics || []).length);
  if (!sections.length) {
    panel.hidden = true;
    $("cv-table").innerHTML = "";
    $("cv-summary").textContent = "";
    return;
  }
  panel.hidden = false;
  const foldCounts = sections.map((item) => item.cv.fold_count).filter((count) => count > 0);
  const foldNote = foldCounts.length ? `${Math.max(...foldCounts)}-fold CV` : "exported CV";
  $("cv-summary").textContent = `${sections.length} stud${sections.length === 1 ? "y" : "ies"} · ${foldNote} · from export/cross_validation`;
  const blocks = sections.map(({ study, cv }) => {
    const rows = (cv.metrics || [])
      .filter((row) => cvMetricMatchesScope(row.metric))
      .sort((left, right) => {
        const leftScore = left.scorer === "OCScore" ? 0 : 1;
        const rightScore = right.scorer === "OCScore" ? 0 : 1;
        return leftScore - rightScore || left.metric.localeCompare(right.metric) || left.scorer.localeCompare(right.scorer);
      });
    if (!rows.length) {
      return `<div class="path">${escapeHtml(studyDisplayName(study))}: no CV metrics for ${scopeLabel()} scope.</div>`;
    }
    const body = rows.map((row) => `
      <tr>
        <td>${escapeHtml(row.scorer)}</td>
        <td>${escapeHtml(row.metric)}</td>
        <td class="numeric">${escapeHtml(numeric(row.mean))} ± ${escapeHtml(numeric(row.std))}</td>
        <td class="numeric">${escapeHtml(row.n_folds || cv.fold_count || "—")}</td>
      </tr>
    `).join("");
    return `
      <div class="cv-study-block">
        <h3>${escapeHtml(studyDisplayName(study))}</h3>
        <table>
          <thead><tr><th>Scorer</th><th>Metric</th><th>Mean ± std</th><th>Folds</th></tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    `;
  }).join("");
  $("cv-table").innerHTML = blocks;
}

function figureFilterSummaryLabel() {
  const filters = state.figureFilters;
  const groupLabels = {
    comparison: "Model comparison",
    selected: "Selected model",
    all: "All figure groups",
  };
  const roleLabels = {
    all: "All roles",
  };
  const group = groupLabels[filters.group] || filters.group;
  const role = roleLabels[filters.role] || titleCase(filters.role);
  const dataset = filters.dataset === "all" ? "All datasets" : filters.dataset;
  const metric = filters.metric === "all" ? "All metrics" : metricMeta(filters.metric).label;
  return `${group} · ${role} · ${dataset} · ${metric}`;
}

function renderFigureControls(study, figures) {
  const datasets = uniqueSorted(figures.map((item) => item.figure.dataset || ""));
  const roles = uniqueSorted(figures.map((item) => item.figure.role || "figure"));
  const metrics = uniqueSorted(figures.map((item) => item.figure.metric_name || ""));
  const groupOptions = [
    optionHtml("comparison", "Model comparison", state.figureFilters.group),
    optionHtml("selected", "Selected model", state.figureFilters.group),
    optionHtml("all", "All figure groups", state.figureFilters.group),
  ].join("");
  const datasetOptions = [optionHtml("all", "All datasets", state.figureFilters.dataset), ...datasets.map((item) => optionHtml(item, item, state.figureFilters.dataset))].join("");
  const roleOptions = [
    optionHtml("all", "All roles", state.figureFilters.role),
    ...roles.map((item) => optionHtml(item, titleCase(item), state.figureFilters.role)),
  ].join("");
  const metricOptions = [optionHtml("all", "All figure metrics", state.figureFilters.metric), ...metrics.map((item) => optionHtml(item, metricMeta(item).label, state.figureFilters.metric))].join("");
  const summary = $("figure-filter-summary");
  if (summary) summary.textContent = figureFilterSummaryLabel();
  $("figure-controls").innerHTML = `
    <label class="filter-field" for="figure-group-filter"><span>Figure group</span><select id="figure-group-filter">${groupOptions}</select></label>
    <label class="filter-field" for="figure-dataset-filter"><span>Dataset</span><select id="figure-dataset-filter">${datasetOptions}</select></label>
    <label class="filter-field" for="figure-role-filter"><span>Role</span><select id="figure-role-filter">${roleOptions}</select></label>
    <label class="filter-field" for="figure-metric-filter"><span>Figure metric</span><select id="figure-metric-filter">${metricOptions}</select></label>
  `;
  const rerenderFigures = () => {
    persistUiState();
    renderDetailPlots(study);
  };
  $("figure-group-filter").addEventListener("change", (event) => {
    state.figureFilters.group = event.target.value;
    rerenderFigures();
  });
  $("figure-dataset-filter").addEventListener("change", (event) => {
    state.figureFilters.dataset = event.target.value;
    rerenderFigures();
  });
  $("figure-role-filter").addEventListener("change", (event) => {
    state.figureFilters.role = event.target.value;
    rerenderFigures();
  });
  $("figure-metric-filter").addEventListener("change", (event) => {
    state.figureFilters.metric = event.target.value;
    rerenderFigures();
  });
}

function slug(value) {
  return String(value || "plot").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "plot";
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function rowsToCsv(rows) {
  const hasStd = rows.some((row) => row.std !== null && row.std !== undefined && row.std !== "");
  const headers = hasStd
    ? ["label", "value", "std", "scope", "study", "metric"]
    : ["label", "value", "scope", "study", "metric"];
  const lines = [headers.join(",")];
  rows.forEach((row) => {
    lines.push(headers.map((key) => csvEscape(row[key] ?? "")).join(","));
  });
  return `${lines.join("\\n")}\\n`;
}

function objectsToCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  rows.forEach((row) => {
    lines.push(headers.map((key) => csvEscape(row[key] ?? "")).join(","));
  });
  return `${lines.join("\\n")}\\n`;
}

function comparisonExportRows(metricHeaders, sortedEntries, selectedMetric) {
  return sortedEntries.map((item) => {
    const summary = item.entry?.metric_summary || null;
    const row = {
      model: modelDisplayName(item),
      type: item.kind,
      reference: isReferenceEntry(item) ? "yes" : "no",
      replicas: item.study?.detected_replica_count ?? "",
      delta: "",
    };
    if (!isReferenceEntry(item)) {
      const delta = metricDecisionDelta(item.entry, selectedMetric);
      row.delta = delta === null ? "" : numeric(delta);
    }
    metricHeaders.forEach((metric) => {
      const stat = metricSummaryLookup(summary, metric.name);
      row[`${metric.name}_mean`] = stat && Number.isFinite(Number(stat.mean)) ? numeric(stat.mean) : "";
      row[`${metric.name}_std`] = stat && Number(stat.count) > 1 && Number.isFinite(Number(stat.std)) ? numeric(stat.std) : "";
    });
    return row;
  });
}

function renderComparisonExportActions(metricHeaders, sortedEntries, selectedMetric) {
  const target = $("comparison-export-actions");
  if (!target) return;
  target.innerHTML = `
    <button class="ghost-button" type="button" data-comparison-export="csv">CSV</button>
    <button class="ghost-button" type="button" data-comparison-export="copy">Copy</button>
  `;
  target.querySelectorAll("button[data-comparison-export]").forEach((button) => {
    button.addEventListener("click", async () => {
      const rows = comparisonExportRows(metricHeaders, sortedEntries, selectedMetric);
      const csv = objectsToCsv(rows);
      const filename = `ocscore_results_${state.resultScope}_${slug(selectedMetric || "metric")}.csv`;
      try {
        if (button.dataset.comparisonExport === "csv") {
          downloadText(filename, csv, "text/csv;charset=utf-8");
          toast("Results table downloaded.");
          return;
        }
        if (!navigator.clipboard?.writeText) {
          throw new Error("Clipboard copy is not supported in this browser");
        }
        await navigator.clipboard.writeText(csv);
        toast("Results table copied to clipboard.");
      } catch (error) {
        toast(error.message || String(error));
      }
    });
  });
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], { type });
  downloadBlob(filename, blob);
}

function downloadBlob(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function svgDimensions(svgString) {
  const match = String(svgString).match(/viewBox="0\\s+0\\s+([\\d.]+)\\s+([\\d.]+)"/);
  if (match) return { width: Number(match[1]), height: Number(match[2]) };
  return { width: 1400, height: 800 };
}

async function svgToPngBlob(svgString, scale = 2) {
  const { width, height } = svgDimensions(svgString);
  const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("SVG render failed"));
      image.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas unavailable");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const pngBlob = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result) resolve(result);
        else reject(new Error("PNG export failed"));
      }, "image/png");
    });
    return pngBlob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function downloadPng(filename, svgString) {
  downloadBlob(filename, await svgToPngBlob(svgString));
}

async function copyPngToClipboard(svgString) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error("Clipboard image copy is not supported in this browser");
  }
  const blob = await svgToPngBlob(svgString);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

const PLOTLY_EXPORT_LAYOUT = {
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: { color: "#202833" },
  xaxis: { color: "#667085", gridcolor: "#ebe5dc", zerolinecolor: "#d8d1c5", titlefont: { color: "#667085" }, tickfont: { color: "#667085" } },
  yaxis: { color: "#202833", tickfont: { color: "#202833" } },
  legend: { font: { color: "#667085" } },
};

function plotlyExportImageOptions(host, format) {
  return {
    format,
    width: Math.max(960, host.offsetWidth || 960),
    height: Math.max(320, host.offsetHeight || 320),
    scale: format === "png" ? 2 : 1,
    layout: PLOTLY_EXPORT_LAYOUT,
  };
}

async function downloadPlotlyImage(filename, host, format) {
  const dataUrl = await Plotly.toImage(host, plotlyExportImageOptions(host, format));
  if (format === "svg") {
    const svg = decodeURIComponent(dataUrl.split(",")[1] || "");
    downloadText(filename, svg, "image/svg+xml;charset=utf-8");
    return;
  }
  downloadDataUrl(filename, dataUrl);
}

function downloadDataUrl(filename, dataUrl) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  link.rel = "noopener";
  link.click();
}

async function copyPlotlyPng(host) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error("Clipboard image copy is not supported in this browser");
  }
  const dataUrl = await Plotly.toImage(host, plotlyExportImageOptions(host, "png"));
  const blob = await (await fetch(dataUrl)).blob();
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

function registerPlotExport(key, title, rows, asset) {
  const payload = { title, rows };
  if (asset === "plotly") payload.engine = "plotly";
  else payload.svg = asset;
  state.plotExports[key] = payload;
  return `
    <div class="export-actions">
      <button class="ghost-button" type="button" data-export-kind="png" data-export-key="${escapeHtml(key)}">PNG</button>
      <button class="ghost-button" type="button" data-export-kind="copy" data-export-key="${escapeHtml(key)}">Copy</button>
      <button class="ghost-button" type="button" data-export-kind="svg" data-export-key="${escapeHtml(key)}">SVG</button>
      <button class="ghost-button" type="button" data-export-kind="csv" data-export-key="${escapeHtml(key)}">CSV</button>
    </div>
  `;
}

function bindPlotExportButtons() {
  document.querySelectorAll("button[data-export-key]").forEach((button) => {
    button.addEventListener("click", async () => {
      const payload = state.plotExports[button.dataset.exportKey];
      if (!payload) return;
      const base = slug(payload.title);
      const kind = button.dataset.exportKind;
      try {
        if (kind === "csv") {
          downloadText(`${base}.csv`, rowsToCsv(payload.rows), "text/csv;charset=utf-8");
          return;
        }
        if (payload.plotlyDivId && window.Plotly) {
          const host = document.getElementById(payload.plotlyDivId);
          if (!host) throw new Error("Chart is not ready yet");
          if (kind === "svg") {
            await downloadPlotlyImage(`${base}.svg`, host, "svg");
          } else if (kind === "png") {
            await downloadPlotlyImage(`${base}.png`, host, "png");
          } else if (kind === "copy") {
            await copyPlotlyPng(host);
            toast("Chart copied to clipboard");
          }
          return;
        }
        if (kind === "svg") {
          downloadText(`${base}.svg`, payload.svg, "image/svg+xml;charset=utf-8");
        } else if (kind === "png") {
          await downloadPng(`${base}.png`, payload.svg);
        } else if (kind === "copy") {
          await copyPngToClipboard(payload.svg);
          toast("Chart copied to clipboard");
        }
      } catch (error) {
        toast(error.message || String(error));
      }
    });
  });
}

function chartSvg(title, rows, options = {}) {
  const rowHeight = 42;
  const top = options.hideTitle ? 54 : 78;
  const bottom = 34;
  const labelWidth = options.labelWidth || 430;
  const valueWidth = options.valueWidth || 150;
  const trackX = labelWidth + 44;
  const trackWidth = options.trackWidth || 760;
  const width = trackX + trackWidth + valueWidth + 34;
  const height = Math.max(180, top + rows.length * rowHeight + bottom);
  const values = rows.map((row) => Number(row.value));
  const minValue = options.zeroCentered
    ? Math.min(0, ...values)
    : options.scaleFromZero
      ? Math.min(0, ...values)
      : Math.min(...values);
  const maxValue = options.zeroCentered
    ? Math.max(0, ...values)
    : options.scaleFromZero
      ? Math.max(...values)
      : Math.max(...values);
  const span = Math.max(maxValue - minValue, 1e-12);
  const zeroX = trackX + ((0 - minValue) / span) * trackWidth;
  const subtitle = options.subtitle
    ? `<text x="24" y="${options.hideTitle ? 28 : 48}" fill="#667085" font-size="18">${escapeHtml(options.subtitle)}</text>`
    : "";
  const titleMarkup = options.hideTitle
    ? ""
    : `<text x="24" y="30" fill="#202833" font-size="24" font-weight="700">${escapeHtml(title)}</text>`;
  const rowMarkup = rows.map((row, index) => {
    const y = top + index * rowHeight;
    const value = Number(row.value);
    const x = trackX + ((value - minValue) / span) * trackWidth;
    const barStart = options.zeroCentered ? Math.min(zeroX, x) : trackX;
    const barWidth = options.dot ? 0 : Math.max(3, Math.abs(x - (options.zeroCentered ? zeroX : trackX)));
    const color = value < 0 && options.zeroCentered ? "#b42318" : "#087f7b";
    const label = `${options.rank ? `${index + 1}. ` : ""}${row.label}`;
    const mark = options.dot
      ? `<circle cx="${x.toFixed(2)}" cy="${y + 17}" r="7" fill="${color}"></circle>`
      : `<rect x="${barStart.toFixed(2)}" y="${y + 10}" width="${barWidth.toFixed(2)}" height="14" rx="7" fill="${color}"></rect>`;
    return `
      <text x="24" y="${y + 21}" fill="#202833" font-size="18">${escapeHtml(label)}</text>
      <rect x="${trackX}" y="${y + 10}" width="${trackWidth}" height="14" rx="7" fill="#ebe5dc"></rect>
      ${options.zeroCentered ? `<line x1="${zeroX.toFixed(2)}" x2="${zeroX.toFixed(2)}" y1="${y + 5}" y2="${y + 29}" stroke="#667085" stroke-width="2"></line>` : ""}
      ${mark}
      <text x="${trackX + trackWidth + 18}" y="${y + 21}" fill="#667085" font-size="17">${escapeHtml(row.display || numeric(value))}</text>
    `;
  }).join("");
  return `
    <svg class="decision-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)}">
      <rect width="${width}" height="${height}" fill="#ffffff"></rect>
      ${titleMarkup}
      ${subtitle}
      ${rowMarkup}
    </svg>
  `;
}

function generatedRankPlot(metricName) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const rows = rankComparisonEntries()
    .map((entry) => {
      const plotted = plotMetricValueDisplay(entry, metricName);
      if (!plotted) return null;
      return {
        entry,
        ...plotted,
        external: Boolean(entry.external),
      baseline_family: entry.baseline_family || null,
      };
    })
    .filter(Boolean)
    .sort((left, right) => metric.direction === "min" ? left.value - right.value : right.value - left.value)
    .map((row) => ({
      label: studyDisplayName(row.entry) + (row.entry.synthesized ? " (approx)" : ""),
      value: row.value,
      std: row.std,
      count: row.count,
      display: row.display,
      barLabel: row.barLabel,
      hoverLabel: row.hoverLabel,
      study_name: row.entry.study_name,
      policy_name: row.entry.policy_name,
      baseline_family: row.entry.baseline_family || null,
      external: row.external,
      scope: metricScope(metricName),
      study: studyDisplayName(row.entry),
      metric: metricName,
    }));
  if (!rows.length) return "";
  const title = `${plotMetricLabel(metric)} rank across studies`;
  const key = `rank_${slug(metricName)}_${state.resultScope}`;
  const divId = `plot-${slug(key)}`;
  const spec = buildRankPlotlySpec(rows, metric);
  state.pendingPlotly.push({ key, divId, spec });
  return `
    <div class="generated-plot">
      <div class="chart-head"><div><strong>${escapeHtml(title)}</strong><div class="scope-note">full_ocscore / Ablation / SF / Other consensus</div></div>${registerPlotExport(key, title, rows, "plotly")}</div>
      <div id="${divId}" class="plotly-host" role="img" aria-label="${escapeHtml(title)}"></div>
    </div>
  `;
}

function generatedStabilityPlot(study, metricName) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const rows = (study.replicas || [])
    .map((replica) => ({ replica, value: replicaMetricValue(replica, metricName) }))
    .filter((row) => row.value !== null)
    .map((row) => ({
      label: row.replica.replica_name,
      value: row.value,
      display: numeric(row.value),
      scope: metricScope(metricName),
      study: study.study_name,
      metric: metricName,
    }));
  const title = `${scopeLabel()} ${metric.label} Replica Stability`;
  if (!rows.length) {
    return `
      <div class="generated-plot">
        <div class="chart-head"><strong>${escapeHtml(title)}</strong></div>
        <span class="path">No replica-level values detected for this metric.</span>
      </div>
    `;
  }
  const values = rows.map((row) => row.value);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const svg = chartSvg(title, rows, { dot: true, subtitle: `Mean ${numeric(mean)} | each dot is one replica.` });
  const key = `stability_${slug(study.study_name)}_${slug(metricName)}_${state.resultScope}`;
  return `
    <div class="generated-plot">
      <div class="chart-head"><div><strong>${escapeHtml(title)}</strong><div class="scope-note">Selected model replica spread</div></div>${registerPlotExport(key, title, rows, svg)}</div>
      ${svg}
    </div>
  `;
}


function isShapFigure(figure) {
  return String(figure.role || "").startsWith("shap");
}

function shapFigureRoleClass(role) {
  const normalized = String(role || "");
  if (normalized === "shap_beeswarm") return "shap-beeswarm";
  if (normalized === "shap_importance") return "shap-importance";
  if (normalized === "shap_dependence") return "shap-dependence";
  return "shap-other";
}

function sortShapFigures(items) {
  const order = { shap_beeswarm: 0, shap_importance: 1, shap_dependence: 2, shap: 3 };
  return [...items].sort((left, right) => {
    const leftOrder = order[left.figure.role] ?? 99;
    const rightOrder = order[right.figure.role] ?? 99;
    if (leftOrder !== rightOrder) return leftOrder - rightOrder;
    const leftDataset = left.figure.dataset || "";
    const rightDataset = right.figure.dataset || "";
    if (leftDataset !== rightDataset) return leftDataset.localeCompare(rightDataset);
    return String(left.figure.path || "").localeCompare(String(right.figure.path || ""));
  });
}

function collectShapFigures(study) {
  return sortShapFigures(collectStudyFigures(study).filter((item) => isShapFigure(item.figure) && figureGroup(item.figure) === "selected"));
}

function shapFigureMatchesFilters(item) {
  const figure = item.figure;
  const filters = state.figureFilters;
  if (filters.dataset !== "all" && (figure.dataset || "") !== filters.dataset) return false;
  if (filters.metric !== "all" && (figure.metric_name || "") !== filters.metric) return false;
  if (filters.role !== "all" && (figure.role || "figure") !== filters.role) return false;
  return true;
}

function figurePreviewMarkup(figure, item, expandable = false) {
  if (!isPreviewableFigure(figure)) return "";
  const src = figureAssetUrl(figure);
  const alt = `${figure.role || "figure"} ${item.study.study_name}`;
  const caption = `${titleCase(figure.role || "figure")} · ${item.study.study_name}${figure.dataset ? ` · ${figure.dataset}` : ""}`;
  if (!expandable) {
    return `<img class="figure-preview" src="${src}" alt="${escapeHtml(alt)}" loading="lazy">`;
  }
  return `<button type="button" class="figure-preview-button" data-figure-src="${escapeHtml(src)}" data-figure-caption="${escapeHtml(caption)}" aria-label="Expand ${escapeHtml(caption)}"><img class="figure-preview figure-preview-expandable" src="${src}" alt="${escapeHtml(alt)}" loading="lazy"></button>`;
}

let figureLightboxBound = false;

function openFigureLightbox(src, caption) {
  const overlay = $("figure-lightbox");
  const image = $("figure-lightbox-image");
  const cap = $("figure-lightbox-caption");
  if (!overlay || !image || !cap || !src) return;
  image.src = src;
  image.alt = caption || "";
  cap.textContent = caption || "";
  overlay.hidden = false;
  document.body.classList.add("figure-lightbox-open");
}

function closeFigureLightbox() {
  const overlay = $("figure-lightbox");
  const image = $("figure-lightbox-image");
  if (!overlay || !image) return;
  overlay.hidden = true;
  image.removeAttribute("src");
  $("figure-lightbox-caption").textContent = "";
  document.body.classList.remove("figure-lightbox-open");
}

function bindFigureLightbox() {
  const overlay = $("figure-lightbox");
  if (!overlay) return;
  $("figure-list").querySelectorAll(".figure-preview-button").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => {
      openFigureLightbox(button.dataset.figureSrc, button.dataset.figureCaption);
    });
  });
  if (figureLightboxBound) return;
  figureLightboxBound = true;
  overlay.querySelector(".figure-lightbox-close")?.addEventListener("click", closeFigureLightbox);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeFigureLightbox();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !overlay.hidden) closeFigureLightbox();
  });
}

function figureCard(item, options = {}) {
  const figure = item.figure;
  const metric = figure.metric_name ? metricMeta(figure.metric_name).label : "unscored";
  const group = figureGroup(figure) === "selected" ? "Selected model" : figureGroup(figure) === "comparison" ? "Model comparison" : "Other artifact";
  return `
    <div class="figure-item">
      <strong>${escapeHtml(titleCase(figure.role || "figure"))}</strong>
      <span>${escapeHtml(item.study.study_name)} / ${escapeHtml(item.label)}${figure.dataset ? ` / ${escapeHtml(figure.dataset)}` : ""}</span>
      <span class="figure-meta"><span>${escapeHtml(group)}</span><span>${escapeHtml(metric)}</span><span>${escapeHtml(figure.suffix || "file")}</span></span>
      ${figurePreviewMarkup(figure, item, Boolean(options.expandable))}
      <a class="asset-link" href="${figureAssetUrl(figure)}" download>Export source file</a>
      <span class="path">${escapeHtml(figure.path)}</span>
    </div>
  `;
}

function figureCardShap(item) {
  const figure = item.figure;
  const roleClass = shapFigureRoleClass(figure.role);
  const metric = figure.metric_name ? metricMeta(figure.metric_name).label : "unscored";
  const subtitle = [figure.dataset, item.label, metric].filter(Boolean).join(" · ");
  return `
    <div class="figure-item shap-figure ${roleClass}">
      <strong>${escapeHtml(titleCase(figure.role || "shap"))}</strong>
      <span class="scope-note">${escapeHtml(subtitle)}</span>
      ${isPreviewableFigure(figure) ? `<img class="figure-preview shap-preview" src="${figureAssetUrl(figure)}" alt="${escapeHtml(figure.role)}" loading="lazy">` : ""}
      <a class="asset-link" href="${figureAssetUrl(figure)}" download>Download</a>
    </div>
  `;
}

function shapFigureSection(study) {
  const items = collectShapFigures(study).filter(shapFigureMatchesFilters);
  if (!items.length) return "";
  const visible = items.slice(0, SHAP_RENDER_LIMIT);
  return `
    <section class="figure-section shap-section">
      <h3>SHAP (selected model)</h3>
      <p class="scope-note">Typical SHAP exports include a beeswarm plot (feature impact by value) and an importance plot (mean |SHAP|) per dataset and replica. Dependence plots explore individual features. Use the figure filters at the top of this section to narrow.</p>
      ${items.length > visible.length ? `<div class="gallery-note">Showing ${visible.length} of ${items.length} SHAP figures. Relax Dataset, Role, or Figure metric to see more.</div>` : ""}
      <div class="shap-figure-grid">${visible.map(figureCardShap).join("")}</div>
    </section>
  `;
}

function figureSection(title, items, options = {}) {
  const visible = items.slice(0, FIGURE_RENDER_LIMIT);
  if (!items.length) return "";
  return `
    <section class="figure-section">
      <h3>${escapeHtml(title)}</h3>
      ${items.length > visible.length ? `<div class="gallery-note">Showing ${visible.length} of ${items.length} matching figures. Use the figure filters above (Figure group, Dataset, Role, Figure metric) to narrow the gallery.</div>` : ""}
      ${options.expandable ? '<p class="scope-note">Click a figure to expand it.</p>' : ""}
      <div class="figure-section-grid">${visible.map((item) => figureCard(item, options)).join("")}</div>
    </section>
  `;
}

function renderDetailPlots(study) {
  const figures = collectStudyFigures(study);
  renderFigureControls(study, figures);
  const selectedMetric = ensureSelectedMetric();
  const filtered = figures.filter(figureMatchesFilters);
  const comparisonFigures = filtered.filter((item) => figureGroup(item.figure) === "comparison");
  const selectedFigures = filtered.filter((item) => figureGroup(item.figure) === "selected" && !isShapFigure(item.figure));
  const otherFigures = filtered.filter((item) => figureGroup(item.figure) === "other");
  $("detail-plots").innerHTML = generatedStabilityPlot(study, selectedMetric);
  bindPlotExportButtons();
  const detected = [
    shapFigureSection(study),
    figureSection("Model Comparison Figures", comparisonFigures, { expandable: true }),
    figureSection("Selected Model Figures", selectedFigures),
    figureSection("Other Matching Figures", otherFigures),
  ].join("");
  $("figure-list").innerHTML = detected || '<div class="path">No figures match the current filters.</div>';
  bindFigureLightbox();
}

function replicaMetricColumns(replicas) {
  const names = new Set();
  replicas.forEach((replica) => {
    (replica.metrics || []).forEach((metric) => names.add(metric.name));
  });
  return Array.from(names)
    .filter((name) => state.resultScope === COMBINED_SCOPE || metricAllowedByScope(name))
    .map((name) => metricMeta(name))
    .sort((left, right) => left.label.localeCompare(right.label));
}

function detailReplicaSortValue(replica, key) {
  if (key === "replica") return replica.replica_name || "";
  if (key === "status") return replica.status || "";
  if (key === "logs") return (replica.log_files || []).length;
  if (key === "path") return replica.path || "";
  if (key === "optuna") return replica.optuna_storage_path ? 1 : 0;
  if (key.startsWith("metric:")) return replicaMetricValue(replica, key.slice(7));
  return "";
}

function sortedDetailReplicas(replicas) {
  const sort = state.detailReplicaSort;
  return [...replicas].sort((left, right) => {
    const compared = compareValues(
      detailReplicaSortValue(left, sort.key),
      detailReplicaSortValue(right, sort.key),
      sort.direction,
    );
    return compared || String(left.replica_name || "").localeCompare(String(right.replica_name || ""));
  });
}

function replicaOptunaCell(replica) {
  if (!replica.exists) return { value: "—" };
  if (!replica.optuna_storage_path) {
    return { value: '<span class="path">no db</span>', title: "No optuna.db found in this replica directory" };
  }
  return {
    value: `<button type="button" class="ghost-button optuna-open" data-replica-path="${escapeHtml(String(replica.path))}">Open</button>`,
    title: `Launch Optuna dashboard for ${replica.optuna_storage_path}`,
  };
}

function bindOptunaDashboardButtons() {
  $("detail-replicas").querySelectorAll("button.optuna-open").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      const replicaPath = button.dataset.replicaPath;
      if (!replicaPath) return;
      button.disabled = true;
      try {
        const payload = await apiPost("/api/optuna-dashboard", { replica_path: replicaPath });
        window.open(payload.url, "_blank", "noopener,noreferrer");
        toast(payload.reused ? `Optuna dashboard already running on port ${payload.port}` : `Optuna dashboard started on port ${payload.port}`);
      } catch (error) {
        toast(error.message || String(error));
      } finally {
        button.disabled = false;
      }
    });
  });
}

function renderDetailReplicaTable(replicas) {
  const metricHeaders = replicaMetricColumns(replicas);
  const headers = [
    { label: "Replica", sortKey: "replica", defaultDirection: "asc" },
    { label: "Status", sortKey: "status", defaultDirection: "asc" },
    ...metricHeaders.map((metric) => ({
      label: metric.label,
      sortKey: `metric:${metric.name}`,
      defaultDirection: metric.direction === "min" ? "asc" : "desc",
      numeric: true,
    })),
    { label: "Optuna", sortKey: "optuna", defaultDirection: "asc" },
    { label: "Logs", sortKey: "logs", defaultDirection: "desc", numeric: true },
    { label: "Path", sortKey: "path", defaultDirection: "asc" },
  ];
  const rows = sortedDetailReplicas(replicas).map((replica) => [
    replica.replica_name,
    statusBadge(replica.status),
    ...metricHeaders.map((metric) => {
      const value = replicaMetricValue(replica, metric.name);
      if (value === null) return { value: "-", numeric: true };
      return { value: numeric(value), numeric: true, title: metric.label };
    }),
    replicaOptunaCell(replica),
    { value: (replica.log_files || []).length, numeric: true },
    { value: `<span class="path">${replica.exists ? replica.path : "missing"}</span>` },
  ]);
  table($("detail-replicas"), headers, rows, () => "", state.detailReplicaSort);
  bindSortButtons($("detail-replicas"), "detailReplicaSort");
  bindOptunaDashboardButtons();
}

function renderDetail(study) {
  if (!study) {
    $("detail-panel").hidden = true;
    state.selectedStudy = null;
    renderProtocolPanel();
    return;
  }
  $("detail-panel").hidden = false;
  state.selectedStudy = study;
  $("detail-title").textContent = `${studyDisplayName(study)} — replicas & figures`;
  const replicas = study.replicas || [];
  if (!replicas.length) {
    $("detail-replicas").innerHTML = '<div class="path">No replica records.</div>';
  } else {
    renderDetailReplicaTable(replicas);
  }
  renderDetailPlots(study);
  renderProtocolPanel();
}

function protocolFact(label, value) {
  if (value === null || value === undefined || value === "") return "";
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd></div>`;
}

function protocolCard(title, factsHtml, fullWidth = false) {
  if (!factsHtml) return "";
  return `
    <section class="protocol-card${fullWidth ? " protocol-card-full" : ""}">
      <h3>${escapeHtml(title)}</h3>
      <dl class="protocol-facts">${factsHtml}</dl>
    </section>
  `;
}

function formatSplitSizes(protocol) {
  const parts = [
    protocol.pdbbind_train_size,
    protocol.pdbbind_validation_size,
    protocol.pdbbind_test_size,
  ].filter((value) => value !== null && value !== undefined && value !== "");
  if (!parts.length) return "";
  return parts.map((value) => `${Math.round(Number(value) * 100)}%`).join(" / ");
}

function activeProtocolSummary() {
  if (state.selectedStudy?.protocol) return state.selectedStudy.protocol;
  return state.workspace?.protocol || null;
}

function renderProtocolPanel() {
  const protocol = activeProtocolSummary();
  const summary = $("protocol-summary");
  const content = $("protocol-content");
  if (!protocol) {
    summary.textContent = "No protocol artifacts found";
    content.innerHTML = '<div class="path">Expected replicas_protocol.json, staged_optuna_protocol.json, or protocol.generated.yml under the workspace or study root.</div>';
    return;
  }

  const viewingStudy = state.selectedStudy?.protocol ? modelDisplayName(state.selectedStudy) : "";
  const titleParts = [
    protocol.protocol_name,
    protocol.feature_policy && protocol.feature_policy !== "baseline" ? protocol.feature_policy : "",
    viewingStudy ? `viewing ${viewingStudy}` : "",
  ].filter(Boolean);
  summary.textContent = titleParts.join(" · ") || protocol.source_kind || "Protocol";

  const replicaList = (protocol.replica_names || []).length
    ? protocol.replica_names.join(", ")
    : protocol.n_replicas ? `${protocol.n_replicas} replicas` : "";

  const overviewFacts = [
    protocolFact("Protocol", protocol.protocol_name),
    protocolFact("Feature policy", protocol.feature_policy),
    protocolFact("Source", protocol.source_path),
    protocolFact("Primary claim", protocol.primary_claim),
    protocolFact("Calibration mode", protocol.calibration_report_mode),
  ].join("");

  const replicationFacts = [
    protocolFact("Replicas", protocol.n_replicas || replicaList),
    protocolFact("Base seed", protocol.base_seed),
    protocolFact("Replica jobs", protocol.replica_jobs),
    protocolFact("Resume completed", protocol.resume_completed === null ? "" : protocol.resume_completed ? "yes" : "no"),
    protocolFact("Replica names", replicaList),
  ].join("");

  const pdbbindFacts = [
    protocolFact("Split strategy", protocol.pdbbind_split_strategy),
    protocolFact("Train / val / test", formatSplitSizes(protocol)),
    protocolFact("Objective", protocol.pdbbind_objective_metric),
    protocolFact("Optuna trials", protocol.pdbbind_trials),
    protocolFact("Epochs", protocol.pdbbind_epochs),
  ].join("");

  const dudezFacts = [
    protocolFact("Primary metric", protocol.dudez_primary_metric),
    protocolFact("BEDROC α", protocol.dudez_bedroc_alpha),
    protocolFact("Scaling", protocol.dudez_scaling_strategy),
    protocolFact("Optuna trials", protocol.dudez_trials),
    protocolFact("Epochs", protocol.dudez_epochs),
  ].join("");

  const stageFacts = (protocol.stage_names || []).length
    ? protocol.stage_names.map((name) => protocolFact("Stage", name)).join("")
    : protocolFact("Stages", "—");

  const activePolicy = state.selectedStudy?.policy_name || state.selectedStudy?.study_name || "";
  const variants = (protocol.ablation_variants || []).map((variant) => `
    <span class="protocol-variant${variant === activePolicy ? " active" : ""}">${escapeHtml(variant)}</span>
  `).join("");

  const variantCard = variants
    ? `
      <section class="protocol-card protocol-card-full">
        <h3>Ablation variants</h3>
        <div class="protocol-variants">${variants}</div>
      </section>
    `
    : "";

  const noteFacts = (protocol.notes || []).map((note) => protocolFact("Note", note)).join("");
  const noteCard = noteFacts ? protocolCard("Artifacts", noteFacts, true) : "";

  content.innerHTML = [
    protocolCard("Overview", overviewFacts),
    protocolCard("Replication", replicationFacts),
    protocolCard("PDBbind", pdbbindFacts),
    protocolCard("DUDEz", dudezFacts),
    protocolCard("Pipeline", stageFacts),
    variantCard,
    noteCard,
  ].join("");
}

function pathBasename(path) {
  let text = String(path || "");
  while (text.length > 0) {
    const last = text.charAt(text.length - 1);
    if (last !== "/" && last !== "\\\\") break;
    text = text.slice(0, -1);
  }
  const slash = text.lastIndexOf("/");
  const backslash = text.lastIndexOf("\\\\");
  const index = slash > backslash ? slash : backslash;
  return index >= 0 ? text.slice(index + 1) : text;
}

function compactSplitSummary(context) {
  const summary = context.pdbbind_split_summary || context.pdbbind_split_strategy || "—";
  const strategy = summary.split("·")[0].trim().replace(/_/g, " ");
  const sizes = summary.match(/(\\d+)%/g);
  if (!sizes || sizes.length < 3) {
    return summary.length > 32 ? `${summary.slice(0, 29)}…` : summary;
  }
  return `${strategy} ${sizes.map((part) => part.replace("%", "")).join("/")}`;
}

function summarizeBaselineSources(sources) {
  if (!sources.length) return null;
  const names = sources.map((source) => pathBasename(source.path));
  const label = names.length > 1 ? `${names[0]} (+${names.length - 1})` : names[0];
  const title = sources.map((source) => {
    const stamp = source.modified_at ? new Date(source.modified_at).toLocaleString() : "";
    return stamp ? `${source.path} · ${stamp}` : String(source.path);
  }).join("\\n");
  return { label, title };
}

function renderRunContext(payload) {
  const context = payload.run_context;
  const strip = $("run-context-items");
  if (!strip) return;
  if (!context) {
    strip.innerHTML = "";
    bindRunContextMarquee();
    return;
  }
  const splitShort = compactSplitSummary(context);
  const baseline = summarizeBaselineSources(context.baseline_sources || []);
  const items = [
    ["Repl", `${context.detected_replica_count}/${context.planned_replica_count}`],
    ["Split", splitShort],
    ["Rank", "DUDEz test"],
    ["Reg", "PDBbind val/test"],
    ["BEDROC", context.dudez_bedroc_alpha != null ? `α=${numeric(context.dudez_bedroc_alpha)}` : "—"],
    ["EF", "1%, 5%"],
  ];
  if (baseline) items.push(["CSV", baseline.label, baseline.title]);
  strip.innerHTML = items.map(([label, value, title]) => `
    <div class="run-context-item${label === "CSV" ? " path-item" : ""}">
      <strong>${escapeHtml(label)}</strong>
      <span${title ? ` title="${escapeHtml(title)}"` : ""}>${escapeHtml(String(value))}</span>
    </div>
  `).join("");
  bindRunContextMarquee();
}

let runContextMarqueeObserver = null;

function resetRunContextMarqueeTrack() {
  const track = $("run-context-items");
  if (!track) return;
  track.style.transition = "";
  track.style.transform = "translateX(0)";
}

function bindRunContextMarquee() {
  const viewport = $("run-context-scroll");
  const track = $("run-context-items");
  if (!viewport || !track) return;

  resetRunContextMarqueeTrack();

  const syncOverflow = () => {
    resetRunContextMarqueeTrack();
    viewport.classList.toggle("is-overflowing", track.scrollWidth > viewport.clientWidth + 1);
  };

  syncOverflow();

  if (runContextMarqueeObserver) {
    runContextMarqueeObserver.disconnect();
    runContextMarqueeObserver = null;
  }
  if (typeof ResizeObserver !== "undefined") {
    runContextMarqueeObserver = new ResizeObserver(syncOverflow);
    runContextMarqueeObserver.observe(viewport);
    runContextMarqueeObserver.observe(track);
  }

  viewport.onmouseenter = () => {
    if (!viewport.classList.contains("is-overflowing")) return;
    const distance = track.scrollWidth - viewport.clientWidth;
    if (distance <= 0) return;
    const seconds = Math.max(6, distance / 42);
    track.style.transition = `transform ${seconds}s linear`;
    track.style.transform = `translateX(-${distance}px)`;
  };
  viewport.onmouseleave = () => {
    track.style.transition = "transform 0.35s ease-out";
    track.style.transform = "translateX(0)";
  };
}

function renderIssues(payload) {
  const issues = payload.issues || [];
  const panel = $("issue-panel");
  panel.hidden = issues.length === 0;
  $("issue-summary").textContent = issues.length ? `${issues.length} issue${issues.length === 1 ? "" : "s"}` : "";
  $("issue-list").innerHTML = issues.map((issue) => `
    <div class="issue-item">
      <strong>${issue.message}</strong>
      <div class="path">${issue.path}</div>
    </div>
  `).join("");
}

function renderWorkspace(payload) {
  state.workspace = payload;
  restoreSelectedStudyFromPersisted();
  const rootNode = $("root-label");
  if (rootNode) {
    rootNode.textContent = pathBasename(payload.root);
    rootNode.title = payload.root || "";
  }
  $("study-count").textContent = payload.study_count;
  $("completed-count").textContent = payload.completed_count;
  $("failed-count").textContent = payload.failed_count;
  $("missing-count").textContent = payload.missing_count;
  renderRunContext(payload);
  renderIssues(payload);
  renderProtocolPanel();
  renderGlobalControls();
  renderComparisonTable();
  renderComparisonCharts();
  renderCrossValidationPanel();
  if (state.selectedStudy) {
    const selected = allStudies().find((study) => study.study_name === state.selectedStudy.study_name);
    renderDetail(selected || null);
  } else {
    renderDetail(null);
  }
  persistUiState();
}

async function refresh() {
  $("health-dot").className = "dot pending";
  $("health-label").textContent = "Loading";
  try {
    const health = await api("/health");
    const payload = await api("/api/ocscore-workspace");
    $("health-dot").className = "dot ok";
    $("health-label").textContent = health.dashboard_model;
    renderWorkspace(payload);
  } catch (error) {
    $("health-dot").className = "dot error";
    $("health-label").textContent = "Error";
    toast(error.message || String(error));
  }
}

$("refresh").addEventListener("click", refresh);
loadPersistedUiState();
bindThemeToggle();
bindAppTabs();
bindCollapsibleZones();
uiStateHydrated = true;
setActiveTab("ablation");
refresh();
"""

# Functions
###############################################################################
## Public ##


def is_workbench_web_asset_path(path: str) -> bool:
    '''Return whether a path is served by embedded Workbench web assets.

    Parameters
    ----------
    path : str
        Request path.

    Returns
    -------
    bool
        True when the path is an embedded web asset route.
    '''

    return path in WORKBENCH_WEB_ROUTES


def _workbench_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workbench_favicon_path() -> Path | None:
    '''Return the browser tab icon path when available beside the package root.'''

    candidate = _workbench_repo_root() / "ocdocker_small_logo.png"
    return candidate if candidate.is_file() else None


def _workbench_brand_logo_path() -> Path | None:
    '''Return the in-page OCDocker wordmark path when available beside the package root.'''

    candidates = (
        _workbench_repo_root() / "OCDocker.png",
        Path(__file__).resolve().parent / "assets" / "OCDocker.png",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_workbench_web_asset(path: str) -> tuple[str, bytes]:
    '''Build one embedded Workbench browser asset.

    Parameters
    ----------
    path : str
        Request path.

    Returns
    -------
    tuple[str, bytes]
        Content type and response body.
    '''

    route = WORKBENCH_WEB_INDEX_ROUTE if path == "/app/" else path
    if route == "/app":
        return "text/html; charset=utf-8", _INDEX_HTML.encode("utf-8")
    if route == "/app.css":
        return "text/css; charset=utf-8", _STYLE_CSS.encode("utf-8")
    if route == "/app.js":
        return "text/javascript; charset=utf-8", _SCRIPT_JS.encode("utf-8")
    if route == WORKBENCH_WEB_FAVICON_ROUTE:
        favicon_path = _workbench_favicon_path()
        if favicon_path is None:
            raise KeyError(f"Unknown Workbench web asset: {path}")
        return "image/png", favicon_path.read_bytes()
    if route == WORKBENCH_WEB_BRAND_LOGO_ROUTE:
        brand_logo_path = _workbench_brand_logo_path()
        if brand_logo_path is None:
            raise KeyError(f"Unknown Workbench web asset: {path}")
        return "image/png", brand_logo_path.read_bytes()
    raise KeyError(f"Unknown Workbench web asset: {path}")


__all__ = [
    "WORKBENCH_WEB_INDEX_ROUTE",
    "WORKBENCH_WEB_FAVICON_ROUTE",
    "WORKBENCH_WEB_BRAND_LOGO_ROUTE",
    "WORKBENCH_WEB_ROUTES",
    "build_workbench_web_asset",
    "is_workbench_web_asset_path",
]
