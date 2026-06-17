#!/usr/bin/env python3

# Description
###############################################################################
'''
Embedded browser assets for the read-only Workbench GUI.
'''

# Imports
###############################################################################
from __future__ import annotations

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
WORKBENCH_WEB_ROUTES: Final[tuple[str, ...]] = (
    "/app",
    "/app/",
    "/app.css",
    "/app.js",
)

_INDEX_HTML: Final[str] = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCDocker Workbench</title>
  <link rel="stylesheet" href="/app.css">
</head>
<body>
  <header class="topbar">
    <div class="identity">
      <span class="product">OCDocker Workbench</span>
      <h1>Decision Console</h1>
    </div>
    <div class="toolbar" aria-label="Workbench controls">
      <label class="field compact" for="depth-input">
        <span>Depth</span>
        <input id="depth-input" type="number" min="0" max="12" value="6">
      </label>
      <button id="refresh-button" type="button">Refresh</button>
    </div>
  </header>

  <section class="connection" aria-live="polite">
    <span id="health-dot" class="dot pending"></span>
    <span id="health-label">Connecting</span>
    <span id="root-label" class="root-label"></span>
  </section>

  <nav class="tabs" aria-label="Workbench views">
    <button class="tab active" type="button" data-view="overview">Overview</button>
    <button class="tab" type="button" data-view="runs">Runs</button>
    <button class="tab" type="button" data-view="metrics">Metrics</button>
    <button class="tab" type="button" data-view="decision">Decision</button>
    <button class="tab" type="button" data-view="artifacts">Artifacts</button>
    <button class="tab" type="button" data-view="report">Report</button>
  </nav>

  <main>
    <section id="view-overview" class="view active" data-view="overview">
      <div class="stat-grid">
        <article class="stat"><span>Runs</span><strong id="run-count">-</strong></article>
        <article class="stat"><span>Results</span><strong id="result-count">-</strong></article>
        <article class="stat"><span>Issues</span><strong id="issue-count">-</strong></article>
        <article class="stat"><span>Missing Artifacts</span><strong id="missing-artifact-count">-</strong></article>
      </div>
      <div class="dashboard-grid">
        <section class="panel">
          <div class="panel-head"><h2>Status</h2></div>
          <div id="status-counts" class="chip-list"></div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Recent Runs</h2></div>
          <div id="recent-runs" class="table-wrap"></div>
        </section>
      </div>
    </section>

    <section id="view-runs" class="view" data-view="runs">
      <section class="panel wide">
        <div class="panel-head"><h2>Run Inventory</h2></div>
        <div id="inventory-runs" class="table-wrap"></div>
      </section>
      <section class="panel wide run-detail-panel">
        <div class="panel-head"><h2>Run Detail</h2><span id="run-detail-title" class="muted">No run selected</span></div>
        <div id="run-detail" class="detail-body"><div class="empty">Select a run to inspect status, metrics, artifacts, and logs.</div></div>
      </section>
    </section>

    <section id="view-metrics" class="view" data-view="metrics">
      <section class="panel wide">
        <div class="panel-head"><h2>Metric Catalog</h2></div>
        <div id="metric-catalog" class="table-wrap"></div>
      </section>
    </section>

    <section id="view-decision" class="view" data-view="decision">
      <div class="control-row">
        <label class="field" for="metric-select"><span>Rank Metric</span><select id="metric-select"></select></label>
        <label class="field" for="mode-select"><span>Mode</span><select id="mode-select"><option value="max">max</option><option value="min">min</option></select></label>
        <label class="field" for="x-select"><span>X Metric</span><select id="x-select"></select></label>
        <label class="field" for="y-select"><span>Y Metric</span><select id="y-select"></select></label>
      </div>
      <div class="dashboard-grid decision-grid">
        <section class="panel">
          <div class="panel-head"><h2>Leaderboard</h2></div>
          <div id="leaderboard" class="table-wrap"></div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Pareto Front</h2></div>
          <div id="pareto" class="table-wrap"></div>
        </section>
        <section class="panel wide">
          <div class="panel-head"><h2>Ablations</h2><span id="ablation-summary" class="muted"></span></div>
          <div id="ablation-table" class="table-wrap"></div>
          <div class="plot-split">
            <div class="plot-block">
              <div class="inline-head"><h3>Ablation Delta</h3><span id="ablation-delta-summary" class="muted"></span></div>
              <div id="ablation-delta-plot" class="mini-plot"></div>
            </div>
            <div class="plot-block">
              <div class="inline-head"><h3>Metric Direction Heatmap</h3><span id="ablation-heatmap-summary" class="muted"></span></div>
              <div id="ablation-heatmap" class="mini-plot"></div>
            </div>
          </div>
        </section>
        <section class="panel plot-panel">
          <div class="panel-head"><h2>Metric Scatter</h2></div>
          <div id="plot" class="plot-area"></div>
        </section>
        <section class="panel wide evidence-panel">
          <div class="panel-head"><h2>Evidence Explorer</h2><span id="evidence-summary" class="muted"></span></div>
          <div class="evidence-grid">
            <div class="plot-block">
              <div class="inline-head"><h3>Performance Profile</h3><span id="evidence-performance-summary" class="muted"></span></div>
              <div id="evidence-performance-plot" class="mini-plot"></div>
            </div>
            <div class="plot-block">
              <div class="inline-head"><h3>Optuna Trace</h3><span id="evidence-optuna-summary" class="muted"></span></div>
              <div id="evidence-optuna-plot" class="mini-plot"></div>
            </div>
            <div class="plot-block">
              <div class="inline-head"><h3>SHAP Importance</h3><span id="evidence-shap-summary" class="muted"></span></div>
              <div id="evidence-shap-plot" class="mini-plot"></div>
            </div>
          </div>
          <div class="evidence-gallery-wrap">
            <div class="inline-head"><h3>Figure Comparison</h3><span id="evidence-gallery-summary" class="muted"></span></div>
            <div class="evidence-controls">
              <label class="field" for="evidence-gallery-group"><span>Figure Set</span><select id="evidence-gallery-group"></select></label>
              <label class="field compact" for="evidence-gallery-limit"><span>Limit</span><input id="evidence-gallery-limit" type="number" min="2" max="80" value="24"></label>
            </div>
            <div id="evidence-gallery" class="evidence-gallery"></div>
          </div>
          <div id="evidence-files" class="table-wrap"></div>
        </section>
      </div>
    </section>

    <section id="view-artifacts" class="view" data-view="artifacts">
      <section class="panel wide">
        <div class="panel-head"><h2>Artifact Index</h2><span id="artifact-summary" class="muted"></span></div>
        <div id="artifact-index" class="table-wrap"></div>
      </section>
    </section>

    <section id="view-report" class="view" data-view="report">
      <section class="panel wide">
        <div class="panel-head"><h2>Analysis Findings</h2><span id="report-summary" class="muted"></span></div>
        <div id="report-findings" class="finding-list"></div>
      </section>
    </section>
  </main>

  <div id="evidence-lightbox" class="evidence-lightbox" hidden>
    <div class="evidence-lightbox-shell" role="dialog" aria-modal="true" aria-labelledby="evidence-lightbox-title">
      <div class="evidence-lightbox-head">
        <div>
          <strong id="evidence-lightbox-title"></strong>
          <span id="evidence-lightbox-subtitle"></span>
        </div>
        <button id="evidence-lightbox-close" type="button">Close</button>
      </div>
      <img id="evidence-lightbox-image" alt="">
      <dl id="evidence-lightbox-meta" class="evidence-lightbox-meta"></dl>
    </div>
  </div>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script src="/app.js" defer></script>
</body>
</html>
"""

_STYLE_CSS: Final[str] = """* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  --bg: #f5f3ee;
  --surface: #ffffff;
  --surface-soft: #eef6f2;
  --ink: #1f2933;
  --muted: #667085;
  --line: #d8d2c6;
  --accent: #087f7b;
  --accent-dark: #065f5b;
  --warn: #b36b00;
  --danger: #b42318;
}

body {
  margin: 0;
  min-width: 320px;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.topbar,
.connection,
.tabs,
main {
  padding-left: clamp(16px, 4vw, 40px);
  padding-right: clamp(16px, 4vw, 40px);
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 88px;
  border-bottom: 1px solid var(--line);
  background: #fbfaf7;
}

.product {
  display: block;
  color: var(--accent-dark);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1,
h2 {
  margin: 0;
  letter-spacing: 0;
}

h1 {
  font-size: clamp(24px, 3vw, 34px);
  line-height: 1.1;
}

h2 {
  font-size: 15px;
}

.toolbar,
.control-row,
.connection,
.tabs,
.panel-head {
  display: flex;
  align-items: center;
}

.toolbar,
.control-row {
  gap: 10px;
  flex-wrap: wrap;
}

button,
select,
input {
  min-height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
}

button {
  cursor: pointer;
  padding: 0 14px;
  font-weight: 700;
}

button:hover,
.tab.active {
  border-color: var(--accent);
  color: var(--accent-dark);
}

button:disabled {
  cursor: progress;
  opacity: 0.65;
}

.field {
  display: grid;
  gap: 4px;
  min-width: 160px;
}

.field.compact {
  min-width: 96px;
}

.field span,
.muted {
  color: var(--muted);
  font-size: 12px;
}

.field input,
.field select {
  width: 100%;
  padding: 0 10px;
}

.connection {
  gap: 10px;
  min-height: 42px;
  border-bottom: 1px solid var(--line);
  background: var(--surface);
}

.dot {
  width: 10px;
  height: 10px;
  flex: 0 0 10px;
  border-radius: 50%;
  background: var(--warn);
}

.dot.ok {
  background: var(--accent);
}

.dot.error {
  background: var(--danger);
}

.root-label {
  min-width: 0;
  overflow: hidden;
  color: var(--muted);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tabs {
  gap: 8px;
  overflow-x: auto;
  min-height: 54px;
  border-bottom: 1px solid var(--line);
  background: #fbfaf7;
}

.tab {
  flex: 0 0 auto;
  min-width: 92px;
}

main {
  padding-top: 18px;
  padding-bottom: 40px;
}

.view {
  display: none;
}

.view.active {
  display: block;
}

.stat-grid,
.dashboard-grid {
  display: grid;
  gap: 14px;
}

.stat-grid {
  grid-template-columns: repeat(4, minmax(140px, 1fr));
  margin-bottom: 14px;
}

.stat,
.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}

.stat {
  min-height: 84px;
  padding: 14px;
}

.stat span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}

.stat strong {
  display: block;
  margin-top: 6px;
  font-size: 28px;
  line-height: 1;
}

.dashboard-grid {
  grid-template-columns: minmax(240px, 0.7fr) minmax(320px, 1.3fr);
}

.decision-grid {
  grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr);
  margin-top: 14px;
}

.panel {
  min-width: 0;
  overflow: hidden;
}

.panel.wide,
.plot-panel {
  grid-column: 1 / -1;
}

.panel-head {
  justify-content: space-between;
  gap: 12px;
  min-height: 48px;
  padding: 12px 14px;
  border-bottom: 1px solid var(--line);
}

.table-wrap {
  overflow: auto;
  max-height: 480px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  padding: 10px 12px;
  border-bottom: 1px solid #ece7dc;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}

th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--surface-soft);
  color: #344054;
  font-size: 12px;
}

td.numeric {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

tr.clickable-row {
  cursor: pointer;
}

tr.clickable-row:focus-visible td,
tr.clickable-row:hover td {
  background: #f4faf7;
  outline: none;
}

.detail-body {
  padding: 14px;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
  gap: 14px;
}

.detail-section {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfaf7;
}

.detail-section.wide {
  grid-column: 1 / -1;
}

.detail-section h3 {
  margin: 0;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  font-size: 13px;
}

.detail-section-body {
  padding: 12px;
}

.key-values {
  display: grid;
  grid-template-columns: minmax(90px, 0.35fr) minmax(0, 0.65fr);
  gap: 8px 12px;
  margin: 0;
}

.key-values dt {
  color: var(--muted);
}

.key-values dd {
  min-width: 0;
  margin: 0;
  overflow-wrap: anywhere;
}

.command-preview,
.log-text {
  overflow: auto;
  max-height: 220px;
  margin: 0;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #ffffff;
  color: var(--ink);
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap;
}

.log-list {
  display: grid;
  gap: 10px;
}

.log-item {
  display: grid;
  gap: 6px;
}

.log-item strong {
  overflow-wrap: anywhere;
}

.chip-list {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  padding: 14px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 30px;
  padding: 4px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfaf7;
}

.chip strong {
  font-variant-numeric: tabular-nums;
}

.plot-area {
  min-height: 540px;
  padding: 14px;
  overflow-x: auto;
}

.plot-area svg,
.mini-plot svg {
  width: 100%;
  min-height: 320px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfaf7;
}

.plot-area svg {
  min-width: 980px;
  min-height: 520px;
}

.plot-split,
.evidence-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(460px, 1fr));
  gap: 14px;
  padding: 14px;
  border-top: 1px solid var(--line);
}

.evidence-grid {
  grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
}

.evidence-gallery-wrap {
  padding: 14px;
  border-top: 1px solid var(--line);
}

.evidence-controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}

.evidence-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 720px), 1fr));
  gap: 16px;
}

.evidence-thumb {
  display: grid;
  gap: 10px;
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfaf7;
}

.evidence-thumb-head {
  display: grid;
  gap: 4px;
}

.evidence-thumb-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
}

.evidence-thumb-title strong,
.evidence-thumb span,
.evidence-file-path {
  overflow-wrap: anywhere;
}

.evidence-thumb-title strong {
  font-size: 15px;
}

.evidence-thumb-title span,
.evidence-thumb span,
.evidence-file-path {
  color: var(--muted);
  font-size: 12px;
}

.evidence-open {
  display: block;
  width: 100%;
  min-height: 0;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: zoom-in;
}

.evidence-open img {
  width: 100%;
  height: clamp(420px, 46vw, 680px);
  border: 1px solid #ece7dc;
  border-radius: 6px;
  background: #ffffff;
  object-fit: contain;
}

.evidence-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 8px;
  margin: 0;
}

.evidence-meta div {
  min-width: 0;
}

.evidence-meta dt {
  color: var(--muted);
  font-size: 11px;
}

.evidence-meta dd {
  margin: 0;
  overflow-wrap: anywhere;
  font-size: 12px;
  font-weight: 700;
}

.evidence-lightbox[hidden] {
  display: none;
}

.evidence-lightbox {
  position: fixed;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 18px;
  background: rgb(31 41 51 / 72%);
}

.evidence-lightbox-shell {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 12px;
  width: min(96vw, 1600px);
  max-height: 96vh;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}

.evidence-lightbox-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.evidence-lightbox-head div {
  display: grid;
  gap: 4px;
  min-width: 0;
}

.evidence-lightbox-head strong,
.evidence-lightbox-head span {
  overflow-wrap: anywhere;
}

.evidence-lightbox-head span {
  color: var(--muted);
  font-size: 12px;
}

.evidence-lightbox img {
  width: 100%;
  max-height: 78vh;
  border: 1px solid #ece7dc;
  border-radius: 6px;
  background: #ffffff;
  object-fit: contain;
}

.evidence-lightbox-meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px 14px;
  margin: 0;
  max-height: 120px;
  overflow: auto;
}

.evidence-lightbox-meta dt {
  color: var(--muted);
  font-size: 11px;
}

.evidence-lightbox-meta dd {
  margin: 0;
  overflow-wrap: anywhere;
  font-size: 12px;
  font-weight: 700;
}

.plot-block {
  min-width: 0;
}

.inline-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  min-height: 30px;
  margin-bottom: 8px;
}

.inline-head h3 {
  margin: 0;
  font-size: 13px;
  font-weight: 800;
}

.mini-plot {
  min-height: 390px;
  overflow-x: auto;
}

.mini-plot svg {
  min-width: 960px;
  min-height: 380px;
}

.plot-point {
  fill: var(--accent);
  stroke: #ffffff;
  stroke-width: 1.5;
}

.plot-point-label,
.plot-tick-label,
.plot-axis-label {
  fill: #344054;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.plot-point-label {
  paint-order: stroke;
  stroke: #fbfaf7;
  stroke-width: 3px;
  font-size: 13px;
  font-weight: 800;
}

.plot-tick-label {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.plot-axis-label {
  font-size: 14px;
  font-weight: 800;
}

.plot-axis {
  fill: none;
  stroke: #667085;
  stroke-width: 1.5;
}

.plot-grid-line {
  stroke: #e6dfd2;
  stroke-width: 1;
}

.plot-zero-line {
  stroke: #667085;
  stroke-dasharray: 4 4;
  stroke-width: 1.2;
}

.plot-bar.performance {
  fill: #0b7d7f;
}

.plot-bar.shap {
  fill: #315c9c;
}

.plot-bar.improved,
.heat-cell.improved {
  fill: #1b8a5a;
}

.plot-bar.regressed,
.heat-cell.regressed {
  fill: #b54708;
}

.plot-bar.neutral,
.heat-cell.neutral {
  fill: #8a94a6;
}

.heat-cell.missing {
  fill: #e6dfd2;
}

.heat-label {
  fill: #101828;
  font-size: 10px;
  font-weight: 700;
}

.heat-value {
  fill: #ffffff;
  font-size: 9px;
  font-variant-numeric: tabular-nums;
}

.heat-cell.missing + .heat-value {
  fill: #344054;
}

.finding-list {
  display: grid;
  gap: 10px;
  padding: 14px;
}

.finding {
  display: grid;
  gap: 4px;
  padding: 12px;
  border: 1px solid var(--line);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  background: #fbfaf7;
}

.finding.warning {
  border-left-color: var(--warn);
}

.finding.error {
  border-left-color: var(--danger);
}

.finding strong {
  font-size: 14px;
}

.toast {
  position: fixed;
  right: 18px;
  bottom: 18px;
  max-width: min(460px, calc(100vw - 36px));
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: 0 10px 30px rgb(31 41 51 / 16%);
  opacity: 0;
  transform: translateY(8px);
  transition: opacity 160ms ease, transform 160ms ease;
  pointer-events: none;
}

.toast.visible {
  opacity: 1;
  transform: translateY(0);
}

.empty {
  padding: 16px;
  color: var(--muted);
}

@media (max-width: 860px) {
  .topbar {
    align-items: flex-start;
    flex-direction: column;
    padding-top: 18px;
    padding-bottom: 18px;
  }

  .stat-grid,
  .dashboard-grid,
  .decision-grid,
  .detail-grid,
  .plot-split,
  .evidence-grid {
    grid-template-columns: 1fr;
  }

  .control-row .field {
    min-width: min(100%, 220px);
  }
}
"""

_APP_JS: Final[str] = """(() => {
  const state = {
    metrics: [],
    selectedMetric: "",
    selectedRunTarget: "",
    xMetric: "",
    yMetric: "",
  };

  const byId = (id) => document.getElementById(id);
  const numericText = (value) => {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return String(value);
    }
    return Math.abs(number) >= 1000 || Math.abs(number) < 0.001 && number !== 0
      ? number.toExponential(3)
      : number.toLocaleString(undefined, { maximumFractionDigits: 5 });
  };
  const valueText = (value) => value === null || value === undefined || value === "" ? "-" : String(value);
  const pathLeaf = (value) => valueText(value).split("/").filter(Boolean).pop() || valueText(value);
  const pathSuffix = (value) => {
    const leaf = pathLeaf(value).toLowerCase();
    const index = leaf.lastIndexOf(".");
    return index >= 0 ? leaf.slice(index) : "";
  };
  const maxDepth = () => Number.parseInt(byId("depth-input").value || "6", 10);

  async function api(path, params = {}) {
    const url = new URL(path, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.filter((item) => item !== "" && item !== null && item !== undefined).forEach((item) => url.searchParams.append(key, item));
      } else if (value !== "" && value !== null && value !== undefined) {
        url.searchParams.set(key, value);
      }
    });
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || response.statusText);
    }
    return payload;
  }

  function evidenceImageUrl(entry) {
    const url = new URL("/api/evidence-asset", window.location.origin);
    url.searchParams.set("path", entry.path);
    url.searchParams.set("max_depth", maxDepth());
    url.searchParams.set("source_depth", "5");
    return url.toString();
  }

  function setBusy(isBusy) {
    byId("refresh-button").disabled = isBusy;
    byId("refresh-button").textContent = isBusy ? "Refreshing" : "Refresh";
  }

  function showToast(message) {
    const toast = byId("toast");
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 4500);
  }

  function setHealth(kind, label, root = "") {
    const dot = byId("health-dot");
    dot.className = `dot ${kind}`;
    byId("health-label").textContent = label;
    byId("root-label").textContent = root;
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function empty(container, message) {
    clearNode(container);
    const node = document.createElement("div");
    node.className = "empty";
    node.textContent = message;
    container.appendChild(node);
  }

  function table(container, columns, rows, options = {}) {
    clearNode(container);
    if (!rows.length) {
      empty(container, options.emptyMessage || "No rows found.");
      return;
    }
    const tableNode = document.createElement("table");
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((column) => {
      const th = document.createElement("th");
      th.textContent = column.label;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    tableNode.appendChild(thead);

    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      if (options.onRowClick) {
        tr.className = "clickable-row";
        tr.tabIndex = 0;
        tr.addEventListener("click", () => options.onRowClick(row));
        tr.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            options.onRowClick(row);
          }
        });
      }
      columns.forEach((column) => {
        const td = document.createElement("td");
        if (column.numeric) {
          td.className = "numeric";
        }
        td.textContent = column.render ? column.render(row) : valueText(row[column.key]);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    tableNode.appendChild(tbody);
    container.appendChild(tableNode);
  }

  function chips(container, payload) {
    clearNode(container);
    const entries = Object.entries(payload || {});
    if (!entries.length) {
      empty(container, "No counts found.");
      return;
    }
    entries.forEach(([name, count]) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = document.createElement("span");
      label.textContent = name;
      const strong = document.createElement("strong");
      strong.textContent = count;
      chip.append(label, strong);
      container.appendChild(chip);
    });
  }

  function messageBlock(message) {
    const node = document.createElement("div");
    node.className = "empty";
    node.textContent = message;
    return node;
  }

  function detailSection(title, options = {}) {
    const section = document.createElement("section");
    section.className = options.wide ? "detail-section wide" : "detail-section";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const body = document.createElement("div");
    body.className = "detail-section-body";
    section.append(heading, body);
    return { section, body };
  }

  function keyValues(items) {
    const list = document.createElement("dl");
    list.className = "key-values";
    items.forEach(([label, value]) => {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = valueText(value);
      list.append(term, description);
    });
    return list;
  }

  function appendDetailTable(parent, columns, rows, emptyMessage) {
    const wrapper = document.createElement("div");
    wrapper.className = "table-wrap";
    parent.appendChild(wrapper);
    table(wrapper, columns, rows, { emptyMessage });
  }

  function showView(viewName) {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.view === viewName);
    });
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    const view = byId(`view-${viewName}`);
    if (view) {
      view.classList.add("active");
    }
  }

  function renderRunDetail(payload) {
    const container = byId("run-detail");
    clearNode(container);
    byId("run-detail-title").textContent = `${payload.run_id} - ${payload.status}`;

    const status = payload.status_report || {};
    const summary = payload.result_summary || {};
    const logs = payload.log_preview ? payload.log_preview.logs || [] : [];
    const artifacts = summary.artifacts || status.artifacts || [];
    const issues = payload.issues || [];

    const grid = document.createElement("div");
    grid.className = "detail-grid";
    container.appendChild(grid);

    const statusSection = detailSection("Status");
    statusSection.body.appendChild(keyValues([
      ["Run", payload.run_id],
      ["Name", payload.name],
      ["Status", payload.status],
      ["Type", payload.spec_type],
      ["Manifest", pathLeaf(payload.manifest_path)],
      ["Workspace", status.workspace],
      ["PID", status.pid || ""],
      ["PID Alive", status.pid_alive === null || status.pid_alive === undefined ? "unknown" : status.pid_alive ? "yes" : "no"],
    ]));
    grid.appendChild(statusSection.section);

    const commandSection = detailSection("Command");
    const command = document.createElement("pre");
    command.className = "command-preview";
    command.textContent = (status.command || []).join(" ") || "No command recorded.";
    commandSection.body.appendChild(command);
    grid.appendChild(commandSection.section);

    const metricsSection = detailSection("Metrics", { wide: true });
    appendDetailTable(metricsSection.body, [
      { label: "Metric", key: "name" },
      { label: "Value", key: "value", numeric: true, render: (row) => numericText(row.value) },
    ], Object.entries(summary.metrics || {}).map(([name, value]) => ({ name, value })), "No result metrics found.");
    grid.appendChild(metricsSection.section);

    const artifactsSection = detailSection("Artifacts", { wide: true });
    appendDetailTable(artifactsSection.body, [
      { label: "Name", key: "name" },
      { label: "Kind", key: "kind", render: (row) => valueText(row.kind || row.role) },
      { label: "Role", key: "role" },
      { label: "Exists", key: "exists", render: (row) => row.exists ? "yes" : "no" },
      { label: "Path", key: "path", render: (row) => pathLeaf(row.path) },
    ], artifacts, "No artifacts declared.");
    grid.appendChild(artifactsSection.section);

    const logsSection = detailSection("Logs", { wide: true });
    if (!logs.length) {
      logsSection.body.appendChild(messageBlock("No logs declared."));
    } else {
      const logList = document.createElement("div");
      logList.className = "log-list";
      logs.forEach((log) => {
        const item = document.createElement("article");
        item.className = "log-item";
        const title = document.createElement("strong");
        title.textContent = `${log.name || pathLeaf(log.path)} - ${log.exists ? "present" : "missing"}`;
        const preview = document.createElement("pre");
        preview.className = "log-text";
        preview.textContent = log.text || log.error || "No preview text.";
        item.append(title, preview);
        logList.appendChild(item);
      });
      logsSection.body.appendChild(logList);
    }
    grid.appendChild(logsSection.section);

    if (issues.length) {
      const issuesSection = detailSection("Issues", { wide: true });
      appendDetailTable(issuesSection.body, [
        { label: "Path", key: "path", render: (row) => pathLeaf(row.path) },
        { label: "Message", key: "message" },
      ], issues, "No detail issues found.");
      grid.appendChild(issuesSection.section);
    }
  }

  async function loadRunDetail(target) {
    if (!target) {
      return;
    }
    state.selectedRunTarget = target;
    byId("run-detail-title").textContent = "Loading";
    empty(byId("run-detail"), "Loading run detail...");
    try {
      const payload = await api("/api/run-detail", { target, lines: 80, max_bytes: 65536 });
      renderRunDetail(payload);
    } catch (error) {
      byId("run-detail-title").textContent = "Could not load run";
      empty(byId("run-detail"), error.message || String(error));
      showToast(error.message || String(error));
    }
  }

  function openRunDetail(row) {
    const target = row.manifest_path || row.workspace;
    if (!target) {
      showToast("Run does not include a manifest path.");
      return;
    }
    showView("runs");
    loadRunDetail(target);
  }

  function numericMetrics() {
    return state.metrics.filter((metric) => metric.numeric_count > 0).map((metric) => metric.metric_name);
  }

  function inferMode(metricName) {
    const lowered = metricName.toLowerCase();
    return ["loss", "error", "rmse", "mse", "mae", "time", "cost"].some((hint) => lowered.includes(hint)) ? "min" : "max";
  }

  function fillSelect(select, names, fallbackIndex = 0) {
    const previous = select.value;
    clearNode(select);
    names.forEach((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    });
    if (names.includes(previous)) {
      select.value = previous;
    } else if (names.length) {
      select.value = names[Math.min(fallbackIndex, names.length - 1)];
    }
  }

  function updateMetricControls() {
    const names = numericMetrics();
    fillSelect(byId("metric-select"), names, 0);
    fillSelect(byId("x-select"), names, 0);
    fillSelect(byId("y-select"), names, 1);
    state.selectedMetric = byId("metric-select").value;
    state.xMetric = byId("x-select").value;
    state.yMetric = byId("y-select").value;
    byId("mode-select").value = state.selectedMetric ? inferMode(state.selectedMetric) : "max";
  }

  function renderOverview(payload) {
    byId("run-count").textContent = payload.run_count ?? 0;
    byId("result-count").textContent = payload.result_manifest_count ?? 0;
    byId("issue-count").textContent = payload.issue_count ?? 0;
    byId("missing-artifact-count").textContent = payload.missing_artifact_count ?? 0;
    chips(byId("status-counts"), payload.status_counts);
    table(byId("recent-runs"), [
      { label: "Run", key: "run_id" },
      { label: "Status", key: "status" },
      { label: "Type", key: "spec_type" },
      { label: "Updated", key: "updated_at", render: (row) => valueText(row.updated_at).replace("T", " ").replace("Z", "") },
    ], payload.recent_runs || [], { onRowClick: openRunDetail });
  }

  function renderInventory(payload) {
    table(byId("inventory-runs"), [
      { label: "Run", key: "run_id" },
      { label: "Name", key: "name" },
      { label: "Status", key: "status" },
      { label: "Type", key: "spec_type" },
      { label: "Artifacts", key: "artifact_count", numeric: true },
      { label: "Missing", key: "missing_artifacts", numeric: true, render: (row) => (row.missing_artifacts || []).length },
      { label: "Workspace", key: "workspace", render: (row) => pathLeaf(row.workspace) },
    ], payload.runs || [], { onRowClick: openRunDetail });
  }

  function renderCatalog(payload) {
    state.metrics = payload.metrics || [];
    updateMetricControls();
    table(byId("metric-catalog"), [
      { label: "Metric", key: "metric_name" },
      { label: "Observed", key: "observed_count", numeric: true },
      { label: "Numeric", key: "numeric_count", numeric: true },
      { label: "Missing", key: "missing_count", numeric: true },
      { label: "Min", key: "min_value", numeric: true, render: (row) => numericText(row.min_value) },
      { label: "Max", key: "max_value", numeric: true, render: (row) => numericText(row.max_value) },
      { label: "Mean", key: "mean_value", numeric: true, render: (row) => numericText(row.mean_value) },
    ], state.metrics);
  }

  function renderArtifacts(payload) {
    byId("artifact-summary").textContent = `${payload.existing_artifact_count || 0} existing / ${payload.missing_artifact_count || 0} missing`;
    table(byId("artifact-index"), [
      { label: "Run", key: "run_id" },
      { label: "Name", key: "name" },
      { label: "Kind", key: "kind" },
      { label: "Role", key: "role" },
      { label: "Exists", key: "exists", render: (row) => row.exists ? "yes" : "no" },
      { label: "Path", key: "path", render: (row) => pathLeaf(row.path) },
    ], (payload.entries || []).slice(0, 200));
  }

  async function renderLeaderboard() {
    const metric = byId("metric-select").value;
    if (!metric) {
      empty(byId("leaderboard"), "No numeric metrics found.");
      return;
    }
    const payload = await api("/api/leaderboard", { metric, mode: byId("mode-select").value, max_depth: maxDepth() });
    table(byId("leaderboard"), [
      { label: "Rank", key: "rank", numeric: true },
      { label: "Run", key: "run_id" },
      { label: "Status", key: "status" },
      { label: metric, key: "metric_value", numeric: true, render: (row) => numericText(row.metric_value) },
      { label: "Missing Artifacts", key: "missing_artifact_count", numeric: true },
    ], payload.ranked_entries || []);
  }

  async function renderPareto() {
    const names = [byId("x-select").value, byId("y-select").value].filter(Boolean);
    const uniqueNames = [...new Set(names)];
    if (!uniqueNames.length) {
      empty(byId("pareto"), "No numeric metrics found.");
      return;
    }
    const payload = await api("/api/pareto", { objective: uniqueNames.map((name) => `${name}:${inferMode(name)}`), max_depth: maxDepth() });
    table(byId("pareto"), [
      { label: "Run", key: "run_id" },
      { label: "Status", key: "status" },
      { label: "Metrics", key: "metric_values", render: (row) => Object.entries(row.metric_values || {}).map(([key, value]) => `${key}: ${numericText(value)}`).join(", ") },
    ], payload.front_entries || []);
  }

  function svgNode(svg, tagName, attributes = {}) {
    const node = document.createElementNS(svg.namespaceURI || "http://www.w3.org/2000/svg", tagName);
    Object.entries(attributes).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        node.setAttribute(key, String(value));
      }
    });
    return node;
  }

  function svgText(svg, value, attributes = {}) {
    const node = svgNode(svg, "text", attributes);
    node.textContent = valueText(value);
    svg.appendChild(node);
    return node;
  }

  function axisTitle(payload, axisName, fallback) {
    const axis = payload.layout && payload.layout[axisName] ? payload.layout[axisName] : {};
    const title = axis.title;
    if (typeof title === "string" && title.trim()) {
      return title;
    }
    if (title && typeof title.text === "string" && title.text.trim()) {
      return title.text;
    }
    return fallback;
  }

  function shortLabel(value, maxLength = 18) {
    const label = valueText(value);
    if (label.length <= maxLength) {
      return label;
    }
    return `${label.slice(0, Math.max(1, maxLength - 3))}...`;
  }

  function paddedDomain(values) {
    const finite = values.map(Number).filter(Number.isFinite);
    if (!finite.length) {
      return [-1, 1];
    }
    const min = Math.min(...finite);
    const max = Math.max(...finite);
    if (min === max) {
      const pad = Math.max(1, Math.abs(min) * 0.1);
      return [min - pad, max + pad];
    }
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }

  function ticks(min, max, count = 5) {
    if (!Number.isFinite(min) || !Number.isFinite(max) || count < 2) {
      return [];
    }
    return Array.from({ length: count }, (_item, index) => min + ((max - min) * index) / (count - 1));
  }

  function shouldPlaceLabel(placed, x, y, minDistance = 72) {
    return placed.every((point) => Math.hypot(point.x - x, point.y - y) >= minDistance);
  }

  function renderSvgScatter(container, payload) {
    clearNode(container);
    const trace = (payload.data || [])[0] || {};
    const rawXs = trace.x || [];
    const rawYs = trace.y || [];
    const points = rawXs.map((x, index) => ({
      x: Number(x),
      y: Number(rawYs[index]),
      label: (trace.text || [])[index] || "run",
    })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (!points.length) {
      empty(container, "No scatter points found.");
      return;
    }

    const metricNames = payload.metric_names || [];
    const xLabel = axisTitle(payload, "xaxis", metricNames[0] || "X metric");
    const yLabel = axisTitle(payload, "yaxis", metricNames[1] || "Y metric");
    const width = 1200;
    const height = 560;
    const plot = { left: 112, right: 54, top: 82, bottom: 86 };
    const xDomain = paddedDomain(points.map((point) => point.x));
    const yDomain = paddedDomain(points.map((point) => point.y));
    const xRange = width - plot.left - plot.right;
    const yRange = height - plot.top - plot.bottom;
    const sx = (value) => plot.left + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * xRange;
    const sy = (value) => height - plot.bottom - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * yRange;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `${payload.title || "Metric scatter"}; x axis ${xLabel}; y axis ${yLabel}`);

    svgText(svg, `Y: ${shortLabel(yLabel, 84)}`, { class: "plot-axis-label", x: plot.left, y: 30 });
    svgText(svg, `X: ${shortLabel(xLabel, 92)}`, { class: "plot-axis-label", x: plot.left + xRange / 2, y: height - 24, "text-anchor": "middle" });

    ticks(xDomain[0], xDomain[1]).forEach((tick) => {
      const x = sx(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: x, y1: plot.top, x2: x, y2: height - plot.bottom }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x, y: height - plot.bottom + 24, "text-anchor": "middle" });
    });
    ticks(yDomain[0], yDomain[1]).forEach((tick) => {
      const y = sy(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: plot.left, y1: y, x2: width - plot.right, y2: y }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x: plot.left - 12, y: y + 4, "text-anchor": "end" });
    });

    svg.appendChild(svgNode(svg, "path", {
      class: "plot-axis",
      d: `M ${plot.left} ${plot.top} V ${height - plot.bottom} H ${width - plot.right}`,
    }));

    const placedLabels = [];
    const labelBudget = Math.min(10, Math.max(3, Math.floor(points.length / 2)));
    points.forEach((point) => {
      const cx = sx(point.x);
      const cy = sy(point.y);
      const marker = svgNode(svg, "circle", { class: "plot-point", cx, cy, r: 7 });
      const title = svgNode(svg, "title");
      title.textContent = `${point.label}: ${xLabel} ${numericText(point.x)}, ${yLabel} ${numericText(point.y)}`;
      marker.appendChild(title);
      svg.appendChild(marker);
      if (placedLabels.length < labelBudget && shouldPlaceLabel(placedLabels, cx, cy)) {
        const anchor = cx > width - 260 ? "end" : "start";
        const labelX = anchor === "end" ? cx - 10 : cx + 10;
        const labelY = Math.max(plot.top + 16, Math.min(height - plot.bottom - 10, cy - 10));
        svgText(svg, shortLabel(point.label, 26), { class: "plot-point-label", x: labelX, y: labelY, "text-anchor": anchor });
        placedLabels.push({ x: labelX, y: labelY });
      }
    });
    container.appendChild(svg);
  }

  async function renderPlot() {
    const xMetric = byId("x-select").value;
    const yMetric = byId("y-select").value;
    if (!xMetric || !yMetric || xMetric === yMetric) {
      empty(byId("plot"), "Select two different numeric metrics.");
      return;
    }
    const payload = await api("/api/plot", { kind: "scatter", x_metric: xMetric, y_metric: yMetric, max_depth: maxDepth() });
    renderSvgScatter(byId("plot"), payload);
  }

  function ablationMetric(row, metricName) {
    return (row.metrics || []).find((item) => item.metric_name === metricName) || null;
  }

  function ablationMetricValue(row, metricName, key) {
    const metric = ablationMetric(row, metricName);
    return metric ? metric[key] : null;
  }

  function ablationMetricClass(metric) {
    if (!metric || metric.direction === "incomplete") {
      return "missing";
    }
    if (metric.improved) {
      return "improved";
    }
    if (metric.regressed) {
      return "regressed";
    }
    return "neutral";
  }

  function renderAblationDeltaPlot(container, payload, metricName) {
    clearNode(container);
    const rows = (payload.candidates || [])
      .map((candidate) => ({ candidate, metric: ablationMetric(candidate, metricName) }))
      .filter((row) => row.metric && Number.isFinite(Number(row.metric.delta)))
      .slice(0, 14);
    if (!rows.length) {
      empty(container, "No numeric ablation deltas found.");
      return;
    }
    const width = 1180;
    const rowHeight = 34;
    const height = Math.max(390, 104 + rows.length * rowHeight);
    const plot = { left: 310, right: 132, top: 56, bottom: 54 };
    const maxAbs = Math.max(...rows.map((row) => Math.abs(Number(row.metric.delta)))) || 1;
    const zeroX = plot.left + (width - plot.left - plot.right) / 2;
    const sx = (value) => zeroX + (Number(value) / maxAbs) * ((width - plot.left - plot.right) / 2);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Ablation delta for ${metricName} against ${payload.baseline_run_id || "baseline"}`);

    [-maxAbs, 0, maxAbs].forEach((tick) => {
      const x = sx(tick);
      svg.appendChild(svgNode(svg, "line", { class: tick === 0 ? "plot-zero-line" : "plot-grid-line", x1: x, y1: plot.top - 18, x2: x, y2: height - plot.bottom + 8 }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x, y: height - 18, "text-anchor": "middle" });
    });
    svgText(svg, `${shortLabel(metricName, 72)} delta vs ${payload.baseline_run_id || "baseline"}`, { class: "plot-axis-label", x: zeroX, y: 28, "text-anchor": "middle" });

    rows.forEach((row, index) => {
      const delta = Number(row.metric.delta);
      const y = plot.top + index * rowHeight;
      const x0 = sx(0);
      const x1 = sx(delta);
      const x = Math.min(x0, x1);
      const widthValue = Math.max(2, Math.abs(x1 - x0));
      svgText(svg, shortLabel(row.candidate.policy_name || row.candidate.run_id, 34), { class: "plot-tick-label", x: plot.left - 12, y: y + 17, "text-anchor": "end" });
      const bar = svgNode(svg, "rect", {
        class: `plot-bar ${ablationMetricClass(row.metric)}`,
        x,
        y,
        width: widthValue,
        height: 18,
        rx: 3,
      });
      const title = svgNode(svg, "title");
      title.textContent = `${row.candidate.policy_name}: baseline ${numericText(row.metric.baseline_value)}, candidate ${numericText(row.metric.candidate_value)}, delta ${numericText(row.metric.delta)}`;
      bar.appendChild(title);
      svg.appendChild(bar);
      svgText(svg, numericText(delta), { class: "plot-tick-label", x: width - plot.right + 10, y: y + 17 });
    });
    container.appendChild(svg);
  }

  function renderAblationHeatmap(container, payload, metricNames) {
    clearNode(container);
    const rows = (payload.candidates || []).slice(0, 14);
    const metrics = metricNames.filter(Boolean).slice(0, 4);
    if (!rows.length || !metrics.length) {
      empty(container, "No ablation metrics found.");
      return;
    }
    const width = 1180;
    const plot = { left: 280, right: 36, top: 62, bottom: 24 };
    const rowHeight = 34;
    const cellWidth = (width - plot.left - plot.right) / metrics.length;
    const height = Math.max(280, plot.top + rows.length * rowHeight + plot.bottom);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Ablation metric direction heatmap for ${metrics.join(", ")}`);

    metrics.forEach((metricName, index) => {
      svgText(svg, shortLabel(metricName, 18), {
        class: "heat-label",
        x: plot.left + index * cellWidth + cellWidth / 2,
        y: 34,
        "text-anchor": "middle",
      });
    });

    rows.forEach((candidate, rowIndex) => {
      const y = plot.top + rowIndex * rowHeight;
      svgText(svg, shortLabel(candidate.policy_name || candidate.run_id, 34), { class: "heat-label", x: plot.left - 12, y: y + 22, "text-anchor": "end" });
      metrics.forEach((metricName, colIndex) => {
        const metric = ablationMetric(candidate, metricName);
        const x = plot.left + colIndex * cellWidth;
        const cell = svgNode(svg, "rect", {
          class: `heat-cell ${ablationMetricClass(metric)}`,
          x: x + 3,
          y: y + 3,
          width: Math.max(8, cellWidth - 6),
          height: rowHeight - 7,
          rx: 4,
        });
        const title = svgNode(svg, "title");
        title.textContent = metric
          ? `${candidate.policy_name}: ${metricName}, baseline ${numericText(metric.baseline_value)}, candidate ${numericText(metric.candidate_value)}, delta ${numericText(metric.delta)}`
          : `${candidate.policy_name}: ${metricName} missing`;
        cell.appendChild(title);
        svg.appendChild(cell);
        svgText(svg, metric ? numericText(metric.delta) : "-", {
          class: "heat-value",
          x: x + cellWidth / 2,
          y: y + 22,
          "text-anchor": "middle",
        });
      });
    });
    container.appendChild(svg);
  }

  function clearAblationPlots(message) {
    byId("ablation-delta-summary").textContent = "";
    byId("ablation-heatmap-summary").textContent = "";
    empty(byId("ablation-delta-plot"), message);
    empty(byId("ablation-heatmap"), message);
  }

  async function renderAblations() {
    const selected = [byId("metric-select").value, byId("x-select").value, byId("y-select").value]
      .filter(Boolean)
      .filter((value, index, array) => array.indexOf(value) === index)
      .slice(0, 3);
    if (!selected.length) {
      byId("ablation-summary").textContent = "";
      empty(byId("ablation-table"), "No numeric metrics found.");
      clearAblationPlots("No numeric metrics found.");
      return;
    }
    let payload;
    try {
      payload = await api("/api/ablations", {
        metric: selected.map((name) => `${name}:${inferMode(name)}`),
        max_depth: maxDepth(),
      });
    } catch (error) {
      byId("ablation-summary").textContent = "";
      empty(byId("ablation-table"), error.message || "No ablations found.");
      clearAblationPlots(error.message || "No ablations found.");
      return;
    }
    byId("ablation-summary").textContent = `${payload.candidate_count || 0} policies vs ${payload.baseline_run_id || "reference"}`;
    const primary = selected[0];
    table(byId("ablation-table"), [
      { label: "Policy", key: "policy_name" },
      { label: "Run", key: "run_id" },
      { label: "Score", key: "net_score", numeric: true },
      { label: "Improved", key: "improved_count", numeric: true },
      { label: "Regressed", key: "regressed_count", numeric: true },
      { label: primary, key: "metrics", numeric: true, render: (row) => numericText(ablationMetricValue(row, primary, "candidate_value")) },
      { label: "Delta", key: "metrics", numeric: true, render: (row) => numericText(ablationMetricValue(row, primary, "delta")) },
      { label: "%", key: "metrics", numeric: true, render: (row) => numericText(ablationMetricValue(row, primary, "percent_delta")) },
    ], payload.candidates || []);
    byId("ablation-delta-summary").textContent = primary;
    byId("ablation-heatmap-summary").textContent = `${selected.length} metrics`;
    renderAblationDeltaPlot(byId("ablation-delta-plot"), payload, primary);
    renderAblationHeatmap(byId("ablation-heatmap"), payload, selected);
  }

  const plotColors = ["#0b7d7f", "#b54708", "#315c9c", "#1b8a5a", "#7a3ea1", "#6b7280"];

  function evidenceMetricHint(metricName) {
    const lowered = valueText(metricName).toLowerCase();
    if (lowered.includes("bedroc")) {
      return "bedroc";
    }
    if (lowered.includes("pr") && lowered.includes("auc")) {
      return "pr-auc";
    }
    if (lowered.includes("roc") || lowered.includes("auc")) {
      return "roc-auc";
    }
    if (lowered.includes("rmse")) {
      return "rmse";
    }
    if (lowered.includes("mae")) {
      return "mae";
    }
    if (lowered.includes("r2")) {
      return "r2";
    }
    return "";
  }

  function pickEvidenceMetric(points, selectedMetric) {
    const metricNames = [...new Set((points || []).map((point) => point.metric_name).filter(Boolean))];
    if (!metricNames.length) {
      return "";
    }
    const hint = evidenceMetricHint(selectedMetric);
    if (hint) {
      const match = metricNames.find((name) => name.toLowerCase().includes(hint));
      if (match) {
        return match;
      }
    }
    return metricNames[0];
  }

  function renderEvidencePerformancePlot(container, points, selectedMetric) {
    clearNode(container);
    const metricName = pickEvidenceMetric(points, selectedMetric);
    const rows = (points || [])
      .filter((point) => point.metric_name === metricName && Number.isFinite(Number(point.value)))
      .sort((left, right) => Number(right.value) - Number(left.value))
      .slice(0, 16);
    if (!rows.length) {
      empty(container, "No performance table points found.");
      return "";
    }
    const width = 1180;
    const rowHeight = 34;
    const height = Math.max(390, 106 + rows.length * rowHeight);
    const plot = { left: 330, right: 96, top: 58, bottom: 48 };
    const values = rows.map((row) => Number(row.value));
    const maxValue = Math.max(...values, 1);
    const minValue = Math.min(0, Math.min(...values));
    const span = maxValue - minValue || 1;
    const sx = (value) => plot.left + ((Number(value) - minValue) / span) * (width - plot.left - plot.right);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", `Performance profile for ${metricName}`);
    svgText(svg, `${shortLabel(metricName, 76)} performance profile`, { class: "plot-axis-label", x: plot.left, y: 30 });
    ticks(minValue, maxValue, 4).forEach((tick) => {
      const x = sx(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: x, y1: plot.top - 12, x2: x, y2: height - plot.bottom }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x, y: height - 18, "text-anchor": "middle" });
    });
    rows.forEach((row, index) => {
      const y = plot.top + index * rowHeight;
      const value = Number(row.value);
      const label = `${row.policy_name || row.run_id} / ${row.label}`;
      svgText(svg, shortLabel(label, 38), { class: "plot-tick-label", x: plot.left - 12, y: y + 20, "text-anchor": "end" });
      const bar = svgNode(svg, "rect", {
        class: "plot-bar performance",
        x: sx(Math.min(0, value)),
        y: y + 3,
        width: Math.max(2, Math.abs(sx(value) - sx(0))),
        height: 21,
        rx: 4,
      });
      const title = svgNode(svg, "title");
      title.textContent = `${label}: ${metricName} ${numericText(value)} from ${row.file_name || "evidence"}`;
      bar.appendChild(title);
      svg.appendChild(bar);
      svgText(svg, numericText(value), { class: "plot-tick-label", x: width - plot.right + 10, y: y + 20 });
    });
    container.appendChild(svg);
    return metricName;
  }

  function renderEvidenceTracePlot(container, points) {
    clearNode(container);
    const grouped = new Map();
    (points || []).forEach((point) => {
      if (!Number.isFinite(Number(point.trial)) || !Number.isFinite(Number(point.best_value))) {
        return;
      }
      const series = point.series || point.run_id || "trace";
      if (!grouped.has(series)) {
        grouped.set(series, []);
      }
      grouped.get(series).push(point);
    });
    const seriesRows = [...grouped.entries()].slice(0, 6).map(([series, rows]) => [series, rows.sort((left, right) => Number(left.trial) - Number(right.trial))]);
    if (!seriesRows.length) {
      empty(container, "No Optuna trial traces found.");
      return;
    }
    const all = seriesRows.flatMap(([_series, rows]) => rows);
    const width = 1180;
    const height = 390;
    const plot = { left: 88, right: 42, top: 62, bottom: 60 };
    const xDomain = paddedDomain(all.map((point) => Number(point.trial)));
    const yDomain = paddedDomain(all.map((point) => Number(point.best_value)));
    const sx = (value) => plot.left + ((Number(value) - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - plot.left - plot.right);
    const sy = (value) => height - plot.bottom - ((Number(value) - yDomain[0]) / (yDomain[1] - yDomain[0])) * (height - plot.top - plot.bottom);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Optuna best-value trace");
    svgText(svg, "Optuna best value by trial", { class: "plot-axis-label", x: plot.left, y: 30 });
    ticks(xDomain[0], xDomain[1], 5).forEach((tick) => {
      const x = sx(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: x, y1: plot.top, x2: x, y2: height - plot.bottom }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x, y: height - 22, "text-anchor": "middle" });
    });
    ticks(yDomain[0], yDomain[1], 5).forEach((tick) => {
      const y = sy(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: plot.left, y1: y, x2: width - plot.right, y2: y }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x: plot.left - 12, y: y + 4, "text-anchor": "end" });
    });
    seriesRows.forEach(([series, rows], index) => {
      const color = plotColors[index % plotColors.length];
      const pathData = rows.map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${sx(point.trial)} ${sy(point.best_value)}`).join(" ");
      svg.appendChild(svgNode(svg, "path", { d: pathData, fill: "none", stroke: color, "stroke-width": 2.4 }));
      svgText(svg, shortLabel(series, 30), { class: "plot-tick-label", x: plot.left + 170 * index, y: 50, fill: color });
    });
    svgText(svg, "Trial", { class: "plot-axis-label", x: plot.left + (width - plot.left - plot.right) / 2, y: height - 12, "text-anchor": "middle" });
    container.appendChild(svg);
  }

  function renderShapFeaturePlot(container, features) {
    clearNode(container);
    const rows = (features || []).filter((row) => Number.isFinite(Number(row.mean_abs_shap))).slice(0, 16);
    if (!rows.length) {
      empty(container, "No SHAP values found.");
      return;
    }
    const width = 1180;
    const rowHeight = 34;
    const height = Math.max(390, 104 + rows.length * rowHeight);
    const plot = { left: 390, right: 96, top: 56, bottom: 48 };
    const maxValue = Math.max(...rows.map((row) => Number(row.mean_abs_shap)), 1);
    const sx = (value) => plot.left + (Number(value) / maxValue) * (width - plot.left - plot.right);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Mean absolute SHAP feature importance");
    svgText(svg, "Mean absolute SHAP", { class: "plot-axis-label", x: plot.left, y: 28 });
    ticks(0, maxValue, 4).forEach((tick) => {
      const x = sx(tick);
      svg.appendChild(svgNode(svg, "line", { class: "plot-grid-line", x1: x, y1: plot.top - 12, x2: x, y2: height - plot.bottom }));
      svgText(svg, numericText(tick), { class: "plot-tick-label", x, y: height - 18, "text-anchor": "middle" });
    });
    rows.forEach((row, index) => {
      const y = plot.top + index * rowHeight;
      const value = Number(row.mean_abs_shap);
      svgText(svg, shortLabel(row.feature, 44), { class: "plot-tick-label", x: plot.left - 12, y: y + 20, "text-anchor": "end" });
      svg.appendChild(svgNode(svg, "rect", { class: "plot-bar shap", x: plot.left, y: y + 3, width: Math.max(2, sx(value) - plot.left), height: 21, rx: 4 }));
      svgText(svg, numericText(value), { class: "plot-tick-label", x: width - plot.right + 10, y: y + 20 });
    });
    container.appendChild(svg);
  }

  function isEvidenceImage(entry) {
    const kind = valueText(entry.kind);
    const suffix = valueText(entry.suffix || pathSuffix(entry.path)).toLowerCase();
    return (kind === "shap" || kind === "figure") && [".png", ".jpg", ".jpeg"].includes(suffix);
  }

  function renderEvidenceGallery(container, entries) {
    clearNode(container);
    const rows = (entries || []).filter(isEvidenceImage).slice(0, 24);
    if (!rows.length) {
      empty(container, "No previewable evidence figures found.");
      return 0;
    }
    rows.forEach((entry) => {
      const item = document.createElement("article");
      item.className = "evidence-thumb";
      const image = document.createElement("img");
      image.loading = "lazy";
      image.decoding = "async";
      image.alt = `${entry.role || entry.kind} ${pathLeaf(entry.path)}`;
      image.src = evidenceImageUrl(entry);
      const title = document.createElement("strong");
      title.textContent = pathLeaf(entry.path);
      const detail = document.createElement("span");
      detail.textContent = [entry.policy_name, entry.dataset, entry.role].filter(Boolean).join(" / ");
      item.append(image, title, detail);
      container.appendChild(item);
    });
    return rows.length;
  }

  async function renderEvidence() {
    let payload;
    try {
      payload = await api("/api/evidence", {
        max_depth: maxDepth(),
        source_depth: 5,
        max_entries: 240,
        max_csv_rows: 400,
        max_series: 6,
        max_shap_features: 30,
      });
    } catch (error) {
      byId("evidence-summary").textContent = "";
      empty(byId("evidence-performance-plot"), error.message || "No evidence found.");
      empty(byId("evidence-optuna-plot"), error.message || "No evidence found.");
      empty(byId("evidence-shap-plot"), error.message || "No evidence found.");
      byId("evidence-gallery-summary").textContent = "";
      empty(byId("evidence-gallery"), error.message || "No evidence found.");
      empty(byId("evidence-files"), error.message || "No evidence found.");
      return;
    }
    byId("evidence-summary").textContent = `${payload.evidence_count || 0} files / ${payload.issue_count || 0} issues`;
    const selectedMetric = byId("metric-select").value;
    const performanceMetric = renderEvidencePerformancePlot(byId("evidence-performance-plot"), payload.performance_points || [], selectedMetric);
    byId("evidence-performance-summary").textContent = performanceMetric || "";
    byId("evidence-optuna-summary").textContent = `${(payload.optimization_points || []).length} points`;
    renderEvidenceTracePlot(byId("evidence-optuna-plot"), payload.optimization_points || []);
    byId("evidence-shap-summary").textContent = `${(payload.shap_features || []).length} features`;
    renderShapFeaturePlot(byId("evidence-shap-plot"), payload.shap_features || []);
    const galleryCount = renderEvidenceGallery(byId("evidence-gallery"), payload.entries || []);
    byId("evidence-gallery-summary").textContent = `${galleryCount} preview images`;
    table(byId("evidence-files"), [
      { label: "Kind", key: "kind" },
      { label: "Role", key: "role" },
      { label: "Dataset", key: "dataset" },
      { label: "Policy", key: "policy_name" },
      { label: "File", key: "path", render: (row) => pathLeaf(row.path) },
      { label: "Metrics", key: "metric_names", render: (row) => (row.metric_names || []).slice(0, 4).join(", ") },
    ], (payload.entries || []).slice(0, 120));
  }

  async function renderReport() {
    const metric = byId("metric-select").value;
    const params = { max_depth: maxDepth(), top_n: 5 };
    if (metric) {
      params.leaderboard = `${metric}:${byId("mode-select").value}`;
    }
    const xMetric = byId("x-select").value;
    const yMetric = byId("y-select").value;
    const objectives = [xMetric, yMetric].filter(Boolean).filter((value, index, array) => array.indexOf(value) === index).map((name) => `${name}:${inferMode(name)}`);
    if (objectives.length) {
      params.objective = objectives;
    }
    const payload = await api("/api/report", params);
    const findings = payload.findings || [];
    byId("report-summary").textContent = `${findings.length} findings / ${payload.issue_count || 0} scan issues`;
    const container = byId("report-findings");
    clearNode(container);
    if (!findings.length) {
      empty(container, "No findings found.");
      return;
    }
    findings.slice(0, 80).forEach((finding) => {
      const item = document.createElement("article");
      item.className = `finding ${finding.severity || "info"}`;
      const title = document.createElement("strong");
      title.textContent = finding.title || finding.kind || "Finding";
      const message = document.createElement("span");
      message.textContent = finding.message || "";
      item.append(title, message);
      container.appendChild(item);
    });
  }

  async function renderDecisionViews() {
    await Promise.all([renderLeaderboard(), renderPareto(), renderPlot(), renderAblations(), renderEvidence(), renderReport()]);
  }

  async function refresh() {
    setBusy(true);
    try {
      const health = await api("/api/health");
      setHealth("ok", "Connected", health.root || "");
      const [overview, inventory, artifacts, catalog] = await Promise.all([
        api("/api/overview", { max_depth: maxDepth() }),
        api("/api/inventory", { max_depth: maxDepth() }),
        api("/api/artifacts", { max_depth: maxDepth() }),
        api("/api/metrics-catalog", { max_depth: maxDepth() }),
      ]);
      renderOverview(overview);
      renderInventory(inventory);
      renderArtifacts(artifacts);
      renderCatalog(catalog);
      await renderDecisionViews();
      if (state.selectedRunTarget) {
        await loadRunDetail(state.selectedRunTarget);
      }
    } catch (error) {
      setHealth("error", "Disconnected");
      showToast(error.message || String(error));
    } finally {
      setBusy(false);
    }
  }

  function bindTabs() {
    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => showView(button.dataset.view));
    });
  }

  function bindControls() {
    byId("refresh-button").addEventListener("click", refresh);
    byId("depth-input").addEventListener("change", refresh);
    ["metric-select", "mode-select", "x-select", "y-select"].forEach((id) => {
      byId(id).addEventListener("change", () => {
        renderDecisionViews().catch((error) => showToast(error.message || String(error)));
      });
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    bindTabs();
    bindControls();
    refresh();
  });
})();
"""

_ASSETS: Final[dict[str, tuple[str, str]]] = {
    "/app": ("text/html; charset=utf-8", _INDEX_HTML),
    "/app.css": ("text/css; charset=utf-8", _STYLE_CSS),
    "/app.js": ("text/javascript; charset=utf-8", _APP_JS),
}

# Functions
###############################################################################
## Private ##


def _normalize_web_path(path: str) -> str:
    '''Normalize a Workbench web asset request path.

    Parameters
    ----------
    path : str
        Raw request path.

    Returns
    -------
    str
        Normalized asset route.
    '''

    clean = str(path).split("?", 1)[0].split("#", 1)[0]
    if clean in {"", "/app/"}:
        return WORKBENCH_WEB_INDEX_ROUTE
    return clean


## Public ##


def is_workbench_web_asset_path(path: str) -> bool:
    '''Return whether a request path targets an embedded Workbench web asset.

    Parameters
    ----------
    path : str
        Raw request path.

    Returns
    -------
    bool
        True when the path can be served as a web asset.
    '''

    return _normalize_web_path(path) in _ASSETS


def build_workbench_web_asset(path: str) -> tuple[str, bytes]:
    '''Build one embedded Workbench web asset response.

    Parameters
    ----------
    path : str
        Raw request path.

    Returns
    -------
    tuple[str, bytes]
        Content type and encoded response body.
    '''

    route = _normalize_web_path(path)
    try:
        content_type, text = _ASSETS[route]
    except KeyError as exc:
        raise KeyError(f"Unknown Workbench web asset: {path}") from exc
    return content_type, text.encode("utf-8")


__all__ = [
    "WORKBENCH_WEB_INDEX_ROUTE",
    "WORKBENCH_WEB_ROUTES",
    "build_workbench_web_asset",
    "is_workbench_web_asset_path",
]
