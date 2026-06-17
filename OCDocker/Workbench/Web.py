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
        <section class="panel plot-panel">
          <div class="panel-head"><h2>Metric Scatter</h2></div>
          <div id="plot" class="plot-area"></div>
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
  min-height: 360px;
  padding: 14px;
}

.plot-area svg {
  width: 100%;
  min-height: 320px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfaf7;
}

.plot-point {
  fill: var(--accent);
  stroke: #ffffff;
  stroke-width: 1.5;
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
  .detail-grid {
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

  function renderSvgScatter(container, payload) {
    clearNode(container);
    const trace = (payload.data || [])[0] || {};
    const xs = (trace.x || []).map(Number);
    const ys = (trace.y || []).map(Number);
    const labels = trace.text || [];
    if (!xs.length || !ys.length) {
      empty(container, "No scatter points found.");
      return;
    }
    const width = 820;
    const height = 320;
    const pad = 44;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spreadX = maxX - minX || 1;
    const spreadY = maxY - minY || 1;
    const sx = (value) => pad + ((value - minX) / spreadX) * (width - pad * 2);
    const sy = (value) => height - pad - ((value - minY) / spreadY) * (height - pad * 2);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", payload.title || "Metric scatter");

    const axis = document.createElementNS(svg.namespaceURI, "path");
    axis.setAttribute("d", `M ${pad} ${pad} V ${height - pad} H ${width - pad}`);
    axis.setAttribute("fill", "none");
    axis.setAttribute("stroke", "#667085");
    axis.setAttribute("stroke-width", "1.5");
    svg.appendChild(axis);

    xs.forEach((x, index) => {
      const point = document.createElementNS(svg.namespaceURI, "circle");
      point.setAttribute("class", "plot-point");
      point.setAttribute("cx", sx(x));
      point.setAttribute("cy", sy(ys[index]));
      point.setAttribute("r", "5");
      const title = document.createElementNS(svg.namespaceURI, "title");
      title.textContent = `${labels[index] || "run"}: ${numericText(x)}, ${numericText(ys[index])}`;
      point.appendChild(title);
      svg.appendChild(point);
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
    await Promise.all([renderLeaderboard(), renderPareto(), renderPlot(), renderReport()]);
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
