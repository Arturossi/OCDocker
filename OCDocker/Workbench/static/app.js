const FIGURE_RENDER_LIMIT = 36;
const SHAP_RENDER_LIMIT = 8;
const TEST_SCOPE = "test";
const VALIDATION_SCOPE = "validation";
const COMBINED_SCOPE = "combined";
const MODEL_COMPARISON_ROLES = new Set(["performance", "cv_mean_std", "cv_heatmap", "cv_fold_comparison", "per_target_validation", "optuna"]);
const SELECTED_MODEL_ROLES = new Set(["shap", "shap_beeswarm", "shap_importance", "shap_dependence", "architecture"]);
const UI_STATE_KEY = "ocscore-workbench-ui";
const JOBS_POLL_INTERVAL_MS = 4000;
const MODEL_CATEGORY_COLORS = {
  full_ocscore: "#7FD4B8",
  ablation: "#9BD4EF",
  sf: "#F5C96A",
  consensus: "#E8B4D4",
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
    protocolSimilarity: false,
    cvTable: false,
    detailReplicas: false,
    detailCharts: false,
    detailFigures: false,
    jobsTable: false,
    jobsLogs: false,
  },
  plotCollapsed: {},
  comparisonSort: { key: "delta", direction: "desc" },
  detailReplicaSort: { key: "replica", direction: "asc" },
  comparisonBaseline: "internal",
  theme: "dark",
  ablationDesign: null,
  ablationDesignContext: null,
  ablationDesignPreview: null,
  ablationDesignPlan: null,
  ablationDesignFeatureCatalog: null,
  ablationDesignFeatureCatalogKey: "",
  ablationDesignFeaturesLoading: false,
  ablationDesignFeatureSelection: [],
  ablationDesignFeatureFilter: "",
  ablationDesignWildcardPattern: "",
  vsDesignContext: null,
  vsDesignPreview: null,
  vsDesignPlan: null,
  rankPlotExpandLabels: false,
  protocolSimilarity: null,
  protocolSimilarityLoading: false,
  protocolSimilarityReference: null,
  protocolSimilarityPlotPayload: null,
  jobs: [],
  jobsLoading: false,
  jobToken: "",
  selectedJobId: null,
  jobLogs: null,
  _persistedSelectedStudyName: null,
};
let ablationDesignPreviewTimer = null;
let protocolSimilarityTimer = null;
let jobsPollTimer = null;
let uiStateHydrated = false;
const $ = (id) => document.getElementById(id);

function readStoredUiStateRaw() {
  try {
    const local = localStorage.getItem(UI_STATE_KEY);
    if (local) return local;
    const session = sessionStorage.getItem(UI_STATE_KEY);
    if (session) {
      localStorage.setItem(UI_STATE_KEY, session);
      sessionStorage.removeItem(UI_STATE_KEY);
      return session;
    }
  } catch (_) {
    /* storage blocked or unavailable */
  }
  return null;
}

function writeStoredUiStateRaw(payload) {
  try {
    localStorage.setItem(UI_STATE_KEY, payload);
    sessionStorage.removeItem(UI_STATE_KEY);
    return;
  } catch (_) {
    /* fall back when localStorage is unavailable */
  }
  try {
    sessionStorage.setItem(UI_STATE_KEY, payload);
  } catch (_) {
    /* ignore quota errors */
  }
}

function loadPersistedUiState() {
  try {
    const raw = readStoredUiStateRaw();
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved.zoneCollapsed && typeof saved.zoneCollapsed === "object") {
      state.zoneCollapsed = { ...state.zoneCollapsed, ...saved.zoneCollapsed };
    }
    if (saved.plotCollapsed && typeof saved.plotCollapsed === "object") {
      state.plotCollapsed = { ...state.plotCollapsed, ...saved.plotCollapsed };
    }
    if (saved.resultScope) state.resultScope = saved.resultScope;
    if (saved.comparisonBaseline) state.comparisonBaseline = saved.comparisonBaseline;
    if (saved.ablationDesign && typeof saved.ablationDesign === "object") {
      state.ablationDesign = { ...defaultAblationDesign(), ...saved.ablationDesign };
    }
    if (typeof saved.ablationDesignWildcardPattern === "string") {
      state.ablationDesignWildcardPattern = saved.ablationDesignWildcardPattern;
    }
    if (typeof saved.rankPlotExpandLabels === "boolean") {
      state.rankPlotExpandLabels = saved.rankPlotExpandLabels;
    }
    if (typeof saved.protocolSimilarityReference === "string" && saved.protocolSimilarityReference) {
      state.protocolSimilarityReference = saved.protocolSimilarityReference;
    }
    if (saved.selectedMetric) state.selectedMetric = saved.selectedMetric;
    if (saved.comparisonSort) state.comparisonSort = saved.comparisonSort;
    if (saved.detailReplicaSort) state.detailReplicaSort = saved.detailReplicaSort;
    if (saved.activeTab) state.activeTab = saved.activeTab;
    if (saved.figureFilters) {
      state.figureFilters = { ...state.figureFilters, ...saved.figureFilters };
      if (state.figureFilters.role === "recommended") state.figureFilters.role = "all";
    }
    if (saved.theme === "light" || saved.theme === "dark") state.theme = saved.theme;
    if (typeof saved.jobToken === "string") state.jobToken = saved.jobToken;
    state._persistedSelectedStudyName = saved.selectedStudyName || null;
  } catch (_) {
    /* ignore corrupt saved state */
  }
}

function persistUiState() {
  if (!uiStateHydrated) return;
  writeStoredUiStateRaw(JSON.stringify({
    zoneCollapsed: state.zoneCollapsed,
    plotCollapsed: state.plotCollapsed,
    selectedStudyName: state.selectedStudy?.study_name || null,
    resultScope: state.resultScope,
    comparisonBaseline: state.comparisonBaseline,
    ablationDesign: readAblationDesignDraft(),
    ablationDesignWildcardPattern: state.ablationDesignWildcardPattern || "",
    rankPlotExpandLabels: Boolean(state.rankPlotExpandLabels),
    protocolSimilarityReference: state.protocolSimilarityReference || null,
    selectedMetric: state.selectedMetric,
    comparisonSort: state.comparisonSort,
    detailReplicaSort: state.detailReplicaSort,
    activeTab: state.activeTab,
    figureFilters: state.figureFilters,
    theme: state.theme,
    jobToken: state.jobToken || "",
  }));
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  state.theme = normalized;
  if (document.documentElement) {
    document.documentElement.setAttribute("data-theme", normalized);
  }
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.classList.toggle("theme-toggle--light", normalized === "light");
    toggle.classList.toggle("theme-toggle--dark", normalized === "dark");
    toggle.setAttribute("aria-checked", normalized === "dark" ? "true" : "false");
    toggle.title = normalized === "dark" ? "Switch to light appearance" : "Switch to dark appearance";
    toggle.setAttribute("aria-label", normalized === "dark" ? "Dark mode on" : "Light mode on");
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
  if (tabId === "design") void ensureAblationDesignContext();
  if (tabId === "vs-design") void ensureVsDesignContext();
  if (tabId === "jobs") {
    void loadJobs();
    startJobsPolling();
  } else {
    stopJobsPolling();
  }
  persistUiState();
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

function plotZoneBodyId(plotKey) {
  return `plot-zone-${slug(plotKey)}`;
}

function applyPlotCollapsedZone(zone, collapsed) {
  zone.classList.toggle("is-collapsed", collapsed);
  const toggle = zone.querySelector(".zone-toggle");
  if (toggle) toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
}

function setPlotCollapsed(plotKey, collapsed, zone = null) {
  state.plotCollapsed[plotKey] = collapsed;
  const target = zone || document.querySelector(`[data-plot-zone="${plotKey}"]`);
  if (target) applyPlotCollapsedZone(target, collapsed);
  persistUiState();
  if (!collapsed && target) {
    requestAnimationFrame(() => resizePlotlyHosts(target));
  }
}

function bindCollapsiblePlots(root = document) {
  const scope = root instanceof Element ? root : document;
  scope.querySelectorAll(".plot-collapsible[data-plot-zone]").forEach((zone) => {
    const plotKey = zone.dataset.plotZone;
    if (!plotKey) return;
    applyPlotCollapsedZone(zone, Boolean(state.plotCollapsed[plotKey]));
    const toggle = zone.querySelector(".zone-toggle");
    if (!toggle) return;
    toggle.addEventListener("click", () => {
      setPlotCollapsed(plotKey, !state.plotCollapsed[plotKey], zone);
    });
  });
}

function collapsiblePlotMarkup(plotKey, title, bodyHtml, options = {}) {
  const collapsed = Boolean(state.plotCollapsed[plotKey]);
  const zoneBodyId = plotZoneBodyId(plotKey);
  const subtitle = options.subtitle ? `<div class="scope-note">${escapeHtml(options.subtitle)}</div>` : "";
  const headActions = options.headActions || "";
  return `
    <div class="generated-plot plot-collapsible zone-block zone-collapsible${collapsed ? " is-collapsed" : ""}" data-plot-zone="${escapeHtml(plotKey)}">
      <div class="zone-head zone-head-split chart-head">
        <button type="button" class="zone-toggle" aria-expanded="${collapsed ? "false" : "true"}" aria-controls="${zoneBodyId}">
          <span class="zone-chevron" aria-hidden="true">▾</span>
          <span class="plot-zone-title">${escapeHtml(title)}</span>
        </button>
        ${headActions}
      </div>
      <div id="${zoneBodyId}" class="zone-body plot-zone-body">
        ${subtitle}
        ${bodyHtml}
      </div>
    </div>
  `;
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

async function apiPost(path, body = {}, headers = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function jobAuthHeaders() {
  return state.jobToken ? { Authorization: `Bearer ${state.jobToken}` } : {};
}

function jobStatusBadge(status) {
  const cssClass = status === "cancelled" ? "missing" : status;
  return `<span class="badge ${escapeHtml(cssClass)}">${escapeHtml(status)}</span>`;
}

function jobActionsCell(job) {
  const logsButton = `<button type="button" class="ghost-button job-logs" data-job-id="${escapeHtml(job.job_id)}">Logs</button>`;
  const cancelButton = job.status === "running"
    ? ` <button type="button" class="ghost-button job-cancel" data-job-id="${escapeHtml(job.job_id)}">Cancel</button>`
    : "";
  return `${logsButton}${cancelButton}`;
}

function bindJobActionButtons() {
  const target = $("jobs-table");
  target.querySelectorAll("button.job-logs").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => void loadJobLogs(button.dataset.jobId));
  });
  target.querySelectorAll("button.job-cancel").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => void cancelJob(button.dataset.jobId));
  });
}

function renderJobsSummary() {
  const summary = $("jobs-summary");
  if (!summary) return;
  const jobs = state.jobs || [];
  if (state.jobsLoading && jobs.length === 0) {
    summary.textContent = "Loading…";
    return;
  }
  const running = jobs.filter((job) => job.status === "running").length;
  summary.textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"} · ${running} running`;
}

function renderJobsTable() {
  renderJobsSummary();
  const target = $("jobs-table");
  const jobs = state.jobs || [];
  if (jobs.length === 0) {
    target.innerHTML = `<p class="muted">${state.jobsLoading ? "Loading jobs…" : "No jobs launched yet."}</p>`;
    return;
  }
  const headers = ["Job", "Kind", "Status", { label: "PID", numeric: true }, "Created", { label: "Exit", numeric: true }, "Actions"];
  const rows = jobs.map((job) => [
    { value: escapeHtml(job.job_id), title: (job.command || []).join(" ") },
    escapeHtml(job.kind),
    jobStatusBadge(job.status),
    { value: job.pid ?? "-", numeric: true },
    job.created_at ? new Date(job.created_at).toLocaleString() : "-",
    { value: job.return_code ?? "-", numeric: true },
    { value: jobActionsCell(job) },
  ]);
  table(target, headers, rows);
  bindJobActionButtons();
}

async function loadJobs() {
  state.jobsLoading = true;
  renderJobsTable();
  try {
    const payload = await api("/api/jobs");
    state.jobs = payload.jobs || [];
  } catch (error) {
    toast(error.message || String(error));
  } finally {
    state.jobsLoading = false;
    renderJobsTable();
  }
}

function startJobsPolling() {
  stopJobsPolling();
  jobsPollTimer = window.setInterval(() => void loadJobs(), JOBS_POLL_INTERVAL_MS);
}

function stopJobsPolling() {
  if (!jobsPollTimer) return;
  window.clearInterval(jobsPollTimer);
  jobsPollTimer = null;
}

async function launchJob() {
  const kind = $("jobs-launch-kind").value;
  const args = $("jobs-launch-args").value.split("\n").map((line) => line.trim()).filter(Boolean);
  const cwd = $("jobs-launch-cwd").value.trim();
  const button = $("jobs-launch");
  button.disabled = true;
  try {
    const body = { kind, args };
    if (cwd) body.cwd = cwd;
    const record = await apiPost("/api/jobs", body, jobAuthHeaders());
    toast(`Launched job ${record.job_id}`);
    $("jobs-launch-args").value = "";
    await loadJobs();
  } catch (error) {
    toast(error.message || String(error));
  } finally {
    button.disabled = false;
  }
}

async function cancelJob(jobId) {
  if (!jobId) return;
  try {
    await apiPost(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {}, jobAuthHeaders());
    toast(`Cancelled job ${jobId}`);
    await loadJobs();
  } catch (error) {
    toast(error.message || String(error));
  }
}

async function loadJobLogs(jobId) {
  if (!jobId) return;
  state.selectedJobId = jobId;
  try {
    state.jobLogs = await api(`/api/jobs/${encodeURIComponent(jobId)}/logs`);
  } catch (error) {
    toast(error.message || String(error));
    state.jobLogs = null;
  }
  renderJobLogs();
}

function renderJobLogs() {
  const panel = $("jobs-logs-panel");
  if (!state.selectedJobId || !state.jobLogs) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("jobs-logs-title").textContent = `Logs · ${state.selectedJobId}`;
  $("jobs-logs-stdout").textContent = state.jobLogs.stdout?.text || "(empty)";
  $("jobs-logs-stderr").textContent = state.jobLogs.stderr?.text || "(empty)";
}

function renderJobTokenStatus() {
  const node = $("jobs-token-status");
  if (!node) return;
  node.textContent = state.jobToken ? "Token configured" : "No token set — job launch/cancel will be rejected";
}

function bindJobsPanel() {
  $("jobs-token-input").value = state.jobToken || "";
  renderJobTokenStatus();
  $("jobs-token-save").addEventListener("click", () => {
    state.jobToken = $("jobs-token-input").value.trim();
    persistUiState();
    renderJobTokenStatus();
    toast(state.jobToken ? "Job token saved" : "Job token cleared");
  });
  $("jobs-token-clear").addEventListener("click", () => {
    state.jobToken = "";
    $("jobs-token-input").value = "";
    persistUiState();
    renderJobTokenStatus();
    toast("Job token cleared");
  });
  $("jobs-launch").addEventListener("click", () => void launchJob());
  $("jobs-logs-close").addEventListener("click", () => {
    state.selectedJobId = null;
    state.jobLogs = null;
    renderJobLogs();
  });
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
    .replace(/\b\w/g, (char) => char.toUpperCase());
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

const LEARNED_SF_BASELINE_NAMES = new Set(["lr_sf", "rf_sf", "xgb_sf", "lgbm_sf", "shuffled_lr_sf"]);

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

function replicaProgressSummary(study) {
  if (!study) {
    return { text: "—", sortValue: 0, title: "", className: "", numeric: false };
  }
  const expected = Number(study.expected_replica_count) || 0;
  const completed = Number(study.completed_count) || 0;
  const detected = Number(study.detected_replica_count) || 0;
  if (!expected) {
    return { text: "—", sortValue: 0, title: "", className: "", numeric: false };
  }
  const text = `${completed}/${expected}`;
  const titleParts = [`${completed} of ${expected} replicas completed`];
  if (detected > completed) {
    titleParts.push(`${detected} replica folders present`);
  }
  return {
    text,
    sortValue: completed,
    title: titleParts.join(" · "),
    className: completed >= expected ? "" : "replica-partial",
    numeric: true,
  };
}

function modelCell(item) {
  const label = modelDisplayName(item);
  const refBadge = isReferenceEntry(item) ? '<span class="role-badge reference">Reference</span>' : "";
  const tip = escapeHtml(modelDescription(item));
  const style = entryPaletteStyle(item);
  return `<span class="model-cell" title="${tip}"><span class="model-pill" style="${style}" data-entry-id="${escapeHtml(item.id)}" title="${tip}">${escapeHtml(label)}</span>${refBadge}</span>`;
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
  if (key === "replicas") return replicaProgressSummary(item.study).sortValue;
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

function isLearnedSfBaseline(item) {
  const family = item?.baseline_family || item?.entry?.baseline_family || "";
  if (family === "learned_sf") return true;
  const name = item?.entry?.baseline_name || item?.study?.baseline_name || "";
  return LEARNED_SF_BASELINE_NAMES.has(name);
}

function kindBadge(item) {
  const tip = escapeHtml(modelDescription(item));
  const learned = isLearnedSfBaseline(item);
  const synthClass = learned ? " synthesized-baseline" : "";
  const label = learned ? `${item.kind} · ML` : item.kind;
  const style = learned ? "" : ` style="${entryPaletteStyle(item)}"`;
  return `<span class="kind-badge${synthClass}"${style} title="${tip}">${escapeHtml(label)}</span>`;
}

function comparisonRowClass(item) {
  const classes = ["selectable"];
  if (isReferenceEntry(item)) classes.push("reference-row");
  return classes.join(" ");
}

function handleRankPlotRowClick(row) {
  if (!row) return;
  const item = comparisonEntries().find((entry) => {
    if (row.external) return entry.external && entry.entry?.baseline_name === row.study_name;
    return !entry.external && entry.study?.study_name === row.study_name;
  });
  if (item) handleComparisonEntryClick(item);
}

function rankPlotLabelToggleMarkup(plotKey) {
  const expanded = Boolean(state.rankPlotExpandLabels);
  return `<button type="button" class="ghost-button" data-rank-expand-labels="${escapeHtml(plotKey)}" aria-pressed="${expanded ? "true" : "false"}">${expanded ? "Compact labels" : "Expand labels"}</button>`;
}

function bindRankPlotLabelToggleButtons() {
  document.querySelectorAll("button[data-rank-expand-labels]").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => {
      state.rankPlotExpandLabels = !state.rankPlotExpandLabels;
      persistUiState();
      renderComparisonCharts();
    });
  });
}

function handleComparisonEntryClick(item) {
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
  if (family === "learned_sf") return "Learned SF";
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

function metricBaseName(name) {
  return String(name || "").replace(/^(test|validation)_/, "");
}

function resolveSelectedMetric(metrics) {
  if (!metrics.length) return null;
  if (state.selectedMetric && metrics.some((metric) => metric.name === state.selectedMetric)) {
    return state.selectedMetric;
  }
  if (state.selectedMetric) {
    const base = metricBaseName(state.selectedMetric);
    const scopedMatch = metrics.find((metric) => metricBaseName(metric.name) === base);
    if (scopedMatch) return scopedMatch.name;
  }
  return metrics[0].name;
}

function ensureSelectedMetric() {
  const metrics = scopedMetrics();
  if (!metrics.length) {
    state.selectedMetric = null;
    return null;
  }
  state.selectedMetric = resolveSelectedMetric(metrics);
  return state.selectedMetric;
}

function isReplicaAggregateCell(item, metric) {
  if (!item || item.external) return false;
  return Number(metric?.count) > 1;
}

function isLearnedSfBaselineCell(item) {
  return isLearnedSfBaseline(item);
}

function learnedSfMetricMark(item) {
  if (!isLearnedSfBaselineCell(item)) return "";
  return '<span class="metric-cell-synth" title="Learned SF classifier (lr / rf / xgb / …)">≈</span>';
}

function metricStatTitle(metric, item) {
  if (!metric) return "";
  if (isLearnedSfBaselineCell(item)) return "Learned SF classifier trained on docking-score columns";
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
  const synthMark = learnedSfMetricMark(item);
  if (!isReplicaAggregateCell(item, metric)) {
    const hint = isLearnedSfBaselineCell(item)
      ? "Learned SF classifier trained on docking-score columns"
      : item?.external ? "External baseline value" : Number(metric.count) === 1 ? "Single replica value" : "Value";
    const valueInner = synthMark
      ? `<span class="metric-stat-line">${synthMark}<span class="metric-mean">${valueText}</span></span>`
      : valueText;
    return `<span class="metric-stat metric-stat-value" title="${escapeHtml(hint)}">${valueInner}</span>`;
  }
  const std = Number(metric.std);
  const stdMarkup = Number.isFinite(std)
    ? `<span class="metric-std" title="Standard deviation (σ) across ${metric.count} replicas">σ ${numeric(std)}</span>`
    : "";
  return `<span class="metric-stat metric-stat-aggregate" title="${escapeHtml(metricStatTitle(metric, item))}"><span class="metric-stat-line">${synthMark}<span class="metric-cell-mu">μ</span><span class="metric-mean">${valueText}</span></span>${stdMarkup}</span>`;
}

function plotMetricLabel(metric) {
  if (typeof metric === "string") {
    return metricMeta(metric).label || metric;
  }
  return metric?.label || metric?.name || "metric";
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

function plotYAxisLeftMargin(maxLabelLen, options = {}) {
  const compact = Boolean(options.compact);
  const rank = Boolean(options.rank);
  const expand = Boolean(options.expand);
  const cap = rank ? (expand ? 560 : 480) : compact ? 156 : 280;
  const floor = rank ? 140 : compact ? 72 : 112;
  const perChar = rank ? 6.4 : compact ? 5.6 : 6.8;
  return Math.min(cap, Math.max(floor, maxLabelLen * perChar));
}

function buildRankCategoryBarTraces(plotRows, categories) {
  return categories.map((category) => {
    const entries = plotRows
      .map((row, index) => ({ row, index }))
      .filter(({ row }) => rankBarCategory(row) === category);
    return {
      type: "bar",
      orientation: "h",
      name: RANK_BAR_LABELS[category],
      legendgroup: category,
      y: entries.map(({ index }) => index),
      x: entries.map(({ row }) => row.value),
      customdata: entries.map(({ row, index }) => [
        row.std,
        row.count,
        row.hoverLabel || row.display,
        rankBarHoverKind(row),
        row.hoverLabel,
        row.study_name,
        row.external,
        row._sourceIndex ?? index,
      ]),
      error_x: {
        type: "data",
        array: entries.map(({ row }) => row.std),
        color: "#8899a6",
        thickness: 1.2,
        width: 5,
      },
      marker: {
        color: RANK_BAR_COLORS[category],
        line: { color: "#ffffff", width: 1 },
      },
      showlegend: true,
      hovertemplate: "<b>%{customdata[4]}</b><br>%{customdata[2]}<br><span style='color:#667085'>%{customdata[3]}</span><extra></extra>",
    };
  });
}

const RANK_LEGEND_ORDER = ["full_ocscore", "ablation", "sf", "consensus"];

function rankPlotNormalizeRows(rows) {
  return rows.map((row, sourceIndex) => {
    const full = String(row.label || "");
    return { ...row, plotLabel: full, hoverLabel: full, _sourceIndex: sourceIndex };
  });
}

function rankPlotCategoriesForRows(rows) {
  return RANK_LEGEND_ORDER.filter((category) => rows.some((row) => rankBarCategory(row) === category));
}

function rankPlotVisibleRows(plotRows, categoryVisibility) {
  if (!categoryVisibility) return plotRows;
  return plotRows.filter((row) => categoryVisibility[rankBarCategory(row)] !== false);
}

function rankPlotExportRows(plotRows, categoryVisibility) {
  return plotRows.filter((row) => categoryVisibility[rankBarCategory(row)] === true);
}

function buildRankPlotLayout(plotRows, metric, options = {}) {
  const expandLabels = Boolean(options.expandLabels);
  const showLegend = options.showLegend !== false;
  const labelOffset = rankPlotLabelOffset(plotRows);
  const maxLabelLen = plotRows.reduce((longest, row) => {
    const lines = String(row.barLabel || row.display || "").split("<br>");
    return Math.max(longest, ...lines.map((line) => line.length));
  }, 8);
  const maxYLabelLen = plotRows.reduce(
    (longest, row) => Math.max(longest, String(row.hoverLabel || row.label || "").length),
    8,
  );
  const yIndices = plotRows.map((_, index) => index);
  const tickFontSize = expandLabels ? 11 : Math.max(9, 12 - Math.floor(maxYLabelLen / 36));
  return {
    template: "plotly_white",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    autosize: true,
    barmode: "overlay",
    margin: {
      l: plotYAxisLeftMargin(maxYLabelLen, { rank: true, expand: expandLabels }),
      r: Math.max(expandLabels ? 180 : 120, maxLabelLen * (expandLabels ? 8 : 7)),
      t: showLegend ? 104 : 72,
      b: 44,
    },
    annotations: plotRows.map((row, index) => ({
      x: row.value + (Number(row.std) || 0) + labelOffset,
      y: index,
      text: row.barLabel || row.display,
      showarrow: false,
      xanchor: "left",
      yanchor: "middle",
      align: "left",
      font: { size: 11, color: "#202833", family: "system-ui, sans-serif" },
      xref: "x",
      yref: "y",
    })),
    legend: showLegend ? {
      orientation: "h",
      yanchor: "bottom",
      y: 1,
      xanchor: "left",
      x: 0,
      font: { color: "#667085", size: 12 },
      bgcolor: "rgba(255,255,255,0)",
      borderwidth: 0,
    } : { visible: false },
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
      type: "linear",
      tickmode: "array",
      tickvals: yIndices,
      ticktext: plotRows.map((row) => row.hoverLabel || row.label || ""),
      range: [Math.max(plotRows.length - 0.5, 0.5), -0.5],
      autorange: false,
      automargin: true,
      ticklabelposition: "outside",
      dtick: 1,
      tickfont: { color: "#202833", size: tickFontSize },
    },
    height: plotLayoutHeight(plotRows.length, 40, 120, 280),
  };
}

function buildRankPlotlySpec(rows, metric, options = {}) {
  const title = `${plotMetricLabel(metric)} rank across studies`;
  const subtitle = "Error bars = σ across replicas";
  const plotRows = rankPlotNormalizeRows(rows);
  const allCategories = options.allCategories || rankPlotCategoriesForRows(plotRows);
  const categoryVisibility = options.categoryVisibility || null;
  const visibleRows = options.forExport
    ? rankPlotExportRows(plotRows, categoryVisibility || {})
    : rankPlotVisibleRows(
      plotRows,
      categoryVisibility || null,
    );
  const visibleCategories = rankPlotCategoriesForRows(
    categoryVisibility ? visibleRows : plotRows,
  );
  const data = buildRankCategoryBarTraces(visibleRows, visibleCategories);
  return {
    data,
    layout: {
      ...buildRankPlotLayout(visibleRows, metric, {
        expandLabels: Boolean(options.expandLabels),
        showLegend: Boolean(options.forExport),
      }),
      title: {
        ...plotTitleLayout(title, subtitle),
        pad: { t: 8, b: 4 },
      },
    },
    config: {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
      toImageButtonOptions: { format: "png", filename: slug(title), scale: 2 },
    },
    meta: {
      allCategories,
      plotRows,
      metric,
      expandLabels: Boolean(options.expandLabels),
    },
  };
}

function rankPlotCategoryVisibilityState(payload) {
  const allCategories = payload.plotRankCategories || rankPlotCategoriesForRows(payload.plotRankAllRows || []);
  if (!payload.plotRankCategoryVisibility) {
    payload.plotRankCategoryVisibility = Object.fromEntries(allCategories.map((category) => [category, true]));
  }
  return payload.plotRankCategoryVisibility;
}

function rankPlotCategoryShown(visibility, category) {
  return visibility[category] !== false;
}

function toggleRankPlotCategory(payload, category) {
  const allCategories = payload.plotRankCategories || rankPlotCategoriesForRows(payload.plotRankAllRows || []);
  const visibility = rankPlotCategoryVisibilityState(payload);
  visibility[category] = !rankPlotCategoryShown(visibility, category);
  if (!allCategories.some((entry) => rankPlotCategoryShown(visibility, entry))) {
    visibility[category] = true;
  }
  return visibility;
}

function buildRankPlotViewSpec(payload, categoryVisibility, { forExport = false } = {}) {
  const allCategories = payload.plotRankCategories || rankPlotCategoriesForRows(payload.plotRankAllRows);
  return buildRankPlotlySpec(payload.plotRankAllRows, payload.plotRankMetric, {
    expandLabels: payload.plotRankExpandLabels ?? state.rankPlotExpandLabels,
    allCategories,
    categoryVisibility,
    forExport,
  });
}

async function reflowRankPlot(host, payload) {
  if (!host || !payload?.plotRankAllRows || !payload?.plotRankMetric) return;
  const categoryVisibility = rankPlotCategoryVisibilityState(payload);
  const spec = buildRankPlotViewSpec(payload, categoryVisibility);
  await Plotly.react(host, spec.data, spec.layout, spec.config);
  payload.plotRankSpec = spec;
  syncPlotlyHostHeight(host, spec.layout);
  syncRankPlotHtmlLegend(payload);
}

function buildRankPlotExportFigure(host, payload) {
  if (!payload?.plotRankAllRows || !payload?.plotRankMetric) return null;
  const categoryVisibility = rankPlotCategoryVisibilityState(payload);
  return buildRankPlotViewSpec(payload, categoryVisibility, { forExport: true });
}

function rankPlotLegendMarkup(exportKey, categories) {
  if (!categories.length) return "";
  const buttons = categories.map((category) => (
    `<button type="button" class="rank-legend-button" data-rank-legend-key="${escapeHtml(exportKey)}" data-rank-legend-category="${escapeHtml(category)}" aria-pressed="true">
      <span class="legend-swatch" style="background:${RANK_BAR_COLORS[category]};border-color:${RANK_BAR_COLORS[category]};"></span>
      <span>${escapeHtml(RANK_BAR_LABELS[category])}</span>
    </button>`
  )).join("");
  return `<div class="rank-plot-legend metric-legend" role="toolbar" aria-label="Filter rank chart categories">${buttons}</div>`;
}

function syncRankPlotHtmlLegend(payload) {
  if (!payload?.plotExportKey) return;
  const visibility = rankPlotCategoryVisibilityState(payload);
  document.querySelectorAll(`button[data-rank-legend-key="${payload.plotExportKey}"]`).forEach((button) => {
    const shown = rankPlotCategoryShown(visibility, button.dataset.rankLegendCategory);
    button.classList.toggle("is-hidden", !shown);
    button.setAttribute("aria-pressed", shown ? "true" : "false");
  });
}

function bindRankPlotLegendButtons() {
  document.querySelectorAll("button[data-rank-legend-key]:not([data-protocol-similarity-legend])").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => {
      const payload = state.plotExports[button.dataset.rankLegendKey];
      const host = payload?.plotlyDivId ? document.getElementById(payload.plotlyDivId) : null;
      if (!payload?.plotRankAllRows || !host) return;
      toggleRankPlotCategory(payload, button.dataset.rankLegendCategory);
      void reflowRankPlot(host, payload);
    });
  });
}

function bindRankPlotInteractions(host, payload) {
  if (host.dataset.rankPlotClickBound === "true") return;
  host.dataset.rankPlotClickBound = "true";
  host.on("plotly_click", (event) => {
    const rowIndex = rankPlotClickRowIndex(event.points?.[0]);
    handleRankPlotRowClick(payload.rows?.[rowIndex]);
  });
}

async function mountPendingPlotlyCharts() {
  const queue = state.pendingPlotly;
  state.pendingPlotly = [];
  for (const item of queue) {
    const host = document.getElementById(item.divId);
    if (!host || !item.spec) continue;
    const payload = state.plotExports[item.key];
    if (!window.Plotly) {
      host.innerHTML = '<span class="path">Plotly failed to load. Check your network connection and refresh.</span>';
      continue;
    }
    if (host.data) Plotly.purge(host);
    await Plotly.newPlot(host, item.spec.data, item.spec.layout, item.spec.config);
    syncPlotlyHostHeight(host, host.layout || item.spec.layout);
    if (payload) {
      payload.plotlyDivId = item.divId;
      if (payload.plotKind === "rank") {
        payload.plotRankSpec = item.spec;
        rankPlotCategoryVisibilityState(payload);
        bindRankPlotInteractions(host, payload);
        syncRankPlotHtmlLegend(payload);
      }
    }
  }
  requestAnimationFrame(() => {
    resizePlotlyHosts();
    requestAnimationFrame(resizePlotlyHosts);
  });
}

function rankPlotClickRowIndex(point) {
  if (!point) return null;
  if (Number.isFinite(Number(point.customdata?.[7]))) return Number(point.customdata[7]);
  return Math.round(Number(point.y));
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

function compactPlotLabel(label, maxLen = 30) {
  const text = String(label || "");
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1)}…`;
}

const PLOT_Y_LABEL_MAX_CHARS = 40;
const PLOT_REPLICA_LABEL_MAX_CHARS = 52;

function uniqueReplicaPlotLabel(modelName, replicaName, maxLen = PLOT_REPLICA_LABEL_MAX_CHARS) {
  const prefix = `${replicaName} · `;
  const budget = maxLen - prefix.length;
  const model = String(modelName || "");
  const tail = model.length <= budget ? model : `${model.slice(0, Math.max(budget - 1, 1))}…`;
  return `${prefix}${tail}`;
}

function plotLabelRows(rows, labelKey = "label", maxChars = PLOT_Y_LABEL_MAX_CHARS) {
  return rows.map((row) => {
    const full = String(row[labelKey] || "");
    return {
      ...row,
      plotLabel: compactPlotLabel(full, maxChars),
      hoverLabel: full,
    };
  });
}

function plotLayoutHeight(rowCount, rowStep = 28, base = 120, minimum = 280) {
  return Math.max(minimum, rowCount * rowStep + base);
}

function syncPlotlyHostHeight(host, layout) {
  if (!host || !layout) return;
  const height = Number(layout.height);
  if (Number.isFinite(height) && height > 0) {
    host.style.height = `${height}px`;
  }
}

function plotTitleLayout(title, subtitle, options = {}) {
  const centered = Boolean(options.centerTitle ?? options.replicaSpread);
  const font = { size: 16, color: "#202833" };
  const layout = { text: title, font };
  if (subtitle) {
    layout.text = `${title}<br><sup>${subtitle}</sup>`;
  }
  layout.x = centered ? 0.5 : 0;
  layout.xanchor = centered ? "center" : "left";
  return layout;
}

function buildSimpleBarPlotlySpec(rows, metric, options = {}) {
  const title = options.title || plotMetricLabel(metric);
  const subtitle = options.subtitle || "";
  const zeroCentered = Boolean(options.zeroCentered);
  const compact = Boolean(options.compact);
  const replicaSpread = Boolean(options.replicaSpread);
  const colorForRow = options.colorForRow || (() => "#9ecae1");
  const maxChars = compact ? 30 : PLOT_Y_LABEL_MAX_CHARS;
  const plotRows = replicaSpread
    ? rows.map((row) => ({
        ...row,
        plotLabel: row.plotLabel || row.label,
        hoverLabel: row.hoverLabel || row.label,
      }))
    : plotLabelRows(rows, "label", maxChars);
  const maxLabelLen = plotRows.reduce((longest, row) => Math.max(longest, String(row.plotLabel || "").length), 8);
  const yIndices = plotRows.map((_, index) => index);
  const trace = {
    type: "bar",
    orientation: "h",
    y: yIndices,
    x: plotRows.map((row) => row.value),
    customdata: plotRows.map((row) => [row.display || row.value, row.hoverLabel || row.plotLabel || row.label]),
    marker: {
      color: plotRows.map((row) => colorForRow(row)),
      line: { color: "#ffffff", width: 1 },
    },
    hovertemplate: "<b>%{customdata[1]}</b><br>%{customdata[0]}<extra></extra>",
  };
  const yaxis = {
    type: "linear",
    tickmode: "array",
    tickvals: yIndices,
    ticktext: plotRows.map((row) => row.plotLabel),
    range: [plotRows.length - 0.5, -0.5],
    autorange: false,
    automargin: false,
    dtick: 1,
    tickfont: { color: "#202833", size: replicaSpread ? 11 : Math.max(10, 12 - Math.floor(maxLabelLen / 28)) },
  };
  return {
    data: [trace],
    layout: {
      template: "plotly_white",
      paper_bgcolor: "#ffffff",
      plot_bgcolor: "#ffffff",
      autosize: true,
      barmode: "overlay",
      bargap: replicaSpread ? 0.15 : 0.2,
      title: plotTitleLayout(title, subtitle, options),
      margin: {
        l: plotYAxisLeftMargin(maxLabelLen, { compact }),
        r: 24,
        t: subtitle ? 88 : 64,
        b: 36,
      },
      xaxis: {
        title: options.xTitle || plotMetricLabel(metric),
        titlefont: { color: "#667085" },
        tickfont: { color: "#667085" },
        gridcolor: "#ebe5dc",
        zeroline: true,
        zerolinecolor: zeroCentered ? "#8a93a0" : "#d8d1c5",
        rangemode: !zeroCentered && metric.direction !== "min" ? "tozero" : "normal",
      },
      yaxis,
      height: plotLayoutHeight(rows.length, 28, 120, 280),
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
  const bodyHtml = `<div id="${divId}" class="plotly-host" role="img" aria-label="${escapeHtml(title)}"></div>`;
  return collapsiblePlotMarkup(key, title, bodyHtml, {
    subtitle: note || subtitle || "",
    headActions: registerPlotExport(key, title, rows, "plotly"),
  });
}

function applyPlotSpan(html, span) {
  if (!html) return html;
  const spanClass = span === "full" ? "plot-span-full" : span === "half" ? "plot-span-half" : "";
  if (!spanClass) return html;
  return html.replace(/class="(generated-plot[^"]*)"/, (_, classes) => `class="${classes} ${spanClass}"`);
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
    const previousMetric = state.selectedMetric;
    state.resultScope = event.target.value;
    state.selectedMetric = previousMetric;
    ensureSelectedMetric();
    if (state.comparisonBaseline !== "internal" && !scopedExternalBaselines().some((item) => externalEntryId(item) === state.comparisonBaseline)) {
      state.comparisonBaseline = "internal";
    }
    persistUiState();
    renderWorkspace(state.workspace);
  };
  $("comparison-baseline-select").onchange = (event) => {
    state.comparisonBaseline = event.target.value || "internal";
    persistUiState();
    renderWorkspace(state.workspace);
  };
  $("decision-metric-select").onchange = (event) => {
    state.selectedMetric = event.target.value || null;
    persistUiState();
    renderComparisonTable();
    renderComparisonCharts();
    scheduleProtocolSimilarityLoad();
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

function buildComparisonColorLegendHtml(entries) {
  if (!entries.length) return "";
  const categoriesPresent = new Set(entries.map((item) => entryModelCategory(item)));
  const items = [
    '<span class="legend-intro">Model / Type pill colors (same palette as charts).</span>',
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
  return `<div class="color-legend-grid">${items.join("")}</div>`;
}

function renderComparisonColorLegend(entries) {
  const html = buildComparisonColorLegendHtml(entries);
  ["comparison-color-legend-top", "comparison-color-legend-bottom"].forEach((id) => {
    const node = $(id);
    if (node) node.innerHTML = html;
  });
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
      (() => {
        const progress = replicaProgressSummary(item.study);
        return {
          value: progress.text,
          numeric: progress.numeric,
          className: progress.className,
          title: progress.title || modelDescription(item),
        };
      })(),
      comparisonDeltaCell(item, selectedMetric),
      ...metricHeaders.map((metric) => comparisonMetricCell(item, metric.name, metricRanks[metric.name], metricRanks[metric.name].size)),
    ]),
    (row, index) => comparisonRowClass(sortedEntries[index]),
    state.comparisonSort,
  );
  bindSortButtons($("comparison-table"), "comparisonSort");
  renderComparisonExportActions(metricHeaders, sortedEntries, selectedMetric);
  renderComparisonColorLegend(sortedEntries);
  $("comparison-table").querySelectorAll("tbody tr.selectable").forEach((row, index) => {
    const item = sortedEntries[index];
    if (!item) return;
    row.dataset.entryId = item.id;
    row.addEventListener("click", () => {
      handleComparisonEntryClick(item);
    });
  });
}

function renderComparisonCharts() {
  const selectedMetric = ensureSelectedMetric();
  state.plotExports = {};
  state.pendingPlotly = [];
  const spreadRows = collectReplicaSpreadRows(selectedMetric);
  const deltaRows = collectDeltaPlotRows(selectedMetric);
  const rankPlot = generatedRankPlot(selectedMetric);
  const spreadPlot = generatedReplicaSpreadPlot(selectedMetric, spreadRows);
  const deltaPlot = generatedAllDeltasPlot(selectedMetric, deltaRows);
  const container = $("comparison-charts");
  container.className = "decision-plots";
  const parts = [];
  if (rankPlot) parts.push(applyPlotSpan(rankPlot, "full"));
  if (spreadPlot && deltaPlot) {
    parts.push(applyPlotSpan(spreadPlot, "full"));
    parts.push(applyPlotSpan(deltaPlot, "full"));
  } else if (spreadPlot) {
    parts.push(applyPlotSpan(spreadPlot, "full"));
  } else if (deltaPlot) {
    parts.push(applyPlotSpan(deltaPlot, "full"));
  }
  if (parts.length === 1) container.classList.add("layout-single");
  container.innerHTML = parts.join("");
  void mountPendingPlotlyCharts();
  bindPlotExportButtons();
  bindRankPlotLabelToggleButtons();
  bindRankPlotLegendButtons();
  bindCollapsiblePlots(container);
}

function collectReplicaSpreadRows(metricName) {
  if (!metricName) return [];
  const entries = comparisonEntries().filter((item) => !item.external);
  const metric = metricMeta(metricName);
  const rows = entries.flatMap((item) => (item.study?.replicas || [])
    .map((replica) => {
      const value = replicaMetricValue(replica, metricName);
      if (value === null) return null;
      const rowKey = `${item.id}::${replica.replica_name}`;
      return {
        rowKey,
        plotLabel: uniqueReplicaPlotLabel(modelDisplayName(item), replica.replica_name),
        label: uniqueReplicaPlotLabel(modelDisplayName(item), replica.replica_name),
        hoverLabel: `${modelDisplayName(item)} · ${replica.replica_name}`,
        replica_name: replica.replica_name,
        value,
        display: numeric(value),
        scope: metricScope(metricName),
        study: item.id,
        metric: metricName,
        color: entryPaletteColor(item),
      };
    })
    .filter(Boolean));
  const deduped = [];
  const seen = new Set();
  rows.forEach((row) => {
    if (seen.has(row.rowKey)) return;
    seen.add(row.rowKey);
    deduped.push(row);
  });
  deduped.sort((left, right) => {
    const valueCmp = metric.direction === "min" ? left.value - right.value : right.value - left.value;
    if (valueCmp !== 0) return valueCmp;
    return String(left.hoverLabel || "").localeCompare(String(right.hoverLabel || ""));
  });
  return deduped;
}

function collectDeltaPlotRows(metricName) {
  if (!metricName) return [];
  const reference = comparisonReferenceSummary();
  const referenceMetric = metricSummaryLookup(reference, metricName);
  if (!referenceMetric) return [];
  return comparisonEntries()
    .filter((item) => !isReferenceEntry(item))
    .map((item) => {
      const delta = metricDecisionDelta(item.entry, metricName);
      if (delta === null) return null;
      const std = item.external ? null : entryMetricStd(item.entry, metricName);
      const withinNoise = std !== null && Math.abs(delta) < std;
      const label = modelDisplayName(item) + (isLearnedSfBaseline(item) ? " (ML)" : "") + (withinNoise ? " · ~noise" : "");
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
}

function generatedReplicaSpreadPlot(metricName, rows = null, options = {}) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const plotRows = rows || collectReplicaSpreadRows(metricName);
  if (plotRows.length < 2) return "";
  const title = `${plotMetricLabel(metric)} · replica values`;
  const key = `replica_spread_${slug(metricName)}_${state.resultScope}`;
  const spec = buildSimpleBarPlotlySpec(plotRows, metric, {
    title,
    subtitle: "Sorted by replica performance (best at top); one bar per replica.",
    colorForRow: (row) => row.color || MODEL_CATEGORY_COLORS.ablation,
    compact: Boolean(options.compact),
    replicaSpread: true,
  });
  return plotlyChartMarkup(key, title, "", "Green = full_ocscore · blue = ablation", plotRows, spec);
}

function generatedAllDeltasPlot(metricName, rows = null, options = {}) {
  if (!metricName) return "";
  const metric = metricMeta(metricName);
  const plotRows = rows || collectDeltaPlotRows(metricName);
  if (!plotRows.length) return "";
  const title = `${plotMetricLabel(metric)} vs ${comparisonReferenceLabel()}`;
  const key = `all_delta_${slug(metricName)}_${state.resultScope}_${slug(state.comparisonBaseline)}`;
  const spec = buildSimpleBarPlotlySpec(plotRows, metric, {
    title,
    subtitle: "Positive = improvement vs reference under metric direction",
    zeroCentered: true,
    xTitle: "Δ vs reference",
    colorForRow: (row) => {
      if (row.withinNoise) return "#c5cad1";
      return row.value > 0 ? "#16703f" : row.value < 0 ? "#b42318" : "#667085";
    },
    compact: Boolean(options.compact),
  });
  return plotlyChartMarkup(key, title, "", "Green/red = directionally better/worse · grey = within replica σ", plotRows, spec);
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
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
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
  return `${lines.join("\n")}\n`;
}

function objectsToCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  rows.forEach((row) => {
    lines.push(headers.map((key) => csvEscape(row[key] ?? "")).join(","));
  });
  return `${lines.join("\n")}\n`;
}

function comparisonExportRows(metricHeaders, sortedEntries, selectedMetric) {
  return sortedEntries.map((item) => {
    const summary = item.entry?.metric_summary || null;
    const row = {
      model: modelDisplayName(item),
      type: item.kind,
      reference: isReferenceEntry(item) ? "yes" : "no",
      replicas: (() => {
        const progress = replicaProgressSummary(item.study);
        return progress.numeric ? progress.text : "";
      })(),
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
  const match = String(svgString).match(/viewBox="0\s+0\s+([\d.]+)\s+([\d.]+)"/);
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

function plotlyExportImageOptions(host, format, exportFigure = null) {
  const layout = exportFigure?.layout || host.layout || {};
  const margin = layout.margin || {};
  const height = Number(exportFigure?.layout?.height || layout.height || host.offsetHeight || 320);
  return {
    format,
    width: Math.max(960, host.offsetWidth || 960),
    height: Math.max(280, height),
    scale: format === "png" ? 2 : 1,
    layout: {
      ...PLOTLY_EXPORT_LAYOUT,
      ...layout,
      margin: {
        ...margin,
        r: Math.max(Number(margin.r) || 24, 160),
        l: Math.max(Number(margin.l) || 112, 112),
      },
    },
  };
}

async function plotlyImageDataUrl(host, format, payload = null) {
  const exportFigure = payload?.plotKind === "rank" ? buildRankPlotExportFigure(host, payload) : null;
  const options = plotlyExportImageOptions(host, format, exportFigure);
  if (exportFigure) {
    return Plotly.toImage(
      { data: exportFigure.data, layout: options.layout },
      { format: options.format, width: options.width, height: options.height, scale: options.scale },
    );
  }
  return Plotly.toImage(host, options);
}

async function downloadPlotlyImage(filename, host, format, payload = null) {
  const dataUrl = await plotlyImageDataUrl(host, format, payload);
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

async function copyPlotlyPng(host, payload = null) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === "undefined") {
    throw new Error("Clipboard image copy is not supported in this browser");
  }
  const dataUrl = await plotlyImageDataUrl(host, "png", payload);
  const blob = await (await fetch(dataUrl)).blob();
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

function registerPlotExport(key, title, rows, asset, options = {}) {
  const payload = { title, rows };
  if (asset === "plotly") payload.engine = "plotly";
  else payload.svg = asset;
  if (options.plotKind) payload.plotKind = options.plotKind;
  state.plotExports[key] = payload;
  const extraActions = options.headActions || "";
  return `
    <div class="export-actions">
      ${extraActions}
      <button class="ghost-button" type="button" data-export-kind="png" data-export-key="${escapeHtml(key)}">PNG</button>
      <button class="ghost-button" type="button" data-export-kind="copy" data-export-key="${escapeHtml(key)}">Copy</button>
      <button class="ghost-button" type="button" data-export-kind="svg" data-export-key="${escapeHtml(key)}">SVG</button>
      <button class="ghost-button" type="button" data-export-kind="csv" data-export-key="${escapeHtml(key)}">CSV</button>
    </div>
  `;
}

function bindPlotExportButtons() {
  document.querySelectorAll("button[data-export-key]").forEach((button) => {
    if (button.dataset.exportBound === "true") return;
    button.dataset.exportBound = "true";
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
            await downloadPlotlyImage(`${base}.svg`, host, "svg", payload);
          } else if (kind === "png") {
            await downloadPlotlyImage(`${base}.png`, host, "png", payload);
          } else if (kind === "copy") {
            await copyPlotlyPng(host, payload);
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
      label: studyDisplayName(row.entry) + (row.entry.baseline_family === "learned_sf" ? " (ML)" : ""),
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
  const categories = rankPlotCategoriesForRows(rows);
  const spec = buildRankPlotlySpec(rows, metric, { expandLabels: state.rankPlotExpandLabels });
  state.pendingPlotly.push({ key, divId, spec });
  const exportActions = registerPlotExport(key, title, rows, "plotly", {
    plotKind: "rank",
    headActions: rankPlotLabelToggleMarkup(key),
  });
  const exportPayload = state.plotExports[key];
  exportPayload.plotExportKey = key;
  exportPayload.plotRankAllRows = rows;
  exportPayload.plotRankMetric = metric;
  exportPayload.plotRankCategories = categories;
  exportPayload.plotRankExpandLabels = state.rankPlotExpandLabels;
  const plotBody = `
    ${rankPlotLegendMarkup(key, categories)}
    <div id="${divId}" class="plotly-host plotly-host-rank" role="img" aria-label="${escapeHtml(title)}"></div>
  `;
  return collapsiblePlotMarkup(key, title, plotBody, {
    subtitle: "Click a bar · legend toggles categories (combine any set) · export matches visible categories",
    headActions: exportActions,
  });
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
  const key = `stability_${slug(study.study_name)}_${slug(metricName)}_${state.resultScope}`;
  if (!rows.length) {
    return collapsiblePlotMarkup(key, title, '<span class="path">No replica-level values detected for this metric.</span>');
  }
  const values = rows.map((row) => row.value);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const svg = chartSvg(title, rows, { dot: true, subtitle: `Mean ${numeric(mean)} | each dot is one replica.` });
  return collapsiblePlotMarkup(key, title, svg, {
    subtitle: "Selected model replica spread",
    headActions: registerPlotExport(key, title, rows, svg),
  });
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
  bindCollapsiblePlots($("detail-plots"));
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
    persistUiState();
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
  renderRunContext();
  persistUiState();
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

function defaultAblationDesign() {
  return {
    templateName: "",
    name: "",
    description: "",
    include_features: "",
    include_patterns: "*",
    exclude_features: "",
    exclude_patterns: "",
    allow_missing_exclude_features: true,
    protocol: "",
    raw_input_dir: "",
    merged_input: "",
    pdbbind_input: "",
    dudez_input: "",
    feature_source: "auto",
    output_dir: "",
    policy_yml_path: "",
  };
}

function ensureAblationDesignDraft() {
  if (!state.ablationDesign) state.ablationDesign = defaultAblationDesign();
  return state.ablationDesign;
}

function readAblationDesignDraft() {
  const draft = ensureAblationDesignDraft();
  return {
    templateName: draft.templateName || "",
    name: draft.name || "",
    description: draft.description || "",
    include_features: draft.include_features || "",
    include_patterns: draft.include_patterns || "",
    exclude_features: draft.exclude_features || "",
    exclude_patterns: draft.exclude_patterns || "",
    allow_missing_exclude_features: draft.allow_missing_exclude_features !== false,
    protocol: draft.protocol || "",
    raw_input_dir: draft.raw_input_dir || "",
    merged_input: draft.merged_input || "",
    pdbbind_input: draft.pdbbind_input || "",
    dudez_input: draft.dudez_input || "",
    feature_source: draft.feature_source || "auto",
    output_dir: draft.output_dir || "",
    policy_yml_path: draft.policy_yml_path || "",
  };
}

function writeAblationDesignDraftFromForm() {
  const draft = ensureAblationDesignDraft();
  draft.templateName = $("ablation-design-template")?.value || "";
  draft.name = $("ablation-design-name")?.value.trim() || "";
  draft.description = $("ablation-design-description")?.value.trim() || "";
  draft.include_features = $("ablation-design-include-features")?.value || "";
  draft.include_patterns = $("ablation-design-include-patterns")?.value || "";
  draft.exclude_features = $("ablation-design-exclude-features")?.value || "";
  draft.exclude_patterns = $("ablation-design-exclude-patterns")?.value || "";
  draft.allow_missing_exclude_features = Boolean($("ablation-design-allow-missing-excludes")?.checked);
  draft.protocol = $("ablation-design-protocol")?.value.trim() || "";
  draft.raw_input_dir = $("ablation-design-raw-input")?.value.trim() || "";
  draft.merged_input = $("ablation-design-merged-input")?.value.trim() || "";
  draft.pdbbind_input = $("ablation-design-pdbbind-input")?.value.trim() || "";
  draft.dudez_input = $("ablation-design-dudez-input")?.value.trim() || "";
  draft.feature_source = $("ablation-design-feature-source")?.value || "auto";
  draft.output_dir = $("ablation-design-output-dir")?.value.trim() || "";
  draft.policy_yml_path = $("ablation-design-policy-path")?.value.trim() || "";
  persistUiState();
  return draft;
}

function ablationDesignHasInputPaths(draft = readAblationDesignDraft()) {
  return Boolean(
    draft.raw_input_dir
    || draft.merged_input
    || draft.pdbbind_input
    || draft.dudez_input
    || state.ablationDesignContext?.discovered_inputs?.ok,
  );
}

function applyAblationDesignDiscoveredInputs(context = state.ablationDesignContext) {
  const draft = ensureAblationDesignDraft();
  const discovered = context?.discovered_inputs;
  if (!discovered?.ok) return;
  if (!draft.raw_input_dir && discovered.raw_input_dir) draft.raw_input_dir = discovered.raw_input_dir;
}

function ablationDesignInputPayload(draft = readAblationDesignDraft()) {
  const payload = { feature_source: draft.feature_source || "auto" };
  if (draft.raw_input_dir) payload.raw_input_dir = draft.raw_input_dir;
  if (draft.merged_input) payload.merged_input = draft.merged_input;
  if (draft.pdbbind_input) payload.pdbbind_input = draft.pdbbind_input;
  if (draft.dudez_input) payload.dudez_input = draft.dudez_input;
  return payload;
}

function ablationDesignLines(value) {
  return String(value || "")
    .split(/\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function ablationDesignPolicyPayload(draft = readAblationDesignDraft()) {
  const payload = {
    name: draft.name,
    description: draft.description,
    allow_missing_exclude_features: draft.allow_missing_exclude_features !== false,
  };
  const includeFeatures = ablationDesignLines(draft.include_features);
  const includePatterns = ablationDesignLines(draft.include_patterns);
  const excludeFeatures = ablationDesignLines(draft.exclude_features);
  const excludePatterns = ablationDesignLines(draft.exclude_patterns);
  if (includeFeatures.length) payload.include_features = includeFeatures;
  if (includePatterns.length) payload.include_patterns = includePatterns;
  if (excludeFeatures.length) payload.exclude_features = excludeFeatures;
  if (excludePatterns.length) payload.exclude_patterns = excludePatterns;
  return payload;
}

function ablationDesignRequestPayload(draft = readAblationDesignDraft()) {
  const payload = {
    policy: ablationDesignPolicyPayload(draft),
    protocol: draft.protocol,
    output_dir: draft.output_dir,
    policy_yml_path: draft.policy_yml_path,
    description: draft.description,
    name: draft.name ? `ablation-${draft.name}` : "",
    ...ablationDesignInputPayload(draft),
  };
  const catalog = state.ablationDesignFeatureCatalog;
  if (catalog?.candidate_features?.length) {
    payload.candidate_features = catalog.candidate_features;
  }
  return payload;
}

function syncAblationDesignOutputPaths(draft = readAblationDesignDraft(), context = state.ablationDesignContext) {
  if (!draft.name || !context) return draft;
  const container = context.ablation_container || "ablations";
  if (!draft.output_dir) draft.output_dir = `${container}/${draft.name}`;
  if (!draft.policy_yml_path) draft.policy_yml_path = `Ablations/${draft.name}.yml`;
  return draft;
}

function renderAblationDesignForm(context = state.ablationDesignContext) {
  const draft = ensureAblationDesignDraft();
  if (context?.protocol_path && !draft.protocol) draft.protocol = context.protocol_path;
  applyAblationDesignDiscoveredInputs(context);
  syncAblationDesignOutputPaths(draft, context);

  const templateSelect = $("ablation-design-template");
  if (templateSelect && context?.catalog) {
    const options = ['<option value="">Custom (blank)</option>']
      .concat(context.catalog.map((item) => {
        const shipped = item.source_kind === "bundled" ? " (shipped)" : "";
        return `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)}${shipped}</option>`;
      }));
    templateSelect.innerHTML = options.join("");
    templateSelect.value = draft.templateName || "";
  }

  $("ablation-design-name").value = draft.name || "";
  $("ablation-design-description").value = draft.description || "";
  $("ablation-design-include-features").value = draft.include_features || "";
  $("ablation-design-include-patterns").value = draft.include_patterns || "";
  $("ablation-design-exclude-features").value = draft.exclude_features || "";
  $("ablation-design-exclude-patterns").value = draft.exclude_patterns || "";
  $("ablation-design-allow-missing-excludes").checked = draft.allow_missing_exclude_features !== false;
  $("ablation-design-protocol").value = draft.protocol || "";
  $("ablation-design-raw-input").value = draft.raw_input_dir || "";
  $("ablation-design-merged-input").value = draft.merged_input || "";
  $("ablation-design-pdbbind-input").value = draft.pdbbind_input || "";
  $("ablation-design-dudez-input").value = draft.dudez_input || "";
  $("ablation-design-feature-source").value = draft.feature_source || "auto";
  $("ablation-design-output-dir").value = draft.output_dir || "";
  $("ablation-design-policy-path").value = draft.policy_yml_path || "";
  if ($("ablation-design-feature-filter")) {
    $("ablation-design-feature-filter").value = state.ablationDesignFeatureFilter || "";
  }
  if ($("ablation-design-wildcard-pattern")) {
    $("ablation-design-wildcard-pattern").value = state.ablationDesignWildcardPattern || "";
  }

  const existing = (context?.existing_ablation_names || []);
  const discoveredFrom = context?.discovered_inputs?.discovered_from;
  const summary = discoveredFrom
    ? `Raw inputs from ${pathBasename(discoveredFrom)}/ · ${existing.length} existing ablations`
    : context?.candidate_source
      ? `Workspace metadata available from ${pathBasename(context.candidate_source)} · ${existing.length} existing ablations`
      : `${existing.length} existing ablations · expected raw_prepare/ under served root`;
  $("ablation-design-summary").textContent = summary;

  renderAblationDesignFeatureBrowser();
  updateAblationDesignTemplateDescription();
  renderAblationDesignTemplateDiff();
  ablationDesignUpdatePatternFeedback();
}

function ablationDesignGroupPatterns(groupName, features = []) {
  if (groupName === "ligand") return ["ligand_*"];
  if (groupName === "receptor") return ["receptor_*"];
  if (groupName === "scoring") {
    const patterns = new Set();
    features.forEach((name) => {
      const text = String(name);
      const idx = text.indexOf("_");
      if (idx > 0) patterns.add(`${text.slice(0, idx + 1)}*`);
    });
    return [...patterns];
  }
  return [];
}

function ablationDesignFnmatch(name, pattern) {
  const escaped = String(pattern)
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`).test(String(name));
}

function ablationDesignFnmatchFolded(name, pattern) {
  return ablationDesignFnmatch(String(name).toLowerCase(), String(pattern).toLowerCase());
}

function ablationDesignPatternCaseMismatchHints(patterns, candidates) {
  const hints = [];
  [...new Set(patterns.map(String).filter(Boolean))].forEach((pattern) => {
    const strictCount = candidates.filter((name) => ablationDesignFnmatch(name, pattern)).length;
    if (strictCount > 0) return;
    const foldedCount = candidates.filter((name) => ablationDesignFnmatchFolded(name, pattern)).length;
    if (foldedCount <= 0) return;
    const suggestion = pattern === pattern.toLowerCase()
      ? pattern
      : pattern.toLowerCase();
    hints.push(
      `Pattern "${pattern}" matches 0 features (case-sensitive). `
      + `"${suggestion}" would match ${foldedCount}.`,
    );
  });
  return hints;
}

function ablationDesignActivePolicyPatterns(draft = readAblationDesignDraft()) {
  return [
    ...ablationDesignWildcardPatterns(),
    ...ablationDesignLines(draft.include_patterns),
    ...ablationDesignLines(draft.exclude_patterns),
  ];
}

function ablationDesignUpdateCaseAndFilterWarnings() {
  const caseNode = $("ablation-design-case-warning");
  const filterNote = $("ablation-design-filter-note");
  const filter = String(state.ablationDesignFeatureFilter || "").trim();
  if (filterNote) {
    filterNote.hidden = !filter;
  }
  const catalog = state.ablationDesignFeatureCatalog;
  if (!caseNode) return;
  if (!catalog?.ok) {
    caseNode.hidden = true;
    caseNode.textContent = "";
    return;
  }
  const candidates = ablationDesignAllCandidateFeatures(catalog);
  const patterns = ablationDesignActivePolicyPatterns();
  const hints = ablationDesignPatternCaseMismatchHints(patterns, candidates);
  if (hints.length) {
    caseNode.hidden = false;
    caseNode.className = "ablation-design-case-warning is-warning";
    caseNode.textContent = hints.slice(0, 2).join(" ");
    return;
  }
  if (patterns.length) {
    caseNode.hidden = false;
    caseNode.className = "ablation-design-case-warning is-note";
    caseNode.textContent =
      "Wildcards and policy patterns are case-sensitive (Python fnmatch). "
      + "Match CSV column names exactly — e.g. ligand_*, not Ligand_*.";
    return;
  }
  caseNode.hidden = true;
  caseNode.textContent = "";
}

function ablationDesignWildcardPatterns() {
  return ablationDesignLines(state.ablationDesignWildcardPattern || "");
}

function ablationDesignAllCandidateFeatures(catalog = state.ablationDesignFeatureCatalog) {
  if (!catalog?.ok) return [];
  const groups = catalog.feature_groups || {};
  const names = Object.values(groups).flatMap((items) => items || []);
  return [...new Set(names.map(String))];
}

function ablationDesignFeaturesMatchingPatterns(features, patterns) {
  if (!patterns.length) return [];
  const matched = new Set();
  features.forEach((name) => {
    if (patterns.some((pattern) => ablationDesignFnmatch(name, pattern))) matched.add(name);
  });
  return [...matched];
}

const ABLATION_DESIGN_FEATURE_GROUPS = [
  ["ligand", "Ligand"],
  ["receptor", "Receptor"],
  ["scoring", "Scoring"],
];
const FEATURE_VIRTUAL_ROW_HEIGHT = 28;
const FEATURE_VIRTUAL_VIEWPORT_HEIGHT = 320;
const FEATURE_VIRTUAL_OVERSCAN = 6;

function ablationDesignTemplateRequest() {
  const templateName = readAblationDesignDraft().templateName;
  if (!templateName) return null;
  const entry = (state.ablationDesignContext?.catalog || []).find((item) => item.name === templateName);
  return entry?.request || null;
}

function ablationDesignPolicyFieldLines(value) {
  if (Array.isArray(value)) return value.map(String).filter(Boolean);
  return ablationDesignLines(value);
}

function ablationDesignDiffFieldLines(field, baseRequest, draftPayload) {
  const base = new Set(ablationDesignPolicyFieldLines(baseRequest?.[field]));
  const current = new Set(ablationDesignPolicyFieldLines(draftPayload?.[field]));
  const added = [...current].filter((item) => !base.has(item));
  const removed = [...base].filter((item) => !current.has(item));
  return { added, removed, changed: added.length > 0 || removed.length > 0 };
}

function renderAblationDesignTemplateDiff() {
  const panel = $("ablation-design-template-diff");
  if (!panel) return;
  const base = ablationDesignTemplateRequest();
  if (!base) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  const draftPayload = ablationDesignPolicyPayload(readAblationDesignDraft());
  const fields = [
    ["include_features", "Include features"],
    ["include_patterns", "Include patterns"],
    ["exclude_features", "Exclude features"],
    ["exclude_patterns", "Exclude patterns"],
  ];
  const chunks = fields.map(([field, label]) => {
    const diff = ablationDesignDiffFieldLines(field, base, draftPayload);
    if (!diff.changed) return "";
    const added = diff.added.length ? `<div class="ablation-design-diff-added">+ ${escapeHtml(diff.added.join(", "))}</div>` : "";
    const removed = diff.removed.length ? `<div class="ablation-design-diff-removed">− ${escapeHtml(diff.removed.join(", "))}</div>` : "";
    return `<div class="ablation-design-diff-field"><strong>${escapeHtml(label)}</strong>${added}${removed}</div>`;
  }).filter(Boolean);
  if (!chunks.length) {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  panel.hidden = false;
  panel.innerHTML = `
    <div class="ablation-design-diff-title">Changes vs template <code>${escapeHtml(readAblationDesignDraft().templateName)}</code></div>
    ${chunks.join("")}
  `;
}

function ablationDesignPatternMatchSummary(patterns) {
  const catalog = state.ablationDesignFeatureCatalog;
  if (!catalog?.ok || !patterns.length) return null;
  const candidates = ablationDesignAllCandidateFeatures(catalog);
  const matched = ablationDesignFeaturesMatchingPatterns(candidates, patterns);
  const unmatchedPatterns = patterns.filter(
    (pattern) => !candidates.some((name) => ablationDesignFnmatch(name, pattern)),
  );
  return { matchedCount: matched.length, unmatchedPatterns };
}

function ablationDesignUpdatePatternFeedback() {
  const wildcardNode = $("ablation-design-wildcard-feedback");
  const rulesNode = $("ablation-design-pattern-feedback");
  const catalog = state.ablationDesignFeatureCatalog;
  if (!catalog?.ok) {
    if (wildcardNode) wildcardNode.textContent = "";
    if (rulesNode) rulesNode.textContent = "";
    ablationDesignUpdateCaseAndFilterWarnings();
    return;
  }
  const wildcardPatterns = ablationDesignWildcardPatterns();
  if (wildcardNode) {
    if (!wildcardPatterns.length) {
      wildcardNode.textContent = "";
    } else {
      const summary = ablationDesignPatternMatchSummary(wildcardPatterns);
      const unmatched = summary?.unmatchedPatterns?.length
        ? ` · ${summary.unmatchedPatterns.length} pattern(s) match nothing`
        : "";
      wildcardNode.textContent = `${summary?.matchedCount ?? 0} feature(s) match wildcard${unmatched}`;
    }
  }
  if (rulesNode) {
    const draft = readAblationDesignDraft();
    const parts = [];
    [
      ["include_patterns", "Include patterns", ablationDesignLines(draft.include_patterns)],
      ["exclude_patterns", "Exclude patterns", ablationDesignLines(draft.exclude_patterns)],
    ].forEach(([, label, patterns]) => {
      if (!patterns.length) return;
      const summary = ablationDesignPatternMatchSummary(patterns);
      const unmatched = summary?.unmatchedPatterns?.length
        ? ` (${summary.unmatchedPatterns.length} unmatched)`
        : "";
      parts.push(`${label}: ${summary?.matchedCount ?? 0} features${unmatched}`);
    });
    rulesNode.textContent = parts.join(" · ");
  }
  ablationDesignUpdateCaseAndFilterWarnings();
}

function mountVirtualFeatureList(container, features) {
  container.innerHTML = "";
  container.classList.add("ablation-design-feature-list-virtual");
  if (!features.length) {
    container.classList.remove("ablation-design-feature-list-virtual");
    container.innerHTML = '<div class="muted">No matches</div>';
    return;
  }
  const totalHeight = features.length * FEATURE_VIRTUAL_ROW_HEIGHT;
  const viewportHeight = Math.min(FEATURE_VIRTUAL_VIEWPORT_HEIGHT, Math.max(totalHeight, FEATURE_VIRTUAL_ROW_HEIGHT * 3));
  container.style.maxHeight = `${viewportHeight}px`;
  container.style.overflowY = "auto";
  container.style.position = "relative";

  const spacer = document.createElement("div");
  spacer.className = "ablation-design-feature-spacer";
  spacer.style.height = `${totalHeight}px`;
  container.appendChild(spacer);

  const viewport = document.createElement("div");
  viewport.className = "ablation-design-feature-viewport";
  container.appendChild(viewport);

  const paint = () => {
    const scrollTop = container.scrollTop;
    const start = Math.max(0, Math.floor(scrollTop / FEATURE_VIRTUAL_ROW_HEIGHT) - FEATURE_VIRTUAL_OVERSCAN);
    const end = Math.min(
      features.length,
      start + Math.ceil(viewportHeight / FEATURE_VIRTUAL_ROW_HEIGHT) + FEATURE_VIRTUAL_OVERSCAN * 2,
    );
    viewport.style.transform = `translateY(${start * FEATURE_VIRTUAL_ROW_HEIGHT}px)`;
    viewport.innerHTML = features.slice(start, end).map((name) => {
      const checked = state.ablationDesignFeatureSelection.includes(name) ? " checked" : "";
      return `
        <label class="ablation-design-feature-item">
          <input type="checkbox" data-feature-name="${escapeHtml(name)}"${checked}>
          <span>${escapeHtml(name)}</span>
        </label>
      `;
    }).join("");
  };

  if (container._virtualPaint) {
    container.removeEventListener("scroll", container._virtualPaint);
    container.removeEventListener("virtual-refresh", container._virtualPaint);
  }
  if (!container._virtualChangeBound) {
    container._virtualChangeBound = true;
    container.addEventListener("change", (event) => {
      const input = event.target.closest("input[data-feature-name]");
      if (!input) return;
      ablationDesignToggleFeatureSelection(input.dataset.featureName, input.checked, { skipRender: true });
      paint();
    });
  }
  container._virtualPaint = paint;
  container.addEventListener("scroll", paint, { passive: true });
  container.addEventListener("virtual-refresh", paint);
  paint();
}

function ablationDesignRefreshFeatureLists() {
  document.querySelectorAll(".ablation-design-feature-list-virtual").forEach((node) => {
    node.dispatchEvent(new Event("virtual-refresh"));
  });
}

function scheduleAblationDesignPreview() {
  window.clearTimeout(ablationDesignPreviewTimer);
  ablationDesignPreviewTimer = window.setTimeout(() => {
    void previewAblationDesign({ silent: true, auto: true });
  }, 420);
}

function ablationDesignVisibleFeatureEntries(catalog = state.ablationDesignFeatureCatalog) {
  if (!catalog?.ok) return [];
  const filter = String(state.ablationDesignFeatureFilter || "").trim().toLowerCase();
  const entries = [];
  ABLATION_DESIGN_FEATURE_GROUPS.forEach(([key, label]) => {
    (catalog.feature_groups?.[key] || []).forEach((name) => {
      const text = String(name);
      if (!filter || text.toLowerCase().includes(filter)) {
        entries.push({ name: text, groupKey: key, groupLabel: label });
      }
    });
  });
  return entries;
}

function ablationDesignSetFeatureSelection(names, mode = "replace") {
  const normalized = [...new Set(names.map(String))];
  if (mode === "replace") {
    state.ablationDesignFeatureSelection = normalized;
  } else if (mode === "add") {
    state.ablationDesignFeatureSelection = [...new Set([
      ...(state.ablationDesignFeatureSelection || []),
      ...normalized,
    ])];
  } else if (mode === "toggle") {
    const current = new Set(state.ablationDesignFeatureSelection || []);
    normalized.forEach((name) => {
      if (current.has(name)) current.delete(name);
      else current.add(name);
    });
    state.ablationDesignFeatureSelection = [...current];
  }
  renderAblationDesignFeatureBrowser();
}

function ablationDesignResolveApplyPatterns(options = {}) {
  const { preferWildcard = true, inferFromSelection = true } = options;
  const wildcardPatterns = ablationDesignWildcardPatterns();
  if (preferWildcard && wildcardPatterns.length) return wildcardPatterns;

  const selected = ablationDesignSelectedFeatures();
  if (!selected.length) return [];

  if (inferFromSelection) {
    const catalog = state.ablationDesignFeatureCatalog;
    if (catalog?.feature_groups) {
      const groupNames = Object.keys(catalog.feature_groups);
      const matchingGroups = groupNames.filter((group) => (
        selected.every((name) => (catalog.feature_groups[group] || []).includes(name))
      ));
      const inferred = [...new Set(matchingGroups.flatMap((group) => (
        ablationDesignGroupPatterns(group, catalog.feature_groups[group] || [])
      )))];
      if (inferred.length) return inferred;
    }
  }
  return selected;
}

function ablationDesignResolveApplyFeatures(options = {}) {
  const { preferWildcard = true } = options;
  const wildcardPatterns = ablationDesignWildcardPatterns();
  if (preferWildcard && wildcardPatterns.length) {
    return ablationDesignFeaturesMatchingPatterns(
      ablationDesignAllCandidateFeatures(),
      wildcardPatterns,
    );
  }
  return ablationDesignSelectedFeatures();
}

function ablationDesignAppendLines(fieldId, values) {
  const node = $(fieldId);
  if (!node || !values.length) return;
  const existing = new Set(ablationDesignLines(node.value));
  const merged = [...existing];
  values.forEach((value) => {
    if (!existing.has(value)) merged.push(value);
  });
  node.value = merged.join("\n");
  writeAblationDesignDraftFromForm();
}

function ablationDesignSelectedFeatures() {
  return [...(state.ablationDesignFeatureSelection || [])];
}

function ablationDesignFeatureSummaryText(catalog = state.ablationDesignFeatureCatalog) {
  if (!catalog?.ok) return "";
  const metadata = (catalog.metadata_columns || []).join(", ") || "—";
  const targets = (catalog.target_columns || []).join(", ") || "—";
  const wildcardPatterns = ablationDesignWildcardPatterns();
  const matchedCount = wildcardPatterns.length
    ? ablationDesignFeaturesMatchingPatterns(ablationDesignAllCandidateFeatures(catalog), wildcardPatterns).length
    : 0;
  const selectionCount = (state.ablationDesignFeatureSelection || []).length;
  let text =
    `${catalog.candidate_feature_count} candidate descriptors from ${catalog.feature_source} `
    + `(metadata stripped: ${metadata}; targets: ${targets})`;
  if (wildcardPatterns.length) text += ` · wildcard matches ${matchedCount}`;
  if (selectionCount) text += ` · ${selectionCount} selected`;
  return text;
}

function ablationDesignUpdateFeatureSummary() {
  const summary = $("ablation-design-feature-summary");
  if (summary) summary.textContent = ablationDesignFeatureSummaryText();
}

function ablationDesignToggleFeatureSelection(name, selected, options = {}) {
  const current = new Set(state.ablationDesignFeatureSelection || []);
  if (selected) current.add(name);
  else current.delete(name);
  state.ablationDesignFeatureSelection = [...current];
  ablationDesignUpdateFeatureSummary();
  if (options.skipRender) return;
  ablationDesignRefreshFeatureLists();
}

function renderAblationDesignFeatureBrowser() {
  const browser = $("ablation-design-feature-browser");
  const actions = $("ablation-design-feature-actions");
  const summary = $("ablation-design-feature-summary");
  const catalog = state.ablationDesignFeatureCatalog;
  if (!browser || !summary) return;

  if (!catalog?.ok) {
    browser.hidden = true;
    if (actions) actions.hidden = true;
    const batchBar = $("ablation-design-batch-bar");
    if (batchBar) batchBar.hidden = true;
    summary.textContent = state.ablationDesignFeaturesLoading
      ? "Loading descriptor columns from raw_prepare/…"
      : (catalog?.error
        || "Expected raw_prepare/ under the served Workbench root.");
    return;
  }

  const filter = String(state.ablationDesignFeatureFilter || "").trim().toLowerCase();
  summary.textContent = ablationDesignFeatureSummaryText(catalog);

  const batchBar = $("ablation-design-batch-bar");
  if (batchBar) batchBar.hidden = false;

  browser.innerHTML = ABLATION_DESIGN_FEATURE_GROUPS.map(([key, label]) => {
    const features = (catalog.feature_groups?.[key] || []).filter((name) => (
      !filter || String(name).toLowerCase().includes(filter)
    ));
    return `
      <section class="ablation-design-feature-group" data-feature-group="${escapeHtml(key)}">
        <div class="ablation-design-feature-group-head">
          <span>${escapeHtml(label)}</span>
          <span class="ablation-design-feature-group-actions">
            <button type="button" class="ablation-design-feature-group-select" data-select-group="${escapeHtml(key)}">All</button>
            <span>${features.length}</span>
          </span>
        </div>
        <div class="ablation-design-feature-list" data-feature-list="${escapeHtml(key)}"></div>
      </section>
    `;
  }).join("");

  browser.hidden = false;
  if (actions) actions.hidden = false;
  ABLATION_DESIGN_FEATURE_GROUPS.forEach(([key]) => {
    const listNode = browser.querySelector(`[data-feature-list="${key}"]`);
    if (!listNode) return;
    const features = (catalog.feature_groups?.[key] || []).filter((name) => (
      !filter || String(name).toLowerCase().includes(filter)
    ));
    mountVirtualFeatureList(listNode, features);
  });
  ablationDesignUpdatePatternFeedback();
  renderAblationDesignTemplateDiff();
  browser.querySelectorAll("[data-select-group]").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.preventDefault();
      const groupKey = button.dataset.selectGroup;
      const visible = ablationDesignVisibleFeatureEntries(catalog)
        .filter((entry) => entry.groupKey === groupKey)
        .map((entry) => entry.name);
      ablationDesignSetFeatureSelection(visible, "add");
    });
  });
}

function ablationDesignInputFingerprint(draft = readAblationDesignDraft()) {
  return JSON.stringify({
    ...ablationDesignInputPayload(draft),
    feature_source: draft.feature_source || "auto",
  });
}

async function loadAblationDesignFeatures(options = {}) {
  const { silent = false, force = false } = options;
  writeAblationDesignDraftFromForm();
  applyAblationDesignDiscoveredInputs();
  const draft = readAblationDesignDraft();
  if (!ablationDesignHasInputPaths(draft)) {
    if (!silent) {
      toast(
        "Expected raw_prepare/raw_pdbbind.csv and raw_dudez.csv under the served Workbench root, "
        + "or set paths in Run settings.",
      );
    }
    return false;
  }

  const fingerprint = ablationDesignInputFingerprint(draft);
  if (!force && state.ablationDesignFeatureCatalog?.ok && state.ablationDesignFeatureCatalogKey === fingerprint) {
    renderAblationDesignFeatureBrowser();
    return true;
  }

  state.ablationDesignFeaturesLoading = true;
  renderAblationDesignFeatureBrowser();
  try {
    const payload = await apiPost("/api/ablation-design/features", ablationDesignInputPayload(draft));
    state.ablationDesignFeatureCatalog = payload;
    state.ablationDesignFeatureCatalogKey = fingerprint;
    state.ablationDesignFeatureSelection = [];
    if (payload.resolved_inputs) {
      Object.assign(ensureAblationDesignDraft(), payload.resolved_inputs);
      renderAblationDesignForm();
    } else {
      renderAblationDesignFeatureBrowser();
    }
    if (!silent) {
      toast(`Loaded ${payload.candidate_feature_count} candidate features.`);
    }
    scheduleAblationDesignPreview();
    return true;
  } catch (error) {
    state.ablationDesignFeatureCatalog = { ok: false, error: error.message || String(error) };
    state.ablationDesignFeatureCatalogKey = "";
    renderAblationDesignFeatureBrowser();
    if (!silent) toast(error.message || String(error));
    return false;
  } finally {
    state.ablationDesignFeaturesLoading = false;
  }
}

async function ensureAblationDesignFeatures(force = false) {
  applyAblationDesignDiscoveredInputs(state.ablationDesignContext);
  return loadAblationDesignFeatures({ silent: true, force });
}

async function loadAblationDesignFeaturesManual() {
  await loadAblationDesignFeatures({ silent: false, force: true });
}

function applyAblationDesignIncludeFeatureRules() {
  const features = ablationDesignResolveApplyFeatures();
  if (!features.length) {
    toast("Enter a wildcard pattern or select features first.");
    return;
  }
  ablationDesignAppendLines("ablation-design-include-features", features);
  toast(`Added ${features.length} include feature${features.length === 1 ? "" : "s"}.`);
}

function applyAblationDesignIncludePatternRules() {
  const patterns = ablationDesignResolveApplyPatterns();
  if (!patterns.length) {
    toast("Enter a wildcard pattern or select features first.");
    return;
  }
  ablationDesignAppendLines("ablation-design-include-patterns", patterns);
  toast(`Added ${patterns.length} include pattern${patterns.length === 1 ? "" : "s"}.`);
}

function applyAblationDesignExcludePatternRules() {
  const patterns = ablationDesignResolveApplyPatterns();
  if (!patterns.length) {
    toast("Enter a wildcard pattern or select features first.");
    return;
  }
  ablationDesignAppendLines("ablation-design-exclude-patterns", patterns);
  toast(`Added ${patterns.length} exclude pattern${patterns.length === 1 ? "" : "s"}.`);
}

function applyAblationDesignExcludeFeatureRules() {
  const features = ablationDesignResolveApplyFeatures();
  if (!features.length) {
    toast("Enter a wildcard pattern or select features first.");
    return;
  }
  ablationDesignAppendLines("ablation-design-exclude-features", features);
  toast(`Added ${features.length} exclude feature${features.length === 1 ? "" : "s"}.`);
}

function selectAblationDesignMatchedFeatures() {
  const patterns = ablationDesignWildcardPatterns();
  if (!patterns.length) {
    toast("Enter a wildcard pattern first (for example ligand_*).");
    return;
  }
  const matched = ablationDesignFeaturesMatchingPatterns(
    ablationDesignAllCandidateFeatures(),
    patterns,
  );
  if (!matched.length) {
    toast("No loaded features matched the wildcard pattern.");
    return;
  }
  ablationDesignSetFeatureSelection(matched, "replace");
  toast(`Selected ${matched.length} matched feature${matched.length === 1 ? "" : "s"}.`);
}

function selectAblationDesignVisibleFeatures() {
  const visible = ablationDesignVisibleFeatureEntries().map((entry) => entry.name);
  if (!visible.length) {
    toast("No visible features to select.");
    return;
  }
  ablationDesignSetFeatureSelection(visible, "replace");
  toast(`Selected ${visible.length} visible feature${visible.length === 1 ? "" : "s"}.`);
}

function invertAblationDesignVisibleFeatures() {
  const visible = new Set(ablationDesignVisibleFeatureEntries().map((entry) => entry.name));
  if (!visible.size) {
    toast("No visible features to invert.");
    return;
  }
  const current = new Set(state.ablationDesignFeatureSelection || []);
  visible.forEach((name) => {
    if (current.has(name)) current.delete(name);
    else current.add(name);
  });
  state.ablationDesignFeatureSelection = [...current];
  renderAblationDesignFeatureBrowser();
  toast("Inverted visible selection.");
}

function updateAblationDesignTemplateDescription() {
  const node = $("ablation-design-template-description");
  const templateName = $("ablation-design-template")?.value || "";
  if (!node) return;
  if (!templateName) {
    node.textContent = "Start from scratch or clone a bundled feature policy.";
    return;
  }
  const entry = (state.ablationDesignContext?.catalog || []).find((item) => item.name === templateName);
  node.textContent = entry?.description || "";
}

function applyAblationDesignTemplate(templateName) {
  const entry = (state.ablationDesignContext?.catalog || []).find((item) => item.name === templateName);
  const draft = ensureAblationDesignDraft();
  draft.templateName = templateName || "";
  if (!entry) {
    renderAblationDesignForm();
    persistUiState();
    return;
  }
  const request = entry.request || {};
  draft.name = templateName.startsWith("custom_") ? templateName : `custom_${templateName}`;
  draft.description = entry.description || request.description || "";
  draft.include_features = (request.include_features || []).join("\n");
  draft.include_patterns = (request.include_patterns || []).join("\n");
  if (!draft.include_features && !draft.include_patterns) draft.include_patterns = "*";
  draft.exclude_features = (request.exclude_features || []).join("\n");
  draft.exclude_patterns = (request.exclude_patterns || []).join("\n");
  draft.allow_missing_exclude_features = request.allow_missing_exclude_features !== false;
  syncAblationDesignOutputPaths(draft, state.ablationDesignContext);
  renderAblationDesignForm();
  persistUiState();
  scheduleAblationDesignPreview();
}

function renderAblationDesignPreview(payload) {
  state.ablationDesignPreview = payload;
  const panel = $("ablation-design-preview-panel");
  const yamlNode = $("ablation-design-yaml");
  if (!panel || !yamlNode) return;

  if (!payload?.ok) {
    panel.hidden = false;
    $("ablation-design-preview-summary").textContent = payload?.error || "Preview failed.";
    $("ablation-design-preview-details").innerHTML = "";
    yamlNode.hidden = true;
    return;
  }

  panel.hidden = false;
  yamlNode.hidden = false;
  yamlNode.textContent = payload.policy_yaml || "";

  if (!payload.preview_available) {
    $("ablation-design-preview-summary").textContent = payload.message || "Preview unavailable.";
    $("ablation-design-preview-details").innerHTML = "";
    return;
  }

  $("ablation-design-preview-summary").textContent =
    `${payload.kept_feature_count} kept · ${payload.excluded_feature_count} excluded · ${payload.candidate_feature_count} candidates`;
  const details = [
    ["Kept (sample)", (payload.kept_features_sample || []).join(", ") || "—"],
    ["Excluded (sample)", (payload.excluded_features_sample || []).join(", ") || "—"],
    ["Missing excludes", (payload.missing_exclude_features || []).join(", ") || "—"],
    ["Unused patterns", (payload.patterns_with_no_matches || []).join(", ") || "—"],
  ];
  $("ablation-design-preview-details").innerHTML = details.map(([label, value]) => `
    <div><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</div>
  `).join("");
  renderAblationDesignTemplateDiff();
}

function renderAblationDesignPlan(payload) {
  state.ablationDesignPlan = payload;
  const commandNode = $("ablation-design-command");
  const preflightNode = $("ablation-design-preflight");
  const copyButton = $("ablation-design-copy-command");
  if (!commandNode || !preflightNode) return;

  if (!payload?.ok) {
    commandNode.hidden = false;
    commandNode.textContent = payload?.error || "Plan failed.";
    preflightNode.hidden = true;
    if (copyButton) copyButton.disabled = true;
    return;
  }

  if (payload.policy_yaml) {
    $("ablation-design-yaml").hidden = false;
    $("ablation-design-yaml").textContent = payload.policy_yaml;
  }
  commandNode.hidden = false;
  commandNode.textContent = payload.planned_command || "";
  if (copyButton) copyButton.disabled = !payload.planned_command;

  const checks = payload.preflight?.checks || [];
  preflightNode.hidden = !checks.length;
  preflightNode.innerHTML = checks.map((check) => {
    const status = check.passed ? "ok" : check.severity === "warning" ? "warning" : "error";
    return `
      <div class="ablation-design-check ${status}">
        <strong>${escapeHtml(check.subject || check.code || "check")}</strong>
        ${escapeHtml(check.message || "")}
      </div>
    `;
  }).join("");
}

async function ensureAblationDesignContext(force = false) {
  if (state.ablationDesignContext && !force) {
    renderAblationDesignForm();
    await ensureAblationDesignFeatures(false);
    return state.ablationDesignContext;
  }
  try {
    const context = await api("/api/ablation-design");
    state.ablationDesignContext = context;
    renderAblationDesignForm();
    await ensureAblationDesignFeatures(force);
    return context;
  } catch (error) {
    toast(error.message || String(error));
    return null;
  }
}

async function previewAblationDesign(options = {}) {
  const { silent = false, auto = false } = options;
  writeAblationDesignDraftFromForm();
  const draft = readAblationDesignDraft();
  if (!draft.name) {
    if (!silent && !auto) toast("Policy name is required.");
    return;
  }
  if (!state.ablationDesignFeatureCatalog?.ok) {
    if (!silent && !auto) toast("Load features before previewing.");
    return;
  }
  const summaryNode = $("ablation-design-preview-summary");
  if (auto && summaryNode) summaryNode.textContent = "Previewing…";
  try {
    const payload = await apiPost("/api/ablation-design/preview", ablationDesignRequestPayload(draft));
    renderAblationDesignPreview(payload);
  } catch (error) {
    if (!silent || !auto) {
      renderAblationDesignPreview({ ok: false, error: error.message || String(error) });
    } else if (summaryNode) {
      summaryNode.textContent = error.message || String(error);
    }
  }
}

async function planAblationDesign() {
  writeAblationDesignDraftFromForm();
  const draft = readAblationDesignDraft();
  if (!draft.name) {
    toast("Policy name is required.");
    return;
  }
  if (!draft.protocol || !ablationDesignHasInputPaths(draft)) {
    toast("Protocol path and at least one input path are required to generate a plan.");
    return;
  }
  try {
    const payload = await apiPost("/api/ablation-design/plan", ablationDesignRequestPayload(draft));
    renderAblationDesignPlan(payload);
    if (payload.policy_yaml) renderAblationDesignPreview(payload);
  } catch (error) {
    renderAblationDesignPlan({ ok: false, error: error.message || String(error) });
  }
}

function downloadAblationDesignYaml() {
  writeAblationDesignDraftFromForm();
  const yamlText = state.ablationDesignPlan?.policy_yaml
    || state.ablationDesignPreview?.policy_yaml
    || "";
  if (!yamlText) {
    toast("Generate a preview or plan first.");
    return;
  }
  const draft = readAblationDesignDraft();
  const filename = `${draft.name || "feature_policy"}.yml`;
  const blob = new Blob([yamlText], { type: "text/yaml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function copyAblationDesignCommand() {
  const command = state.ablationDesignPlan?.planned_command || "";
  if (!command) {
    toast("Generate a plan first.");
    return;
  }
  try {
    await navigator.clipboard.writeText(command);
    toast("Command copied.");
  } catch (_error) {
    toast(command);
  }
}

async function writeAblationDesignPolicy() {
  writeAblationDesignDraftFromForm();
  const draft = readAblationDesignDraft();
  const yamlText = state.ablationDesignPlan?.policy_yaml
    || state.ablationDesignPreview?.policy_yaml
    || "";
  if (!yamlText) {
    toast("Generate a preview or plan first.");
    return;
  }
  if (!draft.policy_yml_path) {
    toast("Policy YAML path is required.");
    return;
  }
  const shippedTemplate = draft.templateName
    && (state.ablationDesignContext?.catalog || []).some(
      (item) => item.name === draft.templateName && item.source_kind === "bundled",
    );
  const overwriteNote = draft.name === draft.templateName && shippedTemplate
    ? "\n\nThis uses a shipped template name — the file will be written under your workspace, not the bundled copy."
    : "";
  const confirmed = window.confirm(
    `Write policy YAML to:\n${draft.policy_yml_path}\n\nThis modifies your served workspace.${overwriteNote}`,
  );
  if (!confirmed) return;
  try {
    const payload = await apiPost("/api/ablation-design/write", {
      ...ablationDesignRequestPayload(draft),
      policy_yaml: yamlText,
      confirm: true,
      overwrite: true,
    });
    toast(`Policy written to ${payload.written_path || draft.policy_yml_path}`);
    await ensureAblationDesignContext(true);
  } catch (error) {
    toast(error.message || String(error));
  }
}

function bindAblationDesignPanel() {
  const templateSelect = $("ablation-design-template");
  if (templateSelect && templateSelect.dataset.bound !== "true") {
    templateSelect.dataset.bound = "true";
    templateSelect.addEventListener("change", (event) => {
      applyAblationDesignTemplate(event.target.value || "");
    });
  }
  const bindInput = (id, handler) => {
    const node = $(id);
    if (!node || node.dataset.bound === "true") return;
    node.dataset.bound = "true";
    node.addEventListener("input", handler);
    node.addEventListener("change", handler);
  };
  bindInput("ablation-design-name", () => {
    writeAblationDesignDraftFromForm();
    syncAblationDesignOutputPaths();
    $("ablation-design-output-dir").value = state.ablationDesign.output_dir || "";
    $("ablation-design-policy-path").value = state.ablationDesign.policy_yml_path || "";
    renderAblationDesignTemplateDiff();
    scheduleAblationDesignPreview();
  });
  const schedulePreviewOnEdit = () => {
    writeAblationDesignDraftFromForm();
    renderAblationDesignTemplateDiff();
    ablationDesignUpdatePatternFeedback();
    scheduleAblationDesignPreview();
  };
  [
    "ablation-design-description",
    "ablation-design-include-features",
    "ablation-design-include-patterns",
    "ablation-design-exclude-features",
    "ablation-design-exclude-patterns",
    "ablation-design-allow-missing-excludes",
    "ablation-design-protocol",
    "ablation-design-output-dir",
    "ablation-design-policy-path",
  ].forEach((id) => bindInput(id, schedulePreviewOnEdit));

  const reloadFeaturesOnInputChange = () => {
    writeAblationDesignDraftFromForm();
    if (state.activeTab === "design") void ensureAblationDesignFeatures(true);
  };
  [
    "ablation-design-raw-input",
    "ablation-design-merged-input",
    "ablation-design-pdbbind-input",
    "ablation-design-dudez-input",
    "ablation-design-feature-source",
  ].forEach((id) => bindInput(id, reloadFeaturesOnInputChange));

  $("ablation-design-feature-filter")?.addEventListener("input", (event) => {
    state.ablationDesignFeatureFilter = event.target.value || "";
    renderAblationDesignFeatureBrowser();
    ablationDesignUpdateCaseAndFilterWarnings();
  });
  $("ablation-design-wildcard-pattern")?.addEventListener("input", (event) => {
    state.ablationDesignWildcardPattern = event.target.value || "";
    ablationDesignUpdateFeatureSummary();
    ablationDesignUpdatePatternFeedback();
    persistUiState();
  });
  $("ablation-design-load-features")?.addEventListener("click", loadAblationDesignFeaturesManual);
  $("ablation-design-select-matched")?.addEventListener("click", selectAblationDesignMatchedFeatures);
  $("ablation-design-select-visible")?.addEventListener("click", selectAblationDesignVisibleFeatures);
  $("ablation-design-invert-visible")?.addEventListener("click", invertAblationDesignVisibleFeatures);
  $("ablation-design-clear-selection")?.addEventListener("click", () => {
    state.ablationDesignFeatureSelection = [];
    renderAblationDesignFeatureBrowser();
  });
  $("ablation-design-apply-include-features")?.addEventListener("click", applyAblationDesignIncludeFeatureRules);
  $("ablation-design-apply-include-pattern")?.addEventListener("click", applyAblationDesignIncludePatternRules);
  $("ablation-design-apply-exclude-pattern")?.addEventListener("click", applyAblationDesignExcludePatternRules);
  $("ablation-design-apply-exclude-features")?.addEventListener("click", applyAblationDesignExcludeFeatureRules);
  $("ablation-design-preview")?.addEventListener("click", () => previewAblationDesign());
  $("ablation-design-plan")?.addEventListener("click", planAblationDesign);
  $("ablation-design-download-yaml")?.addEventListener("click", downloadAblationDesignYaml);
  $("ablation-design-copy-command")?.addEventListener("click", copyAblationDesignCommand);
  $("ablation-design-write-policy")?.addEventListener("click", writeAblationDesignPolicy);
}

function renderVsDesignCandidateSelect(selectId, candidates) {
  const select = $(selectId);
  if (!select) return;
  const current = select.value;
  select.innerHTML = '<option value="">— pick a candidate —</option>' + (candidates || []).map((item) => (
    `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name)}</option>`
  )).join("");
  if (current && (candidates || []).some((item) => item.path === current)) select.value = current;
}

function renderVsDesignDiscover(context) {
  state.vsDesignContext = context;
  const summary = $("vs-design-discover-summary");
  const issuesNode = $("vs-design-discover-issues");
  if (!context) {
    if (summary) summary.textContent = "";
    if (issuesNode) issuesNode.hidden = true;
    return;
  }
  const candidates = context.candidates || { receptors: [], ligands: [], boxes: [] };
  if (summary) {
    summary.textContent = `${candidates.receptors.length} receptor(s) · ${candidates.ligands.length} ligand(s) · ${candidates.boxes.length} box(es)`;
  }
  renderVsDesignCandidateSelect("vs-design-receptor-select", candidates.receptors);
  renderVsDesignCandidateSelect("vs-design-ligand-select", candidates.ligands);
  renderVsDesignCandidateSelect("vs-design-box-select", candidates.boxes);
  if (issuesNode) {
    const issues = context.issues || [];
    issuesNode.hidden = issues.length === 0;
    issuesNode.textContent = issues.join(" ");
  }
}

async function discoverVsDesignCandidates() {
  const inputDir = $("vs-design-input-dir").value.trim();
  try {
    const context = await api("/api/vs-design", inputDir ? { input_dir: inputDir } : {});
    renderVsDesignDiscover(context);
  } catch (error) {
    toast(error.message || String(error));
  }
}

async function ensureVsDesignContext(force = false) {
  if (state.vsDesignContext && !force) return state.vsDesignContext;
  await discoverVsDesignCandidates();
  return state.vsDesignContext;
}

function toggleVsDesignKindFields() {
  const kind = $("vs-design-kind").value;
  document.querySelectorAll("[data-vs-design-kind-fields]").forEach((node) => {
    node.hidden = node.dataset.vsDesignKindFields !== kind;
  });
}

function splitVsDesignEngineList(value) {
  return String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
}

function readVsDesignDraft() {
  const kind = $("vs-design-kind").value;
  const draft = {
    kind,
    receptor: $("vs-design-receptor").value.trim(),
    ligand: $("vs-design-ligand").value.trim(),
    box: $("vs-design-box").value.trim(),
    all_boxes: $("vs-design-all-boxes").checked,
    store_db: $("vs-design-store-db").checked,
  };
  const name = $("vs-design-name").value.trim();
  if (name) draft.name = name;
  const outdir = $("vs-design-outdir").value.trim();
  if (outdir) draft.outdir = outdir;
  const cwd = $("vs-design-cwd").value.trim();
  if (cwd) draft.cwd = cwd;
  const timeout = $("vs-design-timeout").value.trim();
  if (timeout) draft.timeout = Number(timeout);

  if (kind === "vs") {
    draft.engine = $("vs-design-engine").value;
    draft.skip_rescore = $("vs-design-skip-rescore").checked;
    draft.skip_split = $("vs-design-skip-split").checked;
  } else {
    draft.engines = splitVsDesignEngineList($("vs-design-engines").value);
    const rescoringEngines = splitVsDesignEngineList($("vs-design-rescoring-engines").value);
    if (rescoringEngines.length) draft.rescoring_engines = rescoringEngines;
    const clusterMin = $("vs-design-cluster-min").value.trim();
    if (clusterMin) draft.cluster_min = Number(clusterMin);
    const clusterMax = $("vs-design-cluster-max").value.trim();
    if (clusterMax) draft.cluster_max = Number(clusterMax);
    const clusterStep = $("vs-design-cluster-step").value.trim();
    if (clusterStep) draft.cluster_step = Number(clusterStep);
    draft.strict_engines = $("vs-design-strict-engines").checked;
  }
  return draft;
}

function renderVsDesignPreview(payload) {
  state.vsDesignPreview = payload;
  const panel = $("vs-design-preview-panel");
  if (!panel) return;
  panel.hidden = false;

  if (!payload || payload.valid === undefined) {
    $("vs-design-preview-summary").textContent = payload?.error || "Preview failed.";
    $("vs-design-preview-details").innerHTML = "";
    return;
  }

  $("vs-design-preview-summary").textContent = payload.valid
    ? "Draft is valid."
    : `Draft is invalid — ${payload.errors.length} error(s).`;

  const details = [];
  if (payload.errors?.length) details.push(["error", "Errors", payload.errors.join(" ")]);
  if (payload.warnings?.length) details.push(["warning", "Warnings", payload.warnings.join(" ")]);
  $("vs-design-preview-details").innerHTML = details.map(([cssClass, label, value]) => `
    <div class="${cssClass}"><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</div>
  `).join("");
}

async function previewVsDesign() {
  const draft = readVsDesignDraft();
  if (!draft.receptor || !draft.ligand || !draft.box) {
    toast("Receptor, ligand, and box paths are required.");
    return;
  }
  try {
    const payload = await apiPost("/api/vs-design/preview", draft);
    renderVsDesignPreview(payload);
  } catch (error) {
    renderVsDesignPreview({ error: error.message || String(error) });
  }
}

function renderVsDesignPlan(payload) {
  state.vsDesignPlan = payload;
  const commandNode = $("vs-design-command");
  const sendButton = $("vs-design-send-to-jobs");
  if (!commandNode) return;
  commandNode.hidden = false;
  commandNode.textContent = payload?.shell_command || payload?.error || "Plan failed.";
  if (sendButton) sendButton.disabled = !payload?.args;
}

async function planVsDesign() {
  const draft = readVsDesignDraft();
  if (!draft.receptor || !draft.ligand || !draft.box) {
    toast("Receptor, ligand, and box paths are required.");
    return;
  }
  try {
    const payload = await apiPost("/api/vs-design/plan", draft);
    renderVsDesignPlan(payload);
    renderVsDesignPreview({ valid: true, errors: [], warnings: [] });
  } catch (error) {
    renderVsDesignPlan({ error: error.message || String(error) });
  }
}

function sendVsDesignPlanToJobs() {
  const plan = state.vsDesignPlan;
  if (!plan?.args) {
    toast("Generate a plan first.");
    return;
  }
  $("jobs-launch-kind").value = plan.kind;
  $("jobs-launch-args").value = plan.args.join("\n");
  $("jobs-launch-cwd").value = plan.cwd || "";
  setActiveTab("jobs");
  toast("Plan loaded into the Jobs tab — review and click Launch job.");
}

function bindVsDesignPanel() {
  $("vs-design-discover")?.addEventListener("click", () => void discoverVsDesignCandidates());
  $("vs-design-receptor-select")?.addEventListener("change", (event) => {
    if (event.target.value) $("vs-design-receptor").value = event.target.value;
  });
  $("vs-design-ligand-select")?.addEventListener("change", (event) => {
    if (event.target.value) $("vs-design-ligand").value = event.target.value;
  });
  $("vs-design-box-select")?.addEventListener("change", (event) => {
    if (event.target.value) $("vs-design-box").value = event.target.value;
  });
  $("vs-design-kind")?.addEventListener("change", toggleVsDesignKindFields);
  $("vs-design-preview")?.addEventListener("click", () => void previewVsDesign());
  $("vs-design-plan")?.addEventListener("click", () => void planVsDesign());
  $("vs-design-send-to-jobs")?.addEventListener("click", sendVsDesignPlanToJobs);
  toggleVsDesignKindFields();
}

function pathBasename(path) {
  let text = String(path || "");
  while (text.length > 0) {
    const last = text.charAt(text.length - 1);
    if (last !== "/" && last !== "\\") break;
    text = text.slice(0, -1);
  }
  const slash = text.lastIndexOf("/");
  const backslash = text.lastIndexOf("\\");
  const index = slash > backslash ? slash : backslash;
  return index >= 0 ? text.slice(index + 1) : text;
}

function dashboardModelLabel(model) {
  if (model === "strict_ocscore_layout") return "Strict OCScore layout";
  return model || "Unknown";
}

function selectedModelLabel() {
  if (!state.selectedStudy) return "—";
  return studyDisplayName(state.selectedStudy);
}

function runContextSelectionItems() {
  return [
    ["Reference", comparisonReferenceLabel()],
    ["Selected", selectedModelLabel()],
  ];
}

function compactSplitSummary(context) {
  const summary = context.pdbbind_split_summary || context.pdbbind_split_strategy || "—";
  const strategy = summary.split("·")[0].trim().replace(/_/g, " ");
  const sizes = summary.match(/(\d+)%/g);
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
  }).join("\n");
  return { label, title };
}

function renderRunContext(payload = state.workspace) {
  const context = payload?.run_context;
  const strip = $("run-context-items");
  if (!strip) return;
  if (!context) {
    strip.innerHTML = runContextSelectionItems().map(([label, value]) => `
      <div class="run-context-item run-context-item-selection">
        <strong>${escapeHtml(label)}</strong>
        <span>${escapeHtml(String(value))}</span>
      </div>
    `).join("");
    bindRunContextMarquee();
    return;
  }
  const splitShort = compactSplitSummary(context);
  const baseline = summarizeBaselineSources(context.baseline_sources || []);
  const items = [
    ...runContextSelectionItems(),
    ["Repl", `${context.completed_replica_count ?? 0}/${context.planned_replica_count}`],
    ["Split", splitShort],
    ["Rank", "DUDEz test"],
    ["Reg", "PDBbind val/test"],
    ["BEDROC", context.dudez_bedroc_alpha != null ? `α=${numeric(context.dudez_bedroc_alpha)}` : "—"],
    ["EF", "1%, 5%"],
  ];
  if (baseline) items.push(["CSV", baseline.label, baseline.title]);
  strip.innerHTML = items.map(([label, value, title]) => `
    <div class="run-context-item${label === "CSV" ? " path-item" : ""}${label === "Reference" || label === "Selected" ? " run-context-item-selection" : ""}">
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

function protocolSimilarityMetricToken() {
  const metricName = ensureSelectedMetric();
  if (!metricName) return "";
  const meta = metricMeta(metricName);
  const mode = meta.direction === "min" ? "min" : "max";
  return `${metricName}:${mode}`;
}

function scheduleProtocolSimilarityLoad() {
  window.clearTimeout(protocolSimilarityTimer);
  protocolSimilarityTimer = window.setTimeout(() => {
    void loadProtocolSimilarity();
  }, 180);
}

function bindProtocolSimilarityControls() {
  const select = $("protocol-similarity-reference");
  if (!select || select.dataset.bound === "true") return;
  select.dataset.bound = "true";
  select.addEventListener("change", () => {
    state.protocolSimilarityReference = select.value || null;
    persistUiState();
    scheduleProtocolSimilarityLoad();
  });
}

function renderProtocolSimilarityReferenceSelect(payload) {
  const wrap = $("protocol-similarity-reference-wrap");
  const select = $("protocol-similarity-reference");
  if (!wrap || !select || !payload) return;
  const names = protocolSimilarityAllNames(payload);
  if (!names.length) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const selected = (state.protocolSimilarityReference && names.includes(state.protocolSimilarityReference))
    ? state.protocolSimilarityReference
    : (payload.reference_policy && names.includes(payload.reference_policy) ? payload.reference_policy : names[0]);
  select.innerHTML = names.map((name) => (
    `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`
  )).join("");
  select.value = selected;
  if (selected !== state.protocolSimilarityReference) {
    state.protocolSimilarityReference = selected;
  }
}

function protocolSimilarityAllNames(payload) {
  return payload?.protocol_order || [];
}

function protocolSimilarityFilteredTableNames(payload, visibility) {
  const order = protocolSimilarityAllNames(payload);
  const categories = protocolSimilarityClusterCategories(payload);
  if (!categories.length) return order;
  const allShown = categories.every((category) => protocolSimilarityFilterShown(visibility, category.key));
  if (allShown) return order;
  const visible = new Set();
  categories.forEach((category) => {
    if (!protocolSimilarityFilterShown(visibility, category.key)) return;
    category.protocols.forEach((name) => visible.add(name));
  });
  return order.filter((name) => visible.has(name));
}

const PROTOCOL_SIMILARITY_LEGEND_KEY = "protocol-similarity";
const PROTOCOL_SIMILARITY_CLUSTER_COLORS = [
  "#2F6FDE",
  "#12B886",
  "#F59F00",
  "#E64980",
  "#7950F2",
  "#22B8CF",
  "#94D82D",
  "#FF6B6B",
  "#868E96",
  "#FAB005",
];

function protocolSimilarityEntry(payload, policyName) {
  return (payload?.protocols || []).find((item) => item.policy_name === policyName) || null;
}

function protocolSimilarityClusterCategories(payload) {
  const order = payload?.protocol_order || [];
  const labels = payload?.cluster_labels || [];
  const clusters = new Map();
  order.forEach((name, index) => {
    const clusterId = labels[index];
    if (clusterId === undefined) return;
    if (!clusters.has(clusterId)) clusters.set(clusterId, []);
    clusters.get(clusterId).push(name);
  });
  return [...clusters.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([clusterId, protocols], index) => ({
      key: `cluster-${clusterId}`,
      clusterId,
      label: `Cluster ${clusterId + 1}`,
      count: protocols.length,
      protocols,
      color: PROTOCOL_SIMILARITY_CLUSTER_COLORS[index % PROTOCOL_SIMILARITY_CLUSTER_COLORS.length],
    }));
}

function protocolSimilarityPayloadKey(payload) {
  return `${payload.protocol_count}:${payload.reference_policy}:${(payload.protocol_order || []).join("|")}`;
}

function ensureProtocolSimilarityPlotPayload(payload) {
  const sourceKey = protocolSimilarityPayloadKey(payload);
  const categories = protocolSimilarityClusterCategories(payload);
  if (!state.protocolSimilarityPlotPayload || state.protocolSimilarityPlotPayload.sourceKey !== sourceKey) {
    state.protocolSimilarityPlotPayload = {
      sourceKey,
      payload,
      categories,
      categoryVisibility: Object.fromEntries(categories.map((category) => [category.key, true])),
      plotKey: PROTOCOL_SIMILARITY_LEGEND_KEY,
    };
  } else {
    state.protocolSimilarityPlotPayload.payload = payload;
    state.protocolSimilarityPlotPayload.categories = categories;
    categories.forEach((category) => {
      if (state.protocolSimilarityPlotPayload.categoryVisibility[category.key] === undefined) {
        state.protocolSimilarityPlotPayload.categoryVisibility[category.key] = true;
      }
    });
  }
  return state.protocolSimilarityPlotPayload;
}

function protocolSimilarityFilterVisibilityState(plotPayload) {
  const categories = plotPayload.categories || [];
  if (!plotPayload.categoryVisibility) {
    plotPayload.categoryVisibility = Object.fromEntries(categories.map((category) => [category.key, true]));
  }
  categories.forEach((category) => {
    if (plotPayload.categoryVisibility[category.key] === undefined) {
      plotPayload.categoryVisibility[category.key] = true;
    }
  });
  return plotPayload.categoryVisibility;
}

function protocolSimilarityFilterShown(visibility, categoryKey) {
  return visibility[categoryKey] !== false;
}

function protocolSimilarityVisibleNames(payload, visibility) {
  return protocolSimilarityFilteredTableNames(payload, visibility);
}

function protocolSimilarityExportRows(payload, visibleNames) {
  const clusterMap = protocolSimilarityClusterMap(payload);
  const reference = payload.reference_policy;
  return (visibleNames || payload.protocol_order || []).map((name) => {
    const entry = protocolSimilarityEntry(payload, name) || {};
    return {
      protocol: name,
      cluster: clusterMap.has(name) ? clusterMap.get(name) + 1 : "",
      similarity_to_reference: protocolSimilarityToReference(payload, name),
      expanded_features: entry.expanded_feature_count ?? "",
      metric: entry.metric_value ?? "",
      has_run: entry.run_id ? "yes" : "no",
      diff_vs: reference,
    };
  });
}

function renderProtocolSimilarityExportActions(plotPayload, visibleNames) {
  const exportNode = $("protocol-similarity-export");
  if (!exportNode) return;
  const payload = plotPayload.payload;
  const title = "Protocol similarity";
  state.plotExports[PROTOCOL_SIMILARITY_LEGEND_KEY] = {
    title,
    rows: protocolSimilarityExportRows(payload, visibleNames),
    engine: "plotly",
    plotlyDivId: "protocol-similarity-heatmap",
  };
  exportNode.innerHTML = registerPlotExport(
    PROTOCOL_SIMILARITY_LEGEND_KEY,
    title,
    state.plotExports[PROTOCOL_SIMILARITY_LEGEND_KEY].rows,
    "plotly",
  );
  bindPlotExportButtons();
}

function protocolSimilarityFilteredMatrix(payload, visibleNames) {
  const order = payload.protocol_order || [];
  const matrix = payload.similarity_matrix || [];
  const indices = visibleNames.map((name) => order.indexOf(name));
  return indices.map((rowIndex) => indices.map((colIndex) => matrix[rowIndex]?.[colIndex] ?? 0));
}

function protocolSimilarityLegendMarkup(plotPayload) {
  const categories = plotPayload.categories || [];
  if (!categories.length) return "";
  const buttons = categories.map((category) => (
    `<button type="button" class="rank-legend-button" data-protocol-similarity-legend="true" data-rank-legend-key="${escapeHtml(plotPayload.plotKey)}" data-rank-legend-category="${escapeHtml(category.key)}" aria-pressed="true">
      <span class="legend-swatch" style="background:${category.color};border-color:${category.color};"></span>
      <span>${escapeHtml(category.label)} · ${category.count}</span>
    </button>`
  )).join("");
  return `<div class="rank-plot-legend metric-legend" role="toolbar" aria-label="Filter protocol table by cluster">${buttons}</div>`;
}

function syncProtocolSimilarityLegend(plotPayload) {
  if (!plotPayload?.plotKey) return;
  const visibility = protocolSimilarityFilterVisibilityState(plotPayload);
  document.querySelectorAll(`button[data-rank-legend-key="${plotPayload.plotKey}"]`).forEach((button) => {
    const shown = protocolSimilarityFilterShown(visibility, button.dataset.rankLegendCategory);
    button.classList.toggle("is-hidden", !shown);
    button.setAttribute("aria-pressed", shown ? "true" : "false");
  });
}

function toggleProtocolSimilarityCategory(plotPayload, categoryKey) {
  const categories = plotPayload.categories || [];
  const visibility = protocolSimilarityFilterVisibilityState(plotPayload);
  visibility[categoryKey] = !protocolSimilarityFilterShown(visibility, categoryKey);
  if (!categories.some((entry) => protocolSimilarityFilterShown(visibility, entry.key))) {
    visibility[categoryKey] = true;
  }
  return visibility;
}

function bindProtocolSimilarityLegendButtons() {
  document.querySelectorAll("button[data-protocol-similarity-legend]").forEach((button) => {
    if (button.dataset.bound === "true") return;
    button.dataset.bound = "true";
    button.addEventListener("click", () => {
      const plotPayload = state.protocolSimilarityPlotPayload;
      if (!plotPayload || plotPayload.plotKey !== button.dataset.rankLegendKey) return;
      toggleProtocolSimilarityCategory(plotPayload, button.dataset.rankLegendCategory);
      void reflowProtocolSimilarityPlot(plotPayload);
    });
  });
}

function disambiguatePlotLabels(labels) {
  const counts = new Map();
  return labels.map((label) => {
    const text = String(label || "");
    const count = counts.get(text) || 0;
    counts.set(text, count + 1);
    if (count === 0) return text;
    const suffix = ` [${count + 1}]`;
    const budget = Math.max(16, 48 - suffix.length);
    const head = text.length <= budget ? text : `${text.slice(0, budget - 1)}…`;
    return `${head}${suffix}`;
  });
}

function buildProtocolSimilarityHeatmapSpec(payload, visibleNames) {
  const labels = visibleNames || payload.protocol_order || [];
  const matrix = visibleNames
    ? protocolSimilarityFilteredMatrix(payload, visibleNames)
    : (payload.similarity_matrix || []);
  const size = labels.length;
  const plotLabels = disambiguatePlotLabels(labels);
  const height = Math.max(320, 48 + size * 22);
  const marginLeft = Math.min(420, Math.max(160, ...plotLabels.map((label) => label.length * 6)));
  const marginBottom = Math.min(420, Math.max(140, ...plotLabels.map((label) => label.length * 4)));
  return {
    data: [{
      type: "heatmap",
      z: matrix,
      x: plotLabels,
      y: plotLabels,
      customdata: labels.map((label) => [label, label]),
      zmin: 0,
      zmax: 1,
      colorscale: "Viridis",
      hovertemplate: "%{customdata[0]} vs %{customdata[1]}<br>similarity: %{z:.3f}<extra></extra>",
    }],
    layout: {
      title: { text: "Expanded feature similarity", font: { size: 13, color: "#667085" } },
      margin: { l: marginLeft, r: 24, t: 48, b: marginBottom },
      height,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: { tickangle: -45, tickfont: { size: 9, color: "#667085" } },
      yaxis: { tickfont: { size: 9, color: "#667085" }, autorange: "reversed" },
    },
    config: { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
  };
}

async function reflowProtocolSimilarityPlot(plotPayload) {
  const payload = plotPayload.payload;
  const protocolNames = protocolSimilarityAllNames(payload);
  const host = $("protocol-similarity-heatmap");
  renderProtocolSimilarityReferenceSelect(payload);
  renderProtocolSimilarityFilteredSections(payload, protocolNames);
  renderProtocolSimilarityExportActions(plotPayload, protocolNames);
  if (!host || !window.Plotly) return;
  if (!protocolNames.length) {
    host.innerHTML = '<span class="path">No protocols to compare.</span>';
    return;
  }
  const spec = buildProtocolSimilarityHeatmapSpec(payload, protocolNames);
  if (host.data) {
    await Plotly.react(host, spec.data, spec.layout, spec.config);
  } else {
    await Plotly.newPlot(host, spec.data, spec.layout, spec.config);
  }
  syncPlotlyHostHeight(host, spec.layout);
  requestAnimationFrame(() => resizePlotlyHosts(host.parentElement));
}

function protocolSimilarityFamilyRows(payload, visibleNames) {
  const protocolNames = visibleNames || payload.protocol_order || [];
  const familyIds = uniqueSorted(
    (payload.protocols || []).flatMap((entry) => (entry.families || []).map((family) => family.family_id)),
  );
  return familyIds.map((familyId) => ({
    familyId,
    cells: protocolNames.map((policyName) => {
      const entry = (payload.protocols || []).find((item) => item.policy_name === policyName);
      const family = (entry?.families || []).find((item) => item.family_id === familyId);
      if (!family) return { className: "family-absent", label: "—" };
      if (!family.present) return { className: "family-absent", label: "0" };
      if (family.member_count >= family.total_members) {
        return { className: "family-present", label: String(family.member_count) };
      }
      return { className: "family-partial", label: `${family.member_count}/${family.total_members}` };
    }),
  }));
}

function protocolSimilarityFilteredClusterSummaries(payload, visibleNames) {
  const visibleSet = new Set(visibleNames);
  const clusterMap = protocolSimilarityClusterMap(payload);
  const metricValues = Object.fromEntries(
    (payload.protocols || []).map((entry) => [entry.policy_name, entry.metric_value]),
  );
  const byCluster = new Map();
  visibleNames.forEach((name) => {
    const clusterId = clusterMap.get(name);
    if (clusterId === undefined) return;
    if (!byCluster.has(clusterId)) byCluster.set(clusterId, []);
    byCluster.get(clusterId).push(name);
  });
  return [...byCluster.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([clusterId, policyNames]) => {
      const names = [...policyNames].sort();
      const presentValues = names
        .map((name) => metricValues[name])
        .filter((value) => value !== null && value !== undefined);
      return {
        cluster_id: clusterId,
        policy_names: names,
        mean_metric: presentValues.length ? presentValues.reduce((sum, value) => sum + value, 0) / presentValues.length : null,
        metric_count: presentValues.length,
        missing_metric_count: names.length - presentValues.length,
      };
    });
}

function protocolSimilarityToReference(payload, policyName) {
  const order = payload?.protocol_order || [];
  const reference = payload?.reference_policy;
  const row = order.indexOf(policyName);
  const col = order.indexOf(reference);
  if (row < 0 || col < 0) return null;
  const value = payload?.similarity_matrix?.[row]?.[col];
  return Number.isFinite(Number(value)) ? Number(value) : null;
}

function protocolSimilarityClusterMap(payload) {
  const order = payload?.protocol_order || [];
  const labels = payload?.cluster_labels || [];
  const map = new Map();
  order.forEach((name, index) => {
    if (labels[index] !== undefined) map.set(name, labels[index]);
  });
  return map;
}

function protocolSimilarityClusterMeanSimilarity(payload, cluster) {
  const order = payload?.protocol_order || [];
  const matrix = payload?.similarity_matrix || [];
  const indices = (cluster?.policy_names || [])
    .map((name) => order.indexOf(name))
    .filter((index) => index >= 0);
  if (indices.length < 2) return null;
  let sum = 0;
  let count = 0;
  for (let left = 0; left < indices.length; left += 1) {
    for (let right = left + 1; right < indices.length; right += 1) {
      const value = matrix[indices[left]]?.[indices[right]];
      if (Number.isFinite(Number(value))) {
        sum += Number(value);
        count += 1;
      }
    }
  }
  return count ? sum / count : null;
}

function protocolSimilarityDiffLookup(payload, policyName) {
  return (payload?.reference_diffs || []).find((item) => item.policy_name === policyName) || null;
}

function protocolSimilarityMetricsAvailable(payload) {
  return (payload?.protocols || []).filter((entry) => entry.metric_value !== null && entry.metric_value !== undefined).length;
}

function renderProtocolSimilarityOverview(payload, visibleNames) {
  const node = $("protocol-similarity-overview");
  if (!node) return;
  const metricName = ensureSelectedMetric();
  const metricLabel = plotMetricLabel(metricName || payload.metric || "metric");
  const clusterMap = protocolSimilarityClusterMap(payload);
  const reference = payload.reference_policy;
  const names = visibleNames || payload.protocol_order || [];
  const rows = names.map((policyName) => {
    const entry = protocolSimilarityEntry(payload, policyName) || {};
    const diff = protocolSimilarityDiffLookup(payload, policyName);
    const sim = protocolSimilarityToReference(payload, policyName);
    const removedFamilies = (diff?.removed_families || []).join(", ") || "—";
    const addedFamilies = (diff?.added_families || []).join(", ") || "—";
    const metricValue = entry.metric_value;
    const metricCell = metricValue === null || metricValue === undefined
      ? '<span class="muted">—</span>'
      : numeric(metricValue);
    const clusterLabel = clusterMap.has(policyName) ? `Cluster ${clusterMap.get(policyName) + 1}` : "—";
    const sourceStatus = entry.run_id
      ? " · completed"
      : (entry.study_present ? " · partial" : "");
    return `
      <tr>
        <th scope="row">${escapeHtml(policyName)}${policyName === reference ? ' <span class="muted">(ref)</span>' : ""}</th>
        <td>${clusterMap.has(policyName) ? clusterMap.get(policyName) + 1 : "—"}</td>
        <td>${sim === null ? "—" : sim.toFixed(3)}</td>
        <td>${entry.expanded_feature_count ?? "—"}</td>
        <td class="protocol-similarity-family-delta">${escapeHtml(removedFamilies)}</td>
        <td class="protocol-similarity-family-delta">${escapeHtml(addedFamilies)}</td>
        <td>${metricCell}</td>
        <td>${escapeHtml(clusterLabel)}${sourceStatus ? `<span class="muted">${escapeHtml(sourceStatus)}</span>` : ""}</td>
      </tr>
    `;
  }).join("");
  node.hidden = false;
  node.innerHTML = `
    <table>
      <thead>
        <tr>
          <th scope="col">Protocol</th>
          <th scope="col">Cluster</th>
          <th scope="col">Sim to ref</th>
          <th scope="col">Features</th>
          <th scope="col">Families removed vs ref</th>
          <th scope="col">Families added vs ref</th>
          <th scope="col">${escapeHtml(metricLabel)}</th>
          <th scope="col">Source</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderProtocolSimilarityFilteredSections(payload, visibleNames) {
  const overviewNode = $("protocol-similarity-overview");
  const clustersNode = $("protocol-similarity-clusters");
  const familiesNode = $("protocol-similarity-families");
  const diffsNode = $("protocol-similarity-diffs");
  if (!visibleNames.length) {
    if (overviewNode) overviewNode.hidden = true;
    if (clustersNode) clustersNode.hidden = true;
    if (familiesNode) familiesNode.hidden = true;
    if (diffsNode) diffsNode.hidden = true;
    return;
  }

  renderProtocolSimilarityOverview(payload, visibleNames);

  if (clustersNode) {
    clustersNode.hidden = false;
    const metricLabel = plotMetricLabel(ensureSelectedMetric() || payload.metric || "metric");
    const clusters = protocolSimilarityFilteredClusterSummaries(payload, visibleNames);
    clustersNode.innerHTML = clusters.map((cluster) => {
      const meanSim = protocolSimilarityClusterMeanSimilarity(payload, cluster);
      const outcomeText = cluster.mean_metric === null || cluster.mean_metric === undefined
        ? `mean ${metricLabel}: —`
        : `mean ${metricLabel}: ${numeric(cluster.mean_metric)}`;
      const cohesionText = meanSim === null ? "" : `<span>mean similarity: ${meanSim.toFixed(3)}</span>`;
      return `
      <article class="protocol-similarity-cluster">
        <div class="protocol-similarity-cluster-head">
          <strong>Cluster ${cluster.cluster_id + 1}</strong>
          <span>${outcomeText}</span>
          ${cohesionText}
          <span>${cluster.metric_count}/${cluster.policy_names.length} with metric</span>
        </div>
        <div class="muted">${escapeHtml(cluster.policy_names.join(", "))}</div>
      </article>
    `;
    }).join("");
  }

  if (familiesNode) {
    familiesNode.hidden = false;
    const rows = protocolSimilarityFamilyRows(payload, visibleNames);
    const headers = visibleNames.map((name) => `<th>${escapeHtml(compactPlotLabel(name, 20))}</th>`).join("");
    const body = rows.map((row) => `
      <tr>
        <th scope="row">${escapeHtml(row.familyId)}</th>
        ${row.cells.map((cell) => `<td class="${cell.className}">${escapeHtml(cell.label)}</td>`).join("")}
      </tr>
    `).join("");
    familiesNode.innerHTML = `
      <table>
        <thead><tr><th scope="col">Family</th>${headers}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    `;
  }

  if (diffsNode) {
    diffsNode.hidden = false;
    const reference = payload.reference_policy;
    const rankedDiffs = (payload.reference_diffs || [])
      .filter((diff) => diff.policy_name !== reference && visibleNames.includes(diff.policy_name))
      .map((diff) => ({
        diff,
        sim: protocolSimilarityToReference(payload, diff.policy_name),
        deltaFamilies: (diff.removed_families?.length || 0) + (diff.added_families?.length || 0),
      }))
      .sort((left, right) => {
        const simLeft = left.sim ?? 1;
        const simRight = right.sim ?? 1;
        if (simLeft !== simRight) return simLeft - simRight;
        return right.deltaFamilies - left.deltaFamilies;
      })
      .slice(0, 8);
    diffsNode.innerHTML = `
      <h4 class="protocol-similarity-diff-title">Largest family diffs vs ${escapeHtml(reference)} (reference baseline)</h4>
      ${rankedDiffs.map(({ diff, sim }) => `
      <article class="protocol-similarity-diff-card">
        <h4>${escapeHtml(diff.policy_name)} · sim ${sim === null ? "—" : sim.toFixed(3)}</h4>
        <div class="protocol-similarity-diff-grid">
          <span><strong>Removed families:</strong> ${escapeHtml((diff.removed_families || []).join(", ") || "—")}</span>
          <span><strong>Added families:</strong> ${escapeHtml((diff.added_families || []).join(", ") || "—")}</span>
          <span><strong>Shared features:</strong> ${diff.shared_feature_count}</span>
        </div>
      </article>
    `).join("")}
    `;
  }
}

function renderProtocolSimilarityPanel() {
  bindProtocolSimilarityControls();
  const summary = $("protocol-similarity-summary");
  const message = $("protocol-similarity-message");
  const heatmapWrap = $("protocol-similarity-heatmap-wrap");
  const heatmapHost = $("protocol-similarity-heatmap");
  const referenceWrap = $("protocol-similarity-reference-wrap");
  const overviewNode = $("protocol-similarity-overview");
  const clustersNode = $("protocol-similarity-clusters");
  const familiesNode = $("protocol-similarity-families");
  const diffsNode = $("protocol-similarity-diffs");
  if (!summary || !message || !heatmapWrap || !heatmapHost) return;

  if (state.protocolSimilarityLoading) {
    summary.textContent = "Loading expanded protocol comparison…";
    message.hidden = true;
    heatmapWrap.hidden = true;
    if (referenceWrap) referenceWrap.hidden = true;
    if (overviewNode) overviewNode.hidden = true;
    if (clustersNode) clustersNode.hidden = true;
    if (familiesNode) familiesNode.hidden = true;
    if (diffsNode) diffsNode.hidden = true;
    return;
  }

  const payload = state.protocolSimilarity;
  if (!payload) {
    summary.textContent = "Similarity uses expanded feature sets (patterns like ligand_* resolve to columns).";
    message.hidden = true;
    heatmapWrap.hidden = true;
    if (referenceWrap) referenceWrap.hidden = true;
    if (overviewNode) overviewNode.hidden = true;
    if (clustersNode) clustersNode.hidden = true;
    if (familiesNode) familiesNode.hidden = true;
    if (diffsNode) diffsNode.hidden = true;
    return;
  }

  const sourceNote = payload.candidate_source ? ` · candidates from ${pathBasename(payload.candidate_source)}` : "";
  const heatmapCount = protocolSimilarityAllNames(payload).length;
  const metricsAvailable = protocolSimilarityMetricsAvailable(payload);
  const executedCount = (payload.protocols || []).filter((entry) => entry.run_id).length;
  const metricNote = payload.metric
    ? ` · ${metricsAvailable}/${heatmapCount} with ${plotMetricLabel(ensureSelectedMetric() || payload.metric)}`
    : "";
  summary.textContent = payload.preview_available
    ? `${heatmapCount} protocols · ${executedCount} completed · ${protocolSimilarityClusterCategories(payload).length} clusters${sourceNote}${metricNote}`
    : "Candidate features unavailable";

  if (!payload.preview_available || heatmapCount === 0) {
    message.hidden = false;
    message.textContent = payload.message || "No protocols to compare.";
    heatmapWrap.hidden = true;
    if (referenceWrap) referenceWrap.hidden = true;
    if (overviewNode) overviewNode.hidden = true;
    if (clustersNode) clustersNode.hidden = true;
    if (familiesNode) familiesNode.hidden = true;
    if (diffsNode) diffsNode.hidden = true;
    return;
  }

  message.hidden = false;
  const notes = [];
  if (payload.message) {
    notes.push(payload.message);
  }
  if (metricsAvailable < heatmapCount) {
    notes.push(`Metrics for ${metricsAvailable}/${heatmapCount} protocols.`);
  }
  message.textContent = notes.join(" ");
  message.hidden = notes.length === 0;

  const plotPayload = ensureProtocolSimilarityPlotPayload(payload);
  heatmapWrap.hidden = false;
  void reflowProtocolSimilarityPlot(plotPayload);
}

async function loadProtocolSimilarity() {
  bindProtocolSimilarityControls();
  state.protocolSimilarityLoading = true;
  renderProtocolSimilarityPanel();
  try {
    const params = {
      include_catalog_only: "false",
      metric: protocolSimilarityMetricToken() || undefined,
    };
    if (state.protocolSimilarityReference) {
      params.reference = state.protocolSimilarityReference;
    }
    state.protocolSimilarity = await api("/api/ablation-protocol-similarity", params);
  } catch (error) {
    state.protocolSimilarity = {
      preview_available: false,
      protocol_count: 0,
      message: error.message || String(error),
    };
  } finally {
    state.protocolSimilarityLoading = false;
    renderProtocolSimilarityPanel();
  }
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
  scheduleProtocolSimilarityLoad();
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
    $("health-label").textContent = dashboardModelLabel(health.dashboard_model);
    renderWorkspace(payload);
    if (state.activeTab === "design") await ensureAblationDesignContext(true);
    if (state.activeTab === "vs-design") await ensureVsDesignContext(true);
  } catch (error) {
    $("health-dot").className = "dot error";
    $("health-label").textContent = "Error";
    toast(error.message || String(error));
  }
}

$("refresh").addEventListener("click", refresh);
loadPersistedUiState();
bindThemeToggle();
bindAblationDesignPanel();
bindVsDesignPanel();
bindJobsPanel();
bindAppTabs();
bindCollapsibleZones();
uiStateHydrated = true;
setActiveTab(state.activeTab || "ablation");
refresh();
