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
  vsCampaignContext: null,
  vsCampaignPreview: null,
  vsCampaignPlan: null,
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
  jobCampaignProgress: null,
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

