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

// --- BOOTSTRAP --- everything below runs on load and touches the DOM. The JS
// syntax test evaluates the ordered script bundle in a bare engine with no
// `document`, so it cuts the bundle at this marker: keep new top-level wiring
// below the line, not above it.
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
