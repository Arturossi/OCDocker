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

function toggleVsDesignMode() {
  const mode = $("vs-design-mode").value;
  document.querySelectorAll("[data-vs-design-mode-panel]").forEach((node) => {
    node.hidden = node.dataset.vsDesignModePanel !== mode;
  });
}

function renderVsCampaignDiscover(context) {
  state.vsCampaignContext = context;
  const summary = $("vs-campaign-discover-summary");
  const issuesNode = $("vs-campaign-discover-issues");
  if (!context) {
    if (summary) summary.textContent = "";
    if (issuesNode) issuesNode.hidden = true;
    return;
  }
  const manifest = context.manifest || [];
  if (summary) summary.textContent = `${manifest.length} sample(s) discovered`;
  if (manifest.length) $("vs-campaign-manifest").value = JSON.stringify(manifest, null, 2);
  if (issuesNode) {
    const issues = context.issues || [];
    issuesNode.hidden = issues.length === 0;
    issuesNode.textContent = issues.join(" ");
  }
}

async function discoverVsCampaignCandidates() {
  const inputDir = $("vs-campaign-input-dir").value.trim();
  try {
    const context = await api("/api/vs-campaign", inputDir ? { input_dir: inputDir } : {});
    renderVsCampaignDiscover(context);
  } catch (error) {
    toast(error.message || String(error));
  }
}

function readVsCampaignManifest() {
  const raw = $("vs-campaign-manifest").value.trim();
  if (!raw) return [];
  return JSON.parse(raw);
}

function toggleVsCampaignEngineFields() {
  const engine = $("vs-campaign-engine").value;
  document.querySelectorAll("[data-vs-campaign-engine-fields]").forEach((node) => {
    node.hidden = node.dataset.vsCampaignEngineFields !== engine;
  });
}

function readVsCampaignCommonSettings() {
  const settings = { engine: $("vs-campaign-engine").value };
  if (settings.engine === "snakemake") {
    const cores = $("vs-campaign-cores").value.trim();
    if (cores) settings.cores = Number(cores);
  }
  const outdir = $("vs-campaign-outdir").value.trim();
  if (outdir) settings.outdir = outdir;
  const cwd = $("vs-campaign-cwd").value.trim();
  if (cwd) settings.cwd = cwd;
  const timeout = $("vs-campaign-timeout").value.trim();
  if (timeout) settings.timeout = Number(timeout);
  settings.store_db = $("vs-campaign-store-db").checked;
  return settings;
}

function renderVsCampaignPreview(payload) {
  state.vsCampaignPreview = payload;
  const panel = $("vs-campaign-preview-panel");
  if (!panel) return;
  panel.hidden = false;

  if (!payload || payload.valid === undefined) {
    $("vs-campaign-preview-summary").textContent = payload?.error || "Preview failed.";
    $("vs-campaign-preview-details").innerHTML = "";
    return;
  }

  $("vs-campaign-preview-summary").textContent = payload.valid
    ? `Manifest is valid — ${(payload.resolved?.rows || []).length} row(s).`
    : `Manifest is invalid — ${payload.errors.length} error(s).`;

  const details = [];
  if (payload.errors?.length) details.push(["error", "Errors", payload.errors.join(" ")]);
  if (payload.warnings?.length) details.push(["warning", "Warnings", payload.warnings.join(" ")]);
  $("vs-campaign-preview-details").innerHTML = details.map(([cssClass, label, value]) => `
    <div class="${cssClass}"><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</div>
  `).join("");
}

async function previewVsCampaign() {
  let manifest;
  try {
    manifest = readVsCampaignManifest();
  } catch (_error) {
    toast("Manifest is not valid JSON.");
    return;
  }
  if (!manifest.length) {
    toast("Manifest must contain at least one row.");
    return;
  }
  try {
    const payload = await apiPost("/api/vs-campaign/preview", { manifest });
    renderVsCampaignPreview(payload);
  } catch (error) {
    renderVsCampaignPreview({ error: error.message || String(error) });
  }
}

function renderVsCampaignPlan(payload) {
  state.vsCampaignPlan = payload;
  const commandNode = $("vs-campaign-command");
  const sendButton = $("vs-campaign-send-to-jobs");
  if (!commandNode) return;
  commandNode.hidden = false;
  commandNode.textContent = payload?.shell_command || payload?.error || "Plan failed.";
  if (sendButton) sendButton.disabled = !payload?.manifest;
}

async function planVsCampaign() {
  let manifest;
  try {
    manifest = readVsCampaignManifest();
  } catch (_error) {
    toast("Manifest is not valid JSON.");
    return;
  }
  if (!manifest.length) {
    toast("Manifest must contain at least one row.");
    return;
  }
  try {
    const payload = await apiPost("/api/vs-campaign/plan", { manifest, ...readVsCampaignCommonSettings() });
    renderVsCampaignPlan(payload);
    renderVsCampaignPreview({ valid: true, errors: [], warnings: [], resolved: { rows: payload.manifest || [] } });
  } catch (error) {
    renderVsCampaignPlan({ error: error.message || String(error) });
  }
}

function toggleJobsKindFields() {
  const kind = $("jobs-launch-kind").value;
  document.querySelectorAll("[data-jobs-kind-fields]").forEach((node) => {
    node.hidden = node.dataset.jobsKindFields !== kind;
  });
}

function sendVsCampaignPlanToJobs() {
  const plan = state.vsCampaignPlan;
  if (!plan?.manifest) {
    toast("Generate a plan first.");
    return;
  }
  $("jobs-launch-kind").value = plan.kind;
  toggleJobsKindFields();
  $("jobs-launch-args").value = (plan.args || []).join("\n");
  $("jobs-launch-cwd").value = plan.cwd || "";
  $("jobs-launch-manifest").value = JSON.stringify(plan.manifest, null, 2);
  $("jobs-launch-engine").value = plan.engine || "shell";
  $("jobs-launch-cores").value = plan.cores || "";
  $("jobs-launch-results-dir").value = plan.results_dir || "";
  setActiveTab("jobs");
  toast(`Campaign plan (${plan.manifest.length} row(s)) loaded into the Jobs tab — review and click Launch job.`);
}

function bindVsDesignPanel() {
  $("vs-design-mode")?.addEventListener("change", toggleVsDesignMode);
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
  $("vs-campaign-discover")?.addEventListener("click", () => void discoverVsCampaignCandidates());
  $("vs-campaign-engine")?.addEventListener("change", toggleVsCampaignEngineFields);
  $("vs-campaign-preview")?.addEventListener("click", () => void previewVsCampaign());
  $("vs-campaign-plan")?.addEventListener("click", () => void planVsCampaign());
  $("vs-campaign-send-to-jobs")?.addEventListener("click", sendVsCampaignPlanToJobs);
  toggleVsDesignKindFields();
  toggleVsDesignMode();
  toggleVsCampaignEngineFields();
}

