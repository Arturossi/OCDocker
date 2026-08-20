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
  const body = { kind, args };
  if (cwd) body.cwd = cwd;
  if (kind === "vs_campaign") {
    const manifestRaw = $("jobs-launch-manifest").value.trim();
    try {
      body.manifest = manifestRaw ? JSON.parse(manifestRaw) : [];
    } catch (_error) {
      toast("Campaign manifest is not valid JSON.");
      return;
    }
    if (!body.manifest.length) {
      toast("Campaign manifest must contain at least one row.");
      return;
    }
    body.engine = $("jobs-launch-engine").value;
    const cores = $("jobs-launch-cores").value.trim();
    if (cores) body.cores = Number(cores);
    const resultsDir = $("jobs-launch-results-dir").value.trim();
    if (resultsDir) body.results_dir = resultsDir;
  }
  button.disabled = true;
  try {
    const record = await apiPost("/api/jobs", body, jobAuthHeaders());
    toast(`Launched job ${record.job_id}`);
    $("jobs-launch-args").value = "";
    $("jobs-launch-manifest").value = "";
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
  const job = (state.jobs || []).find((item) => item.job_id === jobId);
  if (job && job.kind === "vs_campaign") {
    try {
      state.jobCampaignProgress = await api(`/api/jobs/${encodeURIComponent(jobId)}/campaign-progress`);
    } catch (_error) {
      state.jobCampaignProgress = null;
    }
  } else {
    state.jobCampaignProgress = null;
  }
  renderJobLogs();
}

function renderJobCampaignProgress() {
  const container = $("jobs-campaign-progress");
  const progress = state.jobCampaignProgress;
  if (!progress || progress.engine === "unknown") {
    container.hidden = true;
    return;
  }
  container.hidden = false;
  const overall = progress.overall;
  const outcomeNote = overall && overall.succeeded !== undefined
    ? ` (${overall.succeeded} succeeded, ${overall.failed} failed)`
    : "";
  $("jobs-campaign-progress-summary").textContent = overall
    ? `${progress.engine} engine · ${overall.completed_steps}/${overall.total_steps} steps (${overall.percent}%)${outcomeNote}`
    : `${progress.engine} engine · starting…`;
  const entries = Object.entries(progress.samples || {});
  $("jobs-campaign-progress-samples").innerHTML = entries.map(([name, info]) => (
    `<span class="jobs-campaign-sample ${escapeHtml(info.status)}">${escapeHtml(name)}: ${escapeHtml(info.status)}</span>`
  )).join("");
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
  renderJobCampaignProgress();
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
  $("jobs-launch-kind").addEventListener("change", toggleJobsKindFields);
  toggleJobsKindFields();
  $("jobs-launch").addEventListener("click", () => void launchJob());
  $("jobs-logs-close").addEventListener("click", () => {
    state.selectedJobId = null;
    state.jobLogs = null;
    state.jobCampaignProgress = null;
    renderJobLogs();
  });
}

