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

