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

