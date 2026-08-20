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

