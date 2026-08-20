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

