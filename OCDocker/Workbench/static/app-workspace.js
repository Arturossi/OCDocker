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

function renderProtocolSimilarityLegend(plotPayload) {
  const host = $("protocol-similarity-legend");
  if (!host) return;
  const markup = protocolSimilarityLegendMarkup(plotPayload);
  host.innerHTML = markup;
  host.hidden = !markup;
  if (!markup) return;
  bindProtocolSimilarityLegendButtons();
  syncProtocolSimilarityLegend(plotPayload);
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
  const height = Math.max(420, 48 + size * 30);
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
      // automargin (rather than a hand-rolled margin guessed from label.length) lets Plotly size
      // the margin from the actual rendered tick text, and tickmode "array" with a full tickvals/
      // ticktext list stops Plotly's default auto-thinning from silently dropping every other
      // label when there isn't room -- both were making the heatmap render as a small square
      // crammed into a corner with the rest of the row left blank.
      autosize: true,
      margin: { l: 80, r: 24, t: 48, b: 80 },
      height,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      xaxis: {
        tickangle: -45,
        tickfont: { size: 9, color: "#667085" },
        automargin: true,
        tickmode: "array",
        tickvals: plotLabels,
        ticktext: plotLabels,
      },
      yaxis: {
        tickfont: { size: 9, color: "#667085" },
        autorange: "reversed",
        automargin: true,
        tickmode: "array",
        tickvals: plotLabels,
        ticktext: plotLabels,
      },
    },
    config: { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
  };
}

async function reflowProtocolSimilarityPlot(plotPayload) {
  const payload = plotPayload.payload;
  const visibility = protocolSimilarityFilterVisibilityState(plotPayload);
  const protocolNames = protocolSimilarityVisibleNames(payload, visibility);
  const host = $("protocol-similarity-heatmap");
  renderProtocolSimilarityReferenceSelect(payload);
  renderProtocolSimilarityLegend(plotPayload);
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

