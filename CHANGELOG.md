# Changelog

All notable changes to OCDocker are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-07-12

The release that produced the results reported in the OCScore manuscript. It adds the
MCP server, the OCScore Workbench dashboard, the ablation framework's statistical layer
and the SHAP shortcut-risk analysis.

> **Note on 0.14.0.** That version number was carried in the source tree for a while but
> was never published: no tag, no PyPI release, no archived record. Everything it had
> accumulated since 0.13.5 ships here, in 0.15.0. There is no 0.14.0 to look for.

### Added

- **MCP server** (`OCDocker.MCP`): exposes 19 tools through the Model Context Protocol,
  so a language model can design ablations, plan virtual-screening campaigns, submit
  runs and follow their progress in natural language. Includes Snakemake-backed
  execution and multi-element runs in virtual-screening mode.
- **OCScore Workbench**: a web dashboard for post-ablation analysis, with per-target and
  cross-validation plots, protocol-similarity clustering, an ablation designer, rank-plot
  and cluster legends with category filters, collapsible zones, a theme toggle, and UI
  state persisted to `localStorage`.
- **Statistical comparison of ablations**: paired t-test by seed against the full model,
  with Holm-Bonferroni correction applied both per metric family and globally.
- **SHAP shortcut-risk analysis** (`OCScore.Analysis.SHAP.Dominance`): per-replica
  aggregation of family-level importance and of the single dominant feature, plus
  `classify_policies_by_shortcut_rule`, which discards a configuration when it beats the
  reference model *and* concentrates more than a threshold share of its SHAP importance
  in one feature.
- **Plotting** (`OCScore.Analysis.Plotting.Stats`, `SHAP.DominancePlots`): performance
  versus shortcut-risk scatter with an automatically detected broken axis and a shaded
  discard quadrant, and stacked SHAP family-composition bars. Both take their labels as
  parameters, so a caller can render them in another language without touching the library.
- **Scheduler-friendly pipeline API** and an `OCScoreLayout` module for workspace discovery.
- Ablation-protocol similarity analysis and visualisation.
- Documentation on container usage (Docker, Podman, Singularity), database setup and
  external tools.

### Changed

- The Workbench API moved to FastAPI.
- Ablation policies reworked: `receptor_only` removed, focused policies added.
- Licensing aligned across the tree (BSD-3-Clause; UFRJ, Artur Duque Rossi and
  Pedro Henrique Monteiro Torres as copyright holders).

### Fixed

- The protocol-similarity cluster legend was implemented but never wired into the render
  path, so its category filter never took effect; the dashboard now renders, binds and
  applies it.
- The JavaScript syntax test stripped a hard-coded list of bootstrap calls from `app.js`,
  which silently rotted whenever a new panel added top-level wiring. It now cuts the file
  at an explicit `// --- BOOTSTRAP ---` marker.
- `py-mini-racer`, used by the dashboard's JavaScript syntax test, was imported without
  being declared; it is now part of the `dev` extra.

[0.15.0]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.0
