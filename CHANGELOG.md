# Changelog

All notable changes to OCDocker are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.2] - 2026-07-13

### Added

- **`OCScore_models/`**: four pretrained, ready-to-score OCScore configurations
  from the DUDEz ablation study (`#03`, `#05`, `#09`, `#12` — `#03` is the
  configuration recommended in the manuscript). Each ships a DUDEz `best_model/`
  bundle transfer-linked to its PDBbind `best_model/` bundle, picked as the
  seed replica with the highest validation-split BEDROC (not test-split, to
  keep the selection consistent with the paper's own methodology). See
  `OCScore_models/README.md` for the full metrics table and usage.
- `examples/25_ocscore_score_with_shipped_models.py`: scores a raw pipeline
  archive with any of the shipped `OCScore_models/` configurations via
  `predict_from_export`, resolving both linked bundle paths from
  `OCScore_models/manifest.json`.

### Fixed

- `ocdocker ocscore validate` and `ocdocker ocscore load` could not open a
  DUDEz transfer-model bundle whose linked PDBbind export directory had moved
  (e.g. a bundle copied to another machine, as in `OCScore_models/`) — only
  `score`/`external-blind` accepted a `--pdbbind-export-dir` override. Both
  commands, and the underlying `validate_export_bundle`/`load_exported_model`,
  now accept it too.

### Changed

- `CITATION.cff` now cites Zenodo's **concept DOI** (`10.5281/zenodo.21330171`)
  instead of the version-specific DOI for 0.15.1. The concept DOI always
  resolves to the latest archived version, so it no longer needs to change on
  every release; the manuscript keeps citing the version-specific DOI for the
  release that produced its results.

## [0.15.1] - 2026-07-13

### Fixed

- `.zenodo.json` had a `related_identifiers` entry (the INPI registration number) using
  a `scheme` value Zenodo's metadata schema doesn't recognize, which made every Zenodo
  archival attempt off the GitHub release fail with "Extra metadata load failed". Removed
  the entry; the registration is still documented in the description field.
- Two `tests/cli/test_cli_utilities.py::test_cmd_script_*` tests patched
  `OCDocker.CLI.common._preparse_global_args`/`_bootstrap_ocdocker_env`, but
  `OCDocker.CLI.script` imports those names directly, so the patches never took effect and
  the tests silently exercised the real bootstrap path. It only passed locally because of a
  stray config file outside the repo; CI (no such file) failed with `SystemExit(2)`. Patches
  now target `OCDocker.CLI.script`, where `cmd_script` actually looks the names up.

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

[0.15.2]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.2
[0.15.1]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.1
[0.15.0]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.0
