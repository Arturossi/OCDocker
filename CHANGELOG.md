# Changelog

All notable changes to OCDocker are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `ocdocker vs` and `ocdocker pipeline` generated every missing ODDT model at startup even
  when no ODDT rescoring was requested, which takes several minutes on the first run (a
  `vs --skip-rescore` with Vina took over 4 minutes instead of 23 seconds). `bootstrap()`
  gained `init_oddt: bool = True`; `vs` (which never rescores with ODDT) and `pipeline`
  without `oddt` in `--rescoring-engines` set it to False. `run_oddt` still generates
  missing models on demand.
- Receptor cleaning (`clean_for_dssp`, run by every `Receptor()`, and `clean_pdb_file`)
  kept every ATOM record, so alternate locations and the atoms of later MODELs ended up as
  overlapping copies in the receptor that is docked into (5ar7: 12 His246 side-chain atoms
  instead of 6). Each atom is now kept once, with its alternate location indicator blanked,
  and receptors cleaned by earlier versions are fixed on their next run.
- `subprocess_failures.log` stored the full stderr of every failed command, so a noisy tool
  made it grow without bound (10 failures of a command writing 20000 warning lines: 11 MB).
  It now keeps the same tail as the console summary (`OCDOCKER_SUBPROCESS_TAIL_LINES`, 20
  lines by default, at most 64 KB): 0.02 MB for the same failures.
- `oclogging.configure(log_file=...)` attached a new file handler on every call, so
  configuring the same file again wrote every message once per call.
- `read_vina_log` returned modes that Vina lists in its log but does not write to the
  output (Vina 1.2 lists modes more than `energy_range` above the best one), so the digest
  held scores for poses that do not exist (a test_ptn1 run logged a third mode at
  +12.16 kcal/mol with only two poses written). Modes beyond `vina_energy_range` from the
  best are now dropped.
- The console history (`~/.ocdocker_console_history`) was loaded on top of the history
  already in memory and had no length limit, so each console session started in the same
  process (e.g. the console tests) saved it doubled; it reached 871 MB and stalled the test
  suite. The history is now cleared before loading and limited to 1000 commands.

### Added

- End-to-end `ocdocker vs` tests for gnina and PLANTS. PLANTS runs everywhere with the
  binaries in `bin/docking/plants`; gnina is skipped where it is not installed (as in CI).
  Executables are looked up on PATH, then in `bin/docking`, then in the project's
  `OCDocker.cfg` (or `OCDOCKER_CONFIG`).

### Changed

- CI uses the bundled `bin/docking/smina/smina` instead of downloading the same binary from
  SourceForge.

## [0.17.0] - 2026-10-22

### Added

- `PDBbindSplit` gained the `protein_cluster_heldout` strategy, which assigns whole protein
  clusters (`cluster_column`, `"protein_cluster"` by default) to train, validation and test,
  balancing row counts rather than cluster counts, and checks that no cluster crosses
  splits. A PDB code is not a protein-level independence boundary, so this is the
  meaningful held-out split for PDBbind.
- Ablation protocols `full_leak_safe` and `scoring_function_leak_safe`, which exclude the
  ODDT features (RF-Score, NNScore and PLEC were trained on PDBbind).
- `count_isomorphisms()` in `Toolbox/MoleculeProcessing.py` counts the heavy-atom graph
  automorphisms that the symmetry-corrected RMSD would enumerate, stopping at a limit.
  `get_rmsd_matrix()` raises `TooManyIsomorphisms` for ligands above 100000, instead of
  spyrmsd holding every isomorphism in memory.
- `Ligand.create_box()` gained `max_edge` (clamp the box edge, which grows with the radius
  of gyration, with a warning) and `warn_edge` (warn above this edge, 40 A by default).
- `FeatureReduction` treats `protein_cluster` and `smiles_source` as metadata columns and
  `pocket_` columns as receptor-side.
- End-to-end tests (`tests/integration/test_docking_end_to_end.py`) run the real
  `ocdocker vs` command on `test_files/test_ptn1` with Vina and smina. CI installs Vina,
  DSSP and the official smina static build (pinned by sha256) to run them.
- CI runs the test suite on Python 3.12, 3.13 and 3.14, and a ruff job (syntax errors and
  undefined names).

### Fixed

- `ocdocker vs --engine smina` failed at once with a `TypeError`: `cmd_vs` called smina's
  preparation with Vina's arguments. smina's receptor is now prepared with Open Babel as
  before, and `Smina.run_prepare_ligand()` takes the same `logFile` and `useOpenBabel`
  arguments as Vina's, so `vs` falls back to Open Babel when MGLTools is not available.
- `ocdocker pipeline --engines smina` failed with `UnboundLocalError` (a late local import
  of `OCDocker.Docking.Smina` shadowed the module name in `_run_pipeline_for_box`).
- Ligand and receptor preparation reported success when MGLTools (`pythonsh`) or SPORES was
  missing: the unprepared input was copied to the prepared path, the docking engine then
  failed on it and the Open Babel fallback never ran. An error is now returned. The Vina
  Open Babel receptor preparation writes a rigid PDBQT, which Vina requires.
- The PLANTS docking digest was never generated in batch docking (`generate_digest` was
  indented under `return`).
- `run_oddt` merged every pose into one row when the molecules had no title (split docking
  poses), keeping only the last pose's scores. Rows without a usable name are now named
  after their pose file; rescoring the 36 Vina poses of the test_ptn1 compounds gave 4 rows
  before and 36 after.
- `oddt_models_dir` from the config file was ignored, and generating any PLEC model failed
  with `invalid literal for int() with base 10: 'rf'`.
- The vendored ODDT fork (`ODDT_VENDOR_REF`) moves to Arturossi/oddt `b4288f0`, which works
  with NumPy 2.4 (`np.in1d` removed), scikit-learn 1.6+ and recent SciPy. With the previous
  ref every ODDT score failed under NumPy 2.4 and NNScore could not be trained.
- Log writes (`print_*_log`, `print_section`, `print_subsection`) raised `OSError` when the
  log directory was missing or not writable, so a run reporting an error (e.g. a docking
  failure) died with `PermissionError` instead. They now create the directory and, if the
  file still cannot be written, print the message as a warning. `print_subsection` now
  honours its `logName` argument.
- mypy failed on Python 3.12+ with the NumPy 2.5 type stubs (`PDBbindSplit`).
- `tests/core/test_ligand.py` rewrote the tracked `test_files/test_ptn1/boxes/box0.pdb` on
  every run.

### Removed

- The OCScore Workbench (dashboard, HTTP API, run specs and manifests), the MCP server, the
  `workbench` and `mcp` CLI commands, and the `api` and `mcp` extras (fastapi, starlette,
  uvicorn, mcp, httpx). playwright and py-mini-racer leave the `dev` extra. The code is kept
  on the `archive/workbench-v0.16` branch.
- The PLEC descriptor monkeypatch in `run_oddt`; the vendored ODDT fork fixes the cause
  (`universal_descriptor.build` passing a lazy `map` to `scipy.sparse.vstack`). PLEC scores
  are identical with and without it.
- Unused code: `Docking/Future` (Dock3, Dock6, GEMDOCK, LeDock and rDock stubs),
  `OCScore/DNN` and the empty `OCScore/Dimensionality` package.
- The unused `cupy` dependency (`gpu` extra, also in `all` and `full`), which has no Python
  3.14 wheels.

## [0.16.0] - 2026-09-24

This release also covers the changes that were versioned 0.15.5 but never tagged or given
an entry of their own.

### Added

- **Binding pocket descriptors**: `OCDocker/Pocket.py` adds a `Pocket` class that defines a
  pocket from a reference ligand (every standard receptor residue with a heavy atom within
  `cutoff` angstroms, 8.0 by default, of any reference-ligand heavy atom) and computes
  descriptors on those residues only: residue counts `countA`..`countV`, `TotalAALength`,
  `countChain`, `SASA` (per residue, in the context of the full receptor), `GRAVY`,
  `Aromaticity`, `NetCharge` (side chains only, Henderson-Hasselbalch at pH 7.4) and the
  side-chain hydrogen-bond donor/acceptor atoms lining the pocket. Unlike `Receptor`, where
  `count*` count surface residues, here every pocket residue is counted. Descriptors are
  cached with `to_json()` / `from_json_descriptors`, which also record the cutoff, the
  reference ligand and the residue identifiers.
- `OCDocker/DB/Models/Pockets.py`: a `pockets` table referencing `receptors` through
  `receptor_id`, with `Receptors.pockets` as the reverse relationship. A receptor may hold
  several pockets, so the `receptors` table gains no column and existing databases need no
  migration; `create_tables()` creates the new table.
- `scripts/compute_pocket_descriptors.py`: computes and caches pocket descriptors, in
  parallel, for every receptor of a database that has a reference ligand
  (`reference_ligand.pdb` or `.sdf`), and writes a TSV report of the outcome per receptor.
- `Ligand.load_mol()` gained a `write_mol2: bool = True` option. With `False`, loading a
  non-mol2 file no longer writes a derived `.mol2` next to it.
- `OCScore_models/` ships all 22 ablation-study configurations (#01 to #22), up from 4 in
  0.15.4. Each is the replica with the highest validation BEDROC of its 5 seeds.
  `manifest.json`, the README table and the `--config` choices of
  `examples/25_ocscore_score_with_shipped_models.py` list them all.
- Workbench browser characterization tests (`tests/workbench/test_browser.py`), run with
  Playwright; `playwright>=1.61.0` joins the `dev` extra, and its Chromium binary is
  installed separately with `python -m playwright install chromium`.

### Fixed

- `predict_from_export()` scored raw, unstandardized features. DUDEz bundles never
  persisted a scaler under the default `scaling_strategy="pdbbind_scaler"`, and
  `predict_from_export()` did not forward `pdbbind_export_dir` to `load_exported_model()`,
  so every prediction bypassed the scaler the network was trained with. On the shipped
  configurations' own held-out DUDEz test sets this gave near-random or anti-correlated
  rankings (#03: ROC-AUC 0.558). `load_exported_model()` now falls back to the linked
  PDBbind bundle's `scaler.joblib`, and `predict_from_export()` forwards
  `pdbbind_export_dir` (#03 after the fix: ROC-AUC 0.883).
- `OCScore_models/` shipped configuration #05 where the paper's final candidates are
  #03, #09, #12 and #16; #16 is now included. The README no longer claims that every
  shipped configuration uses `dudez_use_transfer: true` (only #05 does).
- `clean_for_dssp()` rewrote `receptor.pdb` on every `Receptor()` construction, bumping its
  mtime, so under `--rerun-triggers mtime` every completed target sharing that receptor
  looked stale. It now skips files that already match its own output.
- `Ligand.get_centroid()` no longer writes a `.mol2` file as a side effect.
- Receptors prepared with Open Babel were written as flexible ligands: `convert_mols()`
  wrote PDBQT with a torsion tree (`ROOT`/`BRANCH`/`TORSDOF`), which Vina rejects as a
  rigid receptor. This is the receptor path of `OpenBabelPreparationStrategy`, used by
  Smina and Gnina and by the pipeline whenever MGLTools `prepare_receptor4` fails.
  `convert_mols()` gained `rigid: bool = False`, which for PDBQT output adds Open Babel's
  `r` (no torsion tree) and `c` (combine all fragments into one molecule) write options.
  The strategy's receptor path uses it, and `get_receptor_command()` now shows the
  matching `-xr -xc` flags.

### Changed

- Optuna is pinned to the 5.0 series: `optuna==5.0.*` and `optuna-integration==5.0.*`, with
  `optuna-dashboard>=0.21.0` (`environment.yml` moves from Optuna 3.6.1, integration 4.0.0
  and dashboard 0.16.2). The OCScore Optuna stages now build their TPE sampler with
  `multivariate=True` and `constant_liar=True` set explicitly (`_build_tpe_sampler()`),
  instead of inheriting the installed Optuna's defaults, which changed in 5.0. Studies run
  under earlier Optuna versions will not reproduce trial for trial.
- `scikit-learn` is pinned to `==1.8.*`, the version the shipped OCScore scalers and
  calibrators were pickled with, and `sqlalchemy` is capped at `<2.1` until the database
  layer is tested on 2.1.
- `environment.yml` is regenerated from an environment that passed the full test suite.
  It previously listed conda builds that no longer matched the working environment
  (e.g. scikit-learn 1.5.2, numpy 1.26.4, pandas 2.2.3); conda now supplies Python and the
  CUDA runtime, and pip supplies the rest at exact versions.
- `DNNOptimizer.optimize()` types `direction` as `Literal["minimize", "maximize"]`, the
  values `optuna.create_study()` accepts, and `Console.session` gives IPython's `embed` a
  callable type, so mypy passes against Optuna 5.0 and IPython 9.17.
- The Workbench dashboard script is split into modules (`app-core.js`, `app-jobs.js`,
  `app-comparison.js`, `app-plots.js`, `app-results.js`, `app-ablation-design.js`,
  `app-vs-design.js`, `app-workspace.js`), still loaded as classic scripts in that order.
- `environment.yml` no longer installs ODDT from Conda; `scripts/vendor_oddt.sh` supplies
  the required fork before OCDocker is installed.
- `CHANGELOG.md` is the single changelog: the Sphinx page (`docs/source/changelog.md`)
  includes it directly, and notes that existed only in the old `changelog.rst` moved into
  the 0.15.0 entry.
- The `Receptor` docstring now states that `countA`..`countV` count only surface-exposed
  residues, and drops the nonexistent `cleanStructurePath` parameter from
  `count_surface_AA()`.

## [0.15.4] - 2026-08-06

### Fixed

- `Ligand._try_embed_rdkit()` could hang indefinitely (observed in production to exceed
  40 minutes on a single call): RDKit's `AllChem.EmbedMolecule()` can run for a very
  long time on molecules whose specified stereochemistry has no geometrically valid
  strict embedding (e.g. bridged/caged bicyclic bridgeheads in some algorithmically
  generated decoy SMILES), and a Python-level timeout cannot reliably interrupt it since
  the interpreter does not regain control until the C call returns. Each embedding
  attempt now runs in a disposable subprocess (`_embed_worker` /
  `_embed_attempt_with_timeout`) under a hard wall-clock timeout, guaranteeing
  termination regardless of what RDKit is doing internally. On timeout or failure this
  returns `False` rather than relaxing stereochemistry as a fallback, since this code
  has no way to know whether it is processing a benchmark decoy or a real candidate
  whose specified stereocenters matter.

## [0.15.3] - 2026-07-29

### Added

- **LIT-PCBA external validation subset**: `OCDocker/DB/LITPCBA.py` adds `"litpcba"` as a
  third archive type (alongside `dudez`/`pdbbind`), wired through `Config.py`,
  `Initialise.py`, `baseDB.py`, `Prepare.py`, `Digest.py`, and `Dock.py`.
  `scripts/litpcba_validation_subset.py` builds a leakage-checked, compute-tractable
  subset by deduplicating LIT-PCBA candidate receptors against local PDBbind/DUDEz via
  mmseqs2 sequence search, selecting the best-resolution surviving structure per target,
  and subsampling inactives with a deterministic floor/cap/ratio rule (13/15 targets
  kept, 2,699 actives, 131,216 sampled inactives). `scripts/litpcba_build_archive.py`
  converts that subset into the raw archive layout `Prepare`/`Dock` expect. See
  `docs/litpcba_validation_subset.md` for the full methodology and threshold rationale.
- `Ligand.load_mol()`/`Ligand.__init__` gained a `clean: bool = True` option: strips
  known salts/counter-ions and keeps only the largest disconnected fragment, applied
  uniformly across all three load paths (RDKit `Mol` object, file load, SMILES load).

### Fixed

- `scripts/litpcba_validation_subset.py`'s receptor dedup only checked the single
  highest-identity mmseqs2 hit per candidate against both the identity and coverage
  thresholds; a true near-duplicate could hide behind an unrelated hit with higher
  identity but low coverage and go undetected. Now excludes a candidate if any hit
  clears both thresholds.
- Its `--refresh-resolution-cache` started from an empty cache and wrote back only the
  current run's candidates, which could silently shrink the shipped 129-structure
  resolution cache if run against a partial `--litpcba-dir`. Refresh now always merges
  into the existing cache.
- `OCDocker.Rescoring.ODDT.run_oddt` reloaded each ~1-3s gzipped RF/NN/PLEC scorer
  pickle for every single ligand; scorer models are now cached per worker thread
  (`threading.local()`, not shared/global, since ODDT's `scorer.set_protein()` mutates
  the scorer object in place and Snakemake's `--force-use-threads` runs ligand jobs
  concurrently in one process).

### Changed

- `OCScore.Analysis.Plotting.Stats` and `examples/24_ocscore_bedroc_shortcut_risk_scatter.py`:
  ablation eligibility for the shortcut-risk scatter now comes from a formal paired,
  Holm-corrected significance test rather than being inferred from whether a policy
  beats the reference on the plotted axis; the x-axis can now split across up to three
  wide gaps instead of one.

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

### Details

The notes below were written during development of this release and kept in the
Sphinx changelog page under "Unreleased"; they are preserved here in full.

#### Snakemake execution engine and structured progress for VS campaigns

`vs_campaign` jobs gained a second execution engine, chosen via
`engine="shell"` (default, unchanged) or `engine="snakemake"` on
`plan_vs_campaign`/`run_job`/`plan_job` and the Workbench VS tab's
Batch mode: real Snakemake DAG orchestration (parallel via `cores`,
resumable via `--rerun-incomplete`) using a bundled config-driven
multi-sample Snakefile
(`OCDocker.Workbench.Jobs.build_campaign_snakemake_command`,
`OCDocker/Workbench/Snakefiles/vs_campaign.smk` — reads a `samples`
dict, so discovered receptor/ligand/box files never need to be relocated
into a fixed directory layout). Both engines also gained structured
per-sample progress, parsed from the job's own log text: new
`GET /api/jobs/{job_id}/campaign-progress` endpoint
(`OCDocker.Workbench.CampaignProgress`), MCP tool
`get_campaign_progress`, and a live progress view in the Workbench Jobs
tab's log panel. Also fixed: a shared `outdir` across every campaign row
now correctly nests per sample (`<outdir>/<sample>`) instead of every row
writing to the same directory.

#### VS campaign batches (multi-sample)

New `vs_campaign` job kind (`OCDocker.Workbench.Jobs.build_campaign_script`),
`/api/vs-campaign*` endpoints, and matching MCP tools
(`get_vs_campaign_context`, `preview_vs_campaign`, `plan_vs_campaign`)
run many receptor/ligand/box samples as **one** tracked job: discover an
`input/{sample}/...` layout (or hand-author a manifest), validate every
row, and launch the whole batch with a single `run_job(confirm=True)` — no
per-row confirmations, no dependency beyond OCDocker itself with the default
shell engine (see above for the optional Snakemake engine). The generated
shell script continues past a failing row and reports an aggregate pass/fail
count, so one bad sample doesn't abort the rest. The Workbench VS tab gained
a "Single target" / "Batch" mode toggle exposing the same flow in the
browser.

#### VS/pipeline design assistant

New `/api/vs-design*` endpoints and matching MCP tools
(`get_vs_design_context`, `preview_vs_design`, `plan_vs_design`,
`OCDocker.Workbench.VSDesign`) mirror the existing ablation-design flow
for single-target `vs`/`pipeline` docking runs: discover receptor/ligand/box
candidates in a workspace, validate a draft selection (paths, engine names),
and preview the exact command before launching it through the existing
`run_job`/Jobs tab. Read-only and unauthenticated, like ablation design.
Covers one receptor/one ligand/one box per draft — for many samples in one
job, see "VS campaign batches" above.

#### MCP server for LLM-driven OCDocker orchestration

`ocdocker mcp serve` runs a Model Context Protocol server (`OCDocker.MCP`,
new `mcp` optional extra) over stdio for LLM clients such as Claude Code and
Claude Desktop. It is a thin adapter over a running `ocdocker workbench serve`
API: workspace inspection, ablation design preview/plan, protocol similarity,
and job listing/logs are always-available read tools; `run_job` and
`cancel_job` require the Workbench job bearer token and an explicit
`confirm=True` from the calling LLM, so a job is never launched from one
ambiguous instruction. See the optional dependencies page
(`optional_dependencies`).

#### Workbench API can launch and track jobs

`ocdocker workbench serve` now exposes `/api/jobs*` endpoints that launch,
list, poll, tail logs for, and cancel `vs`, `pipeline`, `ocscore train`,
and `ocscore reduce` runs as tracked local subprocesses
(`OCDocker.Workbench.Jobs`). Job state is persisted under
`<served root>/.ocdocker-jobs/` and survives an API restart. Job-execute
endpoints require a bearer token, auto-generated on first run at
`~/.config/ocdocker/workbench_token` or set via `OCDOCKER_WORKBENCH_TOKEN`
(`OCDocker.Workbench.Auth`); all existing read-only inspection endpoints
are unchanged and remain unauthenticated.

#### Workbench API migrated to FastAPI

`ocdocker workbench serve` now runs on FastAPI/uvicorn instead of the stdlib
`http.server`, behind the new `api` optional extra
(`pip install "ocdocker[api]"`). All existing endpoints, response shapes, and
the `/app` browser dashboard are unchanged; the server now also exposes an
OpenAPI schema at `/api/openapi.json`. See the optional dependencies
(`optional_dependencies`) and usage (`usage`) pages.

#### Console and CLI separation

The interactive console lives in `OCDocker.Console`. Use `ocdocker console`
or `python -m OCDocker.Console` to start the REPL; built-in `help` and `exit`
commands are supported. The root `OCDockerConsole.py` wrapper was removed.
See the `OCDocker.Console` and usage (`usage`) pages.

#### Packaging and optional dependencies

OCDocker now ships a **minimal core** install; scientific, ML, and plotting stacks
are optional pip extras (`docking`, `db`, `ml`, `analysis`, `workflow`,
`all`, `full`, `dev`). See the optional dependencies page
(`optional_dependencies`) for the cheat sheet and full reference.

#### Formatting and readability

Priority config and Python modules use expanded multiline formatting (line length
120, one dependency per line in `pyproject.toml`). Long config defaults such as
`reference_column_order` live in module-level constants in `OCDocker/Config.py`.
See the development page (`development`) for contributor formatting rules.

#### OCScore feature-reduction API

Added a granular feature-reduction API in
`OCDocker.OCScore.Utils.FeatureReduction` for descriptor datasets. The new API
keeps feature-reduction behavior in reusable Python functions and dataclasses,
with `run_feature_reduction_protocol` provided only as an orchestration helper.

Highlights:

- descriptor block detection for receptor, ligand, and scoring-function columns
- Ligand/Receptor descriptor metadata support with configurable pattern fallback
- missing-row removal with row-level and block-level reports before data loss
- block-wise constant, near-constant, duplicate, and correlation filtering
- cross-block correlation and Ridge CV predictability diagnostics
- opt-in parallel Ridge CV diagnostics through `CrossBlockDiagnosticsConfig.n_jobs`
- opt-in orchestration progress logging through `FeatureReductionConfig.verbose`
- disabled-by-default conservative cross-block filtering
- reproducibility protocol and stable report filenames

Compatibility and rewiring:

- Existing OCScore training, DNN, autoencoder, SHAP, and downstream evaluation
  paths are not automatically rewired to call this API.
- `OCDocker.OCScore.Utils.IO.load_data` is not changed; the new orchestration
  helper reads raw CSV input directly so missing rows can be reported before
  removal.
- This is additive public API. If released publicly, it fits a minor version bump
  rather than a major version bump.

[Unreleased]: https://github.com/Arturossi/OCDocker/compare/v0.17.0...HEAD
[0.17.0]: https://github.com/Arturossi/OCDocker/releases/tag/v0.17.0
[0.16.0]: https://github.com/Arturossi/OCDocker/releases/tag/v0.16.0
[0.15.4]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.4
[0.15.3]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.3
[0.15.2]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.2
[0.15.1]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.1
[0.15.0]: https://github.com/Arturossi/OCDocker/releases/tag/v0.15.0
