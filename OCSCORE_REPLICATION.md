# OCScore replication guide

End-to-end instructions to reproduce staged OCScore results starting from raw ocdb2-style pipeline CSV outputs: PDBbind and DUDEz feature tables produced after docking, clustering, rescoring, and descriptor extraction.

**Audience:** researchers and collaborators who already have ocdb2-style pipeline tables and want a copy-paste path through raw input preparation, train-only feature reduction, staged Optuna, export, and scoring.

**Related docs:**

- Production protocol details: `docs/ocscore-production-protocol.md`
- General CLI reference: `docs/source/usage.rst` or `ocdocker ocscore --help`
- Upstream docking/rescoring: `MANUAL.md` and `ocdocker pipeline`

---

## Prerequisites

1. Python environment, recommended with conda/mamba and Python 3.11:

   ```bash
   mamba create -n ocdocker python=3.11 -y
   conda activate ocdocker
   pip install "ocdocker[ml]"
   ```

2. Hardware: GPU strongly recommended for Optuna training. CPU works for small budgets.

3. Inputs: raw wide PDBbind and DUDEz pipeline result tables.

4. Example layout:

   ```text
   data/ocdb2/PDBbind/PDBbind.csv
   data/ocdb2/DUDEz/DUDEz.csv
   runs/ocscore/raw_prepare/
   runs/ocscore/train_development/
   runs/ocscore/train_production/
   ```

---

## Input artifacts

Each input file is a raw wide feature table: one row per receptor-ligand or receptor-decoy complex, with metadata columns, scoring-function columns, and engineered descriptors. This guide does not re-run docking.

The preparation loader accepts any of these shapes for each of `--pdbbind-archive` and `--dudez-archive`:

| Shape | Example |
|-------|---------|
| Bare CSV file | `data/ocdb2/PDBbind/PDBbind.csv` |
| Directory containing a canonical CSV | `data/ocdb2/PDBbind/` |
| Tar or tar.gz archive | `data/ocdb2/PDBbind/PDBbind.tar.gz` |

Canonical CSV names inside directories or archives are tried in this order:

1. `pipeline_results.csv`
2. `PDBbind.csv`
3. `DUDEz.csv`

Required columns:

| Dataset | Required column | Notes |
|---------|-----------------|-------|
| PDBbind | `experimental` | Affinity target for regression |
| DUDEz | `kind` | Active rows: `ligands`; decoy rows: `decoys` |
| Both | `receptor` | Used for grouped splits and screening metrics |
| Both | Descriptor / SF columns | Must overlap between PDBbind and DUDEz |

---

## Unified protocol

There is one valid OCScore modeling protocol:

```text
raw PDBbind + raw DUDEz
-> schema validation / alignment
-> fixed outer split
-> train-only feature reduction on PDBbind train rows
-> frozen features applied to PDBbind/DUDEz
-> replicated staged Optuna using the fixed split
-> test evaluation
-> exported model
-> external blind inference only
```

`ocdocker ocscore train` always uses raw unreduced inputs. Training from globally reduced or precomputed feature artifacts is not supported.

Forbidden training artifacts include:

- `reduced_pdbbind.csv`
- `reduced_dudez.csv`
- `reduced_dataset.csv`
- `selected_features.json`
- `selected_features.txt`
- `feature_reduction_protocol.json`

Frozen selected-feature artifacts are valid only as part of export, scoring, or external blind inference workflows.

---

## Stage 1: prepare raw modeling input

`ocscore reduce` is a raw-input preparation and alignment step. It performs schema validation, column alignment, merged raw-table writing, and provenance/hash writing. It does not fit feature selection, feature cleaning, correlation filtering, scaling, or any other data-dependent modeling transform.

```bash
ocdocker ocscore reduce   --pdbbind-archive data/ocdb2/PDBbind/PDBbind.csv   --dudez-archive data/ocdb2/DUDEz/DUDEz.csv   --output-dir runs/ocscore/raw_prepare
```

Tar.gz inputs work the same way:

```bash
ocdocker ocscore reduce   --pdbbind-archive data/ocdb2/PDBbind/PDBbind.tar.gz   --dudez-archive data/ocdb2/DUDEz/DUDEz.tar.gz   --output-dir runs/ocscore/raw_prepare
```

Stage 1 outputs:

| File | Purpose |
|------|---------|
| `merged_input_dataset.csv` | Aligned raw PDBbind+DUDEz table |
| `raw_pdbbind.csv` | PDBbind rows only, unreduced |
| `raw_dudez.csv` | DUDEz rows only, unreduced |
| `prepare_manifest.json` | Content hashes and provenance |

---

## Stage 2: staged Optuna training

Accepted raw training input forms:

- one merged raw input table via `--merged-input`;
- separate raw PDBbind and DUDEz tables via `--pdbbind-input` plus `--dudez-input`;
- a raw input directory produced by `ocscore reduce` via `--raw-input-dir`.

The key requirement is raw/unreduced input, not the physical layout.

### Fast budget preset

Use `smoke-test` for CI and command wiring checks.

```bash
ocdocker ocscore train   --protocol smoke-test   --raw-input-dir runs/ocscore/raw_prepare   --output-dir runs/ocscore/train_smoke
```

### Development budget preset

Use `development` for internal experiments and debugging.

```bash
ocdocker ocscore train   --protocol development   --raw-input-dir runs/ocscore/raw_prepare   --output-dir runs/ocscore/train_development
```

### Production budget preset

Use `production` for production validation and full reporting.

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir runs/ocscore/raw_prepare   --output-dir runs/ocscore/train_production
```

Production preset highlights:

- 3 replicas and 50 Optuna trials per stage by default;
- PDBbind receptor-heldout split;
- train-only feature reduction after the fixed outer split;
- DUDEz `pdbbind_scaler` feature scaling;
- leakage audit, provenance files, baseline reports, and final report;
- minimum budget enforcement through `production_claim`.

### Feature-policy ablations

Feature policies are controlled feature-family ablations applied before train-only feature reduction. They live in `OCDocker/OCScore/Protocols/Ablations/` and use one `.yml` file per policy. Bundled policies can be called by name with `--feature-policy`; custom directories can be added with repeatable `--feature-policy-dir`. Duplicate policy names across bundled, custom, and explicit `.yml` files fail because silent overrides would invalidate scientific controls.

The training order remains:

```text
raw unreduced input
-> schema validation / alignment
-> fixed outer split
-> candidate model feature discovery
-> feature policy
-> train-only feature cleaning/reduction/scaling
-> frozen transforms applied to validation/test/DUDEz/blind rows
```

Reduced CSVs, precomputed `selected_features.json`, and global feature-reduction artifacts are still forbidden as training inputs. External blind evaluation uses the frozen selected features stored in the export bundle and does not re-apply a named policy.

Run default full OCScore:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --output-dir /path/to/output/full_ocscore
```

This is equivalent to:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy full_ocscore     --output-dir /path/to/output/full_ocscore
```

Run one bundled ablation:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy no_pmi     --output-dir /path/to/output/no_pmi
```

Run multiple bundled ablations:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy no_pmi     --feature-policy no_shape_core     --feature-policy shape_only     --output-dir /path/to/output/ablations
```

Run all bundled ablations:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --run-all-feature-policies     --output-dir /path/to/output/all_ablations
```

Add a custom policy directory:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy-dir /path/to/project/Ablations     --feature-policy my_custom_policy     --output-dir /path/to/output/my_custom_policy
```

Run all bundled plus custom policies:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy-dir /path/to/project/Ablations     --run-all-feature-policies     --output-dir /path/to/output/all_policies
```

Run one explicit `.yml` policy:

```bash
ocdocker ocscore train     --protocol production     --raw-input-dir /path/to/raw_inputs     --feature-policy-yml /path/to/project/Ablations/my_policy.yml     --output-dir /path/to/output/my_policy
```

Example custom policy:

```yaml
name: no_custom_ligand_size
description: Remove custom ligand size descriptors used by this project.
include_patterns:
  - "*"
exclude_features:
  - ligand_MolWt
  - ligand_ExactMolWt
  - ligand_HeavyAtomCount
  - ligand_RadiusOfGyration
```

Policy names must be unique across the lookup pool, should use lowercase snake_case, and each `.yml` file should contain one policy. Missing `include_features` fail. Missing `exclude_features` warn and are recorded by default. Metadata, label, group, and target columns are never included, even with `include_patterns: ["*"]`.

Interpretation guide:

| Policy | Interpretation |
|--------|----------------|
| `full_ocscore` | Reference model using all candidate features. |
| `no_pmi` | Tests whether performance depends directly on PMI descriptors. |
| `no_shape_core` | Tests whether performance depends on core ligand 3D shape descriptors. |
| `no_ligand_shape_size` | Tests whether the model reroutes through ligand size/topology proxies. |
| `shape_only` | Measures how much ligand 3D shape alone can separate ligands from decoys. |
| `scoring_function_only` | Measures how much classical scoring functions alone can do under the same ML protocol. |
| `ligand_plus_scoring_function` | Tests ligand descriptors plus docking scores without receptor descriptors. |
| `no_scoring_function` | Tests whether the model depends on docking/scoring-function columns. |
| `ligand_only` | Tests ligand descriptor bias alone. |
| `receptor_plus_scoring_function` | Tests receptor descriptors plus scoring functions without ligand descriptors. |
| `receptor_only` | Tests whether receptor descriptors alone contain split/task artifacts. |

Compare ablations only when they use the same split, replicas, and budget. Multiple-policy runs write `feature_policy_ablation_summary.json` and `feature_policy_ablation_summary.csv`.

### Custom YAML preset

Copy `OCDocker/OCScore/Protocols/example.yml`, edit it, and pass its path to `--protocol`:

```bash
mkdir -p ~/protocols/ocscore
cp OCDocker/OCScore/Protocols/example.yml ~/protocols/ocscore/my-run.yml

ocdocker ocscore train   --protocol ~/protocols/ocscore/my-run.yml   --raw-input-dir runs/ocscore/raw_prepare   --output-dir runs/ocscore/train_custom
```

Bundled names are `smoke-test`, `development`, `production`, and `example`. See `docs/ocscore-production-protocol.md` for the field reference and annotated workflow.

### Replica semantics

All replicas use the same fixed outer split and the same frozen selected feature list and transforms. Replicas differ only by stochastic optimization and training: Optuna sampler seed, model initialization, and training stochasticity. Selected-feature stability across replicas is intentionally fixed by design.

### Stage 2 outputs

| Path / file | Purpose |
|-------------|---------|
| `staged_optuna_protocol.json` / `.md` | Run summary, aggregate metrics, replica table |
| `modeling_pdbbind.csv` / `modeling_dudez.csv` | Train-only selected features applied to all rows |
| `train_only_feature_reduction/` | Frozen feature selection artifacts fitted on training rows |
| `fixed_outer_split.json` | Fixed outer split hashes and row indices |
| `replica_000/`, `replica_001/`, ... | Per-replica staged protocol outputs |
| `replica_*/pdbbind/best_model/` | PDBbind export bundle |
| `replica_*/dudez/best_model/` | DUDEz export bundle |
| `replicas_summary.csv` / `.json` | One row per replica plus aggregate metrics |
| `replicas_protocol.json` | Replicated protocol metadata |

Production reporting outputs include:

| File | Purpose |
|------|---------|
| `final_report.json` | Aggregate metrics and reporting policy |
| `split_assignments.json` | Split indices and diagnostics |
| `leakage_audit.json` | Overlap and scope checks |
| `environment.json` / `command.json` | Reproducibility manifest |
| `baselines_per_fold.csv` | Per-replica baseline metrics |
| `baselines_summary.csv` | Median baseline metrics across replicas |
| `baselines_rank_table.csv` | Baselines ranked by test BEDROC |

---

## Stage 3: export tools

All subcommands require `pip install "ocdocker[ml]"`. Point `--export-dir` at a replica `best_model/` directory.

```bash
ocdocker ocscore validate --export-dir EXPORT
ocdocker ocscore load --export-dir EXPORT --device cpu
```

Cross-validation and plotting with frozen hyperparameters:

```bash
ocdocker ocscore cross-validate   --export-dir EXPORT   --pdbbind-csv runs/ocscore/train_development/modeling_pdbbind.csv   --dudez-csv runs/ocscore/train_development/modeling_dudez.csv   --n-folds 5   --seed 42   --output-dir EXPORT/cross_validation

ocdocker ocscore plot   --export-dir EXPORT   --cv-dir EXPORT/cross_validation
```

Score new raw pipeline data:

```bash
ocdocker ocscore score   --export-dir EXPORT   --raw-archive data/new_screen/pipeline_results.csv   --output-csv runs/ocscore/predictions.csv
```

Scoring uses frozen selected features and model artifacts. It does not re-run feature reduction or refit scalers on new data.

---

## External blind evaluation

External blind evaluation is inference-only. It loads frozen selected features, scalers, model checkpoints, and metadata from a valid export bundle. It may ignore extra blind columns, but it must fail on missing required selected features.

It must not run feature selection, scaling fit, Optuna, model choice, threshold tuning, or any other training-time selection step.

---

## Artifact catalog

| Artifact | Stage | Description |
|----------|-------|-------------|
| `PDBbind.csv` / `DUDEz.csv` | Input | Raw pipeline feature tables |
| `merged_input_dataset.csv` | Reduce | Aligned raw combined input |
| `raw_pdbbind.csv` / `raw_dudez.csv` | Reduce | Dataset-specific raw tables |
| `prepare_manifest.json` | Reduce | Raw input content hashes |
| `modeling_pdbbind.csv` / `modeling_dudez.csv` | Train | Train-only selected features applied to all rows |
| `train_only_feature_reduction/` | Train | Frozen feature selection artifacts |
| `fixed_outer_split.json` | Train | Fixed outer split hashes |
| `staged_optuna_protocol.json` | Train | Protocol summary |
| `replica_*/**/best_model/` | Train | Portable inference bundle |
| `final_report.json` | Production reporting | Headline metrics and policy |
| `split_assignments.json` | Production reporting | Split indices |
| `leakage_audit.json` | Production reporting | Leakage check results |
| `baselines_*.csv` | Production reporting | SF and sklearn baselines |
| `cross_validation_results.json` | Export | CV metrics |
| `predictions.csv` | Export | Scored raw pipeline rows |

---

## Reproducibility checklist

- [ ] Record `ocdocker` version (`pip show ocdocker`) and save `environment.json` for production runs.
- [ ] Keep original raw `PDBbind.csv` and `DUDEz.csv` inputs unchanged alongside outputs.
- [ ] Archive `runs/ocscore/raw_prepare/` and the train output directory.
- [ ] Verify `leakage_audit.json` reports `passed: true` when production reporting is enabled.
- [ ] Document train-only feature selection (`feature_selection_scope=train_only`) in methods.
- [ ] Cite replica aggregate metrics for scientific claims; best-replica exports are deployment candidates.

---

## Known limitations and caveats

1. Starts at pipeline CSVs; generating those tables from structures requires the main OCDocker docking pipeline.
2. The current production protocol uses one fixed outer split and train-only feature reduction shared across replicas. It controls feature-selection leakage for the staged protocol, but it is not a fully nested feature-selection procedure inside every possible outer CV fold unless a future nested-CV mode is implemented.
3. External blind benchmarks remain important for strongest claims.
4. Protein-family or sequence-cluster splits are stricter than receptor-heldout splits.
5. Ligand/scaffold leakage diagnostics remain important.
6. Ranking claims are separate from probability calibration claims.
7. Feature-policy ablations are available through `--feature-policy` and should be compared with matched split, replica, and budget settings.
8. `ocdocker ocscore compare` is not yet a first-class CLI subcommand.

---

## Python API equivalents

| CLI | Example script |
|-----|----------------|
| `ocscore reduce` | `examples/14_feature_reduction_pdbbind_dudez.py` |
| `ocscore train` | `examples/15_ocscore_staged_optuna_from_reduction.py` |
| `ocscore validate|score|...` | `examples/16_ocscore_exported_model_tools.py` |

The CLI modules are the source of truth; examples are thin wrappers.
