# OCScore production protocol

**Start here for the full pipeline:** [OCSCORE_REPLICATION.md](../OCSCORE_REPLICATION.md) (inputs -> reduce -> train -> export tools). This document covers production YAML presets, the unified leakage-safe modeling protocol, replica semantics, and external blind evaluation.

## Core rule

There is one valid OCScore modeling protocol:

```text
raw PDBbind + raw DUDEz
-> schema validation / alignment
-> fixed outer split
-> candidate model feature discovery
-> optional feature policy
-> train-only feature reduction on PDBbind train rows
-> frozen features applied to PDBbind/DUDEz
-> replicated staged Optuna using the fixed split
-> test evaluation
-> exported model
-> external blind inference only
```

Bundled YAML presets change runtime budget, replica count, reporting detail, and diagnostics. They do not change leakage-control rules.

## Quick start

```bash
# Prepare raw aligned input. This is schema validation + column alignment only.
ocdocker ocscore reduce \
  --pdbbind-archive path/to/PDBbind.csv \
  --dudez-archive path/to/DUDEz.csv \
  --output-dir path/to/raw_prepare

# Fast wiring check.
ocdocker ocscore train \
  --protocol smoke-test \
  --raw-input-dir path/to/raw_prepare \
  --output-dir out/smoke-test

# Production validation: full reporting, leakage audit, baselines.
ocdocker ocscore train \
  --protocol production \
  --raw-input-dir path/to/raw_prepare \
  --output-dir out/production

# External blind evaluation is inference only.
ocdocker ocscore external-blind \
  --export-dir path/to/best_model \
  --blind-csv path/to/external_blind.csv \
  --output-dir path/to/blind_eval \
  --label-column label \
  --group-column receptor \
  --kind-column kind
```

## Protocol files

Staged training is configured through YAML protocol files. The train CLI accepts either:

- a path to a custom `.yml` protocol file, or
- a bundled name resolved from `OCDocker/OCScore/Protocols/`.

Bundled protocols:

| Protocol | Replicas | Trials per stage | Reporting | Use case |
|----------|----------|------------------|-----------|----------|
| `smoke-test` | 1 | 3 | minimal | CI and wiring checks |
| `development` | 2 | 15 | light | Internal iteration |
| `production` | 3 | 50 | full | Production validation and reporting |
| `example` | 2 | 10 | off | Annotated template to copy and edit |

Copy `OCDocker/OCScore/Protocols/example.yml` to any location, edit it, and pass that path to `--protocol`. The train command copies the resolved protocol into the output directory for provenance.

## Accepted training inputs

`ocscore train` accepts raw unreduced modeling inputs in any one of these forms:

- `--merged-input path/to/merged_input_dataset.csv`
- `--pdbbind-input path/to/PDBbind.csv` plus `--dudez-input path/to/DUDEz.csv`
- `--raw-input-dir path/to/raw_prepare` produced by `ocscore reduce`

The key requirement is raw/unreduced input. The physical layout can be merged, separate, or a raw input directory.

The hidden `--reduction-archive` option is deprecated and is not a training path. Training from globally reduced or precomputed feature artifacts is not supported. Forbidden training artifacts include:

- `reduced_pdbbind.csv`
- `reduced_dudez.csv`
- `reduced_dataset.csv`
- `selected_features.json`
- `selected_features.txt`
- `feature_reduction_protocol.json`

Frozen selected-feature artifacts are only valid as part of export, scoring, or external blind inference workflows.

## What `ocscore reduce` does

`ocscore reduce` is a raw-input preparation step. It performs:

- schema validation;
- column alignment;
- merging raw PDBbind and DUDEz tables;
- provenance and content-hash writing.

It does not fit feature selection, feature cleaning, correlation filtering, scaling, or any other data-dependent modeling transform. Those transforms are fitted inside `ocscore train` after the fixed outer split is created.

## Train-only feature reduction

Every staged training run follows this order:

1. Load raw unreduced PDBbind and DUDEz inputs.
2. Perform schema-only validation.
3. Create one fixed outer split.
4. Fit feature cleaning, feature reduction, and scaling only on training rows.
5. Freeze selected features and transforms.
6. Apply frozen artifacts to validation, test, DUDEz, and blind data.
7. Train and tune on train/validation.
8. Evaluate test and blind data once.

Every train run records `feature_selection.json` under `train_only_feature_reduction/` with the `train_only` scope, selected-feature hash, removed-feature hash, fit split, fit row count, and transform artifact hashes.

## Feature Policies

Feature policies are named feature-family ablations applied after candidate model feature discovery and before train-only feature reduction. Each policy constrains the candidate descriptor pool, then OCScore fits its own train-only missingness, feature-cleaning, correlation, reduction, and scaling artifacts for that policy.

Bundled policies live in `OCDocker/OCScore/Protocols/Ablations/`. They use `.yml` files, not `.yaml`, with one policy per file. Bundled policies can be called by name:

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --feature-policy no_pmi   --output-dir out/no_pmi
```

When no policy is supplied, `full_ocscore` is used. This command:

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --output-dir out/full_ocscore
```

is equivalent to adding `--feature-policy full_ocscore`.

Run several policies as independent production runs:

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --feature-policy no_pmi   --feature-policy no_shape_core   --feature-policy shape_only   --output-dir out/ablations
```

Run all discovered policies:

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --run-all-feature-policies   --output-dir out/all_ablations
```

Add a custom lookup directory or an explicit one-off `.yml` file:

```bash
ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --feature-policy-dir /path/to/project/Ablations   --feature-policy my_custom_policy   --output-dir out/my_custom_policy

ocdocker ocscore train   --protocol production   --raw-input-dir path/to/raw_prepare   --feature-policy-yml /path/to/project/Ablations/my_policy.yml   --output-dir out/my_policy
```

Duplicate policy names across bundled policies, `--feature-policy-dir` entries, and explicit `.yml` files fail clearly and list all conflicting paths. Unknown policies fail and list available policies.

Policy schema:

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

Supported fields are `name`, `description`, `include_features`, `include_patterns`, `exclude_features`, `exclude_patterns`, `allow_missing_exclude_features`, and `allow_empty_policy`. Missing `include_features` fail. Missing `exclude_features` warn and are recorded by default. Metadata, label, group, and target columns are never model features, even if `include_patterns: ["*"]` matches them. Final feature order follows the deterministic production candidate-feature order.

Multiple-policy runs write `feature_policy_ablation_summary.json` and `feature_policy_ablation_summary.csv`. Each policy output contains `feature_policy_metadata.json`; export bundles and SHAP reports include the policy metadata. External blind evaluation uses the frozen selected features from the exported model and does not re-apply policy logic.

Interpretation table:

| Policy | Interpretation |
|--------|----------------|
| `full_ocscore` | Reference model using all candidate features. |
| `no_pmi` | Tests whether performance depends directly on PMI descriptors. |
| `no_shape_core` | Tests whether performance depends on core ligand 3D shape descriptors. |
| `no_ligand_shape_size` | Tests whether the model reroutes through ligand size/topology proxies. |
| `shape_only` | Measures how much ligand 3D shape alone can separate ligands from decoys. |
| `scoring_function_only` | Measures how much classical scoring functions alone can do under the same ML protocol. |
| `ligand_plus_scoring_function` | Tests ligand descriptors plus docking scores without receptor descriptors. |
| `ligand_plus_scoring_function_no_shape_core` | Tests ligand descriptors plus docking scores after removing core 3D shape descriptors. |
| `ligand_plus_scoring_function_no_shape_size` | Tests ligand descriptors plus docking scores after removing shape and ligand size/topology proxies. |
| `ligand_plus_scoring_function_no_shape_size_no_autocorr2d` | Tests ligand descriptors plus docking scores after removing shape/size proxies and AUTOCORR2D descriptors. |
| `ligand_plus_scoring_function_no_pmi` | Tests ligand descriptors plus docking scores after removing PMI descriptors only. |
| `ligand_plus_scoring_function_no_plants` | Tests ligand descriptors plus non-PLANTS scoring-function columns. |
| `ligand_plus_scoring_function_clean_receptor` | Tests ligand plus scoring-function columns with receptor descriptors but without receptor amino-acid counts and length proxies. |
| `no_scoring_function` | Tests whether the model depends on docking/scoring-function columns. |
| `ligand_only` | Tests ligand descriptor bias alone. |
| `receptor_plus_scoring_function` | Tests receptor descriptors plus scoring functions without ligand descriptors. |

Compare full vs ablated policies only under the same split, replica count, trial budget, and reporting settings.

## Replica semantics

Replicas reuse the same fixed outer split and the same frozen selected features/transforms. Replica seeds affect Optuna sampler behavior, model initialization, and training stochasticity. Replica seeds do not change the split or the selected feature list.

Selected-feature stability across replicas is therefore not expected to vary in this design: the selected features are intentionally fixed across replicas so aggregate metrics isolate stochastic optimization/training variation.

## Production reporting

When `reporting.generate_final_report: true` in a protocol, the training output includes production reporting artifacts such as:

- `final_report.json` with aggregate metrics and reporting policy;
- `leakage_audit.json` when `reporting.run_leakage_audit: true`;
- baseline CSVs when `reporting.run_baselines: true`;
- `split_assignments.json`, `environment.json`, and `command.json` provenance files.

The bundled `production` protocol enables final reporting, leakage audit, baselines, and minimum budget enforcement through `production_claim`.

## External blind evaluation

External blind evaluation is inference-only. It loads a frozen export bundle and applies the training-time selected features, scalers, checkpoints, and metadata to new raw rows.

External blind evaluation may ignore extra blind columns. It must fail when required selected features are missing. It must not run feature selection, scaling fit, Optuna, model choice, threshold tuning, or any other training-time selection step.

## Scientific caveats

The current production protocol uses one fixed outer split and train-only feature reduction shared across replicas. It controls feature-selection leakage for the staged protocol, but it is not a fully nested feature-selection procedure inside every possible outer CV fold unless a future nested-CV mode is implemented.

External blind benchmarks remain important for the strongest claims. Protein-family or sequence-cluster splits are stricter than receptor-heldout splits. Ligand/scaffold leakage diagnostics remain important. Ranking claims are separate from probability calibration claims; use `calibration_validated` reporting only when calibration is explicitly validated.
