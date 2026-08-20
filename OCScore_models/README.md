# OCScore_models

Pretrained OCScore DUDEz screening models, ready to score new data without
retraining. All 22 feature-ablation configurations from the OCScore paper
(`data/ocdb2/OCScore/output/latex/main.tex`, Sections 4.4/4.6) are shipped
here. Config numbers match the paper's `#NN` identifiers and the labels in
`examples/24_ocscore_bedroc_shortcut_risk_scatter.py`.

| Config | Feature policy | Replica (seed) | Val. BEDROC | Test BEDROC | Notes |
|---|---|---|---|---|---|
| #01 | `ligand_plus_scoring_function` | replica_004 (46) | 0.6113 | 0.4956 | Ligand + scoring functions, no receptor |
| #02 | `ligand_plus_scoring_function_no_shape_core` | replica_000 (42) | 0.6143 | 0.4327 | Worse than the full model (p=0.042) |
| **#03** | `ligand_plus_scoring_function_no_shape_size` | replica_002 (44) | 0.6036 | 0.4030 | Recommended |
| #04 | `ligand_plus_scoring_function_no_plants` | replica_003 (45) | 0.6150 | 0.4568 | #01 without PLANTS scores |
| #05 | `ligand_plus_scoring_function_no_pmi` | replica_001 (43) | 0.6187 | 0.4579 | Fails the shortcut-risk screen; not a final candidate |
| #06 | `no_shape_core_no_receptor_surface_counts` | replica_002 (44) | 0.6145 | 0.3988 | Full set minus shape-core, receptor surface counts |
| #07 | `ligand_plus_scoring_function_no_pmi_no_autocorr2d` | replica_000 (42) | 0.6131 | 0.4416 | #01 minus PMI, AUTOCORR2D |
| #08 | `ligand_plus_scoring_function_no_pmi_no_plants` | replica_003 (45) | 0.6153 | 0.4574 | #01 minus PMI, PLANTS |
| #09 | `ligand_plus_scoring_function_no_shape_size_no_autocorr2d` | replica_004 (46) | 0.6035 | 0.4574 | Final candidate |
| #10 | `ligand_plus_scoring_function_clean_receptor` | replica_000 (42) | 0.6181 | 0.4179 | Full set, reduced receptor descriptors |
| #11 | `no_shape_core_no_receptor_length_pair` | replica_000 (42) | 0.6212 | 0.4274 | Full set minus shape-core, receptor chain-length |
| #12 | `no_pmi` | replica_004 (46) | 0.6183 | 0.4547 | Final candidate |
| #13 | `no_shape_core_no_receptor_surface_size` | replica_003 (45) | 0.6189 | 0.4761 | Full set minus shape-core, receptor surface size |
| #14 | `full_ocscore` | replica_000 (42) | 0.6218 | 0.3926 | Reference: complete 363-feature set |
| #15 | `no_shape_core` | replica_000 (42) | 0.6235 | 0.4050 | Full set minus shape-core |
| #16 | `no_ligand_shape_size` | replica_000 (42) | 0.6172 | 0.4383 | Final candidate |
| #17 | `ligand_only` | replica_003 (45) | 0.6030 | 0.4129 | Ligand descriptors only |
| #18 | `ligand_plus_scoring_function_no_pmi_no_shape_size_no_autocorr2d_no_vsa` | replica_002 (44) | 0.5827 | 0.3713 | Worse than the full model (p=0.0003) |
| #19 | `scoring_function_only` | replica_004 (46) | 0.4207 | 0.3537 | Control: scoring functions only; worse (p<0.0001) |
| #20 | `no_scoring_function` | replica_004 (46) | 0.6077 | 0.3662 | Full set minus scoring functions |
| #21 | `receptor_plus_scoring_function` | replica_003 (45) | 0.3958 | 0.3770 | Control: no ligand descriptors; worse (p<0.0001) |
| #22 | `shape_only` | replica_003 (45) | 0.2178 | 0.1768 | Control: shape descriptors only; worse (p<0.0001) |

"Worse"/"Control" notes are Holm-Bonferroni-corrected significance results
against the full model (`#14`) on validation BEDROC (Section 4.4); configs
without a note showed no statistically detectable difference. Full per-config
metadata (feature-policy name, source replica/seed, selection rule, both
metrics) is in [`manifest.json`](manifest.json).

## Selection

Each policy was trained 5 times with different seeds. The shipped replica is
the one with the highest validation-split BEDROC among its 5 seeds — the
same criterion the paper uses (Section 4.6).

The paper's shortcut-risk screen keeps four final candidates: **#03**
(recommended), **#09**, **#12**, **#16**. **#05** fails that screen and is
not a final candidate, but is shipped for reference/comparison.

## Layout

Each `NN_<feature_policy_name>/` directory holds two independent export
bundles in the format written by
`OCDocker.OCScore.Optimization.ModelExport.export_best_model_bundle`:

```
03_ligand_plus_scoring_function_no_shape_size/
├── dudez/best_model/       # DUDEz screening classifier
│   ├── best_model.pt
│   ├── architecture.json
│   ├── retrain_config.json
│   ├── feature_metadata.json
│   ├── best_trial_summary.json
│   ├── probability_calibrator.joblib
│   └── split_indices.npz
└── pdbbind/best_model/     # linked PDBbind regressor
    ├── best_model.pt
    ├── scaler.joblib
    └── ... (same files as above)
```

**Always pass both directories together** — see below. For configs where
`retrain_config.json`'s `resolved_model_config.dudez_use_transfer` is `true`
(#05, #19, #20, #22), the DUDEz classifier reuses the PDBbind bundle's
feature extractor and `pdbbind_export_dir` is required — loading without it
raises `ValueError`. Every config's DUDEz classifier was also trained on
features standardized by the PDBbind bundle's `scaler.joblib`, so
`pdbbind_export_dir` is required for correct scoring in all 22 cases, not
just the transfer ones. `retrain_config.json`'s
`extra.pdbbind_best_model_export_dir` field is left blank on purpose (it
recorded a local training-machine path with no meaning after the bundle is
copied elsewhere); pass `pdbbind_export_dir` explicitly instead.

## Scoring new data

### Python API

```python
import OCDocker.OCScore.Optimization.ModelExport as ocexport
import OCDocker.OCScore.Utils.IO as ocscoreio

raw = ocscoreio.load_pipeline_results_from_archive("pipeline_results.csv")
dataframe = ocscoreio.prepare_dudez_dataframe(raw)  # requires a "kind" column (ligands/decoys)

predictions = ocexport.predict_from_export(
    "OCScore_models/03_ligand_plus_scoring_function_no_shape_size/dudez/best_model",
    dataframe,
    pdbbind_export_dir="OCScore_models/03_ligand_plus_scoring_function_no_shape_size/pdbbind/best_model",
)
```

`predictions` adds `ocscore_prediction`, `ocscore_probability`, and
`ocscore_probability_calibrated` columns to the input rows. Scoring uses the
bundle's frozen `selected_features` list; it does not refit scalers or
feature reduction on the new data.

### CLI

```bash
ocdocker ocscore score \
  --export-dir OCScore_models/03_ligand_plus_scoring_function_no_shape_size/dudez/best_model \
  --pdbbind-export-dir OCScore_models/03_ligand_plus_scoring_function_no_shape_size/pdbbind/best_model \
  --raw-archive /path/to/pipeline_results.csv \
  --output-csv /path/to/predictions.csv
```

`ocdocker ocscore validate` and `ocdocker ocscore load` also accept
`--pdbbind-export-dir` for sanity-checking a bundle without raw data.

### Runnable example

```bash
python examples/25_ocscore_score_with_shipped_models.py --list
python examples/25_ocscore_score_with_shipped_models.py \
  --config 03 --raw-archive /path/to/pipeline_results.csv --output-csv /path/to/predictions.csv
```

## Legacy models

This directory previously held single-file models (`OCScore.pt` +
`OCScore_scaler.pkl` + `OCScore_mask.pkl`) for an older architecture, loaded
through `OCDocker.OCScore.Scoring.get_score()`. That loader does not
understand the `best_model/` export-bundle format above and will raise
`ValueError: Model file contains a dict but no model object found` if
pointed at these files. Use `predict_from_export` / `ocdocker ocscore score`
for everything shipped here.
