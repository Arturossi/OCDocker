# OCScore_models

Pretrained OCScore models, ready to score new data without retraining.

All 22 configurations from the DUDEz feature-ablation study described in the
OCScore paper (`data/ocdb2/OCScore/output/latex/main.tex`, Sections 4.4/4.6)
are shipped here. Each one is a real, trained model bundle picked from the
study's per-seed replicas — not a toy or placeholder.

| Config | Feature policy | Selected replica (seed) | Validation BEDROC | Test BEDROC | Notes |
|---|---|---|---|---|---|
| #01 | `ligand_plus_scoring_function` | replica_004 (46) | 0.6113 | 0.4956 | Ligand descriptors + scoring functions, no receptor; base config the `#0x`/`#1x` "no_..." variants below are derived from |
| #02 | `ligand_plus_scoring_function_no_shape_core` | replica_000 (42) | 0.6143 | 0.4327 | Statistically worse than the full model (Holm-family p=0.042) |
| **#03** | `ligand_plus_scoring_function_no_shape_size` | replica_002 (44) | 0.6036 | 0.4030 | Recommended in the paper: best validation rank, distributed SHAP explanation (no shortcut risk) |
| #04 | `ligand_plus_scoring_function_no_plants` | replica_003 (45) | 0.6150 | 0.4568 | `#01` without PLANTS scoring-function columns |
| #05 | `ligand_plus_scoring_function_no_pmi` | replica_001 (43) | 0.6187 | 0.4579 | Screened out in the paper's shortcut-risk step (hit the 20% cutoff exactly); not one of the paper's four final candidates. Shipped here for reference/comparison, not as a validated recommendation. |
| #06 | `no_shape_core_no_receptor_surface_counts` | replica_002 (44) | 0.6145 | 0.3988 | Complete feature set minus shape-core minus receptor surface counts |
| #07 | `ligand_plus_scoring_function_no_pmi_no_autocorr2d` | replica_000 (42) | 0.6131 | 0.4416 | `#01` without PMI and AUTOCORR2D |
| #08 | `ligand_plus_scoring_function_no_pmi_no_plants` | replica_003 (45) | 0.6153 | 0.4574 | `#01` without PMI and PLANTS scores |
| #09 | `ligand_plus_scoring_function_no_shape_size_no_autocorr2d` | replica_004 (46) | 0.6035 | 0.4574 | One of the paper's four final candidates (#03/#09/#12/#16), survives the shortcut-risk screen |
| #10 | `ligand_plus_scoring_function_clean_receptor` | replica_000 (42) | 0.6181 | 0.4179 | Complete feature set, "clean receptor" (no shape-core, no receptor surface, no SASA) |
| #11 | `no_shape_core_no_receptor_length_pair` | replica_000 (42) | 0.6212 | 0.4274 | Complete feature set minus shape-core minus receptor chain-length pair |
| #12 | `no_pmi` | replica_004 (46) | 0.6183 | 0.4547 | One of the paper's four final candidates, survives the shortcut-risk screen |
| #13 | `no_shape_core_no_receptor_surface_size` | replica_003 (45) | 0.6189 | 0.4761 | Complete feature set minus shape-core minus receptor surface size |
| #14 | `full_ocscore` | replica_000 (42) | 0.6218 | 0.3926 | Reference/baseline: complete 363-feature set, no ablation |
| #15 | `no_shape_core` | replica_000 (42) | 0.6235 | 0.4050 | Complete feature set minus shape-core only |
| #16 | `no_ligand_shape_size` | replica_000 (42) | 0.6172 | 0.4383 | One of the paper's four final candidates; the one previously missing from this directory (see below) |
| #17 | `ligand_only` | replica_003 (45) | 0.6030 | 0.4129 | Ligand descriptors only — no receptor, no scoring functions |
| #18 | `ligand_plus_scoring_function_no_pmi_no_shape_size_no_autocorr2d_no_vsa` | replica_002 (44) | 0.5827 | 0.3713 | Most-pruned `#01` variant; statistically worse than the full model (Holm-family p=0.0003) |
| #19 | `scoring_function_only` | replica_004 (46) | 0.4207 | 0.3537 | Control: only the 16 raw scoring functions, no molecular descriptors; statistically worse (p<0.0001) |
| #20 | `no_scoring_function` | replica_004 (46) | 0.6077 | 0.3662 | Complete feature set minus all scoring functions — ligand + receptor descriptors only |
| #21 | `receptor_plus_scoring_function` | replica_003 (45) | 0.3958 | 0.3770 | Control: receptor descriptors + scoring functions, no ligand descriptors; statistically worse (p<0.0001) |
| #22 | `shape_only` | replica_003 (45) | 0.2178 | 0.1768 | Control: ligand shape-core descriptors only (5 features); weakest of all 22, statistically worse (p<0.0001) |

Statistical-significance notes above are from the paper's Holm-Bonferroni-corrected
paired t-test against the full model (`#14`) on validation BEDROC (Section 4.4);
configs without a note showed no statistically detectable difference from `#14`.

Config numbers match the `#NN` identifiers used throughout the paper and in
`examples/24_ocscore_bedroc_shortcut_risk_scatter.py`. Full per-config metadata
(feature-policy name, source replica/seed, selection rule, both metrics) is in
[`manifest.json`](manifest.json).

**#05 vs #16**: the paper's shortcut-risk screen (Section on SHAP-based selection)
keeps exactly four configurations as final candidates: **#03, #09, #12, #16**
(`main.tex` line 711: *"risco ≥20% descartou doze e deixou quatro (#03, #09,
#12, #16)"*). **#05 is explicitly discarded** in the paper (line 640: *"a #05...
atingiu exatamente 20,0% e foi descartada"*) and does not appear in the final
ranking table at all. This directory originally shipped #05 in #16's place;
#16 has since been added and #05 is kept alongside it for reference, not as a
stand-in for a validated candidate.

## Why these numbers and not others

Each ablation policy was trained 5 times with different seeds. Within each of
the configurations above, the shipped replica is the one with the
**highest validation-split BEDROC** among its 5 seeds — the same criterion the
paper uses to rank and select configurations (Section 4.6), so the shipped
weights were not cherry-picked on the DUDEz test split. The recorded
`dudez_test_bedroc` in the table/manifest is that replica's test score,
reported for reference only.

## Layout

Each `NN_<feature_policy_name>/` directory holds two independent export
bundles in the format written by
`OCDocker.OCScore.Optimization.ModelExport.export_best_model_bundle`:

```
03_ligand_plus_scoring_function_no_shape_size/
├── dudez/best_model/       # DUDEz screening classifier (transfer-linked to pdbbind/best_model)
│   ├── best_model.pt
│   ├── architecture.json
│   ├── retrain_config.json
│   ├── feature_metadata.json
│   ├── best_trial_summary.json
│   ├── probability_calibrator.joblib
│   └── split_indices.npz
└── pdbbind/best_model/     # PDBbind regressor whose feature extractor DUDEz was transferred from
    ├── best_model.pt
    ├── scaler.joblib
    └── ... (same files as above)
```

For configs where `retrain_config.json`'s `resolved_model_config.dudez_use_transfer`
is `true` (currently #05, #19, #20, #22), the DUDEz classifier reuses the
PDBbind bundle's feature extractor and `pdbbind_export_dir` is required —
loading without it raises `ValueError`. For the rest (18 of 22 configs,
each trained from scratch rather than transferred), `pdbbind_export_dir` is
unused but harmless to pass. **Always pass both directories together** — see
below — so the same call works regardless of a given config's transfer
setting. `retrain_config.json`'s `extra.pdbbind_best_model_export_dir` field
is left blank on purpose (it recorded a local training-machine path that has
no meaning after the bundle is copied elsewhere); pass `pdbbind_export_dir`
explicitly instead.

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

## What these models are *not*

This directory previously held single-file models (`OCScore.pt` +
`OCScore_scaler.pkl` + `OCScore_mask.pkl`) for an older architecture, loaded
through `OCDocker.OCScore.Scoring.get_score()`. That loader expects a
pickled model object (or a raw `state_dict`) and a boolean feature mask; it
does **not** understand the `best_model/` export-bundle format above and will
raise `ValueError: Model file contains a dict but no model object found` if
pointed at these files. Use `predict_from_export` / `ocdocker ocscore score`
for everything shipped here.
