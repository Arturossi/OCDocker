# OCScore_models

Pretrained OCScore models, ready to score new data without retraining.

These six configurations come from the DUDEz feature-ablation study described
in the OCScore paper (`data/ocdb2/OCScore/output/latex/main.tex`, Sections
4.4/4.6). Each one is a real, trained model bundle picked from the study's
per-seed replicas — not a toy or placeholder.

| Config | Feature policy | Selected replica (seed) | Validation BEDROC | Test BEDROC | Notes |
|---|---|---|---|---|---|
| **#03** | `ligand_plus_scoring_function_no_shape_size` | replica_002 (44) | 0.6036 | 0.4030 | Recommended in the paper: best validation rank, distributed SHAP explanation (no shortcut risk) |
| #05 | `ligand_plus_scoring_function_no_pmi` | replica_001 (43) | 0.6187 | 0.4579 | Screened out in the paper's shortcut-risk step (hit the 20% cutoff exactly); not one of the paper's four final candidates. Shipped here for reference/comparison, not as a validated recommendation. |
| #09 | `ligand_plus_scoring_function_no_shape_size_no_autocorr2d` | replica_004 (46) | 0.6035 | 0.4574 | One of the paper's four final candidates (#03/#09/#12/#16), survives the shortcut-risk screen |
| #12 | `no_pmi` | replica_004 (46) | 0.6183 | 0.4547 | One of the paper's four final candidates, survives the shortcut-risk screen |
| #14 | `full_ocscore` | replica_000 (42) | 0.6218 | 0.3926 | Reference/baseline: complete 363-feature set, no ablation |
| #16 | `no_ligand_shape_size` | replica_000 (42) | 0.6172 | 0.4383 | One of the paper's four final candidates; the one previously missing from this directory (see below) |

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
is `true` (currently only #05), the DUDEz classifier reuses the PDBbind
bundle's feature extractor and `pdbbind_export_dir` is required — loading
without it raises `ValueError`. For the rest (currently #03/#09/#12/#14/#16,
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
