# Manual Run Log (AE+NN Training)

This file is manually curated from terminal outputs shared by the user.

## Curation Rules
- `--no_invert` runs are excluded.
- If the same `model_name` was run multiple times and artifacts were overwritten, only the newest successful run is kept.

## Run 001
Date: 2026-03-04
Source: user-provided terminal output

Command:
```bash
conda run -n ocdocker python examples/11_python_api_train_model_from_db.py \
  --model_type DNN \
  --model_name OCScore_AE_NN_7 \
  --df_path /data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz \
  --storage postgresql+psycopg://ocdocker:<password>@localhost:5432/optimization \
  --studies NN_Optimization_7 \
  --use_gpu
```

Data summary:
- Train shape: (2256, 454)
- Test shape: (752, 454)
- Validation shape: (10438, 454)

Study selection:
- Mode: global best trial across provided studies
- Studies provided: [NN_Optimization_7]
- Complete trials found: 625
- Selected NN study/trial: NN_Optimization_7 / 841
- Selected NN Optuna metrics: RMSE 0.6056, AUC 0.7465, Combined -0.1409

Autoencoder/mask resolution during training:
- AO search scope: AO_Optimization_7 (derived from NN study id)
- Selected AO study/trial: AO_Optimization_7 / 1606
- Selected AO metrics: RMSE 0.4100, Val_RMSE 0.4955
- Mask source: NN_Ablation_Optimization_1
- Mask shape/active: (454) / 446 active

Final retrain settings:
- Epochs: 968
- Batch size: 32
- Learning rate: 0.03789006794194488
- Optimizer: SGD
- GPU: True

Final evaluation (after retrain):
- RMSE: 1.6925
- AUC: 0.7542
- AUC_adj: 0.7542
- Combined (RMSE - AUC): 0.9384

Evaluation inference note:
- The original terminal output for this run did not print final evaluation.
- Metrics above were inferred by loading `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_7.pt`
  and evaluating with the same preprocessing settings used in script 11
  (`invert_conditionally=True`, `normalize=True`, `scaler=standard`,
  `random_seed=42`, reference column order enforced).

Artifacts saved:
- Model: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_7.pt
- Mask: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_7_mask.pkl
- Scaler: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_7_scaler.pkl

Checksums (SHA256):
- Model: `82db67b30b8300c829f1a062cfc9944cffc1ac8c08531f9a8aac624a1ba8cd9c`
- Mask: `7f0ffb858236a22ae782ed39629b100eed16ffff5849e5c086893c7b7f3950f0`
- Scaler: `7ae8327821737a47b69f374a531a233be08ab0e12c9c6f9e0b5b8c1e1d85ebc3`

Artifact timestamps:
- Model: `2026-03-04 15:28:46.045308290 -0300`
- Mask: `2026-03-04 15:28:46.046045940 -0300`
- Scaler: `2026-03-04 20:38:38.959990633 -0300`

---

## Run 002
Date: 2026-03-04
Source: user-provided terminal output

Command:
```bash
conda run -n ocdocker python examples/11_python_api_train_model_from_db.py \
  --model_type DNN \
  --df_path /data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz \
  --storage 'postgresql+psycopg://ocdocker:<password>@localhost:5432/optimization' \
  --pair_by_major \
  --major_numbers 6 7 8 9 10 \
  --pair_select_by rmse \
  --model_name OCScore_AE_NN_6to10_best_rmse \
  --output_dir /data/hd4tb/OCDocker/OCDocker/OCScore_models \
  --use_gpu
```

Data summary:
- Train shape: (2256, 454)
- Test shape: (752, 454)
- Validation shape: (10438, 454)

Per-major selected studies/trials (before final pick):
- Major 6: NN trial 869 (RMSE 0.6140, AUC 0.6980, Combined -0.0841), AO trial 1940 (RMSE 0.4075, Val_RMSE 0.4864)
- Major 7: NN trial 737 (RMSE 0.6042, AUC 0.7185, Combined -0.1142), AO trial 1606 (RMSE 0.4100, Val_RMSE 0.4955)
- Major 8: NN trial 774 (RMSE 0.6117, AUC 0.7291, Combined -0.1174), AO trial 1672 (RMSE 0.3952, Val_RMSE 0.4784)
- Major 9: NN trial 773 (RMSE 0.6145, AUC 0.7421, Combined -0.1276), AO trial 1837 (RMSE 0.4143, Val_RMSE 0.4992)
- Major 10: NN trial 829 (RMSE 0.6098, AUC 0.7380, Combined -0.1283), AO trial 1611 (RMSE 0.3982, Val_RMSE 0.4821)

Selection result:
- Criterion: `rmse`
- Selected major: 7
- Selected NN study/trial: NN_Optimization_7 / 737
- Selected AO study/trial: AO_Optimization_7 / 1606

Final retrain settings:
- Epochs: 1000
- Batch size: 32
- Learning rate: 0.04038070571338472
- Optimizer: SGD
- GPU: True
- Mask source: NN_Ablation_Optimization_1
- Mask shape/active: (454) / 446 active

Final evaluation (after retrain):
- RMSE: 1.6774
- AUC: 0.7527
- AUC_adj: 0.7527
- Combined (RMSE - AUC): 0.9246

Drift vs selected Optuna NN trial (737):
- RMSE drift: +1.0732 (1.6774 - 0.6042)
- AUC drift: +0.0342 (0.7527 - 0.7185)

Artifacts saved:
- Model: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_rmse.pt
- Mask: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_rmse_mask.pkl
- Scaler: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_rmse_scaler.pkl
- Pairwise summary: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_rmse_pairwise_major_results.csv

Checksums (SHA256):
- Model: `e1ea32061d04f8b1546747ae66a1aa8824d9373f5087183f9a97e186f8d49b49`
- Mask: `7f0ffb858236a22ae782ed39629b100eed16ffff5849e5c086893c7b7f3950f0`
- Scaler: `7ae8327821737a47b69f374a531a233be08ab0e12c9c6f9e0b5b8c1e1d85ebc3`
- Pairwise summary: `fdd4604a9cedf59a52572cee22431f88bb1c0c316a10efab481ce0b1a43f8ead`

Artifact timestamps:
- Model: `2026-03-04 18:08:26.328259199 -0300`
- Mask: `2026-03-04 18:08:26.328578412 -0300`
- Scaler: `2026-03-04 18:08:26.329418947 -0300`
- Pairwise summary: `2026-03-04 18:08:26.331578420 -0300`

---

## Run 003
Date: 2026-03-04
Source: user-provided terminal output

Command:
```bash
conda run -n ocdocker python examples/11_python_api_train_model_from_db.py \
  --model_type DNN \
  --df_path /data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz \
  --storage 'postgresql+psycopg://ocdocker:<password>@localhost:5432/optimization' \
  --pair_by_major \
  --major_numbers 6 7 8 9 10 \
  --pair_select_by auc \
  --model_name OCScore_AE_NN_6to10_best_auc \
  --output_dir /data/hd4tb/OCDocker/OCDocker/OCScore_models \
  --use_gpu
```

Data summary:
- Train shape: (2256, 454)
- Test shape: (752, 454)
- Validation shape: (10438, 454)

Per-major selected studies/trials (before final pick):
- Major 6: NN trial 357 (RMSE 0.6631, AUC 0.7653, Combined -0.1021), AO trial 1940 (RMSE 0.4075, Val_RMSE 0.4864)
- Major 7: NN trial 386 (RMSE 0.8218, AUC 0.7569, Combined 0.0649), AO trial 1606 (RMSE 0.4100, Val_RMSE 0.4955)
- Major 8: NN trial 108 (RMSE 0.6269, AUC 0.7612, Combined -0.1343), AO trial 1672 (RMSE 0.3952, Val_RMSE 0.4784)
- Major 9: NN trial 710 (RMSE 0.6242, AUC 0.7628, Combined -0.1386), AO trial 1837 (RMSE 0.4143, Val_RMSE 0.4992)
- Major 10: NN trial 468 (RMSE 0.7083, AUC 0.7607, Combined -0.0524), AO trial 1611 (RMSE 0.3982, Val_RMSE 0.4821)

Selection result:
- Criterion: `auc`
- Selected major: 6
- Selected NN study/trial: NN_Optimization_6 / 357
- Selected AO study/trial: AO_Optimization_6 / 1940

Final retrain settings:
- Epochs: 851
- Batch size: 32
- Learning rate: 0.07032442713233411
- Optimizer: SGD
- GPU: True
- Mask source: NN_Ablation_Optimization_1
- Mask shape/active: (454) / 446 active

Final evaluation (after retrain):
- RMSE: 1.8943
- AUC: 0.6722
- AUC_adj: 0.6722
- Combined (RMSE - AUC): 1.2221

Drift vs selected Optuna NN trial (357):
- RMSE drift: +1.2312 (1.8943 - 0.6631)
- AUC drift: -0.0931 (0.6722 - 0.7653)

Artifacts saved:
- Model: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_auc.pt
- Mask: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_auc_mask.pkl
- Scaler: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_auc_scaler.pkl
- Pairwise summary: /data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_best_auc_pairwise_major_results.csv

Checksums (SHA256):
- Model: `77ac603c7bd82af8e398bd3a9687a6980eba613edc9ea387853d764eaf2d17c7`
- Mask: `7f0ffb858236a22ae782ed39629b100eed16ffff5849e5c086893c7b7f3950f0`
- Scaler: `7ae8327821737a47b69f374a531a233be08ab0e12c9c6f9e0b5b8c1e1d85ebc3`
- Pairwise summary: `808a48fc22936f07c12cbbb7e2f617b87ad6b559d28626611075dc6859d64c33`

Artifact timestamps:
- Model: `2026-03-04 18:17:09.467027552 -0300`
- Mask: `2026-03-04 18:17:09.467329512 -0300`
- Scaler: `2026-03-04 18:17:09.467614039 -0300`
- Pairwise summary: `2026-03-04 18:17:09.468614041 -0300`

---

## Attempt Log
Date: 2026-03-04
Source: user-provided terminal output

- Attempt A1: `OCScore_AE_NN_7` run interrupted (`KeyboardInterrupt`) before completion.
- Attempt A2: `--pair_select_by rmse` failed with `unrecognized arguments` (script version at that moment lacked parser arg).
- Attempt A3: one `--pair_select_by rmse` run interrupted (`^C`) before completion.

---

## Run 004
Date: 2026-03-04
Source: reconstructed manually from saved artifacts and terminal logs

Run identity:
- Model name: `OCScore_AE_NN_6to10_bestpair`
- Selection mode: `pair_by_major`
- Selection criterion: `combined`
- Selected major: 9

Input data:
- Data file: `/data/hd4tb/OCDocker/data/ocdb/OCDocker.csv.gz`
- Storage: `postgresql+psycopg://ocdocker:<password>@localhost:5432/optimization`
- Train shape: (2256, 454)
- Test shape: (752, 454)
- Validation shape: (10438, 454)

Selection context:
- Selected NN study/trial: `NN_Optimization_9` / 721
- Selected AO study/trial: `AO_Optimization_9` / 1837
- Selected NN Optuna RMSE: 0.615866418159702
- Selected NN Optuna AUC: 0.7605762935392184
- Selected NN Optuna Combined: -0.1447098753795164
- Selected AO Optuna RMSE: 0.4142875382248137
- Selected AO Optuna Val_RMSE: 0.4992234977276637

Final evaluation:
- RMSE: 1.6894085640784238
- AUC: 0.762107813034694
- AUC_adj: 0.762107813034694
- Combined (RMSE - AUC): 0.9273007510437298

Artifacts:
- Model: `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_bestpair.pt`
- Mask: `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_bestpair_mask.pkl`
- Scaler: `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_bestpair_scaler.pkl`
- Pairwise summary: `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_bestpair_pairwise_major_results.csv`
- Reconstructed stats JSON: `/data/hd4tb/OCDocker/OCDocker/OCScore_models/OCScore_AE_NN_6to10_bestpair_run_stats_reconstructed.json`

Checksums (SHA256):
- Model: `341d0f34a14f83bd80db94f5e96dec019bf29f382d395df9fc92cc67d0680dc4`
- Mask: `7f0ffb858236a22ae782ed39629b100eed16ffff5849e5c086893c7b7f3950f0`
- Scaler: `7ae8327821737a47b69f374a531a233be08ab0e12c9c6f9e0b5b8c1e1d85ebc3`
- Pairwise summary: `55702980d5c3b56b009958d97989f115df05ab7f1284371ac054253a23c5642a`
- Reconstructed stats JSON: `1ee4d704c53674d3d78d5ccd29fa26515898d34e4e4860a69066554483880479`

Artifact timestamps:
- Model: `2026-03-04 17:21:02.757857808 -0300`
- Mask: `2026-03-04 17:21:02.758259795 -0300`
- Scaler: `2026-03-04 17:21:02.758606464 -0300`
- Pairwise summary: `2026-03-04 17:21:02.758839140 -0300`
- Reconstructed stats JSON: `2026-03-04 20:36:53.643477497 -0300`

---

## Summary Table
| Entry | Model Name | Selection Criterion | Selected NN Trial | Selected AO Trial | Final RMSE | Final AUC | Final Combined | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | `OCScore_AE_NN_7` | `combined` (single study) | 841 | 1606 | 1.6925 | 0.7542 | 0.9384 | Same as Thesis |
| 002 | `OCScore_AE_NN_6to10_best_rmse` | `rmse` | 737 | 1606 | 1.6774 | 0.7527 | 0.9246 | Select 5 networks, one for each major run then pick the best based on RMSE |
| 003 | `OCScore_AE_NN_6to10_best_auc` | `auc` | 357 | 1940 | 1.8943 | 0.6722 | 1.2221 | Select 5 networks, one for each major run then pick the best based on AUC |
| 004 | `OCScore_AE_NN_6to10_bestpair` | `combined` | 721 | 1837 | 1.6894 | 0.7621 | 0.9273 | Select 5 networks, one for each major run then pick the best based on RMSE - AUC |
