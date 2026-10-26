# OCScore Gen3B model bundle

Frozen models for scoring a docked library. Nothing here trains, fits or selects: the weights,
the feature list and the preprocessing statistics were all fixed at release time, and scoring a
new table only applies them.

Run them with:

    ocdocker ocscore predict --csv <table>.csv --output-dir <dir>

For a target whose table lacks the context columns, compute them once from the structure first:

    ocdocker ocscore descriptors --receptor <protein>.pdb \
        --reference-ligand <site>.pdb --ligand-resname <RES> --out context.json
    ocdocker ocscore predict --csv <table>.csv --receptor-descriptors context.json \
        --output-dir <dir>

Nothing in this bundle is tied to a particular target. Several targets can share one table and
one descriptor file: give each `ocscore descriptors` run a `--target-id` matching the value in
the table's receptor column and the same `--out`, and the file accumulates them. Scoring then
applies each target's own context to its own rows, and a receptor present in the table but
missing from the file is an error rather than a silently empty block.

## What is shipped

| Directory | Models | What it is for |
| --- | --- | --- |
| `release_m1_1/` | 3 seeds | The release model. Report screening results from this. |
| `ablation/<arm>/` | 8 arms x 3 seeds | Block factorial, for attributing performance to input blocks. |
| `control/shuffled_labels/` | 3 seeds | Trained on labels permuted within target. A diagnostic, never a scorer. |
| `baselines/` | 9 comparators | The floor a result has to clear. Not torch checkpoints. |

The baselines are mean and median consensus of the score block, ridge and elastic-net logistic
regression at the `S`, `S+L` and `S+L+R+P` arms, maximum ECFP4 similarity to the training actives
(skipped unless `--smiles-column` is given), and a random ranking. The linear models were fitted
on the same training rows as the networks and their coefficient vectors span the whole contract,
zero outside their arm. The ECFP4 and random rows are controls in the same sense as the
randomised-label network: ECFP4 asks whether a ranking could have come from recognising familiar
chemistry rather than from the target.

Arm directories name the blocks each arm takes: `S` scores, `L` ligand descriptors, `R` receptor
descriptors, `P` pocket descriptors. An arm reads only the columns of its own blocks, so `S` is
an 11-input network and `S_L` a 310-input one. `S_L_R_P` is the full input and reproduces the
release.

Each model is reported per seed and as a three-seed ensemble, the ensemble being the mean of
within-receptor percentile ranks.

Every family shares one feature contract, `release_m1_1/feature_manifest.json` and
`release_m1_1/preprocessing.npz`: 365 columns in a fixed order, made up of 11 docking scores,
299 ligand descriptors, 29 receptor descriptors and 26 pocket descriptors.

## The control is not optional

The control model never saw a true label. On an evaluation set whose labels reflect binding it
must land at chance, and `ocdocker ocscore predict` fails the run's `control_check` when it does
not. A control above chance means the labels are partly predictable from the inputs without any
activity supervision. That is a statement about the set, not about any model, and while it holds
the ranking of the other models cannot be attributed to the models.

This is not hypothetical. On the 7UL2 collection the control reaches ROC-AUC 0.798 against 0.713
for the release, because its actives and inactives are separable by bulk ligand chemistry before
any docking is consulted.

## Input table

A merged OCDocker results table, one row per docked compound. Required columns:

- the 11 docking scores, in either the contract names (`vina_vina`, `plants_plp`, ...) or the
  pipeline's upper-case convention (`VINA_VINA`, `PLANTS_PLP`, ...), which is mapped automatically
- the 299 ligand descriptors, with or without the `ligand_` prefix
- the 55 receptor and pocket descriptors, as `receptor_*` and `pocket_*` columns, or supplied
  separately with `--receptor-descriptors <json>`. A full pipeline run writes both blocks
  prefixed and needs nothing further. A table assembled outside the pipeline often carries only
  the receptor set, and unprefixed: the receptor and pocket sets share 24 of their 29 and 26
  names, so unprefixed columns are resolved using the names unique to one set
  (`TotalAALength`, `IsoelectricPoint`, `DipoleMoment`, `AvgAALength`, `InstabilityIndex` for the
  receptor; `countHBondDonors`, `countHBondAcceptors` for the pocket) and left alone when both or
  neither appear
- `receptor`, naming the target each row was docked against
- `kind`, naming each compound `active` or `inactive`, when metrics are wanted

A table needs only the blocks of the models it is scored with. By default every requested model
must be scorable, and a table that falls short is refused with the missing columns named per
block, so it is clear whether what is absent is a score, a ligand descriptor or a context block.
With `--skip-unavailable` the models the table can support are scored and the rest are listed in
the report; a table without context columns still gets the `S` and `S_L` arms and the baselines
that read only scores and ligand descriptors. The release model and the control take every
block, so a run that skips them has no control check. Rows that carry every needed column but
leave some of them empty are dropped and counted in the report.

## Two properties of the contract that constrain use

Scores and ligand descriptors are standardised **within receptor**. A whole library is ranked
against one target at a time, and a single compound on its own cannot be scored; a table with one
row per target produces nothing meaningful.

The 55 context features describe the receptor, not the compound. A new target needs them computed
once before its table can be scored, with `ocdocker ocscore descriptors`, which takes the
receptor structure and a ligand defining the site. Receptor and pocket descriptors are not
interchangeable -- the receptor set describes the whole protein and the pocket set only the
residues lining the site, and none of the 24 shared names agree between them.

## Output

`ocscore_predictions.csv` carries one column per model seed and per ensemble.
`ocscore_metrics.csv` is written when the table carries labels, one row per model.
`ocscore_prediction_report.json` records what ran, how many rows were scored and dropped, and the
control verdict.
