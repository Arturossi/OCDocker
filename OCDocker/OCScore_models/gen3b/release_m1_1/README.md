# OCScore M1.1 release candidate

Internal scientific release freeze, 19 October 2026. Not public: do not publish the repository,
the weights or the manuscript from this directory.

A 365-feature OCScore model. It supersedes the 366-feature M1 release in `../release/`, which
remains unchanged and available for reproducibility. The single difference is that
`smina_ad4_scoring` was excluded from the candidate score block before selection was refitted.

## Why this version exists

smina and gnina report different values for `ad4_scoring` on identical poses. Their unweighted
term values agree to the precision the logs carry; the two apply opposite signs to the AutoDock4
hydrogen-bond weight, -0.0974 against +0.0974, and that single difference accounts for the whole
gap.

A sign discrepancy was identified in the tested smina AD4 scoring configuration. Numerical
experiments and comparison with the AutoDock4 reference support the positive coefficient used by
gnina. The affected smina output was conservatively excluded from OCScore M1.1. The historical
intent of smina's negative coefficient and its status across other versions remain unconfirmed,
and no change has been made to smina or to OCDocker's smina configuration. The evidence is in `../ad4_audit.md`: the term values go to
exactly zero when the ligand is displaced out of the site, so a negative hydrogen-bond term is
attractive and the negative coefficient penalises favourable hydrogen bonding; smina is alone in
pairing a negative weight with it while its other four AD4 weights are positive; its five
magnitudes are exactly the AutoDock 4.0 coefficients, which are all positive; and smina run with
+0.0974 reproduces gnina to 1e-5.

In M1 the two AD4 columns both survived redundancy filtering only because they disagree. M1.1
removes the smina column from the candidate set and refits, so the retained basis is chosen
without that pair.

**This is not a predictive improvement and must not be reported as one.** Against M1 under
identical seeds, folds, budgets and evaluation rules, M1.1 is +0.0023 BEDROC on DUDEz
[-0.0028, +0.0075] and -0.0058 on LIT-PCBA [-0.0195, +0.0049], both spanning zero, and -0.0001
[-0.0071, +0.0072] on LIT-PCBA once the 24-active OPRK1 family is set aside. The complete
12-family LIT-PCBA result is primary; the OPRK1 exclusion is sensitivity analysis. Full
comparison in `../m011_v11_results.md`.

## Contents

| File | What it is |
| --- | --- |
| `feature_manifest.json` | the 365 selected features, their group, order, every removal decision and the reconstruction diagnostics |
| `ocscore_m1_1_seed{4200,4300,4400}.pt` | the three ensemble members, each carrying its architecture config and the manifest hash it was trained against |
| `preprocessing.npz` | fitted context mean and sd, the context column order and the model feature order |
| `model_card.json` | methodology, architecture, hyperparameters, training-data counts, environment, limitations |
| `reference_predictions_*.npy` | ensemble scores produced at training time, for reload verification |
| `SHA256SUMS`, `environment.json` | checksums and the software versions the checkpoints were produced under |

Code one directory up: `build_release_manifest_v11.py`, `train_release_v11.py`,
`ocscore_release_v11.py`, `verify_release_v11.py`, `check_preprocessing_v11.py`.

## The model

Frozen Gen3B screening methodology, unchanged from M1: cross-entropy with a pairwise screening
term at weight 0.1, 50/50 DUDEz and LIT-PCBA domain sampling, 8 sampled negatives per active. No
PDBbind affinity pretraining stage and no affinity latent. Architecture and hyperparameters
unchanged. The deployed predictor is the three-seed ensemble, seeds 4200, 4300 and 4400,
aggregating by equal-weight mean of within-receptor percentile ranks. No family is held out,
because nothing is evaluated at this stage.

## Features

| Group | Retained | Of |
| --- | --- | --- |
| A, engine scores | 11 | 16 |
| B, ligand descriptors | 299 | 409 |
| C, receptor descriptors | 29 | 29 |
| D, pocket descriptors | 26 | 28 |
| total | 365 | 482 |

The count was not forced; 365 is what the frozen procedure returned.

Scores retained: `vina_vina`, `vina_vinardo`, `smina_vina`, `gnina_vinardo`, `gnina_ad4_scoring`,
`gnina_dkoes_fast`, `gnina_dkoes_scoring`, `gnina_dkoes_scoring_old`, `plants_chemplp`,
`plants_plp`, `plants_plp95`.

All four docking engines remain represented, so no program leaves the pipeline. Relative to M1
the set loses `smina_ad4_scoring` by exclusion and swaps `smina_dkoes_scoring_old` for
`gnina_dkoes_scoring_old`, because removing a column changes the pivot order rather than simply
subtracting from the retained set.

Excluded by design and verified absent: the five ODDT machine-learned scores and any gnina CNN
output. The manifest builder asserts on either.

### Selection

Per-group column-pivoted QR; a feature is dropped only when the retained set explains at least
99% of its variance on held-out protein clusters. Refitted on all eligible development data,
16,008 PDBbind rows over 2,055 clusters, with the reconstruction measured on 411 clusters held
out of the fit that produced the selection. A refit, not a union or intersection of the 27 fold
manifests; as a diagnostic only it overlaps them by 356-365 of 365 features.

Two decisions must be reported separately. The feature names and the 99% threshold were chosen
with no reference to any screening label or metric. Choosing the filtered model over the full one
was made after comparing them on the development benchmarks, and is not independent of screening
results.

The receptor block reports no minimum reconstruction R2 because nothing in it was removed.

## Preprocessing

Two populations for two purposes, both matching the development arms. Feature selection is fitted
on the full PDBbind feature table, 16,008 rows. Context standardisation is fitted on the
affinity-eligible subset, 5,765 rows over 169 clusters, with an empty exclusion set because
nothing is held out.

Scores are negated to the literature convention and standardised within receptor; ligand
descriptors are standardised within receptor; context features use the stored mean and sd.
Within-receptor standardisation uses the sample standard deviation, matching the pandas groupby
transform in the training code.

## Verification

`check_preprocessing_v11.py` builds the inputs through the frozen development path and through
the released inference module and compares all 365 columns: maximum absolute difference 0.000e+00
on both DUDEz and LIT-PCBA, and the saved context statistics match the development values exactly.

`verify_release_v11.py` reloads the checkpoints on CPU and compares against the training-time
reference:

| | DUDEz | LIT-PCBA |
| --- | --- | --- |
| max logit difference, CPU against GPU | 6.68e-06 | 1.34e-05 |
| ensemble score drift, in rank positions | 1.0 | 2.0 |
| identical ensemble scores | 99.87% | 99.39% |
| Spearman against reference, pooled and worst receptor | 1.0000 | 1.0000 |
| top-1% set agreement | 100% | 100% |
| top-1% crossings beyond a tie | 0 | 0 |

Both pass. The LIT-PCBA drift sits exactly at the two-rank-position tolerance rather than below
it. That threshold was fixed during the M1 release, before this run, and was not adjusted
afterwards.

A tolerance in score units is not meaningful for a rank-aggregated output: the smallest possible
nonzero change in the ensemble score is one rank position, 1/(3n). Numerical agreement is
therefore checked on the raw logits and ranking agreement on the ranks. The drift tolerance is
two rank positions rather than one because `rankdata` averages exact ties and a tie resolving
differently between devices moves a compound half a position.

## Limitations

The ensemble ranks a library against one target. Scores and ligand descriptors are standardised
within receptor and the ensemble aggregates within-receptor percentile ranks, so the output is a
property of the screened library and a single compound cannot be scored in isolation.

The filtering produces no demonstrated runtime saving. Every docking engine still runs, the
consensus medoid still needs all three, and the descriptor routine still computes the full RDKit
set before the unused columns are discarded.

Neither the filtering nor the AD4 exclusion is established as an accuracy improvement. Both were
adopted for parsimony, determinism and auditability.

The ROC-AUC values printed by the verification script are computed on the training data and are a
reload check, not a performance estimate.

## External validation boundary

The planned case studies were untouched throughout selection, training and checkpoint selection,
and nothing in this procedure reads them. This checkpoint is frozen before their labels or
results are examined. When their identities become available, protein-family and compound overlap
against the development data must be documented before any independence claim.
