# LIT-PCBA external validation subset

This document describes, precisely enough to cite in a methods section, how the
LIT-PCBA "full data" release is filtered and subsampled into the set used for
external blind evaluation of OCDocker/OCScore. The procedure is implemented in
[`scripts/litpcba_validation_subset.py`](../scripts/litpcba_validation_subset.py)
and is deterministic: running it with default arguments against the same
local PDBbind/DUDEz archives reproduces the table below exactly.

## Why this filtering exists

LIT-PCBA ships, per target, several pre-aligned crystal structures plus a
single `actives.smi`/`inactives.smi` compound list. Used as-is, it is not a
valid external validation set for a model trained on PDBbind/DUDEz, for two
reasons:

1. **Receptor leakage.** Several LIT-PCBA structures are the same deposited
   PDB entry (or a near-identical re-deposition of the same complex) as an
   entry already present in PDBbind or DUDEz. Docking against those would
   partly validate the model against its own training receptors rather than
   against unseen ones (see the production protocol's note that
   "protein-family or sequence-cluster splits are stricter than
   receptor-heldout splits", [`docs/ocscore-production-protocol.md`](ocscore-production-protocol.md)).
2. **Intractable inactive counts.** LIT-PCBA's inactive sets range up to
   ~362k compounds for a single target (~2.8M across all 15 targets), which
   is not tractable to dock end to end with four docking engines
   (Vina/Smina/PLANTS/Gnina).

## Procedure

### Step 1 — Receptor sequence-identity dedup

For every LIT-PCBA candidate receptor (`<target>/<pdb_code>_protein.mol2`),
every PDBbind receptor (`<pdb_code>/receptor.pdb`), and every DUDEz receptor
(`<target>/receptor.pdb`), a CA-trace one-letter protein sequence is
extracted directly from atom records (no external structure parser).

LIT-PCBA candidate sequences are searched against the union of
PDBbind + DUDEz sequences with **mmseqs2** (`mmseqs search`, version 18.8cc5c
at the time this subset was built), using a loose prefilter
(`--min-seq-id 0.5 -c 0.7 --cov-mode 0`) that only bounds the alignment
search space and never makes the exclusion decision itself.

A candidate is excluded as a **near-duplicate** if, against its best hit:

```
identity >= 0.99   AND   coverage >= 0.95
```

where `identity` is mmseqs2's `fident` and `coverage = alnlen / qlen` (query
length = the LIT-PCBA candidate). These thresholds intentionally target
literal or near-literal structure duplication (the same complex, possibly
re-refined or re-deposited), not general protein-family membership — see
[Threshold rationale](#threshold-rationale) below.

**If every candidate structure for a target is excluded, the whole target is
dropped.** There is no partial fallback (e.g. relaxing the threshold for that
one target), because that would make the exclusion rule target-dependent and
non-reproducible.

### Step 2 — Representative receptor selection

Among the surviving (non-duplicate) candidates for a kept target, the
structure with the **best (lowest) crystallographic resolution** is selected
as that target's receptor. Resolutions are read from
[`scripts/litpcba_resolution_cache.tsv`](../scripts/litpcba_resolution_cache.tsv)
(shipped with the script, covering all 129 structures in the "full data"
release, fetched from the [RCSB Data API](https://data.rcsb.org) on
2026-07-17) and only re-queried over the network for codes missing from the
cache.

### Step 3 — Inactive subsampling

All actives are always kept. For a target with `n_actives`, the number of
inactives sampled is:

```
n_sampled = min( max(ratio * n_actives, floor), cap, n_available )
```

with defaults `ratio = 100`, `floor = 2000`, `cap = 20000`.

- The **floor** keeps low-active targets above the pool size generally
  considered necessary for statistically stable low-percentile enrichment
  metrics (EF1%, BEDROC) — with too few inactives, "the top 1%" is too small
  a sample to be meaningful.
- The **cap** prevents the largest targets (up to 362k available inactives)
  from dominating total docking time.
- The **ratio** (1 active : 100 inactives) sets the nominal class imbalance
  between the floor and cap regimes, in line with commonly used
  active:decoy ratios in virtual-screening benchmarks.

Sampling draws indices with `random.Random(seed).sample(...)` (default
`seed = 42`) over the inactives in their original file order, then keeps
them in that original order. Given the same input files and the same
parameters, this is fully deterministic.

## Threshold rationale

An earlier, stricter policy considered here was excluding any candidate with
**≥90% sequence identity** to an existing PDBbind/DUDEz receptor, regardless
of coverage. That policy was rejected: LIT-PCBA's targets were selected by
its authors specifically for having rich PDB structural coverage, which is
the same property that guarantees they are also heavily represented in
PDBbind (built from the whole PDB). Applying ≥90% identity uniformly leaves
**zero** usable candidates for 9 of the 15 targets (ADRB2, ALDH1, ESR1_ago,
ESR1_ant, GBA, MAPK1, PKM2, PPARG, VDR) — it filters out the protein, not a
duplicated structure. The ≥99% identity + ≥95% coverage rule instead targets
literal/near-literal structure duplication, which keeps 13 of 15 targets
usable; only ALDH1 and GBA still have no non-duplicate candidate under this
rule and are dropped.

## Reproducing this subset

```bash
python scripts/litpcba_validation_subset.py \
  --litpcba-dir /path/to/extracted/full_data \
  --pdbbind-dir /path/to/ocdb2/PDBbind \
  --dudez-dir /path/to/ocdb2/DUDEz \
  --output-dir /path/to/litpcba_validation_subset
```

Requires the `mmseqs2` binary on `PATH` (`mamba install -c bioconda -c
conda-forge mmseqs2`). Every threshold above is a CLI flag
(`--ratio`, `--floor`, `--cap`, `--seed`, `--dup-min-identity`,
`--dup-min-coverage`) if a different policy is needed; the defaults reproduce
the table below exactly.

## Resulting subset

**13 of 15 targets kept, 2,699 actives, 131,216 sampled inactives
(133,915 compounds total).**

| Target | Receptor (PDB) | Resolution (Å) | Actives | Inactives sampled | Inactives available | Realized ratio |
|---|---|---|---|---|---|---|
| ADRB2 | 3sn6 | 3.20 | 17 | 2,000 | 312,483 | 117.6 |
| ESR1_ago | 2qzo | 1.72 | 13 | 2,000 | 5,583 | 153.8 |
| ESR1_ant | 5ufx | 1.55 | 102 | 4,948 (all) | 4,948 | 48.5 |
| FEN1 | 5fv7 | 2.84 | 369 | 20,000 | 355,402 | 54.2 |
| IDH1 | 4umx | 1.88 | 39 | 3,900 | 362,049 | 100.0 |
| KAT2A | 5mlj | 1.80 | 194 | 19,400 | 348,548 | 100.0 |
| MAPK1 | 4zzn | 1.33 | 308 | 20,000 | 62,629 | 64.9 |
| MTORC1 | 4dri | 1.45 | 97 | 9,700 | 32,972 | 100.0 |
| OPRK1 | 6b73 | 3.10 | 24 | 2,400 | 269,816 | 100.0 |
| PKM2 | 3gr4 | 1.60 | 546 | 20,000 | 245,523 | 36.6 |
| PPARG | 3b1m | 1.60 | 27 | 2,700 | 5,211 | 100.0 |
| TP53 | 3zme | 1.35 | 79 | 4,168 (all) | 4,168 | 52.8 |
| VDR | 3a2j | 2.70 | 884 | 20,000 | 355,388 | 22.6 |

**Realized ratio only equals the nominal 100:1 for the 5 targets bound by
neither the floor nor the cap** (IDH1, KAT2A, MTORC1, OPRK1, PPARG). For the
others, availability (ESR1_ant, TP53) or the 20,000 cap (FEN1, MAPK1, PKM2,
VDR) or the 2,000 floor (ADRB2, ESR1_ago) determines the realized ratio
instead. This is reported per target rather than describing the subset with
a single blanket ratio.

### Statistical power caveat

For rank-based metrics (AUC-ROC, BEDROC, EF<sub>x%</sub>), precision is
bottlenecked by `n_actives`, not by the inactive pool size, once the pool is
past a few hundred compounds. Using the Hanley–McNeil AUC variance formula
at an illustrative AUC of 0.70, taking ADRB2 from its sampled 2,000
inactives up to all 312,483 available moves `SE(AUC)` from 0.0715 to 0.0713
— under a 2% relative change for 156x more docking. The four
lowest-active targets (ESR1_ago: 13, ADRB2: 17, OPRK1: 24, PPARG: 27) have
95% CI half-widths of roughly ±0.11 to ±0.16 on AUC at that same illustrative
AUC, a limit set by how few confirmed actives LIT-PCBA has for those
targets, not by any parameter of this script. Report per-target confidence
intervals (Hanley–McNeil or bootstrap) alongside point estimates, and treat
these four targets as lower-power/exploratory rather than presenting all 13
targets as equally precise.

**Dropped targets:**

| Target | Reason |
|---|---|
| ALDH1 | All 8 candidate structures are near-duplicates of an existing PDBbind/DUDEz receptor. |
| GBA | All 6 candidate structures are near-duplicates of an existing PDBbind/DUDEz receptor. |

The script also writes `excluded_structures.tsv` (every excluded candidate,
its best match, and the identity/coverage that triggered exclusion) and
`manifest.json`/`manifest.csv` (the table above plus the exact parameters
used) into `--output-dir`, for full provenance.

## Limitations

- Sequence-identity dedup at the receptor level does not guarantee the
  *ligands* in LIT-PCBA are novel relative to PDBbind/DUDEz training ligands
  — only the receptor structure is checked here. Ligand/scaffold overlap is a
  separate diagnostic (see `docs/ocscore-production-protocol.md`'s scientific
  caveats).
- Resolution is used as a single, checkable proxy for structure quality when
  selecting a representative receptor per target. It does not account for
  binding-site completeness, alternate conformations, or apo/holo state.
- Only one receptor per target is used (no ensemble docking across a
  target's multiple structures), which is a scope decision made for this
  validation run, not a limitation of the script itself.
