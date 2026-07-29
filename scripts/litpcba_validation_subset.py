#!/usr/bin/env python3

# Description
###############################################################################
'''
Builds a leakage-checked, compute-tractable LIT-PCBA validation subset for
external blind evaluation of OCDocker/OCScore.

LIT-PCBA ("full data" release) ships, per target, several pre-aligned crystal
structures plus a single ``actives.smi``/``inactives.smi`` compound list. Two
problems make the raw release unusable as-is for validating a model trained
on PDBbind/DUDEz:

1. Several LIT-PCBA structures are the same deposited PDB entry (or a
   near-identical re-deposition of the same complex) as an entry already
   present in PDBbind/DUDEz. Docking against those would partly validate the
   model against its own training receptors.
2. The inactive sets are enormous (up to ~362k compounds for one target,
   ~2.8M across all targets), which is intractable to dock with four engines
   (Vina/Smina/PLANTS/Gnina) end to end.

This script resolves both, deterministically:

Step 1 -- Receptor sequence-identity dedup
    Extract a CA-trace one-letter sequence for every LIT-PCBA candidate
    receptor, every PDBbind receptor, and every DUDEz receptor. Search LIT-PCBA
    candidates against the union of PDBbind+DUDEz sequences with mmseqs2. A
    candidate is excluded as a near-duplicate if its best hit has sequence
    identity >= ``--dup-min-identity`` (default 0.99) AND coverage
    (alignment length / query length) >= ``--dup-min-coverage`` (default
    0.95). A target with zero surviving candidates is dropped entirely --
    excluding a subset of its structures is not possible when every available
    structure is a duplicate.

Step 2 -- Representative receptor selection
    Among the surviving (non-duplicate) candidates for a target, the one with
    the best (lowest) crystallographic resolution is selected as that
    target's receptor. Resolutions are read from ``--resolution-cache`` (a
    TSV shipped alongside this script, covering all 129 structures in the
    "full data" release) and only queried from the RCSB Data API
    (https://data.rcsb.org) for codes missing from the cache.

Step 3 -- Inactive subsampling
    For a target with ``n_actives`` (all actives are always kept), the number
    of inactives sampled is:

        n_sampled = min(max(ratio * n_actives, floor), cap, n_available)

    with defaults ``ratio=100``, ``floor=2000``, ``cap=20000``. The floor
    keeps low-active targets above the pool size generally considered needed
    for statistically stable low-percentile enrichment metrics (EF1%,
    BEDROC); the cap prevents the largest targets (up to 362k inactives) from
    dominating total docking time. Sampling is done with
    ``random.Random(seed)`` (default seed 42) over the inactives in their
    original file order, so the same input + same parameters always produce
    the same subset.

Usage
-----

    python scripts/litpcba_validation_subset.py \\
        --litpcba-dir /path/to/extracted/full_data \\
        --pdbbind-dir /path/to/ocdb2/PDBbind \\
        --dudez-dir /path/to/ocdb2/DUDEz \\
        --output-dir /path/to/litpcba_validation_subset

Running with no tuning flags reproduces the exact subset documented in
``docs/litpcba_validation_subset.md``. Every threshold above is a CLI flag if
a different policy is needed.

Requires the ``mmseqs2`` binary on PATH (``mamba install -c bioconda -c
conda-forge mmseqs2``).
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RESOLUTION_CACHE = os.path.join(SCRIPT_DIR, "litpcba_resolution_cache.tsv")

# Inactive subsampling rule: n_sampled = min(max(RATIO * n_actives, FLOOR), CAP, n_available)
DEFAULT_RATIO = 100
DEFAULT_FLOOR = 2000
DEFAULT_CAP = 20000
DEFAULT_SEED = 42

# A candidate receptor is a near-duplicate of an existing PDBbind/DUDEz
# receptor if its best hit reaches both of these thresholds.
DEFAULT_DUP_MIN_IDENTITY = 0.99
DEFAULT_DUP_MIN_COVERAGE = 0.95

# mmseqs2 prefilter thresholds (loose on purpose -- only used to bound the
# alignment search space, never to make the dedup decision itself). These
# must stay below the dup-* thresholds above or true near-duplicates could be
# filtered out before mmseqs2 even reports a percent identity for them.
DEFAULT_PREFILTER_MIN_SEQ_ID = 0.5
DEFAULT_PREFILTER_MIN_COVERAGE = 0.7

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{code}"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
}

_MOL2_SUBST_RE = re.compile(r"^([A-Za-z]{3})(\d+)")

# Classes
###############################################################################

@dataclass
class Candidate:
    '''A single LIT-PCBA crystal structure candidate for a target.

    Attributes
    ----------
    target : str
        The LIT-PCBA target name.
    code : str
        The PDB code of this candidate structure.
    best_identity : float, optional
        Sequence identity of the best PDBbind/DUDEz hit, by default 0.0.
    best_coverage : float, optional
        Alignment coverage (of this candidate) of the best hit, by default 0.0.
    best_match : str, optional
        Name of the PDBbind/DUDEz entry that is the best hit, by default "-".
    excluded : bool, optional
        True if this candidate is a near-duplicate of an existing PDBbind/DUDEz
        receptor, by default False.
    resolution : float | None, optional
        Crystallographic resolution in Angstrom, by default None.
    '''
    target: str
    code: str
    best_identity: float = 0.0
    best_coverage: float = 0.0
    best_match: str = "-"
    excluded: bool = False
    resolution: Optional[float] = None


@dataclass
class TargetResult:
    '''The outcome of processing one LIT-PCBA target.

    Attributes
    ----------
    target : str
        The LIT-PCBA target name.
    dropped : bool, optional
        True if every candidate for this target was excluded as a
        near-duplicate, by default False.
    drop_reason : str, optional
        Human-readable reason the target was dropped, by default "".
    receptor_code : str | None, optional
        PDB code of the selected representative receptor, by default None.
    receptor_resolution : float | None, optional
        Crystallographic resolution of the selected receptor, by default None.
    n_actives : int, optional
        Number of active compounds for this target, by default 0.
    n_inactives_available : int, optional
        Number of inactive compounds available before subsampling, by default 0.
    n_inactives_sampled : int, optional
        Number of inactive compounds kept after subsampling, by default 0.
    candidates : List[Candidate], optional
        Every candidate structure considered for this target, by default [].
    '''
    target: str
    dropped: bool = False
    drop_reason: str = ""
    receptor_code: Optional[str] = None
    receptor_resolution: Optional[float] = None
    n_actives: int = 0
    n_inactives_available: int = 0
    n_inactives_sampled: int = 0
    candidates: List[Candidate] = field(default_factory=list)


# Functions
###############################################################################
## Private ##

def _seq_from_pdb(path: str) -> str:
    '''Extract a CA-trace one-letter sequence from a PDB file.

    Parameters
    ----------
    path : str
        Path to the PDB file.

    Returns
    -------
    str
        The one-letter amino-acid sequence traced through CA atoms.
    '''

    seq = []
    last_key = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3TO1:
                continue
            key = (line[21:22], line[22:26].strip(), line[26:27])
            if key == last_key:
                continue
            last_key = key
            seq.append(AA3TO1[resname])
    return "".join(seq)


def _seq_from_mol2(path: str) -> str:
    '''Extract a CA-trace one-letter sequence from a Tripos mol2 file.

    Parameters
    ----------
    path : str
        Path to the mol2 file.

    Returns
    -------
    str
        The one-letter amino-acid sequence traced through CA atoms.
    '''

    seq = []
    last_key = None
    in_atom = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("@<TRIPOS>"):
                in_atom = line.strip() == "@<TRIPOS>ATOM"
                continue
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 8 or parts[1] != "CA":
                continue
            m = _MOL2_SUBST_RE.match(parts[7])
            if not m:
                continue
            resname, resseq = m.group(1).upper(), m.group(2)
            if resname not in AA3TO1:
                continue
            if resseq == last_key:
                continue
            last_key = resseq
            seq.append(AA3TO1[resname])
    return "".join(seq)


def _write_fasta(records: List[Tuple[str, str]], out_path: str) -> None:
    '''Write (name, sequence) records as FASTA, skipping sequences shorter than 20 residues.

    Parameters
    ----------
    records : List[Tuple[str, str]]
        (name, sequence) pairs to write.
    out_path : str
        Path of the FASTA file to write.
    '''

    with open(out_path, "w") as f:
        for name, seq in records:
            if len(seq) < 20:
                continue
            f.write(f">{name}\n{seq}\n")


def _run(cmd: List[str]) -> None:
    '''Run a subprocess command, raising RuntimeError with its stderr on failure.

    Parameters
    ----------
    cmd : List[str]
        The command and its arguments, as passed to ``subprocess.run``.

    Raises
    ------
    RuntimeError
        If the command exits with a non-zero return code.
    '''

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")


## Public ##

def check_mmseqs_available(mmseqs_bin: str) -> None:
    '''Raise a clear error if the mmseqs2 binary cannot be found.

    Parameters
    ----------
    mmseqs_bin : str
        Name or path of the mmseqs2 binary to look up on PATH.

    Raises
    ------
    SystemExit
        If ``mmseqs_bin`` cannot be resolved.
    '''

    if shutil.which(mmseqs_bin) is None:
        raise SystemExit(
            f"mmseqs2 binary '{mmseqs_bin}' not found on PATH. Install it with "
            "'mamba install -c bioconda -c conda-forge mmseqs2' (or point "
            "--mmseqs-bin at an existing install)."
        )


def collect_litpcba_candidates(litpcba_dir: str) -> List[Candidate]:
    '''Enumerate every ``<target>/<code>_protein.mol2`` candidate receptor.

    Parameters
    ----------
    litpcba_dir : str
        Path to the extracted LIT-PCBA "full data" release.

    Returns
    -------
    List[Candidate]
        One Candidate per discovered ``<code>_protein.mol2`` file.
    '''

    candidates = []
    for target_dir in sorted(glob(os.path.join(litpcba_dir, "*"))):
        if not os.path.isdir(target_dir):
            continue
        target = os.path.basename(target_dir)
        for prot in sorted(glob(os.path.join(target_dir, "*_protein.mol2"))):
            code = os.path.basename(prot).replace("_protein.mol2", "")
            candidates.append(Candidate(target=target, code=code))
    return candidates


def build_query_fasta(litpcba_dir: str, candidates: List[Candidate], out_path: str) -> None:
    '''Write a FASTA of every LIT-PCBA candidate receptor sequence, keyed as ``<target>__<code>``.

    Parameters
    ----------
    litpcba_dir : str
        Path to the extracted LIT-PCBA "full data" release.
    candidates : List[Candidate]
        Candidates to extract sequences for.
    out_path : str
        Path of the FASTA file to write.
    '''

    records = []
    for c in candidates:
        prot = os.path.join(litpcba_dir, c.target, f"{c.code}_protein.mol2")
        records.append((f"{c.target}__{c.code}", _seq_from_mol2(prot)))
    _write_fasta(records, out_path)


def build_pdbbind_fasta(pdbbind_dir: str, out_path: str) -> int:
    '''Write a FASTA of every local PDBbind receptor sequence and return how many were written.

    Parameters
    ----------
    pdbbind_dir : str
        Path to the local PDBbind archive (one subdir per PDB code, each
        containing ``receptor.pdb``).
    out_path : str
        Path of the FASTA file to write.

    Returns
    -------
    int
        Number of receptor sequences written.
    '''

    records = []
    for entry_dir in sorted(glob(os.path.join(pdbbind_dir, "*"))):
        if not os.path.isdir(entry_dir):
            continue
        rec = os.path.join(entry_dir, "receptor.pdb")
        if not os.path.isfile(rec):
            continue
        records.append((os.path.basename(entry_dir), _seq_from_pdb(rec)))
    _write_fasta(records, out_path)
    return len(records)


def build_dudez_fasta(dudez_dir: str, out_path: str) -> int:
    '''Write a FASTA of every local DUDEz receptor sequence and return how many were written.

    Parameters
    ----------
    dudez_dir : str
        Path to the local DUDEz archive (one subdir per target, each
        containing ``receptor.pdb``).
    out_path : str
        Path of the FASTA file to write.

    Returns
    -------
    int
        Number of receptor sequences written.
    '''

    records = []
    for target_dir in sorted(glob(os.path.join(dudez_dir, "*"))):
        if not os.path.isdir(target_dir):
            continue
        rec = os.path.join(target_dir, "receptor.pdb")
        if not os.path.isfile(rec):
            continue
        records.append((os.path.basename(target_dir), _seq_from_pdb(rec)))
    _write_fasta(records, out_path)
    return len(records)


def run_mmseqs_search(
    query_fasta: str,
    target_fasta: str,
    work_dir: str,
    mmseqs_bin: str,
    prefilter_min_seq_id: float,
    prefilter_min_coverage: float,
) -> str:
    '''Run mmseqs2 search and return the path to the tabular hits file.

    Parameters
    ----------
    query_fasta : str
        FASTA of the LIT-PCBA candidate receptor sequences (the query DB).
    target_fasta : str
        FASTA of the combined PDBbind+DUDEz receptor sequences (the target DB).
    work_dir : str
        Directory to create the mmseqs2 databases and hits file in.
    mmseqs_bin : str
        Path to the mmseqs2 binary.
    prefilter_min_seq_id : float
        ``mmseqs search --min-seq-id`` prefilter threshold (loose; only bounds
        the alignment search space).
    prefilter_min_coverage : float
        ``mmseqs search -c`` prefilter threshold (loose; only bounds the
        alignment search space).

    Returns
    -------
    str
        Path to the tabular ``hits.tsv`` file written by ``mmseqs convertalis``.
    '''

    os.makedirs(work_dir, exist_ok=True)
    query_db = os.path.join(work_dir, "queryDB")
    target_db = os.path.join(work_dir, "targetDB")
    result_db = os.path.join(work_dir, "resultDB")
    tmp_dir = os.path.join(work_dir, "tmp")
    hits_tsv = os.path.join(work_dir, "hits.tsv")

    _run([mmseqs_bin, "createdb", query_fasta, query_db])
    _run([mmseqs_bin, "createdb", target_fasta, target_db])
    _run([
        mmseqs_bin, "search", query_db, target_db, result_db, tmp_dir,
        "--min-seq-id", str(prefilter_min_seq_id),
        "-c", str(prefilter_min_coverage),
        "--cov-mode", "0",
        "-a",
    ])
    _run([
        mmseqs_bin, "convertalis", query_db, target_db, result_db, hits_tsv,
        "--format-output", "query,target,fident,alnlen,qlen,tlen,evalue,bits",
    ])
    return hits_tsv


def apply_dedup(
    candidates: List[Candidate],
    hits_tsv: str,
    dup_min_identity: float,
    dup_min_coverage: float,
) -> None:
    '''Fill in best_identity/best_coverage/best_match/excluded on each candidate, in place.

    Parameters
    ----------
    candidates : List[Candidate]
        Candidates to update in place.
    hits_tsv : str
        Tabular mmseqs2 hits file, as produced by ``run_mmseqs_search``.
    dup_min_identity : float
        Minimum sequence identity, together with ``dup_min_coverage``, to flag
        a candidate as a near-duplicate.
    dup_min_coverage : float
        Minimum alignment coverage (of the query), together with
        ``dup_min_identity``, to flag a candidate as a near-duplicate.
    '''

    by_key = {(c.target, c.code): c for c in candidates}
    hits_by_key: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = {}

    with open(hits_tsv) as f:
        for line in f:
            query, target_hit, fident, alnlen, qlen, tlen, evalue, bits = line.strip().split("\t")
            target, code = query.split("__", 1)
            fident = float(fident)
            coverage = int(alnlen) / int(qlen)
            key = (target, code)
            hits_by_key.setdefault(key, []).append((fident, coverage, target_hit))

    for key, c in by_key.items():
        hits = hits_by_key.get(key)
        if not hits:
            continue
        # A candidate is a near-duplicate if ANY hit clears both thresholds --
        # picking only the single highest-identity hit could miss a
        # lower-identity, higher-coverage hit that is the true duplicate,
        # while a spurious high-identity/low-coverage hit against an
        # unrelated receptor masks it.
        duplicating = [h for h in hits if h[0] >= dup_min_identity and h[1] >= dup_min_coverage]
        if duplicating:
            c.best_identity, c.best_coverage, c.best_match = max(duplicating, key=lambda h: h[0])
            c.excluded = True
        else:
            c.best_identity, c.best_coverage, c.best_match = max(hits, key=lambda h: h[0])
            c.excluded = False


def load_resolution_cache(path: str) -> Dict[Tuple[str, str], Optional[float]]:
    '''Load a resolution cache TSV into a ``(target, code) -> resolution`` dict (empty dict if the file doesn't exist).

    Parameters
    ----------
    path : str
        Path to the resolution cache TSV.

    Returns
    -------
    Dict[Tuple[str, str], float | None]
        Mapping from ``(target, code)`` to resolution (None if unknown).
    '''

    cache: Dict[Tuple[str, str], Optional[float]] = {}
    if not os.path.isfile(path):
        return cache
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            res = row["resolution"]
            cache[(row["target"], row["code"])] = float(res) if res and res != "None" else None
    return cache


def fetch_resolution_rcsb(code: str) -> Optional[float]:
    '''Fetch a PDB entry's crystallographic resolution from the RCSB Data API, or None on any failure.

    Parameters
    ----------
    code : str
        The 4-character PDB code to query.

    Returns
    -------
    float | None
        The resolution in Angstrom, or None if unavailable or the request failed.
    '''

    try:
        with urllib.request.urlopen(RCSB_ENTRY_URL.format(code=code), timeout=10) as r:
            data = json.load(r)
        res = data.get("rcsb_entry_info", {}).get("resolution_combined")
        return res[0] if res else None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return None


def resolve_resolutions(
    candidates: List[Candidate],
    cache_path: str,
    refresh: bool,
) -> None:
    '''Fill in resolution on each non-excluded candidate, in place.

    Parameters
    ----------
    candidates : List[Candidate]
        Candidates to update in place. Excluded candidates are skipped.
    cache_path : str
        Path to the resolution cache TSV, refreshed in place for any code
        that was queried.
    refresh : bool
        If True, re-fetch every non-excluded candidate's resolution from the
        RCSB Data API instead of trusting the existing cache entry.
    '''

    # Always start from the full existing cache, even under --refresh: a run
    # over a partial/test --litpcba-dir must not shrink the write-back to
    # only the candidates it happens to see. --refresh instead forces a
    # re-fetch for this run's candidates specifically, via the `refresh`
    # check below.
    cache = load_resolution_cache(cache_path)
    updated = False

    for c in candidates:
        if c.excluded:
            continue
        key = (c.target, c.code)
        if not refresh and key in cache and cache[key] is not None:
            c.resolution = cache[key]
            continue
        res = fetch_resolution_rcsb(c.code)
        c.resolution = res
        cache[key] = res
        updated = True

    if updated:
        with open(cache_path, "w") as f:
            f.write("target\tcode\tresolution\n")
            for (target, code), res in sorted(cache.items()):
                f.write(f"{target}\t{code}\t{res if res is not None else ''}\n")


def select_representative_receptors(candidates: List[Candidate]) -> Dict[str, TargetResult]:
    '''Group candidates by target, dropping targets with no usable candidate
    and picking the best-resolution structure among the rest.

    Parameters
    ----------
    candidates : List[Candidate]
        All candidates, across all targets, with ``excluded``/``resolution``
        already filled in.

    Returns
    -------
    Dict[str, TargetResult]
        One TargetResult per target, keyed by target name.
    '''

    by_target: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_target.setdefault(c.target, []).append(c)

    results: Dict[str, TargetResult] = {}
    for target, cands in sorted(by_target.items()):
        usable = [c for c in cands if not c.excluded]
        tr = TargetResult(target=target, candidates=cands)
        if not usable:
            tr.dropped = True
            tr.drop_reason = (
                f"all {len(cands)} candidate structure(s) are near-duplicates "
                "of an existing PDBbind/DUDEz receptor"
            )
        else:
            with_res = [c for c in usable if c.resolution is not None]
            pool = with_res if with_res else usable
            best = min(pool, key=lambda c: c.resolution if c.resolution is not None else float("inf"))
            tr.receptor_code = best.code
            tr.receptor_resolution = best.resolution
        results[target] = tr
    return results


def sample_inactives(
    inactives_path: str,
    n_actives: int,
    ratio: int,
    floor: int,
    cap: int,
    seed: int,
) -> Tuple[int, List[str]]:
    '''Apply n_sampled = min(max(ratio * n_actives, floor), cap, n_available)
    and return (n_available, sampled_lines) preserving original file order.

    Parameters
    ----------
    inactives_path : str
        Path to the target's ``inactives.smi`` file.
    n_actives : int
        Number of actives for this target (all actives are always kept).
    ratio : int
        Inactives sampled per active, before floor/cap.
    floor : int
        Minimum inactives sampled per target.
    cap : int
        Maximum inactives sampled per target.
    seed : int
        Random seed for ``random.Random(seed).sample(...)``.

    Returns
    -------
    Tuple[int, List[str]]
        The number of inactives available, and the sampled lines (in their
        original file order).
    '''

    with open(inactives_path, "r", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    n_available = len(lines)
    n_target = max(ratio * n_actives, floor)
    n_sampled = min(n_target, cap, n_available)

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n_available), n_sampled))
    return n_available, [lines[i] for i in indices]


def materialize_target(
    litpcba_dir: str,
    output_dir: str,
    target_result: TargetResult,
    ratio: int,
    floor: int,
    cap: int,
    seed: int,
) -> None:
    '''Write the actives/sampled-inactives/receptor files for one kept target.

    Parameters
    ----------
    litpcba_dir : str
        Path to the extracted LIT-PCBA "full data" release.
    output_dir : str
        Root output directory; a ``<target>`` subdirectory is created under it.
    target_result : TargetResult
        The kept target to materialize; updated in place with actives/inactives counts.
    ratio : int
        Inactives sampled per active, before floor/cap.
    floor : int
        Minimum inactives sampled per target.
    cap : int
        Maximum inactives sampled per target.
    seed : int
        Random seed for inactive subsampling.
    '''

    target = target_result.target
    src_dir = os.path.join(litpcba_dir, target)
    dst_dir = os.path.join(output_dir, target)
    os.makedirs(dst_dir, exist_ok=True)

    shutil.copyfile(
        os.path.join(src_dir, f"{target_result.receptor_code}_protein.mol2"),
        os.path.join(dst_dir, "receptor_protein.mol2"),
    )
    shutil.copyfile(
        os.path.join(src_dir, f"{target_result.receptor_code}_ligand.mol2"),
        os.path.join(dst_dir, "receptor_ligand.mol2"),
    )

    actives_path = os.path.join(src_dir, "actives.smi")
    shutil.copyfile(actives_path, os.path.join(dst_dir, "actives.smi"))
    with open(actives_path, "r", errors="replace") as f:
        n_actives = sum(1 for line in f if line.strip())
    target_result.n_actives = n_actives

    n_available, sampled = sample_inactives(
        os.path.join(src_dir, "inactives.smi"), n_actives, ratio, floor, cap, seed,
    )
    target_result.n_inactives_available = n_available
    target_result.n_inactives_sampled = len(sampled)
    with open(os.path.join(dst_dir, "inactives_sampled.smi"), "w") as f:
        f.write("\n".join(sampled) + "\n")


def write_manifest(results: Dict[str, TargetResult], output_dir: str, params: dict) -> None:
    '''Write manifest.csv/.json, dropped_targets.tsv, and excluded_structures.tsv into output_dir.

    Parameters
    ----------
    results : Dict[str, TargetResult]
        Every target's outcome, keyed by target name.
    output_dir : str
        Directory to write the manifest and provenance files into.
    params : dict
        The run's CLI parameters, recorded in manifest.json for provenance.
    '''

    kept = [r for r in results.values() if not r.dropped]
    dropped = [r for r in results.values() if r.dropped]

    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "receptor_code", "receptor_resolution", "n_actives",
                    "n_inactives_available", "n_inactives_sampled", "realized_ratio"])
        for r in sorted(kept, key=lambda r: r.target):
            realized_ratio = r.n_inactives_sampled / r.n_actives if r.n_actives else None
            w.writerow([r.target, r.receptor_code, r.receptor_resolution, r.n_actives,
                        r.n_inactives_available, r.n_inactives_sampled,
                        f"{realized_ratio:.1f}" if realized_ratio is not None else ""])

    with open(os.path.join(output_dir, "dropped_targets.tsv"), "w") as f:
        f.write("target\treason\n")
        for r in sorted(dropped, key=lambda r: r.target):
            f.write(f"{r.target}\t{r.drop_reason}\n")

    with open(os.path.join(output_dir, "excluded_structures.tsv"), "w") as f:
        f.write("target\tcode\tbest_match\tidentity\tcoverage\n")
        for r in results.values():
            for c in sorted(r.candidates, key=lambda c: c.code):
                if c.excluded:
                    f.write(f"{c.target}\t{c.code}\t{c.best_match}\t{c.best_identity:.3f}\t{c.best_coverage:.3f}\n")

    manifest = {
        "parameters": params,
        "targets_kept": [
            {
                "target": r.target,
                "receptor_code": r.receptor_code,
                "receptor_resolution": r.receptor_resolution,
                "n_actives": r.n_actives,
                "n_inactives_available": r.n_inactives_available,
                "n_inactives_sampled": r.n_inactives_sampled,
                "realized_ratio": r.n_inactives_sampled / r.n_actives if r.n_actives else None,
            }
            for r in sorted(kept, key=lambda r: r.target)
        ],
        "targets_dropped": [
            {"target": r.target, "reason": r.drop_reason}
            for r in sorted(dropped, key=lambda r: r.target)
        ],
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    '''Build the CLI argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    '''

    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-checked, compute-tractable LIT-PCBA validation "
            "subset (receptor dedup against PDBbind/DUDEz + principled "
            "inactive subsampling)."
        ),
    )
    parser.add_argument("--litpcba-dir", required=True, help="Path to the extracted LIT-PCBA 'full data' release.")
    parser.add_argument("--pdbbind-dir", required=True, help="Path to the local PDBbind archive (one subdir per PDB code, each containing receptor.pdb).")
    parser.add_argument("--dudez-dir", required=True, help="Path to the local DUDEz archive (one subdir per target, each containing receptor.pdb).")
    parser.add_argument("--output-dir", required=True, help="Where to write the manifest and the materialized per-target subset.")

    parser.add_argument("--ratio", type=int, default=DEFAULT_RATIO, help=f"Inactives sampled per active, before floor/cap. Default {DEFAULT_RATIO}.")
    parser.add_argument("--floor", type=int, default=DEFAULT_FLOOR, help=f"Minimum inactives sampled per target. Default {DEFAULT_FLOOR}.")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help=f"Maximum inactives sampled per target. Default {DEFAULT_CAP}.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for inactive sampling. Default {DEFAULT_SEED}.")

    parser.add_argument("--dup-min-identity", type=float, default=DEFAULT_DUP_MIN_IDENTITY, help=f"Minimum sequence identity to flag a candidate as a near-duplicate. Default {DEFAULT_DUP_MIN_IDENTITY}.")
    parser.add_argument("--dup-min-coverage", type=float, default=DEFAULT_DUP_MIN_COVERAGE, help=f"Minimum alignment coverage (of the query) to flag a candidate as a near-duplicate. Default {DEFAULT_DUP_MIN_COVERAGE}.")
    parser.add_argument("--prefilter-min-seq-id", type=float, default=DEFAULT_PREFILTER_MIN_SEQ_ID, help="mmseqs2 search prefilter --min-seq-id (loose; must stay below --dup-min-identity).")
    parser.add_argument("--prefilter-min-coverage", type=float, default=DEFAULT_PREFILTER_MIN_COVERAGE, help="mmseqs2 search prefilter -c (loose; must stay below --dup-min-coverage).")

    parser.add_argument("--resolution-cache", default=DEFAULT_RESOLUTION_CACHE, help="TSV cache of PDB resolutions, refreshed in place for any code missing from it.")
    parser.add_argument("--refresh-resolution-cache", action="store_true", help="Ignore the existing cache and re-query the RCSB Data API for every candidate.")

    parser.add_argument("--mmseqs-bin", default="mmseqs", help="Path to the mmseqs2 binary. Default 'mmseqs' (resolved via PATH).")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep the intermediate mmseqs2 databases/FASTA files under <output-dir>/_work.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    '''Run the full LIT-PCBA validation subset pipeline (dedup, resolution, subsampling, manifest).

    Parameters
    ----------
    argv : List[str] | None, optional
        Command-line arguments to parse, by default None (uses ``sys.argv``).

    Returns
    -------
    int
        Process exit code (0 on success).
    '''

    args = build_arg_parser().parse_args(argv)

    if args.prefilter_min_seq_id >= args.dup_min_identity or args.prefilter_min_coverage >= args.dup_min_coverage:
        raise SystemExit(
            "--prefilter-min-seq-id/--prefilter-min-coverage must be strictly "
            "below --dup-min-identity/--dup-min-coverage, otherwise true "
            "near-duplicates could be discarded before mmseqs2 reports a "
            "percent identity for them."
        )

    check_mmseqs_available(args.mmseqs_bin)
    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    print("[1/5] Enumerating LIT-PCBA candidate receptors...")
    candidates = collect_litpcba_candidates(args.litpcba_dir)
    print(f"      {len(candidates)} candidate structures across "
          f"{len({c.target for c in candidates})} targets.")

    print("[2/5] Extracting receptor sequences and running mmseqs2 dedup search...")
    query_fasta = os.path.join(work_dir, "litpcba.fasta")
    existing_fasta = os.path.join(work_dir, "existing.fasta")
    build_query_fasta(args.litpcba_dir, candidates, query_fasta)
    n_pdbbind = build_pdbbind_fasta(args.pdbbind_dir, os.path.join(work_dir, "pdbbind.fasta"))
    n_dudez = build_dudez_fasta(args.dudez_dir, os.path.join(work_dir, "dudez.fasta"))
    with open(existing_fasta, "w") as out:
        for name in ("pdbbind.fasta", "dudez.fasta"):
            with open(os.path.join(work_dir, name)) as f:
                out.write(f.read())
    print(f"      {n_pdbbind} PDBbind + {n_dudez} DUDEz existing receptor sequences.")

    hits_tsv = run_mmseqs_search(
        query_fasta, existing_fasta, os.path.join(work_dir, "mmseqs"), args.mmseqs_bin,
        args.prefilter_min_seq_id, args.prefilter_min_coverage,
    )
    apply_dedup(candidates, hits_tsv, args.dup_min_identity, args.dup_min_coverage)
    n_excluded = sum(1 for c in candidates if c.excluded)
    print(f"      {n_excluded}/{len(candidates)} candidates excluded as near-duplicates.")

    print("[3/5] Resolving crystallographic resolutions and picking representative receptors...")
    resolve_resolutions(candidates, args.resolution_cache, args.refresh_resolution_cache)
    results = select_representative_receptors(candidates)
    dropped = [r for r in results.values() if r.dropped]
    for r in dropped:
        print(f"      DROPPED {r.target}: {r.drop_reason}")

    print("[4/5] Subsampling inactives and materializing the per-target subset...")
    for r in results.values():
        if r.dropped:
            continue
        materialize_target(args.litpcba_dir, args.output_dir, r, args.ratio, args.floor, args.cap, args.seed)
        print(f"      {r.target}: receptor={r.receptor_code} ({r.receptor_resolution} A), "
              f"actives={r.n_actives}, inactives {r.n_inactives_sampled}/{r.n_inactives_available}")

    print("[5/5] Writing manifest...")
    params = {
        "ratio": args.ratio, "floor": args.floor, "cap": args.cap, "seed": args.seed,
        "dup_min_identity": args.dup_min_identity, "dup_min_coverage": args.dup_min_coverage,
        "prefilter_min_seq_id": args.prefilter_min_seq_id, "prefilter_min_coverage": args.prefilter_min_coverage,
    }
    write_manifest(results, args.output_dir, params)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    n_kept = len(results) - len(dropped)
    print(f"\nDone. {n_kept}/{len(results)} targets kept. See {args.output_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
