#!/usr/bin/env python3

# Description
###############################################################################
'''
Computes pocket descriptors for every receptor of one or more OCDocker
database roots (PDBbind, DUDEz, LIT-PCBA ocdb2 layouts) and caches them as
``pocket_descriptors.json`` next to each ``receptor.pdb``.

Each receptor holds one pocket, defined by its docking centre: every
standard residue with a heavy atom within ``--radius`` angstroms of the centre
(see :class:`OCDocker.Pocket.Pocket`). The centre is read from the receptor's
docking box (``boxes/box0.pdb``, otherwise the first compound box), so the
pocket is the site every compound of that receptor was docked into and no
reference ligand is used.

Existing ``pocket_descriptors.json`` files are kept unless ``--overwrite`` is
given, so an interrupted run can be resumed. A TSV report with one line per
receptor (status and reason) is written to ``--report``.

Usage
-----

    python scripts/compute_pocket_descriptors.py \\
        --database-dir /path/to/ocdb2/PDBbind \\
        --database-dir /path/to/ocdb2/DUDEz \\
        --database-dir /path/to/ocdb2/LITPCBA \\
        --report /path/to/pocket_descriptors_report.tsv \\
        --workers 4
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import csv
import os
import sys

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
# Receptor-level box first; every compound box of a receptor shares its centre
BOX_FILE = os.path.join("boxes", "box0.pdb")

DEFAULT_RADIUS = 12.0
DEFAULT_WORKERS = 4
POCKET_NAME = "pocket"

# Functions
###############################################################################
## Private ##

def _find_box(receptor_dir: str) -> Optional[str]:
    '''Return the docking box file holding the centre of a receptor directory.

    Parameters
    ----------
    receptor_dir : str
        The receptor directory.

    Returns
    -------
    str | None
        The box file path, or None if none exists.
    '''

    receptor_box = os.path.join(receptor_dir, BOX_FILE)
    if os.path.isfile(receptor_box):
        return receptor_box

    # Stop at the first compound box, since a receptor can hold many thousands of compounds
    compounds_dir = os.path.join(receptor_dir, "compounds")
    if not os.path.isdir(compounds_dir):
        return None
    for kind in sorted(os.listdir(compounds_dir)):
        kind_dir = os.path.join(compounds_dir, kind)
        if not os.path.isdir(kind_dir):
            continue
        with os.scandir(kind_dir) as entries:
            for entry in entries:
                box = os.path.join(entry.path, BOX_FILE)
                if os.path.isfile(box):
                    return box
    return None

def _compute_one(receptor_dir: str, radius: float, overwrite: bool) -> Tuple[str, str, str]:
    '''Compute and cache the pocket descriptors of one receptor directory.

    Parameters
    ----------
    receptor_dir : str
        The receptor directory (containing ``receptor.pdb``).
    radius : float
        The pocket radius around the docking centre in angstroms.
    overwrite : bool
        Whether an existing ``pocket_descriptors.json`` is recomputed.

    Returns
    -------
    Tuple[str, str, str]
        The receptor directory, the status (``ok``, ``skipped`` or ``failed``) and a reason.
    '''

    import OCDocker.Pocket as ocpocket

    output_json = os.path.join(receptor_dir, f"{POCKET_NAME}_descriptors.json")
    if os.path.isfile(output_json) and not overwrite:
        return receptor_dir, "skipped", "already computed"

    box = _find_box(receptor_dir)
    if box is None:
        return receptor_dir, "failed", "no docking box"

    try:
        pocket = ocpocket.Pocket(
            os.path.join(receptor_dir, "receptor.pdb"),
            box,
            name=POCKET_NAME,
            radius=radius,
            canonicalize_pdb=False,
        )
    except Exception as e:
        return receptor_dir, "failed", f"{type(e).__name__}: {e}"

    if not pocket.is_valid():
        return receptor_dir, "failed", "invalid pocket (see log)"

    if pocket.to_json(overwrite=overwrite) != 0:
        return receptor_dir, "failed", "could not write json"
    return receptor_dir, "ok", f"{pocket.totalAALength} residues"

## Public ##

def collect_receptor_dirs(database_dirs: List[str]) -> List[str]:
    '''List every receptor directory (one containing ``receptor.pdb``) of the database roots.

    Parameters
    ----------
    database_dirs : List[str]
        The database root directories.

    Returns
    -------
    List[str]
        The receptor directories, sorted.
    '''

    receptor_dirs: List[str] = []
    for database_dir in database_dirs:
        for entry in os.scandir(database_dir):
            if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "receptor.pdb")):
                receptor_dirs.append(entry.path)
    return sorted(receptor_dirs)

def build_arg_parser() -> argparse.ArgumentParser:
    '''Build the CLI argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    '''

    parser = argparse.ArgumentParser(
        description="Compute and cache pocket descriptors (pocket_descriptors.json) for every receptor of OCDocker database roots.",
    )
    parser.add_argument("--database-dir", action="append", required=True, help="Database root (one subdir per receptor, each containing receptor.pdb). Repeat for several databases.")
    parser.add_argument("--report", required=True, help="TSV report written with one line per receptor (directory, status, reason).")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help=f"Pocket radius around the docking centre in angstroms. Default {DEFAULT_RADIUS}.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of worker processes. Default {DEFAULT_WORKERS}.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute receptors that already have pocket_descriptors.json.")
    return parser

def main(argv: Optional[List[str]] = None) -> int:
    '''Compute the pocket descriptors of every receptor and write the report.

    Parameters
    ----------
    argv : List[str] | None, optional
        Command-line arguments to parse, by default None (uses ``sys.argv``).

    Returns
    -------
    int
        Process exit code (0 on success, 1 if any receptor failed).
    '''

    args = build_arg_parser().parse_args(argv)

    print("[1/2] Enumerating receptor directories...")
    receptor_dirs = collect_receptor_dirs(args.database_dir)
    print(f"      {len(receptor_dirs)} receptors across {len(args.database_dir)} database roots.")

    print(f"[2/2] Computing pockets (radius {args.radius} A, {args.workers} workers)...")
    results: List[Tuple[str, str, str]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_compute_one, d, args.radius, args.overwrite) for d in receptor_dirs]
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i % 500 == 0 or i == len(futures):
                print(f"      {i}/{len(futures)} done", flush=True)

    with open(args.report, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["receptor_dir", "status", "reason"])
        writer.writerows(sorted(results))

    counts = {status: sum(1 for _, s, _ in results if s == status) for status in ("ok", "skipped", "failed")}
    print(f"\nDone. ok={counts['ok']} skipped={counts['skipped']} failed={counts['failed']}. See {args.report}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
