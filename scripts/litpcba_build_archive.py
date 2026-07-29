#!/usr/bin/env python3

# Description
###############################################################################
'''
Converts the output of ``scripts/litpcba_validation_subset.py`` into the raw
archive layout OCDocker's ``Prepare``/``Dock`` pipeline expects for a new
archive type (see ``OCDocker/DB/LITPCBA.py``, mirroring how DUDEz/PDBbind are
laid out under ``config.paths.ocdb_path``).

For each kept target directory in the subset (``<target>/actives.smi``,
``inactives_sampled.smi``, ``receptor_protein.mol2``, ``receptor_ligand.mol2``),
this writes:

    <archive-dir>/<target>/receptor.pdb              (converted from receptor_protein.mol2 via obabel)
    <archive-dir>/<target>/reference_ligand.mol2      (copied from receptor_ligand.mol2)
    <archive-dir>/<target>/reference_ligand.pdb       (converted from the .mol2 via obabel --
                                                        OCDockerPipeline's box-generation only
                                                        recognizes .pdb/.sdf, not .mol2)
    <archive-dir>/<target>/compounds/ligands/<id>.smi (one file per active)
    <archive-dir>/<target>/compounds/decoys/<id>.smi  (one file per sampled inactive)

The flat per-compound .smi files are the raw-input form Prepare.py already
knows how to organize into per-molecule folders (see
``OCDocker.Processing.Preprocessing.Prepare.__sub_core_prepare``), matching
how the existing DUDEz raw archive is laid out.

Usage:

    python scripts/litpcba_build_archive.py \\
        --subset-dir /path/to/litpcba_validation_subset \\
        --archive-dir /path/to/ocdb2/LITPCBA
'''

# Imports
###############################################################################
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from glob import glob
from typing import List, Optional

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
_NON_TARGET_ENTRIES = {"manifest.json", "manifest.csv", "dropped_targets.tsv", "excluded_structures.tsv", "_work"}

# Functions
###############################################################################
## Private ##

def _write_compound_files(smi_path: str, out_dir: str) -> int:
    '''Split a whitespace-delimited SMILES-list file into one per-compound
    subfolder holding a ``ligand.smi`` (named by its ID, the second column).

    Pre-organized subfolders are used rather than flat ``<id>.smi`` files:
    Prepare.py's flat-file auto-organize path builds its glob pattern as
    ``*.{ligandExt}`` where ``ligandExt`` already includes a leading dot,
    producing a double-dot pattern (``*..smi``) that matches nothing. The
    existing DUDEz raw archive sidesteps this by shipping pre-organized
    subfolders directly, so this does the same rather than depending on that
    dead code path.

    Parameters
    ----------
    smi_path : str
        Path to the whitespace-delimited SMILES-list file (``<smiles> <id> ...``).
    out_dir : str
        Directory to create one ``<id>/ligand.smi`` subfolder per compound in.

    Returns
    -------
    int
        Number of compounds written.
    '''

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(smi_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            smiles, compound_id = parts[0], parts[1]
            compound_dir = os.path.join(out_dir, compound_id)
            os.makedirs(compound_dir, exist_ok=True)
            with open(os.path.join(compound_dir, "ligand.smi"), "w") as out:
                out.write(f"{smiles}\t{compound_id}\n")
            n += 1
    return n


def _convert_mol2_to_pdb(mol2_path: str, pdb_path: str, obabel_bin: str) -> None:
    '''Convert a mol2 file to PDB via obabel, raising RuntimeError on failure.

    Parameters
    ----------
    mol2_path : str
        Path to the source mol2 file.
    pdb_path : str
        Path of the PDB file to write.
    obabel_bin : str
        Path to the obabel binary.

    Raises
    ------
    RuntimeError
        If obabel exits with a non-zero return code or the output file wasn't created.
    '''

    proc = subprocess.run(
        [obabel_bin, mol2_path, "-O", pdb_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.isfile(pdb_path):
        raise RuntimeError(f"obabel failed converting {mol2_path} -> {pdb_path}\n{proc.stderr}")


## Public ##

def list_target_dirs(subset_dir: str) -> List[str]:
    '''List kept target subdirectories under a subset directory, excluding manifest/work entries.

    Parameters
    ----------
    subset_dir : str
        Output directory of ``scripts/litpcba_validation_subset.py``.

    Returns
    -------
    List[str]
        Names of the kept target subdirectories.
    '''

    targets = []
    for entry in sorted(os.listdir(subset_dir)):
        if entry in _NON_TARGET_ENTRIES or entry.startswith("."):
            continue
        full = os.path.join(subset_dir, entry)
        if os.path.isdir(full):
            targets.append(entry)
    return targets


def build_target(subset_dir: str, archive_dir: str, target: str, obabel_bin: str, overwrite: bool) -> dict:
    '''Materialize one target's raw archive layout and return its compound counts.

    Parameters
    ----------
    subset_dir : str
        Output directory of ``scripts/litpcba_validation_subset.py``.
    archive_dir : str
        Destination archive directory (e.g. ``<ocdb_path>/LITPCBA``).
    target : str
        Name of the target subdirectory to build.
    obabel_bin : str
        Path to the obabel binary.
    overwrite : bool
        If True, rebuild ``receptor.pdb``/``reference_ligand.mol2`` even if
        already present.

    Returns
    -------
    dict
        ``{"target": target, "n_actives": int, "n_inactives": int}``.
    '''

    src = os.path.join(subset_dir, target)
    dst = os.path.join(archive_dir, target)
    os.makedirs(dst, exist_ok=True)

    receptor_pdb = os.path.join(dst, "receptor.pdb")
    if overwrite or not os.path.isfile(receptor_pdb):
        _convert_mol2_to_pdb(os.path.join(src, "receptor_protein.mol2"), receptor_pdb, obabel_bin)

    reference_ligand_mol2 = os.path.join(dst, "reference_ligand.mol2")
    if overwrite or not os.path.isfile(reference_ligand_mol2):
        shutil.copyfile(os.path.join(src, "receptor_ligand.mol2"), reference_ligand_mol2)

    # OCDockerPipeline's snakefile (_REFERENCE_LIGAND_FILENAMES) only looks
    # for reference_ligand.pdb/.sdf when auto-generating boxes/box0.pdb, not
    # .mol2 -- convert one so box generation doesn't silently fail for every
    # ligand under this target.
    reference_ligand_pdb = os.path.join(dst, "reference_ligand.pdb")
    if overwrite or not os.path.isfile(reference_ligand_pdb):
        _convert_mol2_to_pdb(reference_ligand_mol2, reference_ligand_pdb, obabel_bin)

    n_actives = _write_compound_files(os.path.join(src, "actives.smi"), os.path.join(dst, "compounds", "ligands"))
    n_inactives = _write_compound_files(os.path.join(src, "inactives_sampled.smi"), os.path.join(dst, "compounds", "decoys"))

    return {"target": target, "n_actives": n_actives, "n_inactives": n_inactives}


def build_arg_parser() -> argparse.ArgumentParser:
    '''Build the CLI argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    '''

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset-dir", required=True, help="Output directory of scripts/litpcba_validation_subset.py.")
    parser.add_argument("--archive-dir", required=True, help="Destination archive directory (e.g. <ocdb_path>/LITPCBA).")
    parser.add_argument("--targets", nargs="*", default=None, help="Only build these targets (default: all kept targets in --subset-dir).")
    parser.add_argument("--obabel-bin", default="obabel", help="Path to the obabel binary. Default 'obabel' (resolved via PATH).")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild receptor.pdb/reference_ligand.mol2 even if already present.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    '''Build the raw LIT-PCBA archive for every kept target (or --targets subset).

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

    if shutil.which(args.obabel_bin) is None:
        raise SystemExit(f"obabel binary '{args.obabel_bin}' not found on PATH.")

    targets = args.targets if args.targets else list_target_dirs(args.subset_dir)
    if not targets:
        raise SystemExit(f"No target directories found under {args.subset_dir}.")

    os.makedirs(args.archive_dir, exist_ok=True)
    total_actives = total_inactives = 0
    for target in targets:
        result = build_target(args.subset_dir, args.archive_dir, target, args.obabel_bin, args.overwrite)
        total_actives += result["n_actives"]
        total_inactives += result["n_inactives"]
        print(f"{target}: {result['n_actives']} actives, {result['n_inactives']} inactives -> {args.archive_dir}/{target}")

    print(f"\nDone. {len(targets)} targets, {total_actives} actives, {total_inactives} inactives "
          f"written to {args.archive_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
