#!/usr/bin/env python3

# Description
###############################################################################
'''
Pytest configuration: session-level cleanup of common, top-level artifacts
that may be produced by tests when run from the repo root.

We only remove directories that did not exist before the test session and are
known to be ephemeral (e.g., plots/, csvs/). This avoids deleting any user
data that predates the test run.
'''

# Imports
###############################################################################

from __future__ import annotations
from pathlib import Path
import shutil
import pytest

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################

@pytest.fixture(scope="session", autouse=True)
def _cleanup_generated_top_level_dirs():
    '''Record selected top-level dirs before tests and remove if newly created.'''

    repo_root = Path(__file__).resolve().parent.parent

    # Common top-level artifacts
    candidates = [
        repo_root / "plots",
        repo_root / "csvs",
        repo_root / "ocdocker_out",
    ]
    existed_before = {p: p.exists() for p in candidates}

    # Docking artifacts under test_files (created by PLANTS tests)
    tf = repo_root / "test_files" / "test_ptn1"
    lig_dir = tf / "compounds" / "ligands" / "ligand"
    plants_dir = lig_dir / "plantsFiles"
    plant_files = [
        # prepared structures (mol2/pdbqt)
        tf / "prepared_receptor.mol2",
        tf / "prepared_receptor.pdbqt",
        lig_dir / "prepared_ligand.mol2",
        lig_dir / "prepared_ligand.pdbqt",
        lig_dir / "ligand.mol2",
        lig_dir / "ligand_tmp.mol",
        # descriptors
        tf / "test_receptor_descriptors.json",
        lig_dir / "test_ligand_descriptors.json",
        # engine outputs/configs
        vina_dir / "vina_config.txt",
        vina_dir / "vina_out.pdbqt",
        smina_dir / "smina_config.txt",
        smina_dir / "smina_out.pdbqt",
    ]
    file_existed_before = {p: p.exists() for p in plant_files}
    dir_existed_before = {
        plants_dir: plants_dir.exists(),
        vina_dir: vina_dir.exists(),
        smina_dir: smina_dir.exists(),
    }

    yield

    # Remove top-level dirs that were created by this test session
    for p, existed in existed_before.items():
        if not existed and p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    # Remove engine-specific dirs if created during tests
    for d, existed in dir_existed_before.items():
        if not existed and d.exists():
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

    # Remove generated docking files if they did not exist before
    for f, existed in file_existed_before.items():
        if not existed and f.exists():
            try:
                f.unlink()
            except Exception:
                pass
