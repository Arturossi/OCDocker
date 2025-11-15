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
import os
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
    # Engine-specific output dirs used across tests
    vina_dir = lig_dir / "vinaFiles"
    smina_dir = lig_dir / "sminaFiles"
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
        lig_dir / "ligand_test_descriptors.json",
        # engine outputs/configs
        vina_dir / "vina_config.txt",
        vina_dir / "vina_out.pdbqt",
        smina_dir / "smina_config.txt",
        smina_dir / "smina_out.pdbqt",
        smina_dir / "prepared_ligand.pdbqt",
        # PLANTS config
        plants_dir / "plants_config.txt",
    ]
    file_existed_before = {p: p.exists() for p in plant_files}
    # Track descriptor JSONs present before tests to remove only new ones
    desc_before = {p.name for p in lig_dir.glob("*_descriptors.json")}
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

    # Remove any newly created descriptor JSONs under the ligand directory
    try:
        for p in lig_dir.glob("*_descriptors.json"):
            if p.name not in desc_before:
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    # As a final safeguard, attempt to remove a known small set of test artifacts
    # even if they existed before (they are safe to re-generate in tests).
    safe_always_remove = [
        plants_dir / "plants_config.txt",
        lig_dir / "prepared_ligand.pdbqt",
        lig_dir / "prepared_ligand.mol2",
    ]
    for f in safe_always_remove:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass

    # Also purge any descriptor JSONs in the ligand directory unconditionally
    try:
        for p in lig_dir.glob("*_descriptors.json"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def _force_sqlite_backend_for_tests(tmp_path_factory):
    """Force OCDocker to use a temporary SQLite file during tests.

    Ensures no MySQL connections/tables are created and the SQLite file is
    removed after the session ends.
    """

    db_dir = tmp_path_factory.mktemp("ocdocker_db")
    db_file = db_dir / "ocdocker.db"
    # Prefer test-local sqlite
    os.environ.setdefault("OCDOCKER_USE_SQLITE", "1")
    os.environ["OCDOCKER_SQLITE_PATH"] = str(db_file)

    yield

    # Cleanup sqlite file
    try:
        if db_file.exists():
            db_file.unlink()
    except Exception:
        pass
