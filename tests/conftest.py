#!/usr/bin/env python3

# Description
###############################################################################
'''
Pytest configuration and shared fixtures for OCDocker tests.

Usage:

pytest tests
'''

# Imports
###############################################################################
import os
import pytest
import shutil

from pathlib import Path
from typing import Optional, Set

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    '''Clean up test files before and after test runs to ensure clean state.
    
    This fixture runs automatically before and after all tests and cleans up:
    - Generated output files from previous test runs
    - Temporary directories
    - Log files
    - Config files
    - Descriptor JSON files
    '''

    # Get the project root (assuming tests are in tests/ directory)
    project_root = Path(__file__).resolve().parent.parent
    test_files_dir = project_root / "test_files"
    
    # List of patterns/directories to clean
    # Note: We exclude box files and input files since they are test fixtures
    cleanup_patterns = [
        "**/*.pdbqt",
        "**/*.mol2",
        "**/*plantsFiles*",
        "**/*plants_config.txt",
        "**/*vina_config.txt",
        "**/*smina_config.txt",
        "**/*.log",
        "**/run/",
        "**/*_descriptors.json",
        "**/*_tmp.mol",
    ]
    
    # Files/directories to explicitly exclude from cleanup (test fixtures)
    exclude_patterns = [
        "**/boxes/",
        "**/ligand.smi",
        "**/receptor.pdb",
        "**/receptor.cif",
    ]
    
    # Helper function to check if path should be excluded
    def should_exclude(path: Path) -> bool:
        '''Check if a path matches any exclusion pattern.
        
        Parameters
        ----------
        path : Path
            The file or directory path to check.

        Returns
        -------
        bool
            True if the path should be excluded from cleanup, False otherwise.
        '''

        path_str = str(path)
        # Check if path is within any boxes directory
        if "/boxes/" in path_str or path_str.endswith("/boxes") or path.name == "boxes":
            return True
        # Check if path is a box file
        if path.name.startswith("box") and path.suffix == ".pdb":
            return True
        # Check other exclusion patterns
        for exclude_pattern in exclude_patterns:
            # Use Path.match to respect glob semantics and avoid substring matches
            try:
                if path.match(exclude_pattern) or path.name == exclude_pattern:
                    return True
            except (TypeError, ValueError):
                # If pattern is malformed, fall back to name match only
                if path.name == exclude_pattern:
                    return True
        return False
    
    # Clean up files matching patterns BEFORE tests
    if test_files_dir.exists():
        for pattern in cleanup_patterns:
            for path in test_files_dir.glob(pattern):
                # Skip if path matches exclusion patterns
                if should_exclude(path):
                    continue
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                except (OSError, PermissionError, FileNotFoundError):
                    # Ignore errors during cleanup
                    pass
    
    # Yield control to tests
    yield
    
    # Clean up files matching patterns AFTER tests
    if test_files_dir.exists():
        for pattern in cleanup_patterns:
            for path in test_files_dir.glob(pattern):
                # Skip if path matches exclusion patterns
                if should_exclude(path):
                    continue
                try:
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                except (OSError, PermissionError, FileNotFoundError):
                    # Ignore errors during cleanup
                    pass


@pytest.fixture(autouse=True)
def ensure_clean_test_state(tmp_path):
    '''Ensure each test starts with a clean temporary directory.
    
    This fixture automatically runs before each test and ensures
    that the tmp_path is clean and ready for use.
    '''
    
    # Clear any existing files in tmp_path
    if tmp_path.exists():
        for item in tmp_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except (OSError, PermissionError):
                pass
    
    yield tmp_path


def _exe_available(path: Optional[str]) -> bool:
    if not path:
        return False
    # Expand user (~) and env vars for configured paths
    path = os.path.expandvars(os.path.expanduser(str(path))).strip()
    if not path:
        return False
    # Allow command strings with args by checking the first token
    try:
        import shlex
        parts = shlex.split(path)
        path = parts[0] if parts else path
    except ValueError:
        # Fallback to raw string if parsing fails
        pass
    if os.path.isabs(path):
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil.which(path) is not None


def _missing_external_tools() -> Set[str]:
    # Try to resolve tool paths from config, fallback to common names
    try:
        from OCDocker.Config import get_config, OCDockerConfig
        cfg = get_config()
        # Prefer explicit config file if present (tests may run before bootstrap)
        config_file = os.getenv("OCDOCKER_CONFIG", "")
        if not config_file:
            project_root = Path(__file__).resolve().parent.parent
            candidate = project_root / "OCDocker.cfg"
            if candidate.is_file():
                config_file = str(candidate)
        if config_file and os.path.isfile(config_file):
            try:
                cfg = OCDockerConfig.from_config_file(config_file)
            except Exception:
                # Keep existing cfg if loading fails
                pass
        tools = {
            "vina": (getattr(cfg.vina, "executable", "") or "vina"),
            "smina": (getattr(cfg.smina, "executable", "") or "smina"),
            "plants": (getattr(cfg.plants, "executable", "") or "plants"),
            "prepare_ligand4": (getattr(cfg.tools, "prepare_ligand", "") or "prepare_ligand4.py"),
            "prepare_receptor4": (getattr(cfg.tools, "prepare_receptor", "") or "prepare_receptor4.py"),
            "obabel": (getattr(cfg.tools, "obabel", "") or "obabel"),
        }
    except Exception:
        tools = {
            "vina": "vina",
            "smina": "smina",
            "plants": "plants",
            "prepare_ligand4": "prepare_ligand4.py",
            "prepare_receptor4": "prepare_receptor4.py",
            "obabel": "obabel",
        }

    missing = {name for name, exe in tools.items() if not _exe_available(exe)}
    if os.getenv("OCDOCKER_DEBUG_EXTERNAL", "").lower() in ("1", "true", "yes"):
        for name, exe in tools.items():
            status = "OK" if name not in missing else "MISSING"
            print(f"[external-check] {name}: {exe} => {status}")

    # DSSP (used for surface AA counts in Receptor)
    try:
        from OCDocker.Config import get_config, OCDockerConfig
        cfg = get_config()
        config_file = os.getenv("OCDOCKER_CONFIG", "")
        if not config_file:
            project_root = Path(__file__).resolve().parent.parent
            candidate = project_root / "OCDocker.cfg"
            if candidate.is_file():
                config_file = str(candidate)
        if config_file and os.path.isfile(config_file):
            try:
                cfg = OCDockerConfig.from_config_file(config_file)
            except Exception:
                pass
        dssp_candidate = getattr(cfg.tools, "dssp", "dssp")
    except Exception:
        dssp_candidate = "dssp"
    dssp_candidates = [dssp_candidate, "mkdssp", "dssp"]
    if not any(_exe_available(c) for c in dssp_candidates if c):
        missing.add("dssp")

    # Python OpenBabel bindings (used by Receptor/Conversion)
    try:
        import openbabel  # type: ignore
        _ = openbabel
    except Exception:
        missing.add("openbabel-py")

    return missing


def pytest_collection_modifyitems(config, items):
    # Allow forcing external tests on (e.g., local dev with binaries installed)
    if os.getenv("OCDOCKER_FORCE_EXTERNAL_TESTS", "").lower() in ("1", "true", "yes"):
        return

    missing = _missing_external_tools()
    if not missing:
        return

    skip_external = pytest.mark.skip(
        reason=f"Missing external tools/binaries: {', '.join(sorted(missing))}"
    )

    # Map test modules to required tools
    required_by_file = {
        "test_vina_prepare.py": {"prepare_ligand4", "prepare_receptor4"},
        "test_Smina.py": {"smina", "prepare_ligand4", "prepare_receptor4", "openbabel-py", "dssp"},
        "test_Smina_utils.py": {"dssp"},
        "test_PLANTS.py": {"plants", "obabel", "openbabel-py", "dssp"},
        "test_Vina.py": {"vina", "prepare_ligand4", "prepare_receptor4", "openbabel-py", "dssp"},
        "test_preparation_strategy.py": {"dssp"},
        "test_integration_docking_workflow.py": {"dssp"},
        "test_plants_prepare.py": {"plants", "obabel"},
        "test_Receptor.py": {"openbabel-py", "dssp"},
    }

    for item in items:
        fpath = str(item.fspath)
        for filename, required in required_by_file.items():
            if fpath.endswith(filename) and (missing & required):
                item.add_marker(skip_external)
                break


def pytest_configure(config):
    '''Configure pytest hooks for test file cleanup.'''
    pass
