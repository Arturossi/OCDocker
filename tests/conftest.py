"""Pytest configuration and shared fixtures for OCDocker tests.

This module provides:
- Automatic cleanup of test files before and after test runs
- Shared fixtures used across multiple test modules
"""

import os
import shutil
from pathlib import Path
import pytest

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
'''
###############################################################################

def pytest_configure(config):
    '''Configure pytest hooks for test file cleanup.'''
    pass


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
            # Convert glob pattern to check
            if "**" in exclude_pattern:
                pattern = exclude_pattern.replace("**", "")
                if pattern.startswith("/"):
                    pattern = pattern[1:]
                if pattern in path_str or path_str.endswith(pattern):
                    return True
            elif path_str.endswith(exclude_pattern) or path.name == exclude_pattern:
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
