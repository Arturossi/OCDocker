#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for Initialise helpers and doc-build detection.

Usage:

pytest tests/test_Initialise.py
'''

# Imports
###############################################################################
import argparse
import ast
import inspect
import pytest
import sys

from pathlib import Path

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def load_is_doc_build():
    '''Load the is_doc_build function from Initialise.py for testing.
    
    SECURITY NOTE: This function uses exec(compile()) to extract a specific function
    from a known source file (Initialise.py) for testing purposes. The file path is
    constructed from the test file's location and never comes from user input.
    Only the 'is_doc_build' function is extracted and executed, making this safe.
    
    Returns
    -------
    callable
        The is_doc_build function from Initialise.py.
        
    Raises
    ------
    RuntimeError
        If the is_doc_build function is not found in Initialise.py.
    '''
    # Construct path to Initialise.py relative to this test file
    # This is a known, trusted path within the project structure
    path = Path(__file__).resolve().parents[2] / "OCDocker" / "Initialise.py"
    
    # Validate that the path is within the project directory (security check)
    project_root = Path(__file__).resolve().parents[2]
    if not str(path.resolve()).startswith(str(project_root.resolve())):
        raise RuntimeError(f"Security check failed: path {path} is outside project root")
    
    # Validate that the file exists and is a Python file
    if not path.exists() or path.suffix != '.py':
        raise RuntimeError(f"Expected Python file not found at {path}")
    
    source = path.read_text()
    tree = ast.parse(source)
    
    # Extract only the specific function we need (is_doc_build)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "is_doc_build":
            # Create a minimal module containing only this function
            mod = ast.Module([node], [])
            ast.fix_missing_locations(mod)
            # Execute in isolated namespace
            ns = {}
            # SECURITY: Only the extracted function is executed, not arbitrary code
            exec(compile(mod, filename=str(path), mode="exec"), ns)
            return ns["is_doc_build"]
    raise RuntimeError("is_doc_build not found")


is_doc_build = load_is_doc_build()



def _make_runtime_namespace(tmp_path, threads=None, tmp_dir=None):
    '''Make a minimal bootstrap namespace for scheduler resource tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    threads : int, optional
        Requested thread count, by default None.
    tmp_dir : str, optional
        Requested temporary directory, by default None.

    Returns
    -------
    argparse.Namespace
        Namespace accepted by ``OCDocker.Initialise.bootstrap``.
    '''

    return argparse.Namespace(
        update=False,
        config_file=str(tmp_path / "OCDocker.cfg"),
        output_level=2,
        overwrite=False,
        no_splash=True,
        multiprocess=True,
        threads=threads,
        tmp_dir=tmp_dir,
    )


def _make_runtime_config(tmp_path, tmp_dir=""):
    '''Make a minimal OCDockerConfig for lightweight bootstrap tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    tmp_dir : str, optional
        Temporary directory recorded in the config object, by default "".

    Returns
    -------
    OCDocker.Config.OCDockerConfig
        Minimal configuration object with required paths populated.
    '''

    from OCDocker.Config import OCDockerConfig

    config = OCDockerConfig()
    config.paths.ocdb_path = str(tmp_path / "ocdb")
    config.tmp_dir = str(tmp_dir)
    config.database.backend = "sqlite"
    config.database.sqlite_path = str(tmp_path / "ocdocker.sqlite")
    config.oddt_models_dir = str(tmp_path / "oddt_models")
    return config


def _bootstrap_with_config(monkeypatch, tmp_path, config, namespace):
    '''Bootstrap OCDocker with a mocked config loader.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary test directory.
    config : OCDocker.Config.OCDockerConfig
        Configuration returned by the mocked loader.
    namespace : argparse.Namespace
        Runtime namespace passed into bootstrap.

    Returns
    -------
    module
        Imported ``OCDocker.Initialise`` module.
    '''

    import OCDocker.Initialise as ocinit
    from OCDocker.Config import OCDockerConfig

    cfg_path = tmp_path / "OCDocker.cfg"
    cfg_path.write_text("[paths]\nocdb_path = test\n", encoding="utf-8")
    ocinit.reset_runtime(cleanup=False)
    monkeypatch.setenv("OCDOCKER_SKIP_ODDT", "1")
    monkeypatch.setenv("OCDOCKER_DB_BACKEND", "sqlite")
    monkeypatch.setattr(
        ocinit,
        "_resolve_config_file_path",
        lambda requested_config, include_package_locations=True: str(cfg_path),
    )
    monkeypatch.setattr(
        OCDockerConfig,
        "from_config_file",
        classmethod(lambda cls, config_file: config),
    )
    monkeypatch.setattr(ocinit, "print_description", lambda: None)
    monkeypatch.setattr(ocinit, "initialise_oddt_models", lambda *args, **kwargs: None)
    ocinit.bootstrap(namespace, init_db=False)
    return ocinit

## Public ##

@pytest.mark.order(1)
def test_is_doc_build_pytest_and_after_clearing(monkeypatch):
    # Should detect the pytest environment
    assert is_doc_build() is True

    with monkeypatch.context() as mp:
        # Remove doc/test related modules
        for name in ["pytest", "unittest", "doctest", "sphinx", "sphinx.ext.autodoc"]:
            mp.delitem(sys.modules, name, raising=False)
        # Empty call stack
        mp.setattr(inspect, "stack", lambda: [], raising=False)
        assert is_doc_build() is False

@pytest.mark.order(2)
def test_bootstrap_respects_scheduler_threads_and_tmp_dir(monkeypatch, tmp_path):
    '''Bootstrap should honor explicit scheduler resources.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    tmp_dir = tmp_path / "scheduler_tmp"
    tmp_dir.mkdir()
    sentinel = tmp_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    namespace = _make_runtime_namespace(tmp_path, threads=3, tmp_dir=str(tmp_dir))
    config = _make_runtime_config(tmp_path)

    ocinit = _bootstrap_with_config(monkeypatch, tmp_path, config, namespace)

    from OCDocker.Config import get_config

    runtime = get_config()
    assert runtime.available_cores == 3
    assert runtime.multiprocess is True
    assert runtime.tmp_dir == str(tmp_dir)
    assert sentinel.exists()
    ocinit.reset_runtime(cleanup=False)


@pytest.mark.order(3)
def test_bootstrap_uses_snakemake_threads_and_preserves_config_tmp(monkeypatch, tmp_path):
    '''Bootstrap should use Snakemake thread env and preserve configured tmp.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    tmp_dir = tmp_path / "config_tmp"
    tmp_dir.mkdir()
    sentinel = tmp_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    namespace = _make_runtime_namespace(tmp_path)
    config = _make_runtime_config(tmp_path, tmp_dir=str(tmp_dir))
    monkeypatch.setenv("SNAKEMAKE_THREADS", "5")

    ocinit = _bootstrap_with_config(monkeypatch, tmp_path, config, namespace)

    from OCDocker.Config import get_config

    runtime = get_config()
    assert runtime.available_cores == 5
    assert runtime.multiprocess is True
    assert runtime.tmp_dir == str(tmp_dir)
    assert sentinel.exists()
    ocinit.reset_runtime(cleanup=False)


@pytest.mark.order(4)
def test_bootstrap_clamps_invalid_scheduler_threads(monkeypatch, tmp_path):
    '''Bootstrap should clamp non-positive scheduler thread counts.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Monkeypatch fixture.
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    namespace = _make_runtime_namespace(tmp_path, threads=0)
    config = _make_runtime_config(tmp_path, tmp_dir=str(tmp_path / "tmp"))

    ocinit = _bootstrap_with_config(monkeypatch, tmp_path, config, namespace)

    from OCDocker.Config import get_config

    runtime = get_config()
    assert runtime.available_cores == 1
    assert runtime.multiprocess is False
    ocinit.reset_runtime(cleanup=False)

