#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for non-executing Workbench launch plans.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

import pytest

from OCDocker.Workbench import RunManifest
from OCDocker.Workbench import build_launch_script
from OCDocker.Workbench import build_run_launch_plan
from OCDocker.Workbench import write_launch_script
from OCDocker.Workbench import write_model

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

# Functions
###############################################################################
## Private ##


def _write_launch_manifest(run_dir: Path, *, command: tuple[str, ...]) -> Path:
    '''Write a test run manifest for launch-plan tests.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run directory.
    command : tuple[str, ...]
        Manifest command.

    Returns
    -------
    pathlib.Path
        Written manifest path.
    '''

    run_dir.mkdir(parents=True)
    return write_model(
        run_dir / "run_manifest.yml",
        RunManifest(
            run_id="run-001",
            spec_type="ocscore_study",
            name="launch-study",
            workspace=Path("."),
            command=command,
        ),
    )


## Public ##


def test_build_run_launch_plan_resolves_bundle_paths(tmp_path) -> None:
    '''Launch plans expose commands, logs, PID files, and script paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "run-001"
    manifest_path = _write_launch_manifest(
        run_dir,
        command=("ocdocker", "ocscore", "train"),
    )

    plan = build_run_launch_plan(run_dir, log_dir="logs", script_path="run.sh")

    assert plan.manifest_path == manifest_path
    assert plan.cwd == run_dir
    assert plan.log_dir == run_dir / "logs"
    assert plan.stdout_log == run_dir / "logs" / "run-001.stdout.log"
    assert plan.stderr_log == run_dir / "logs" / "run-001.stderr.log"
    assert plan.pid_file == run_dir / "logs" / "run-001.pid"
    assert plan.script_path == run_dir / "run.sh"
    assert plan.shell_command == "ocdocker ocscore train"
    assert "nohup ocdocker ocscore train" in plan.background_command
    assert plan.script_written is False


def test_write_launch_script_writes_but_does_not_execute(tmp_path) -> None:
    '''Launch script writing creates a script without launching a process.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "run-001"
    _write_launch_manifest(run_dir, command=("ocdocker", "--help"))
    plan = build_run_launch_plan(run_dir, script_path="run.sh")

    updated_plan = write_launch_script(plan)
    script_text = plan.script_path.read_text(encoding="utf-8")

    assert updated_plan.script_written is True
    assert script_text == build_launch_script(plan)
    assert "set -euo pipefail" in script_text
    assert "ocdocker --help" in script_text
    assert not (run_dir / "logs" / "run-001.pid").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_launch_script(plan)

    overwritten = write_launch_script(plan, overwrite=True)
    assert overwritten.script_written is True


def test_build_run_launch_plan_rejects_empty_manifest_command(tmp_path) -> None:
    '''Launch plans require a command in the run manifest.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary test directory.
    '''

    run_dir = tmp_path / "run-001"
    _write_launch_manifest(run_dir, command=())

    with pytest.raises(ValueError, match="no command"):
        build_run_launch_plan(run_dir)
